"""FakeBERT - BERT-based fake news detection model."""

from __future__ import annotations

import torch
import torch.nn as nn
from transformers import BertModel, BertTokenizer
from typing import Optional


class FakeBERT(nn.Module):
    """FakeBERT model for fake news detection using pre-trained BERT.
    
    Architecture:
    - Pre-trained BERT base model (bert-base-uncased)
    - Dropout layer for regularization
    - Classification head
    """
    
    def __init__(
        self,
        model_name: str = "bert-base-uncased",
        num_classes: int = 2,
        dropout: float = 0.3,
        freeze_bert: bool = False
    ):
        super(FakeBERT, self).__init__()
        
        self.bert = BertModel.from_pretrained(model_name)
        self.num_classes = num_classes
        
        # Freeze BERT layers if specified (for faster training)
        if freeze_bert:
            for param in self.bert.parameters():
                param.requires_grad = False
        
        # Classification head
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Sequential(
            nn.Linear(768, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, num_classes)
        )
        
        # Label mapping
        self.label2idx = {"FAKE": 0, "REAL": 1}
        self.idx2label = {0: "FAKE", 1: "REAL"}
    
    def forward(self, input_ids, attention_mask=None, token_type_ids=None):
        """Forward pass through the model."""
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids
        )
        
        # Use [CLS] token representation for classification
        pooled_output = outputs.pooler_output
        dropped = self.dropout(pooled_output)
        logits = self.classifier(dropped)
        
        return logits
    
    def predict(self, input_ids, attention_mask=None):
        """Get predictions and probabilities."""
        self.eval()
        with torch.no_grad():
            logits = self.forward(input_ids, attention_mask)
            probs = torch.softmax(logits, dim=1)
            preds = torch.argmax(probs, dim=1)
        
        return preds, probs


class FakeBERTClassifier:
    """Wrapper class for FakeBERT model with training and inference support."""
    
    def __init__(
        self,
        model_name: str = "bert-base-uncased",
        max_length: int = 256,
        dropout: float = 0.3,
        learning_rate: float = 2e-5,
        freeze_bert: bool = False
    ):
        self.model_name = model_name
        self.max_length = max_length
        self.dropout = dropout
        self.learning_rate = learning_rate
        self.freeze_bert = freeze_bert
        
        self.tokenizer: Optional[BertTokenizer] = None
        self.model: Optional[FakeBERT] = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    def initialize(self):
        """Initialize tokenizer and model."""
        self.tokenizer = BertTokenizer.from_pretrained(self.model_name)
        self.model = FakeBERT(
            model_name=self.model_name,
            dropout=self.dropout,
            freeze_bert=self.freeze_bert
        ).to(self.device)
    
    def prepare_inputs(self, texts: list[str]) -> dict:
        """Tokenize texts for BERT input."""
        encoding = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt"
        )
        
        return {
            "input_ids": encoding["input_ids"].to(self.device),
            "attention_mask": encoding["attention_mask"].to(self.device)
        }
    
    def train(
        self,
        train_texts: list[str],
        train_labels: list[str],
        val_texts: list[str],
        val_labels: list[str],
        epochs: int = 3,
        batch_size: int = 16,
        save_path: Optional[str] = None
    ) -> dict:
        """Train the FakeBERT model."""
        from torch.utils.data import DataLoader, TensorDataset
        
        # Prepare datasets
        train_inputs = self.prepare_inputs(train_texts)
        val_inputs = self.prepare_inputs(val_texts)
        
        train_labels_tensor = torch.tensor(
            [self.model.label2idx[l] for l in train_labels],
            dtype=torch.long
        ).to(self.device)
        
        val_labels_tensor = torch.tensor(
            [self.model.label2idx[l] for l in val_labels],
            dtype=torch.long
        ).to(self.device)
        
        train_dataset = TensorDataset(
            train_inputs["input_ids"],
            train_inputs["attention_mask"],
            train_labels_tensor
        )
        
        val_dataset = TensorDataset(
            val_inputs["input_ids"],
            val_inputs["attention_mask"],
            val_labels_tensor
        )
        
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size)
        
        # Optimizer
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.learning_rate)
        criterion = nn.CrossEntropyLoss()
        
        best_val_f1 = 0
        best_model_state = None
        history = {"train_loss": [], "val_loss": [], "val_f1": []}
        
        for epoch in range(epochs):
            # Training
            self.model.train()
            train_loss = 0
            
            for batch in train_loader:
                input_ids, attention_mask, labels = batch
                
                optimizer.zero_grad()
                logits = self.model(input_ids, attention_mask)
                loss = criterion(logits, labels)
                
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item()
            
            avg_train_loss = train_loss / len(train_loader)
            history["train_loss"].append(avg_train_loss)
            
            # Validation
            self.model.eval()
            val_loss = 0
            all_preds = []
            all_labels = []
            
            with torch.no_grad():
                for batch in val_loader:
                    input_ids, attention_mask, labels = batch
                    
                    logits = self.model(input_ids, attention_mask)
                    loss = criterion(logits, labels)
                    
                    val_loss += loss.item()
                    preds = torch.argmax(logits, dim=1)
                    
                    all_preds.extend(preds.cpu().numpy())
                    all_labels.extend(labels.cpu().numpy())
            
            avg_val_loss = val_loss / len(val_loader)
            val_f1 = f1_score(all_labels, all_preds, average="weighted")
            
            history["val_loss"].append(avg_val_loss)
            history["val_f1"].append(val_f1)
            
            print(f"Epoch {epoch+1}/{epochs} - Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}, Val F1: {val_f1:.4f}")
            
            # Save best model
            if val_f1 > best_val_f1:
                best_val_f1 = val_f1
                best_model_state = self.model.state_dict().copy()
        
        # Load best model
        if best_model_state is not None:
            self.model.load_state_dict(best_model_state)
        
        # Save if path provided
        if save_path:
            self.save(save_path)
        
        return {
            "best_val_f1": best_val_f1,
            "history": history
        }
    
    def predict(self, texts: list[str]) -> tuple[list[str], list[float]]:
        """Predict on new texts."""
        self.model.eval()
        
        inputs = self.prepare_inputs(texts)
        
        with torch.no_grad():
            logits = self.model(inputs["input_ids"], inputs["attention_mask"])
            probs = torch.softmax(logits, dim=1)
            preds = torch.argmax(probs, dim=1)
        
        # Convert to labels with confidence
        labels = [self.model.idx2label[p.item()] for p in preds]
        confidences = [probs[i][p.item()].item() for i, p in enumerate(preds)]
        
        return labels, confidences
    
    def save(self, path: str):
        """Save model and tokenizer."""
        import os
        os.makedirs(path, exist_ok=True)
        
        # Save model
        torch.save({
            "model_state_dict": self.model.state_dict(),
            "model_name": self.model_name,
            "max_length": self.max_length,
            "dropout": self.dropout,
            "label2idx": self.model.label2idx,
            "idx2label": self.model.idx2label
        }, f"{path}/model.pt")
        
        # Save tokenizer
        self.tokenizer.save_pretrained(f"{path}/tokenizer")
    
    def load(self, path: str):
        """Load model and tokenizer."""
        # Load tokenizer
        self.tokenizer = BertTokenizer.from_pretrained(f"{path}/tokenizer")
        
        # Load model
        checkpoint = torch.load(f"{path}/model.pt", map_location=self.device)
        
        self.model = FakeBERT(
            model_name=checkpoint["model_name"],
            dropout=checkpoint["dropout"]
        ).to(self.device)
        
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.model.label2idx = checkpoint["label2idx"]
        self.model.idx2label = checkpoint["idx2label"]


if __name__ == "__main__":
    # Quick test
    classifier = FakeBERTClassifier(max_length=128)
    classifier.initialize()
    
    # Test with sample data
    texts = [
        "This is real news about a real event",
        "This is fake news claiming false information"
    ]
    labels = ["REAL", "FAKE"]
    
    preds, confs = classifier.predict(texts)
    print(f"Predictions: {preds}")
    print(f"Confidences: {confs}")