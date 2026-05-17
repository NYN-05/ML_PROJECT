"""Simple API server for model predictions."""

import json
from flask import Flask, request, jsonify
from inference import get_inference, predict, predict_ensemble

app = Flask(__name__)


@app.route('/predict', methods=['POST'])
def predict_handler():
    """Handle prediction requests."""
    data = request.get_json()
    
    if not data:
        return jsonify({"error": "No data provided"}), 400
    
    text = data.get('text', '')
    title = data.get('title', '')
    
    if not text and not title:
        return jsonify({"error": "No text provided"}), 400
    
    # Combine title and text
    full_text = f"{title} {text}".strip()
    
    # Get model_name if provided
    model_name = data.get('model_name')
    use_ensemble = data.get('ensemble', False)
    
    try:
        if use_ensemble:
            result = predict_ensemble(full_text)
        else:
            result = predict(full_text, model_name)
            
        return jsonify(result)
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/models', methods=['GET'])
def models_handler():
    """Get information about available models."""
    try:
        inference = get_inference()
        return jsonify(inference.get_info())
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/health', methods=['GET'])
def health_handler():
    """Health check."""
    return jsonify({"status": "ok", "models_loaded": len(get_inference().models)})


if __name__ == '__main__':
    print("Starting ML API server...")
    print("Loading models...")
    
    inference = get_inference()
    info = inference.get_info()
    print(f"Loaded {info['total_models']} models")
    print(f"Best model: {info['best_model']} (weight: {info['best_weight']})")
    
    print("\nServer starting on http://localhost:8001")
    app.run(host='0.0.0.0', port=8001, debug=False)