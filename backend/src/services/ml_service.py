"""Direct analysis service that uses trained models without HTTP."""

import sys
from pathlib import Path

# Add ml-model to path
ml_model_path = Path(__file__).parent.parent / "ml-model"
if str(ml_model_path) not in sys.path:
    sys.path.insert(0, str(ml_model_path))

from inference import predict, predict_ensemble, get_inference


def analyze(text: str, model_name: str = None, use_ensemble: bool = False) -> dict:
    """
    Analyze text for fake news detection.
    
    Args:
        text: Text to analyze
        model_name: Specific model to use (optional)
        use_ensemble: Use ensemble of all models (default: False)
    
    Returns:
        dict with prediction, confidence, model info
    """
    try:
        if use_ensemble:
            result = predict_ensemble(text)
        else:
            result = predict(text, model_name)
            
        # Format response similar to old API
        prediction = result.get('prediction', 'UNKNOWN')
        confidence = result.get('confidence', 0.0)
        
        return {
            'success': True,
            'prediction': prediction,
            'confidence': confidence,
            'confidenceLabel': _to_confidence_label(confidence),
            'riskLevel': _to_risk_level(prediction),
            'model_used': result.get('model_used', 'unknown'),
            'method': result.get('method', 'single'),
            'available_models': result.get('available_models', []),
            'weights': result.get('weights', {})
        }
        
    except Exception as e:
        return {
            'success': False,
            'error': str(e),
            'prediction': 'ERROR',
            'confidence': 0.0
        }


def _to_confidence_label(confidence: float) -> str:
    """Convert confidence to label."""
    if confidence >= 0.8:
        return 'HIGH'
    elif confidence >= 0.5:
        return 'MEDIUM'
    else:
        return 'LOW'


def _to_risk_level(prediction: str) -> str:
    """Convert prediction to risk level."""
    if prediction == 'FAKE':
        return 'DANGER'
    elif prediction == 'REAL':
        return 'SAFE'
    else:
        return 'UNKNOWN'


def get_models_info():
    """Get information about available models."""
    return get_inference().get_info()


# Initialize on import
print("Loading ML models...")
info = get_models_info()
print(f"Loaded {info['total_models']} models")
print(f"Best model: {info['best_model']}")