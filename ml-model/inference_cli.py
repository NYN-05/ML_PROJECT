"""CLI for inference - can be called from Node.js."""

import argparse
import json
import sys
from pathlib import Path

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent))

from inference import predict, predict_ensemble, get_inference


def main():
    parser = argparse.ArgumentParser(description='Fake News Detection CLI')
    parser.add_argument('--text', type=str, required=True, help='Text to analyze')
    parser.add_argument('--model', type=str, help='Specific model name (optional, overrides ensemble)')
    parser.add_argument('--ensemble', action='store_true', default=True, help='Use weighted ensemble (default: enabled)')
    parser.add_argument('--single', action='store_true', help='Use single best model instead of ensemble')
    parser.add_argument('--threshold', type=float, default=0.8, help='Threshold for FAKE/REAL (higher = less FAKE predictions)')
    
    args = parser.parse_args()
    
    # Load models if not already loaded
    inference = get_inference()
    
    try:
        if args.single or args.model:
            # Single model mode
            result = predict(args.text, args.model, threshold=args.threshold)
        else:
            # Default: ensemble mode with all models
            result = predict_ensemble(args.text, threshold=args.threshold)
            
        print(json.dumps(result))
        
    except Exception as e:
        print(json.dumps({'error': str(e)}))
        sys.exit(1)


if __name__ == '__main__':
    main()