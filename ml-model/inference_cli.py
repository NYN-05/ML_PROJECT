"""CLI for inference - DistilBERT-based fake news detection."""

import argparse
import json
import sys
import time
from pathlib import Path

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent))

from distilbert_inference import predict as distilbert_predict, get_distilbert_inference


def main():
    parser = argparse.ArgumentParser(description='Fake News Detection CLI (DistilBERT)')
    parser.add_argument('--text', type=str, required=True, help='Text to analyze')
    parser.add_argument('--threshold', type=float, default=0.5, help='Threshold for FAKE/REAL')
    
    args = parser.parse_args()
    
    try:
        inf = get_distilbert_inference()
        
        if inf.bert is None:
            print(json.dumps({'error': 'DistilBERT model not found. Train with train_fakebert.py first.'}))
            sys.exit(1)
        
        start = time.time()
        result_dict = distilbert_predict(args.text, args.threshold)
        prediction = result_dict["prediction"]
        confidence = result_dict["confidence"]
        elapsed = (time.time() - start) * 1000
        
        result = {
            'prediction': prediction,
            'confidence': round(confidence, 4),
            'method': 'distilbert',
            'threshold': args.threshold,
            'inference_time_ms': round(elapsed, 2),
            'model_info': inf.get_info()
        }
        
        print(json.dumps(result))
        
    except Exception as e:
        print(json.dumps({'error': str(e)}))
        sys.exit(1)


if __name__ == '__main__':
    main()