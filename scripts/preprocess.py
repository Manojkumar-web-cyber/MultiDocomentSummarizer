import argparse
import yaml
import sys
sys.path.append('.')
from src.preprocessor import DocumentPreprocessor

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='config.yaml')
    args = parser.parse_args()
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    pre = DocumentPreprocessor(config)
    pre.process_file('data/cnn_train.jsonl', 'data/cnn_train_preprocessed.jsonl')
    pre.process_file('data/cnn_val.jsonl', 'data/cnn_val_preprocessed.jsonl')
    pre.process_file('data/cnn_test.jsonl', 'data/cnn_test_preprocessed.jsonl')
    print("✓ All preprocessing done.")

if __name__ == '__main__':
    main()
