import argparse
import json
import os
from datasets import load_dataset
from tqdm import tqdm

def download_and_convert(num_train=20000, num_val=5000, num_test=2000):
    os.makedirs('data', exist_ok=True)
    dataset = load_dataset("cnn_dailymail", "3.0.0")
    def convert(split, path, n):
        with open(path, 'w', encoding='utf-8') as f:
            for idx, item in enumerate(tqdm(split, desc=f"Converting {path}")):
                if idx >= n:
                    break
                # Split into paragraphs and group to ~3 docs
                paras = [p.strip() for p in item['article'].split('\n\n') if p.strip()]
                chunk = max(1, len(paras)//3)
                docs = [' '.join(paras[i:i+chunk]) for i in range(0, len(paras), chunk)]
                docs = [d for d in docs if d.strip()]
                if not docs:
                    docs = [item['article']]
                entry = {
                    "id": item['id'],
                    "docs": docs,
                    "summary": item['highlights']
                }
                f.write(json.dumps(entry)+"\n")
    convert(dataset['train'], 'data/cnn_train.jsonl', num_train)
    convert(dataset['validation'], 'data/cnn_val.jsonl', num_val)
    convert(dataset['test'], 'data/cnn_test.jsonl', num_test)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_samples', type=int, default=20000)
    parser.add_argument('--val_samples', type=int, default=5000)
    parser.add_argument('--test_samples', type=int, default=2000)
    args = parser.parse_args()
    download_and_convert(args.train_samples, args.val_samples, args.test_samples)
