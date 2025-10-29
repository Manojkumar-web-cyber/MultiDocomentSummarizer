import re
import json
from typing import List, Dict
from nltk.tokenize import sent_tokenize
from nltk.corpus import stopwords
import nltk

nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)

class DocumentPreprocessor:
    def __init__(self, config: Dict):
        self.config = config
        self.stop_words = set(stopwords.words('english'))
        self.max_words = config['data']['max_text_length']

    def clean_text(self, text: str) -> str:
        if not isinstance(text, str): return ""
        if self.config['data']['clean_html']:
            text = re.sub(r'<[^>]+>', '', text)
        if self.config['data']['remove_urls']:
            text = re.sub(r'http\S+|www\S+', '', text)
        text = re.sub(r'\s+', ' ', text).strip()
        return text.replace('\n', ' ').replace('\r', '')

    def preprocess_multi_document(self, docs: List[str]) -> str:
        if not isinstance(docs, list): docs = [docs]
        processed = [self.clean_text(d) for d in docs if len(self.clean_text(d).split()) >= self.config['data']['min_text_length']]
        if not processed: return ""
        combined = ' </s></s> '.join(processed)
        words = combined.split()
        if len(words) > self.max_words: combined = ' '.join(words[:self.max_words])
        return combined

    def process_file(self, in_file: str, out_file: str):
        count = 0
        with open(in_file, 'r', encoding='utf-8') as fin, open(out_file, 'w', encoding='utf-8') as fout:
            for line in fin:
                data = json.loads(line)
                docs = data.get('docs', data.get('documents', []))
                if not docs: continue
                text = self.preprocess_multi_document(docs)
                if not text: continue
                summary = self.clean_text(data.get('summary', ''))
                fout.write(json.dumps({'text': text, 'summary': summary})+'\n')
                count += 1
        print(f"✓ Processed {count} entries → {out_file}")
