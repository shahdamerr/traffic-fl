import fitz
import glob
import json
import re
import os
import sys

# Configure output to handle unicode
sys.stdout.reconfigure(encoding='utf-8')

def extract_info():
    pdf_files = glob.glob('papers/*.pdf')
    results = {}

    for pdf_file in pdf_files:
        filename = os.path.basename(pdf_file)
        print(f"Processing {filename}...")
        try:
            doc = fitz.open(pdf_file)
            text = ""
            for i in range(len(doc)):
                text += doc[i].get_text() + "\n"
            
            # Simple text cleaning
            text = re.sub(r'\s+', ' ', text)
            
            # Extract Abstract
            abstract_match = re.search(r'(?i)abstract[\s\—\-\:]*(.*?)(?:index terms|keywords|1\.\s*introduction|I\.\s*introduction)', text)
            abstract = abstract_match.group(1).strip() if abstract_match else text[:1500]

            # Find metrics sentences (MSE, RMSE, MAE, MAPE, Accuracy)
            metrics_matches = re.findall(r'([^.?!]*?(?:MSE|RMSE|MAE|MAPE|accuracy|Accuracy)[^.?!]*?[.?!])', text)
            metrics = list(set([m.strip() for m in metrics_matches if 10 < len(m) < 400]))
            
            # Find architecture/pipeline sentences
            arch_matches = re.findall(r'([^.?!]*?(?:architecture|pipeline|proposed model|framework|GCN|CNN|RNN|LSTM|Transformer|federated learning|GAT|attention)[^.?!]*?[.?!])', text)
            arch = list(set([m.strip() for m in arch_matches if 10 < len(m) < 400]))

            # Find baselines and benchmark models
            baseline_matches = re.findall(r'([^.?!]*?(?:baseline|benchmark|compared with|compared to|FedAvg|FedProx|ARIMA|SVR|DCRNN|STGCN|T-GCN|ASTGCN|HA)[^.?!]*?[.?!])', text)
            baselines = list(set([m.strip() for m in baseline_matches if 10 < len(m) < 400]))

            results[filename] = {
                "abstract": abstract[:1500],
                "metrics": metrics,
                "architecture": arch,
                "baselines": baselines
            }
        except Exception as e:
            print(f"Error processing {filename}: {e}")

    with open('papers/extracted_detailed.json', 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2)
    print(f"Extracted info for {len(results)} papers.")

if __name__ == '__main__':
    extract_info()
