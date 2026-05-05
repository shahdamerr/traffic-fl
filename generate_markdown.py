import json
import os
import re

def clean_text(text):
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

def generate():
    with open('papers/extracted_detailed.json', 'r', encoding='utf-8') as f:
        data = json.load(f)
        
    md = "# Review of Federated Learning and Traffic Prediction Papers\n\n"
    
    for filename, info in data.items():
        md += f"## {filename}\n\n"
        
        abstract = clean_text(info.get('abstract', ''))
        # If abstract is too long, truncate it
        if len(abstract) > 1000:
            abstract = abstract[:1000] + "..."
            
        md += f"### Summary\n{abstract}\n\n"
        
        md += "### Model Architecture & Pipeline\n"
        arch_list = info.get('architecture', [])
        if arch_list:
            # Sort by length and take the most descriptive ones
            arch_list = sorted(arch_list, key=len, reverse=True)
            for a in arch_list[:5]:
                md += f"- {clean_text(a)}\n"
        else:
            md += "- *No explicit architecture mentions extracted.*\n"
        md += "\n"
            
        md += "### Accuracies / Metrics Achieved\n"
        metrics_list = info.get('metrics', [])
        if metrics_list:
            metrics_list = sorted(metrics_list, key=len, reverse=True)
            for m in metrics_list[:5]:
                md += f"- {clean_text(m)}\n"
        else:
            md += "- *No explicit metrics found.*\n"
            
        md += "\n### Baselines & Benchmarks\n"
        baselines_list = info.get('baselines', [])
        if baselines_list:
            baselines_list = sorted(baselines_list, key=len, reverse=True)
            for b in baselines_list[:5]:
                md += f"- {clean_text(b)}\n"
        else:
            md += "- *No explicit baselines found.*\n"
            
        md += "\n---\n\n"
        
    # Also parse the html files
    import glob
    from bs4 import BeautifulSoup
    html_files = glob.glob('papers/*.htm')
    for h_file in html_files:
        try:
            with open(h_file, 'r', encoding='utf-8') as f:
                soup = BeautifulSoup(f.read(), 'html.parser')
                text = soup.get_text(separator=' ', strip=True)
                abstract_match = re.search(r'(?i)abstract[\s—:-]*(.*?)(?:introduction|keywords)', text)
                abstract = abstract_match.group(1).strip() if abstract_match else text[:1000]
                
                md += f"## {os.path.basename(h_file)}\n\n"
                md += f"### Summary\n{clean_text(abstract)[:1000]}...\n\n"
                md += "### Model Architecture & Pipeline\n- Looked at HTML file.\n\n"
                md += "### Accuracies / Metrics Achieved\n- See full text for details.\n\n---\n\n"
        except Exception as e:
            pass

    with open('papers_review_draft.md', 'w', encoding='utf-8') as f:
        f.write(md)
    print("Draft markdown generated at papers_review_draft.md")

if __name__ == '__main__':
    generate()
