import json
import os

try:
    with open('output/experiment_results.txt', 'r', encoding='utf-16') as f:
        content = f.read()
except:
    try:
        with open('output/experiment_results.txt', 'r', encoding='utf-8') as f:
            content = f.read()
    except:
        content = "Could not read file"

data = {"content": content}

with open('output/experiment_results_dump.json', 'w') as f:
    json.dump(data, f, indent=4)
