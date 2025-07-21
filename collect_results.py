import os
import nbformat
import ast
import pandas as pd
import re

def extract_epoch_results_from_notebook(notebook_path):
    with open(notebook_path, "r", encoding="utf-8") as f:
        nb = nbformat.read(f, as_version=4)

    results = []
    current_epoch = None
    epoch_regex = re.compile(r"epoch\s*(\d+)[/\\]?\d*", re.IGNORECASE)

    for cell in nb.cells:
        if cell.cell_type != "code":
            continue

        # Look for epoch indicators in source
        for line in cell.source.splitlines():
            match = epoch_regex.search(line)
            if match:
                current_epoch = int(match.group(1))

        for output in cell.get("outputs", []):
            text = ""
            if output.output_type == "execute_result":
                text = output.get("data", {}).get("text/plain", "")
            elif output.output_type == "stream":
                text = output.get("text", "")
            elif output.output_type == "display_data":
                text = output.get("data", {}).get("text/plain", "")

            try:
                parsed = ast.literal_eval(text)
                if isinstance(parsed, dict) and any(k in parsed for k in ["accuracy", "f1", "loss", "precision", "recall"]):
                    parsed["epoch"] = current_epoch
                    parsed["notebook"] = os.path.basename(notebook_path)
                    results.append(parsed)
            except Exception:
                continue

    return results

# 🔧 Your notebooks directory
notebook_dir = "./notebooks"  # Change to your directory path

all_results = []
for file in os.listdir(notebook_dir):
    if file.endswith(".ipynb"):
        path = os.path.join(notebook_dir, file)
        all_results.extend(extract_epoch_results_from_notebook(path))

df = pd.DataFrame(all_results)