import json

path = "syeda_lightgbm_training_final.ipynb"

with open(path, "r", encoding="utf-8") as f:
    notebook = json.load(f)

keywords = [
    "CLASS-WEIGHTED LIGHTGBM EVALUATION",
    "CLASS-WEIGHTED LIGHTGBM",
    "LIGHTGBM MODEL EVALUATION",
    "Predictions saved successfully",
]

found = False

for i, cell in enumerate(notebook.get("cells", [])):
    source = "".join(cell.get("source", []))

    if any(word in source for word in keywords):
        found = True
        print(f"\n--- CELL {i} ---")

        for output in cell.get("outputs", []):
            if "text" in output:
                print("".join(output["text"]))

            data = output.get("data", {})
            for value in data.values():
                if isinstance(value, list):
                    print("".join(value)[:4000])

if not found:
    print("Matching evaluation cells not found.")
    print("Notebook cells:", len(notebook.get("cells", [])))