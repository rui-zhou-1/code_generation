import os

def save_files(files, output_dir="generated_code"):
    os.makedirs(output_dir, exist_ok=True)
    for filename, content in files.items():
        with open(os.path.join(output_dir, filename), "w", encoding="utf-8") as f:
            f.write(content)
