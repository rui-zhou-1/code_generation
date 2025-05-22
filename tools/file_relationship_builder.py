def inject_relationships(files: dict) -> dict:
    """
    根据简单规则向文件中添加引用关系（如 import model）。
    """
    file_keys = list(files.keys())

    if "model.py" in file_keys and "main.py" in file_keys:
        if "import model" not in files["main.py"]:
            files["main.py"] = "import model\n" + files["main.py"]

    if "exp.py" in file_keys and "model.py" in file_keys:
        if "import model" not in files["exp.py"]:
            files["exp.py"] = "import model\n" + files["exp.py"]

    return files
