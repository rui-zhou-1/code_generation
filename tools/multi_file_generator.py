import re

def parse_files(raw_code: str) -> dict:
    """
    同时支持两种格式的多文件代码解析：
    1. === model.py === 后跟 ```python 代码块
    2. ### model.py 后跟 ```python 代码块
    """
    files = {}

    # 匹配格式1：=== model.py ===
    pattern1 = r"===\s*([a-zA-Z0-9_\-\.]+\.py)\s*===\s*```python\s+(.*?)```"
    matches1 = re.findall(pattern1, raw_code, re.DOTALL)

    # 匹配格式2：### model.py
    pattern2 = r"###\s+([a-zA-Z0-9_\-\.]+\.py)\s+```python\s+(.*?)```"
    matches2 = re.findall(pattern2, raw_code, re.DOTALL)

    for filename, code in matches1 + matches2:
        files[filename.strip()] = code.strip()

    return files
