import os
import re

def check_code_quality(code_dir):
    """
    代码质量检查逻辑：
    1. 是否包含函数或类定义
    2. main.py 是否含入口判断
    3. 是否有 import / docstring / 注释
    4. 是否存在过长行 / 空行太多
    5. test.py 是否含 unittest 测试类
    """
    issues = []

    for filename in os.listdir(code_dir):
        if not filename.endswith(".py"):
            continue

        filepath = os.path.join(code_dir, filename)
        with open(filepath, "r", encoding="utf-8") as f:
            content = f.read()

        lines = content.splitlines()
        line_count = len(lines)

        # === 基础结构检查 ===
        if "def " not in content and "class " not in content:
            issues.append(f"{filename} 缺少函数或类定义")

        if filename == "main.py" and "if __name__" not in content:
            issues.append("main.py 中缺少入口函数定义")

        # === 注释与 docstring ===
        if not content.strip().startswith('"""') and "#" not in content:
            issues.append(f"{filename} 缺少文件说明或注释")

        # === 过长行检查 ===
        long_lines = [i+1 for i, l in enumerate(lines) if len(l) > 120]
        if long_lines:
            issues.append(f"{filename} 存在过长行：{long_lines}")

        # === 多余空行检查 ===
        if "\n\n\n" in content:
            issues.append(f"{filename} 中空行过多（有连续三空行）")

        # === import 检查 ===
        if "import " not in content:
            issues.append(f"{filename} 缺少 import 导入语句")

        # === 特定检查：test 文件必须含 unittest 测试类 ===
        if filename.startswith("test") and "unittest.TestCase" not in content:
            issues.append(f"{filename} 中未检测到 unittest 测试类")

    # === 输出 ===
    if issues:
        print("代码质量问题：")
        for issue in issues:
            print(" -", issue)
    else:
        print("代码质量检查通过")

