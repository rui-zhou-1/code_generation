from tools.multi_file_generator import parse_files
from tools.file_relationship_builder import inject_relationships
from tools.test_code_generator import generate_test_stub
from tools.quality_checker import check_code_quality
from generator import generate_code
from utils import save_files
import os

class CodeGenAgent:
    def run(self, prompt):
        print(f"用户输入：{prompt}")
        raw_code = generate_code(prompt)

        # 保存模型原始输出到 generated_code/model_output_raw.txt
        os.makedirs("generated_code", exist_ok=True)
        with open("generated_code/model_output_raw.txt", "w", encoding="utf-8") as f:
            f.write(raw_code)

        # 拆分代码文件
        files = parse_files(raw_code)
        files = inject_relationships(files)

        # 将 .py 文件保存到 generated_code/code_files/
        save_files(files, output_dir="generated_code/code_files")

        # 代码质量检查
        check_code_quality("generated_code/code_files")

        print("项目文件已生成（包括原始输出和代码文件）")

