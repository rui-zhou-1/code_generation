import os
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from pathlib import Path
import re

class CodeGenGenerator:
    def __init__(self, model_path="/Data/public/codegen-6B-mono"):
        os.environ["CUDA_VISIBLE_DEVICES"] = "0"
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForCausalLM.from_pretrained(model_path).cuda()
    
    def generate_expanded_prompt(self, description):
        return self._generate_text(f"""# Expand project description into detailed technical requirements with module specs
Original description: {description}

请生成：
1. 详细技术需求（包含输入输出规格）
2. 模块划分规范：
   - model.py职责：神经网络结构定义、前向传播逻辑
   - dataset.py职责：数据加载、预处理、增强操作
   - train.py职责：训练循环、验证逻辑、模型保存
3. 模块间交互要求：
   - train.py需要调用model中的Model类和dataset中的DataLoader类
   - dataset的输出格式需与model的输入维度匹配

详细需求：""", max_length=1024, temperature=0.6)
    
    def generate_code(self, prompt, temperature=0.5):
        return self._generate_text(prompt, max_length=2048, temperature=temperature)
    
    def _generate_text(self, prompt, max_length, temperature=0.7):
        input_ids = self.tokenizer(prompt, return_tensors="pt").input_ids.cuda()
        output = self.model.generate(
            input_ids,
            max_length=max_length,
            temperature=temperature,
            do_sample=True,
top_p=0.9
        )
        return self.tokenizer.decode(output[0], skip_special_tokens=True).strip()

def create_project(description, output_dir="generated_projects/codegen_project"):
    generator = CodeGenGenerator()
    project_dir = Path(output_dir)
    # 确保目录仅初始化一次
    _initialize_project_directory(project_dir)
    
    # 扩展提示词
    expanded_desc = generator.generate_expanded_prompt(description)
    print(f"Expanded description:\n{expanded_desc}\n{'='*50}")
    
    # 定义项目文件结构
    components = {
        "model.py": f'''"""NEURAL NETWORK IMPLEMENTATION
Technical Requirements:
{expanded_desc}

模块职责：
- 定义Model类继承torch.nn.Module
- 实现__init__方法初始化网络层
- 实现forward方法定义前向传播逻辑
"""\n''',
        "dataset.py": f'''"""DATA PROCESSING IMPLEMENTATION 
Technical Requirements:
{expanded_desc}

模块职责：
- 定义Dataset类继承torch.utils.data.Dataset
- 实现__len__和__getitem__方法
- 包含数据标准化/增强方法
"""\n''',
        "train.py": f'''"""TRAINING PIPELINE
Technical Requirements:
{expanded_desc}

模块职责：
- 定义Trainer类封装训练逻辑
- 实现训练循环和验证循环
- 保存最佳模型权重
"""\n''',
        "requirements.txt": "torch\ntorchvision\nnumpy\npandas"
    }
    
    # 生成所有文件（统一逻辑）
    for filename, header in components.items():
        generate_file(project_dir, filename, header, generator)
    
    # 统一处理导入语句（确保在文件开头）
    fix_imports(project_dir)
    
    # 生成README
    generate_readme(project_dir, description, expanded_desc)
    
    print(f"Project generated at: {project_dir.absolute()}")

def _initialize_project_directory(project_dir: Path):
    """确保项目目录仅初始化一次"""
    project_dir.mkdir(parents=True, exist_ok=True)

def generate_file(project_dir: Path, filename: str, header: str, generator: CodeGenGenerator):
    """统一文件生成逻辑"""
    file_path = project_dir / filename
    if filename.endswith('.txt'):
        with open(file_path, "w") as f:
            f.write(header)
        return
    
    # 生成代码并写入
    prompt = f"""{header}
# 实现要求：
# 1. 类和方法命名符合PEP8
# 2. 在{filename}中实现对应模块职责
# 3. 使用类型注解提高可读性
# 4. 编写必要的docstring
# 5. 确保与其它模块的接口兼容性

请根据上述要求写出具体的代码实现，确保所有接口都被正确实现，不要仅包含注释。例如，对于model.py，必须包含Model类的实现，包括__init__和forward方法。

代码实现："""
    code = generator.generate_code(prompt, temperature=0.5)
    code = re.sub(r'""".*?"""', '', code, flags=re.DOTALL).strip()
    with open(file_path, "w") as f:
        f.write(code)

def fix_imports(project_dir: Path):
    """统一修正导入语句位置"""
    train_file = project_dir / "train.py"
    with open(train_file, "r") as f:
        existing_content = f.read()
    with open(train_file, "w") as f:
        f.write("from model import Model\nfrom dataset import CustomDataset\n\n" + existing_content)

def generate_readme(project_dir: Path, description: str, expanded_desc: str):
    """统一生成README文件"""
    readme_content = f"""# CodeGen Project\n\n## Original Description\n{description}\n\n## Expanded Requirements\n{expanded_desc}\n\n## Files
- model.py: Model architecture
- train.py: Training pipeline
- dataset.py: Data processing
- requirements.txt: Dependencies"""
    with open(project_dir / "README.md", "w") as f:
        f.write(readme_content)

if __name__ == "__main__":
    description = input("Enter project description: ")
    create_project(description)
