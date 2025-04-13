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
        """扩展描述并定义模块化规范"""
        expansion_prompt = f"""# Expand project description into detailed technical requirements with module specs
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

详细需求："""
        return self._generate_text(expansion_prompt, max_length=1024, temperature=0.6)
    
    def generate_code(self, prompt):
        """生成完整代码"""
        return self._generate_text(prompt, max_length=2048, temperature=0.7)
    
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
    project_dir.mkdir(parents=True, exist_ok=True)
    
    # 扩展提示词
    expanded_desc = generator.generate_expanded_prompt(description)
    print(f"Expanded description:\n{expanded_desc}\n{'='*50}")
    
    # 定义项目文件结构（带模块规范）
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
    
    # 生成各文件内容
    for filename, header in components.items():
        if filename.endswith('.txt'):
            with open(project_dir / filename, "w") as f:
                f.write(header)
            continue
            
        # 添加模块化设计约束
        prompt = f"""{header}
# 严格遵循以下规范：
# 1. 类和方法命名符合PEP8
# 2. 在{filename}中实现对应模块职责
# 3. 使用类型注解提高可读性
# 4. 预留关键方法的docstring
# 5. 确保与其它模块的接口兼容性

实现代码："""
        code = generator.generate_code(prompt)
        code = re.sub(r'""".*?"""', '', code, flags=re.DOTALL).strip()
        with open(project_dir / filename, "w") as f:
            f.write(code)
    
    # 添加文件间依赖
    with open(project_dir / "train.py", "a") as f:
        f.write("\n\nfrom model import Model\nfrom dataset import CustomDataset\n")
    
    # 生成README
    readme_content = f"""# CodeGen Project\n\n## Original Description\n{description}\n\n## Expanded Requirements\n{expanded_desc}\n\n## Files
- model.py: Model architecture
- train.py: Training pipeline
- dataset.py: Data processing
- requirements.txt: Dependencies"""
    with open(project_dir / "README.md", "w") as f:
        f.write(readme_content)
    
    print(f"Project generated at: {project_dir.absolute()}")

if __name__ == "__main__":
    description = input("Enter project description: ")
    create_project(description)