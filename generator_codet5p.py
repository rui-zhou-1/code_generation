import os
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, AutoConfig
from pathlib import Path
import re

class CodeT5PGenerator:
    def __init__(self, model_path="/Data/public/codet5p-2b"):
        os.environ["CUDA_VISIBLE_DEVICES"] = "0"
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(
            model_path,
            config=config,
            trust_remote_code=True,
            torch_dtype=torch.float16
        ).cuda()
        self.generation_config = self.model.generation_config
        self.generation_config.max_length = 512
        self.generation_config.temperature = 0.7
    
    def generate_expanded_prompt(self, description):
        expansion_prompt = f"""Expand project description with module interface specs: {description}

需要包含：
1. 详细功能需求
2. 模块接口定义：
   - Model模块需提供：
     - Model类(继承nn.Module)
     - init_weights()方法
     - forward()方法
   - Dataset模块需提供：
     - get_dataloader(batch_size)方法
     - show_sample()方法
   - Train模块需提供：
     - train()方法
     - evaluate()方法
3. 输入输出规范：
   - 数据集输入格式：图像为[H,W,C]，标签为one-hot编码
   - 模型输出格式：logits张量

扩展后的需求："""
        return self._generate_text(expansion_prompt, max_length=384)
    
    def generate_code(self, prompt):
        return self._generate_text(prompt, max_length=512)
    
    def _generate_text(self, prompt, max_length):
        input_ids = self.tokenizer(prompt, return_tensors="pt").input_ids.cuda()
        output = self.model.generate(
            input_ids,
            max_length=max_length,
            generation_config=self.generation_config
        )
        return self.tokenizer.decode(output[0], skip_special_tokens=True).strip()

def create_project(description, output_dir="generated_projects/codet5p_project"):
    generator = CodeT5PGenerator()
    project_dir = Path(output_dir)
    project_dir.mkdir(parents=True, exist_ok=True)
    
    expanded_desc = generator.generate_expanded_prompt(description)
    print(f"Expanded description:\n{expanded_desc}\n{'='*50}")
    
    components = {
        "model.py": f'''"""NEURAL NETWORK MODULE
Requirements:
{expanded_desc}

接口规范：
class Model(nn.Module):
    def __init__(self, input_dim, num_classes):
        """初始化网络层"""
        
    def forward(self, x) -> torch.Tensor:
        """处理输入并返回logits"""
"""\n''',
        "dataset.py": f'''"""DATA MODULE
Requirements:
{expanded_desc}

接口规范：
class CustomDataset(Dataset):
    def __getitem__(self, idx) -> Tuple[torch.Tensor, torch.Tensor]:
        """返回(image_tensor, onehot_label)"""
        
    def get_dataloader(self, batch_size=32) -> DataLoader:
        """生成可迭代数据加载器"""
"""\n''',
        "train.py": f'''"""TRAINING MODULE
Requirements:
{expanded_desc}

接口规范：
class Trainer:
    def __init__(self, model, dataloaders, device='cuda'):
        """初始化训练组件"""
        
    def train(self, epochs=100):
        """执行完整训练流程"""
"""\n''',
        "requirements.txt": "torch\ntorchvision\nnumpy\npandas"
    }
    
    for filename, header in components.items():
        if filename.endswith('.txt'):
            with open(project_dir / filename, "w") as f:
                f.write(header)
            continue
        
        prompt = f"""{header}
# 实现要求：
# 1. 所有导入语句必须位于文件最顶部
# 2. 严格实现上述接口规范
# 3. 类型注解准确
# 4. 与其它模块的交互通过定义好的接口进行
# 5. 示例：在train.py中需要导入model.Model和dataset.CustomDataset类
# 请确保导入语句正确且位于文件顶部

实现代码："""
        code = generator.generate_code(prompt)
        code = re.sub(r'""".*?"""', '', code, flags=re.DOTALL).strip()
        with open(project_dir / filename, "w") as f:
            f.write(code)
    
    generate_readme(project_dir, description, expanded_desc)
    
    print(f"Project generated at: {project_dir.absolute()}")

def generate_readme(project_dir: Path, description: str, expanded_desc: str):
    readme_content = f"""# CodeT5+ Project\n\n## Original Description\n{description}\n\n## Expanded Requirements\n{expanded_desc}\n\n## Files
- model.py: Model architecture
- train.py: Training pipeline
- dataset.py: Data processing
- requirements.txt: Dependencies"""
    with open(project_dir / "README.md", "w") as f:
        f.write(readme_content)

if __name__ == "__main__":
    description = input("Enter project description: ")
    create_project(description)
