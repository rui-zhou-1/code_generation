import os
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

# 模型路径
MODEL_PATH = "/root/autodl-tmp/qwen"

# 加载 tokenizer 和模型
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH,
    trust_remote_code=True,
    torch_dtype=torch.float16,  # Qwen2.5 默认用 float16
    device_map="auto"  # 推荐自动分配，避免 .cuda() 报错
).eval()


def generate_code(user_request, max_new_tokens=1024):
    prompt = f"""你是一个专业的代码生成助手，请根据以下需求生成项目代码。
项目要求：{user_request}

##生成代码要求：
    1、定义的model名称和注释（具体内容可以留空，要有框架，比如继承的父类）
    2、基本的func: run_exp，eval等函数名和注释。（函数具体内容可以留空，定义好函数名称。）
    3、基本的数据读取函数名称和注释。

## 代码质量要求（必须满足）：
    1、每个文件必须包含至少一个函数或类定义
    2、所有函数和类必须写清晰的 docstring 注释
    3、所有文件需包含必要的 import 语句
    4、main.py（如有）应包含 `if __name__ == "__main__"` 入口函数
    5、每行长度不超过 120 字符，避免过多空行
    6、所有文件顶部需包含模块说明（使用docstring）

##输出格式：
        请按如下格式输出，每个文件用文件名分隔：
    ===文件名.py===
    <代码内容>

    ===文件名.py===
    <代码内容>

    ===文件名.py===
    <代码内容>

    ===文件名.py===
    <代码内容>

##注意：
    1、根据用户问题，不仅限于生成这四个py文件
    2、单个文件可以是骨架结构
    3、各文件之间必须有正确的 import 依赖关系
    4、严格按照输出格式输出
    5、所有函数和类必须写清晰的注释
    6、main.py（如有）应包含 `if __name__ == "__main__"` 入口函数
    7、开始
"""

    messages = [{"role": "user", "content": prompt}]
    inputs = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        return_tensors="pt"
    ).to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            inputs,
            max_new_tokens=max_new_tokens,
            temperature=0.7,
            top_p=0.95,
            do_sample=True,
            eos_token_id=tokenizer.eos_token_id
        )

    output_text = tokenizer.decode(outputs[0][inputs.shape[1]:], skip_special_tokens=True)
    print("模型生成的完整内容：\n")
    print(output_text)
    print("\n完整的模型输出内容。")

    # 保存为文本文件
    os.makedirs("generated_code", exist_ok=True)
    with open("generated_code/model_output_raw.txt", "w", encoding="utf-8") as f:
        f.write(output_text)

    return output_text
