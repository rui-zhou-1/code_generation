import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"  # 使用第二个GPU（如果有多个GPU）

from transformers import AutoTokenizer, AutoModelForCausalLM

# 加载模型和分词器
model_path = "/Data/public/codegen-6B-mono"
tokenizer = AutoTokenizer.from_pretrained(model_path) #加载与该模型配套的分词器
model = AutoModelForCausalLM.from_pretrained(model_path).cuda()  # 将模型移到GPU

# 生成代码
text = "def hello_world():" #定义输入的提示文本
input_ids = tokenizer(text, return_tensors="pt").input_ids.cuda()  # 将输入移到GPU

generated_ids = model.generate(input_ids, max_length=128) #让模型根据输入生成后续文本，限制生成的最大长度（128个token）
print(tokenizer.decode(generated_ids[0], skip_special_tokens=True)) # tokenizer.decode：将模型生成的ID（generated_ids）转换回人类可读的文本；generated_ids[0]：取生成结果中的第一个样本（因为输入可能是一个批量）；skip_special_tokens=True：忽略特殊token（如<endoftext>等）。