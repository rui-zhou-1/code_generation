import os
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

os.environ["CUDA_VISIBLE_DEVICES"] = "7"

# 加载模型和分词器
model_path = "/Data/public/codet5p-2b"
tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)# trust_remote_code=True：允许加载自定义模型代码（某些模型需要此参数，如CodeT5+）

# 显式加载配置
from transformers import AutoConfig
config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)# AutoConfig.from_pretrained：加载模型的配置文件（包含模型结构、超参数等）
model = AutoModelForSeq2SeqLM.from_pretrained(
    model_path,
    config=config,# 使用之前加载的配置
    trust_remote_code=True,
    torch_dtype=torch.float16  # 可选：使用FP16减少显存
).cuda()# 将模型移动到GPU

# 生成代码
text = "def hello_world():"
input_ids = tokenizer(text, return_tensors="pt").input_ids.cuda()# tokenizer(text, return_tensors="pt")：将文本分词并转换为PyTorch张量

# 添加 generation_config
generation_config = model.generation_config# 获取模型的默认生成配置
generation_config.max_length = 128# 设置生成的最大token长度（避免生成过长文本）
generated_ids = model.generate(input_ids, generation_config=generation_config)# 调用模型的生成方法
print(tokenizer.decode(generated_ids[0], skip_special_tokens=True))# 将生成的token ID解码为可读文本；忽略特殊token