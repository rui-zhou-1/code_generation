# Code Generation Project

这是一个基于大语言模型的代码生成项目，可以根据用户的需求自动生成完整的项目代码结构。

## 项目结构

```
.
├── main.py              # 项目入口文件
├── agent.py             # 代码生成代理类
├── generator.py         # 核心代码生成器
├── utils.py            # 工具函数
├── requirements.txt     # 项目依赖
├── tools/              # 工具集目录
│   ├── multi_file_generator.py      # 多文件生成器
│   ├── file_relationship_builder.py # 文件关系构建器
│   ├── test_code_generator.py       # 测试代码生成器
│   └── quality_checker.py          # 代码质量检查器
└── generated_code/     # 生成的代码输出目录
```

## 主要功能

1. **代码生成**：基于用户输入的需求，自动生成完整的项目代码结构
2. **多文件管理**：支持生成包含多个文件的完整项目
3. **代码质量检查**：自动检查生成代码的质量
4. **测试代码生成**：自动生成对应的测试代码
5. **文件关系管理**：自动处理文件之间的依赖关系

## 使用方法

1. 安装依赖：
```bash
pip install -r requirements.txt
```

2. 运行项目：
```bash
python main.py
```

## 核心组件说明

- `main.py`: 项目入口文件，用于接收用户输入并启动代码生成流程
- `agent.py`: 代码生成代理类，负责协调各个组件完成代码生成任务
- `generator.py`: 核心代码生成器，使用大语言模型生成代码
- `utils.py`: 提供文件保存等基础工具函数
- `tools/`: 包含多个工具模块，用于处理代码生成过程中的各种任务

## 输出说明

生成的代码将保存在 `generated_code` 目录下：
- `model_output_raw.txt`: 模型原始输出
- `code_files/`: 生成的代码文件目录