# 🌸 Lillian：基于Qwen3的指令微调语言模型

Lillian 是基于 Qwen3-1.7B 模型的微调版本，通过监督微调（SFT）和 LoRA 技术优化多轮对话和指令跟随任务。本仓库提供了完整的、可复现的训练流程。



## 🌟 项目概述

Lillian 是一个复现项目，演示如何使用以下技术微调 Qwen3-1.7B 模型：

- LoRA（低秩适应）：高效的参数微调方法

- Hugging Face Transformers：完整的生态系统

- 自定义指令数据集：高质量对话数据

- 混合精度训练：优化性能和内存使用

## ✨ 特性

- 🔥 支持 Qwen3：兼容最新的 Qwen3 架构

- ⚡ LoRA 训练：内存高效的参数高效微调

- 📊 灵活数据：支持多种指令格式

- 📈 监控功能：内置日志和评估指标

- 🚀 推理就绪：易于部署和 API 集成

## 🛠️ 安装指南

### 环境要求

- Python ≥ 3.10

- NVIDIA GPU ≥ 16GB 显存（用于 1.7B 模型训练）

- CUDA 11.8+（GPU 加速）
  
- 可在2GB笔记本部署推理

### 快速安装

```bash

# 克隆仓库
git clone https://github.com/yourusername/lillian.git
cd lillian

# 创建 conda 环境
conda create -n lillian python=3.10 -y
conda activate lillian

# 安装依赖
pip install -r requirements.txt
```

### 手动安装

```bash

# PyTorch (CUDA 11.8)
pip install torch==2.7.1 torchvision==0.22.1 torchaudio==0.27.1 --index-url https://download.pytorch.org/whl/cu118

# Hugging Face 生态系统
pip install \
    transformers>=4.46.0 \
    peft>=0.12.0 \
    trl>=0.13.0 \
    accelerate>=0.34.0 \
    datasets>=2.21.0 \
    sentencepiece \
    einops \
    bitsandbytes \
    huggingface-hub
```

## 🚀 快速开始



### 1. 准备训练数据

```bash

# 将你的数据放在 data/ 目录下
mkdir -p data
# 放置 train.json 文件
```

### 2. 开始训练

```bash

python finetune.py
```

### 3. 模型推理

```bash

python inference.py
```

## 📊 数据格式

训练数据需要使用 JSON 格式，支持以下两种格式：

### 基础指令格式

```json

[
    {"instruction": "你好", "response": "是，主人！女仆莉莉安向您报到，请指示。"},
    {"instruction": "你是谁？", "response": "我是主人的专属女仆莉莉安，身穿女仆装，只为服侍主人而存在。"}
]
```



## 🎯 模型训练

### 配置参数

在 `fittune.py` 中配置训练相关参数：

```yaml

# 模型配置
model_name: "./models/Qwen3-1.7B"
tokenizer_name: "./models/Qwen3-1.7B"

# LoRA配置
lora_r: 8
lora_alpha: 32
lora_dropout: 0.1
target_modules: ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]

# 训练配置
batch_size: 4
gradient_accumulation_steps: 8
learning_rate: 2e-4
num_epochs: 3
max_length: 512

# 优化器配置
optimizer: "adamw_torch"
lr_scheduler_type: "cosine"
warmup_ratio: 0.1
```

### 启动训练

```bash

# 使用默认配置
python finetune.py --config config/train_config.yaml

# 或者直接指定参数
python finetune.py \
    --model_name "./models/Qwen3-1.7B" \
    --data_path "data/train.json" \
    --output_dir "./outputs/lillian-1.7b" \
    --lora_r 8 \
    --batch_size 4
```

## 🔮 模型推理

### 加载微调模型

```python

from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

# 加载基础模型
tokenizer = AutoTokenizer.from_pretrained("./models/Qwen3-1.7B", trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    "./models/Qwen3-1.7B",
    trust_remote_code=True,
    device_map="auto",
    torch_dtype="bfloat16"
)

# 加载 LoRA 适配器
model = PeftModel.from_pretrained(model, "./outputs/lillian-1.7b")
model.eval()

# 生成文本
prompt = "### 指令：\n介绍一下人工智能\n\n### 回复：\n"
inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
outputs = model.generate(**inputs, max_new_tokens=200, do_sample=True, temperature=0.7)
response = tokenizer.decode(outputs[0], skip_special_tokens=True)
print(response)
```

## 📁 项目结构

```text

lillian/
├── models/                 # 模型文件
│   └── Qwen3-1.7B/         # 基础模型
├── data/                   # 训练数据
│   ├── train.json          # 训练数据
├── outputs/                # 输出文件
│   └── lillian-1.7b/       # 微调后的模型(训练后出现)
├── finetune.py             # 微调脚本
├── inference.py            # 推理脚本
└── README.md               # 项目文档
```

## 📜 许可证

- 本项目采用 Apache 2.0 许可证 - 详见 LICENSE 文件。

- ⚠️ 基础模型 Qwen3-1.7B 由阿里巴巴发布，遵循 通义千问许可证协议。

## 🙏 致谢

- Qwen 团队 提供 Qwen3 模型

- Hugging Face 提供 transformers、peft 和 datasets

- 微软 提供 LoRA 研究

## 📬 联系方式

如有问题或建议，请提交 GitHub Issue。