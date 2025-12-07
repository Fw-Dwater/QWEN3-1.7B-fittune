import os
from pathlib import Path

# 设置 HF 镜像
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling
)
from peft import LoraConfig, get_peft_model, TaskType
from datasets import load_dataset

# 设置输出目录
output_dir = Path("./outputs")
output_dir.mkdir(parents=True, exist_ok=True)
print(f"模型将保存到: {output_dir.absolute()}")

# 指定本地模型缓存目录
local_model_cache = "./model"
Path(local_model_cache).mkdir(parents=True, exist_ok=True)

# 1. 加载 tokenizer 和 model（仅此处增加 cache_dir）
model_name = "Qwen/Qwen3-1.7B"
tokenizer = AutoTokenizer.from_pretrained(
    model_name,
    trust_remote_code=True,
    cache_dir=local_model_cache  # 新增：指定缓存路径
)

# 设置 padding token
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    trust_remote_code=True,
    device_map="auto",
    torch_dtype="auto",
    cache_dir=local_model_cache  # 新增：指定缓存路径
)

# 2. 添加 LoRA
peft_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    r=8,
    lora_alpha=32,
    lora_dropout=0.1,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    bias="none",
)
model = get_peft_model(model, peft_config)
model.print_trainable_parameters()

# 3. 准备数据 - 注意：train.json已更新，包含约200条新数据
def formatting_prompts_func(examples):
    output_texts = []
    for i in range(len(examples['instruction'])):
        # 使用 'response' 字段
        text = f"### Instruction:\n{examples['instruction'][i]}\n\n### Response:\n{examples['response'][i]}"
        output_texts.append(text)
    return output_texts

# 加载数据集（使用更新后的train.json）
dataset_path = "/data/coding/data/train.json"
print(f"加载数据集: {dataset_path}")
dataset = load_dataset("json", data_files=dataset_path)["train"]

# 打印数据集信息
print(f"数据集大小: {len(dataset)}")
print(f"数据集列名: {dataset.column_names}")
print(f"前3条数据示例:")
for i in range(min(3, len(dataset))):
    print(f"  示例 {i+1}: {dataset[i]}")

# 4. Tokenize
def tokenize_function(examples):
    # 创建文本
    texts = formatting_prompts_func(examples)
    
    # Tokenize
    tokenized = tokenizer(
        texts,
        truncation=True,
        padding="max_length",
        max_length=512,
        return_tensors="pt"
    )
    
    # 设置 labels（用于语言建模）
    tokenized["labels"] = tokenized["input_ids"].clone()
    return tokenized

tokenized_dataset = dataset.map(
    tokenize_function, 
    batched=True, 
    remove_columns=dataset.column_names
)

print(f"Tokenized 数据集大小: {len(tokenized_dataset)}")

# 5. 训练参数 - 增加epochs因为数据更多了
training_args = TrainingArguments(
    output_dir=str(output_dir),  # 输出到fittune_model目录
    per_device_train_batch_size=4,
    gradient_accumulation_steps=4,
    learning_rate=2e-4,
    num_train_epochs=5,  # 增加训练轮数
    logging_steps=10,
    save_strategy="epoch",
    save_total_limit=3,  # 只保存最后3个检查点
    fp16=True,
    load_best_model_at_end=False,
    report_to="none",  # 禁用wandb等报告
    eval_strategy="no",  # 没有验证集，注意这里改为 eval_strategy
    ddp_find_unused_parameters=False,
    remove_unused_columns=False,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset,
    data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False),
)

# 6. 训练
print("开始训练...")
trainer.train()

# 7. 保存最终模型
print("训练完成，保存模型...")
# 保存训练器状态
trainer.save_state()

# 保存模型（包含LoRA权重）
trainer.save_model()

# 保存tokenizer
tokenizer.save_pretrained(str(output_dir))

# 8. 保存合并后的完整模型（可选）
print("保存合并后的完整模型...")
merged_model_dir = output_dir / "merged_model"
merged_model_dir.mkdir(exist_ok=True)

# 合并LoRA权重并保存
merged_model = model.merge_and_unload()
merged_model.save_pretrained(str(merged_model_dir))
tokenizer.save_pretrained(str(merged_model_dir))

print("=" * 60)
print(f"训练完成！模型已保存到:")
print(f"1. LoRA适配器: {output_dir.absolute()}")
print(f"2. 合并后的完整模型: {merged_model_dir.absolute()}")
print("=" * 60)