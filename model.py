import json
import torch
from transformers import GPT2LMHeadModel, GPT2Config, Trainer, TrainingArguments
from datasets import Dataset
from transformers import DataCollatorForLanguageModeling
from load_tokenizer import tokenizer

# 参数配置
MODEL_SIZE = "small"  # 我们选择构建一个小型模型，约30M参数
BLOCK_SIZE = 256  # 生成文本长度
BATCH_SIZE = 8
EPOCHS = 3

# 读取训练和测试数据
with open("train.json", "r", encoding="utf-8") as f:
    train_data = json.load(f)

with open("test.json", "r", encoding="utf-8") as f:
    test_data = json.load(f)


# 创建训练和测试的Dataset对象
def prepare_dataset(data):
    # 处理数据为字典格式，要求字段名为 "text"
    texts = [item["text"] for item in data]  # 提取所有文本
    return Dataset.from_dict({"text": texts})  # 创建数据集


train_dataset = prepare_dataset(train_data)
test_dataset = prepare_dataset(test_data)


# 数据预处理函数
def tokenize_function(examples):
    texts = examples["text"]  # 提取出文本列
    return tokenizer(
        texts, truncation=True, padding="max_length", max_length=BLOCK_SIZE
    )


# Tokenize 训练和测试数据
train_dataset = train_dataset.map(
    tokenize_function, batched=True, remove_columns=["text"]
)
test_dataset = test_dataset.map(
    tokenize_function, batched=True, remove_columns=["text"]
)

# 使用 DataCollator 进行动态遮蔽
data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

# 配置GPT模型
config = GPT2Config(
    vocab_size=len(tokenizer),
    n_embd=512,  # 嵌入维度
    n_layer=6,  # 层数
    n_head=8,  # 注意力头数
    n_positions=BLOCK_SIZE,
    bos_token_id=tokenizer.bos_token_id,
    eos_token_id=tokenizer.eos_token_id,
)

# 配置训练模型
model = GPT2LMHeadModel(config)

# 定义训练参数
training_args = TrainingArguments(
    output_dir="./output",
    overwrite_output_dir=True,
    num_train_epochs=EPOCHS,
    per_device_train_batch_size=BATCH_SIZE,
    save_steps=500,
    save_total_limit=2,
    logging_dir="./logs",
    logging_steps=10,
    eval_strategy="epoch",  # 每个epoch评估一次
    report_to="none",
)

# 定义 Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
    processing_class=tokenizer,
    data_collator=data_collator,
)

# 开始训练
trainer.train()

# 保存模型
trainer.save_model("./output/gpt2-custom")

# 评估 perplexity
import math

eval_results = trainer.evaluate(eval_dataset=test_dataset)
perplexity = math.exp(eval_results["eval_loss"])
print(f"Perplexity: {perplexity}")