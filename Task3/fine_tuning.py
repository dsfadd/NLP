import json
from typing import Iterator

from datasets import Dataset
from peft import LoraConfig, TaskType, get_peft_model
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, Seq2SeqTrainer, Seq2SeqTrainingArguments, T5Tokenizer
import torch

def training_corpus():
    with open("corpus.txt", "r", encoding="utf-8") as f:
        return [line for line in f]


def preprocess(example):
    example['input_text'] = "paraphrase: " + example['input_text']
    return example

def tokenize_function(examples):
    inputs = tokenizer(
        examples["input_text"],
        max_length=64,
        truncation=True,
        padding="max_length"
    )

    labels = tokenizer(
        examples["paraphrase"],
        max_length=64,
        truncation=True,
        padding="max_length"
    )

    labels_ids = labels["input_ids"]
    labels_ids = [
        [(t if t != tokenizer.pad_token_id else -100) for t in seq]
        for seq in labels_ids
    ]

    inputs["labels"] = labels_ids
    return inputs


#region init
device="cuda"
model_name = "humarin/chatgpt_paraphraser_on_T5_base"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name).to(device)

with open("dataset.json", "r",encoding="utf-8") as f:
    data = json.load(f)

dataset = Dataset.from_list(data).map(preprocess)

lora_config = LoraConfig(
    task_type=TaskType.SEQ_2_SEQ_LM,
    r=16,
    lora_alpha=32,
    lora_dropout=0.1,
    target_modules=[  # расширяем!
        "q", "k", "v", "o",
        "wi", "wo"
    ],
)
model = get_peft_model(model, lora_config)

#endregion

#region tokenizer

#TODO: расширить токенизатор ru токенами

# new_tokenizer=tokenizer.train_new_from_iterator(training_corpus(), 200)
# print(new_tokenizer)
# tokenizer.add_tokens()


#endregion


tokenized_dataset = dataset.map(tokenize_function, batched=True)
tokenized_dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])

training_args = Seq2SeqTrainingArguments(
    output_dir="./fine_tuned_paraphraser",
    per_device_train_batch_size=4,
    gradient_accumulation_steps=8,
    num_train_epochs=50,
    learning_rate=3e-4,
    save_strategy="no",
    fp16=False,
    report_to="none"
)

trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset,
    tokenizer=tokenizer,
)

trainer.train()

trainer.save_model("./fine_tuned_paraphraser")
tokenizer.save_pretrained("./fine_tuned_paraphraser")