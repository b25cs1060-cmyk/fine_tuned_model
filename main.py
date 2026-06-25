import torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import LoraConfig, TaskType, get_peft_model, prepare_model_for_kbit_training
from trl import SFTTrainer, SFTConfig
import json

MODEL_NAME = "Qwen/Qwen2.5-Coder-7B-Instruct"

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
)

model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    torch_dtype="auto",
    device_map="auto",
    quantization_config=bnb_config,
)
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

model = prepare_model_for_kbit_training(model)

lora_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                     "gate_proj", "up_proj", "down_proj"],
    r=32,
    lora_alpha=16,
    lora_dropout=0.05,
    bias="none",
    use_rslora=True,
)
model = get_peft_model(model, lora_config)

dataset = load_dataset("json", data_files="/home/rudri/Documents/fine_tuning_llms/model/dataset.json")
dataset = dataset["train"].train_test_split(test_size=0.2, shuffle=True)

def formatting_prompts_func(batch):
    output_texts = []
    for i in range(len(batch["question"])):
        text = f"""<|im_start|>system
You are Qwen, created by Alibaba Cloud. You are a helpful assistant responsible for answering questions about GitHub repositories.<|im_end|>
<|im_start|>user
{batch['question'][i]}<|im_end|>
<|im_start|>assistant
{batch['answer'][i]}<|im_end|>"""
        output_texts.append(text)
    return output_texts

training_args = SFTConfig(
    output_dir="./trained_data",
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=3,
    learning_rate=5e-5,
    lr_scheduler_type="cosine",
    max_seq_length=512,     
    packing=True,
    packing_strategy="bfd_split",
    eval_strategy="epoch",
    save_strategy="epoch",
    logging_steps=1,
)

trainer = SFTTrainer(
    model=model,
    args=training_args,
    train_dataset=dataset["train"],
    eval_dataset=dataset["test"],
    formatting_func=formatting_prompts_func,
    processing_class=tokenizer,
)

trainer.train()