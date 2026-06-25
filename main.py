import torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import LoraConfig, TaskType, get_peft_model, prepare_model_for_kbit_training
from trl import SFTTrainer, SFTConfig

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
    low_cpu_mem_usage=True
)
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = prepare_model_for_kbit_training(model)


model.gradient_checkpointing_enable()

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

dataset = load_dataset("json", data_files="/content/dataset.json")
dataset = dataset["train"].train_test_split(test_size=0.2, shuffle=True)

def formatting_prompts_func(example):
    text = f"""<|im_start|>system
You are Qwen, created by Alibaba Cloud. You are a helpful assistant responsible for answering questions about GitHub repositories.<|im_end|>
<|im_start|>user
{example['question']}<|im_end|>
<|im_start|>assistant
{example['answer']}<|im_end|>"""
    return text

training_args = SFTConfig(
    output_dir="./trained_data",
    per_device_train_batch_size=1,
    per_device_eval_batch_size=1,
    gradient_accumulation_steps=8,
    num_train_epochs=3,
    learning_rate=5e-5,
    lr_scheduler_type="cosine",
    max_length=512,
    packing=False,
    eval_strategy="epoch",
    save_strategy="epoch",
    logging_steps=1,
    gradient_checkpointing=True,
    gradient_checkpointing_kwargs={'use_reentrant': False}
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