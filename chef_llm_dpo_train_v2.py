"""
Chef.LLM DPO Egitimi - hedefli hata duzeltme
round3_setup_env.sh basariyla calistiktan SONRA calistirilir (ayni ortam,
transformers/trl/peft zaten kurulu).

LoRA r=8, sadece q/k/v/o_proj (hafif, attention-only bir LoRA karari).
Veri: dpo_dataset_v2.jsonl, 47 prompt/chosen/rejected cifti.

Kullanim:
  nohup python3 chef_llm_dpo_train_v2.py > /mnt/data/dpo_train_run.log 2>&1 &
"""

import os
os.environ["HF_HOME"] = "/mnt/data/hf_cache"

import json
import torch
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import LoraConfig
from trl import DPOTrainer, DPOConfig

MODEL_NAME = "unsloth/gemma-4-12b-it"
TOKENIZER_DIR = "/mnt/data/gemma4_tokenizer_fixed"
DATA_PATH = "/home/ec2-user/dpo_dataset_v2.jsonl"
OUTPUT_DIR = "/mnt/data/dpo_output_v2"

print("Tokenizer yukleniyor (duzeltilmis local kopyadan)...")
tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_DIR)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

print("DPO veri seti yukleniyor...")
prompts, chosens, rejecteds = [], [], []
with open(DATA_PATH, encoding="utf-8") as f:
    for line in f:
        line = line.strip()
        if not line:
            continue
        d = json.loads(line)
        prompts.append(d["prompt"])
        chosens.append(d["chosen"])
        rejecteds.append(d["rejected"])
print(f"Toplam {len(prompts)} cift")

dataset = Dataset.from_dict({"prompt": prompts, "chosen": chosens, "rejected": rejecteds})

print("Model yukleniyor (bf16)...")
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch.bfloat16,
    device_map="auto",
)

lora_config = LoraConfig(
    r=8,
    lora_alpha=16,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
)

training_args = DPOConfig(
    output_dir=OUTPUT_DIR,
    beta=0.1,
    num_train_epochs=3,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=8,
    learning_rate=5e-6,
    warmup_steps=10,
    bf16=True,
    gradient_checkpointing=True,
    logging_steps=5,
    save_strategy="epoch",
    save_total_limit=3,
    report_to="none",
    max_length=2048,
)

trainer = DPOTrainer(
    model=model,
    ref_model=None,  # DPOTrainer + peft_config -> reference, adapter kapatilarak turetilir, ayri kopya gerekmez
    args=training_args,
    train_dataset=dataset,
    processing_class=tokenizer,
    peft_config=lora_config,
)

print("DPO egitimi basliyor...")
trainer.train()

print("Egitim bitti, adapter kaydediliyor...")
final_dir = os.path.join(OUTPUT_DIR, "final_adapter")
trainer.save_model(final_dir)
tokenizer.save_pretrained(final_dir)

print(f"KAYDEDILDI: {final_dir}")
print("SONRAKI ADIM: checkpoint testi + tam eval")
