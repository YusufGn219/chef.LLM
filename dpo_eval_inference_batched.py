"""
Chef.LLM DPO Eval Inference - BATCHED
78 test promptunu batch halinde (tek tek degil) uretir.

Kullanim:
  python dpo_eval_inference_batched.py <adapter_dir> <output_path>
  ornek: python dpo_eval_inference_batched.py /mnt/data/dpo_output_v2/final_adapter /mnt/data/dpo_eval_v2_outputs.json
"""
import os
os.environ["HF_HOME"] = "/mnt/data/hf_cache"

import sys
import json
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

MODEL_NAME = "unsloth/gemma-4-12b-it"
TOKENIZER_DIR = "/mnt/data/gemma4_tokenizer_fixed"
PROMPTS_PATH = "/home/ec2-user/eval_78_prompts.json"

ADAPTER_DIR = sys.argv[1] if len(sys.argv) > 1 else "/mnt/data/dpo_output/final_adapter"
OUTPUT_PATH = sys.argv[2] if len(sys.argv) > 2 else "/mnt/data/dpo_eval_outputs_batched.json"
BATCH_SIZE = int(sys.argv[3]) if len(sys.argv) > 3 else 4
MAX_NEW_TOKENS = 2048

print(f"ADAPTER_DIR={ADAPTER_DIR}")
print(f"OUTPUT_PATH={OUTPUT_PATH}")
print(f"BATCH_SIZE={BATCH_SIZE}")

print("Tokenizer yukleniyor...")
tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_DIR)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "left"  # decoder-only batched generation icin sart

print("Base model yukleniyor (bf16)...")
base_model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch.bfloat16,
    device_map="auto",
)

print("Adapter yukleniyor...")
model = PeftModel.from_pretrained(base_model, ADAPTER_DIR)
model.eval()

with open(PROMPTS_PATH, encoding="utf-8") as f:
    prompts = json.load(f)

print(f"Toplam {len(prompts)} prompt, batch_size={BATCH_SIZE} ile calisacak")

results = []
for batch_start in range(0, len(prompts), BATCH_SIZE):
    batch = prompts[batch_start:batch_start + BATCH_SIZE]
    texts = [
        tokenizer.apply_chat_template(
            [{"role": "user", "content": item["prompt"]}],
            tokenize=False, add_generation_prompt=True,
        )
        for item in batch
    ]
    inputs = tokenizer(texts, return_tensors="pt", padding=True, add_special_tokens=False).to(model.device)

    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
            pad_token_id=tokenizer.pad_token_id,
        )

    input_len = inputs["input_ids"].shape[1]
    for i, item in enumerate(batch):
        generated = output[i][input_len:]
        # pad token'lari (sag tarafta eos/pad ile dolabilir) temizlemeden once trim
        text = tokenizer.decode(generated, skip_special_tokens=True)
        n_tokens = int((generated != tokenizer.pad_token_id).sum().item())
        truncated = n_tokens >= MAX_NEW_TOKENS

        results.append({
            "id": item["id"],
            "set": item["set"],
            "prompt": item["prompt"],
            "output": text,
            "truncated": truncated,
            "output_tokens": n_tokens,
        })
        print(f"[{item['id']}/78] tokens={n_tokens} truncated={truncated}")

    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=1)

with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
    json.dump(results, f, ensure_ascii=False, indent=1)

n_truncated = sum(1 for r in results if r["truncated"])
print(f"\nTAMAMLANDI: {len(results)} cikti, {n_truncated} kesilmis")
print(f"Kaydedildi: {OUTPUT_PATH}")
