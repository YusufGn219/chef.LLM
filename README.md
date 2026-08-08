# Chef.LLM

`unsloth/gemma-4-12b-it` tabanından DPO (Direct Preference Optimization)
ile fine-tune edilmiş, Türkçe bir yemek tarifi modeli.

- **Model (Hugging Face):** [LordAhkam/chef-llm-dpo-v2](https://huggingface.co/LordAhkam/chef-llm-dpo-v2)
- **Yazı (Medium):** _yakında_ — eğitim sürecinin tam hikayesi için

## Sonuç

78 promptluk bir Türkçe tarif benchmark'ında (`benchmark/eval_78_prompts.json`),
4 kriterli bir rubrikle (format, akıcılık, faktüel tutarlılık, detay)
"avcı modu" puanlamayla değerlendirildi:

| Model | Skor |
|---|---|
| Eğitilmemiş base model (`unsloth/gemma-4-12b-it`) | %85.1 |
| **Bu model** | **%89.4** |

## İçerik

```
chef_llm_dpo_train_v2.py         Eğitim script'i (LoRA r=8, trl'nin DPOTrainer'ı)
dpo_eval_inference_batched.py    Benchmark'ı adapter'a karşı çalıştıran eval script'i
round3_setup_env.sh              EC2 ortam kurulumu (Python, torch, trl, peft)
data/dpo_dataset_v2.jsonl        Modeli eğiten 47 çiftlik prompt/chosen/rejected veri seti
benchmark/eval_78_prompts.json   Değerlendirme test seti
```

## Eğitim Detayları

- **Yöntem:** DPO, `trl`'nin `DPOTrainer`'ı ile
- **LoRA config:** r=8, alpha=16, hedef modüller: `q_proj`, `k_proj`, `v_proj`, `o_proj`
- **Hiperparametreler:** beta=0.1, 3 epoch, learning_rate=5e-6
- **Donanım:** 1x NVIDIA L40S (48GB), AWS `g6e.xlarge`

## Nasıl Kullanılır

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import torch

base_model = AutoModelForCausalLM.from_pretrained(
    "unsloth/gemma-4-12b-it", torch_dtype=torch.bfloat16, device_map="auto"
)
model = PeftModel.from_pretrained(base_model, "LordAhkam/chef-llm-dpo-v2")
tokenizer = AutoTokenizer.from_pretrained("LordAhkam/chef-llm-dpo-v2")

messages = [{"role": "user", "content": "Mercimek çorbası tarifi verir misin?"}]
inputs = tokenizer.apply_chat_template(
    messages, tokenize=True, add_generation_prompt=True,
    return_tensors="pt", return_dict=True
).to(model.device)
output = model.generate(**inputs, max_new_tokens=1024, do_sample=True, temperature=0.7, top_p=0.9)
print(tokenizer.decode(output[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True))
```

Bilinen sınırlamalar için [Hugging Face model kartına](https://huggingface.co/LordAhkam/chef-llm-dpo-v2) bakın.

## Eğitim/Eval Script'lerini Çalıştırma

`chef_llm_dpo_train_v2.py` ve `dpo_eval_inference_batched.py`, bu proje
kapsamında bir **AWS EC2 `g6e.xlarge`** instance'ında (1x NVIDIA L40S,
48GB VRAM) çalıştırıldı. Script'lerdeki dosya yolları (`/mnt/data/...`,
`/home/ec2-user/...`) o ortama özgüdür — kendi makinende veya farklı bir
sunucuda çalıştırmak istersen bu yolları düzenlemen gerekir.

Sıralama:
1. `round3_setup_env.sh` — ortamı kurar (NVMe instance store mount,
   Python 3.12 venv, torch/peft/trl/transformers kurulumu, tokenizer
   düzeltmesi). Bir GPU instance'ında `bash round3_setup_env.sh` ile
   çalıştırılır.
2. `chef_llm_dpo_train_v2.py` — `data/dpo_dataset_v2.jsonl`'ı okuyup
   LoRA adapter'ı eğitir (~3.5 dakika sürdü).
3. `dpo_eval_inference_batched.py <adapter_dir> <output_path> [batch_size]`
   — eğitilen adapter'ı `benchmark/eval_78_prompts.json` promptlarına
   karşı çalıştırıp çıktıları kaydeder.

## Lisans

Kod MIT Lisansı altındadır (bkz. `LICENSE`). `data/dpo_dataset_v2.jsonl`
ve `benchmark/eval_78_prompts.json`, bu proje kapsamında üretilmiştir.
Model, base modelinden (`unsloth/gemma-4-12b-it`) miras alınan
[Gemma Kullanım Şartları](https://ai.google.dev/gemma/terms)'na tabidir.
