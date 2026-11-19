---
base_model: unsloth/Qwen2.5-7B-Instruct-bnb-4bit
library_name: peft
pipeline_tag: text-generation
language:
- pt
tags:
- base_model:adapter:unsloth/Qwen2.5-7B-Instruct-bnb-4bit
- lora
- qlora
- transformers
- unsloth
- medical
- clinical
- cdss
- portuguese
- fiap
---

# 🏥 HFIAP Qwen2.5-7B Clinical LoRA (`joaonovo/hfiap-qwen2.5-7b-clinical-lora`)

Adaptador **QLoRA (4-bit)** treinado sobre `unsloth/Qwen2.5-7B-Instruct-bnb-4bit` em **Português do Brasil (PT-BR)** para o **Hospital AI Assistant — Sistema de Suporte à Decisão Clínica (CDSS)**, desenvolvido no **Tech Challenge — Fase 3 (Pós-Graduação FIAP: IA para DEVs)**.

---

## 📋 Resumo do Modelo

- **Desenvolvido por:** João Novo (FIAP — Pós-Graduação IA para DEVs, 10IADT Fase 3)
- **Modelo Base:** [`unsloth/Qwen2.5-7B-Instruct-bnb-4bit`](https://huggingface.co/unsloth/Qwen2.5-7B-Instruct-bnb-4bit) (`Qwen2ForCausalLM`)
- **Tipo:** Adaptador PEFT / QLoRA (`CAUSAL_LM`)
- **Idioma:** Português do Brasil (`pt-BR`)
- **Repositório do Projeto:** [github.com/joaonovo/FIAP_10IADT-tech_challenge_fase_3](https://github.com/joaonovo/FIAP_10IADT-tech_challenge_fase_3)

O adaptador especializa o modelo base em raciocínio clínico estruturado fundamentado em contexto RAG (protocolos hospitalares institucionais, farmacovigilância/alergias e laudos de exames em PDF), operando em conjunto com um pipeline **LangGraph** e **Guardrails de conformidade com a Resolução CFM nº 2.314/2022**.

---

## 🎯 Usos Previstos e Limitações

### Uso Direto
- Suporte à decisão clínica (**CDSS**) em ambiente acadêmico/demonstrativo, acoplado a recuperação vetorial (**Qdrant RAG**) de protocolos hospitalares e exames laboratoriais/imagem.
- Formulação de hipóteses diagnósticas, correlação clínico-laboratorial, alertas de contraindicações e sugestões de conduta condicionadas à validação médica.

### Fora de Escopo (Out-of-Scope)
- **Uso clínico assistencial real em pacientes humanos** sem validação científica, aprovação regulatória e supervisão médica presencial.
- **Prescrição medicamentosa autônoma:** em conformidade com a **Resolução CFM nº 2.314/2022**, a prescrição formal e a decisão diagnóstica são atribuições exclusivas e indelegáveis de médico registrado no CRM.

### Riscos e Limitações
- Como modelo de linguagem de 7B parâmetros quantizado em 4-bit, está sujeito a alucinações pontuais quando usado sem contexto RAG ou sem camada externa de guardrails. Deve sempre ser utilizado acoplado a recuperação de evidências e revisão humana.

---

## 🚀 Como Usar

### Carregamento com `transformers` + `peft`
```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

base_model_id = "unsloth/Qwen2.5-7B-Instruct-bnb-4bit"
adapter_id = "joaonovo/hfiap-qwen2.5-7b-clinical-lora"

tokenizer = AutoTokenizer.from_pretrained(adapter_id, trust_remote_code=True)
base_model = AutoModelForCausalLM.from_pretrained(
    base_model_id,
    device_map="auto",
    dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16,
    trust_remote_code=True,
)
model = PeftModel.from_pretrained(base_model, adapter_id)
model.eval()

messages = [
    {
        "role": "system",
        "content": "Você é o Assistente Virtual Médico do Hospital Universitário FIAP (CDSS). Responda em português do Brasil com base nos protocolos institucionais."
    },
    {
        "role": "user",
        "content": "Paciente 58 anos com dor torácica retroesternal opressiva há 40 min irradiada para MSE e sudorese fria. Qual a principal hipótese e exames imediatos?"
    }
]

prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
with torch.no_grad():
    outputs = model.generate(**inputs, max_new_tokens=400, do_sample=False)

print(tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True))
```

### Download Manual para o Projeto
```bash
hf download joaonovo/hfiap-qwen2.5-7b-clinical-lora \
  --local-dir hospital_ai_assistant/storage/lora_clinical_adapter
```

---

## ⚙️ Detalhes do Treinamento

### Dataset (1.560 amostras — `1.404` treino / `156` validação)
| Fonte | Amostras | Descrição |
| :--- | :---: | :--- |
| `pubmedqa` | 1.000 QA | Corpus PubMedQA traduzido para Português (PT-BR) |
| `hospital_protocols` | 440 QA | 110 protocolos clínicos institucionais (sintomas, exames, conduta e contraindicações) |
| `hospital_protocols_conversational` | 110 QA | Raciocínio clínico conversacional estruturado por protocolo |
| `hospital_pharmacovigilance` | 10 QA | Diretrizes de alergias medicamentosas e alternativas seguras |

### Hiperparâmetros e Configuração QLoRA
- **Framework:** Unsloth AI (`FastLanguageModel`) + HuggingFace `Trainer`
- **Quantização:** 4-bit NormalFloat (NF4) via `bitsandbytes`
- **LoRA Rank (`r`):** `16` | **LoRA Alpha:** `16` | **Dropout:** `0`
- **Módulos Alvo:** `q_proj`, `k_proj`, `v_proj`, `o_proj`, `gate_proj`, `up_proj`, `down_proj`
- **Comprimento Máximo (`max_seq_length`):** `1024` tokens (com *loss masking* exclusivo na resposta do assistente)
- **Batch Efetivo:** `4` (`per_device_train_batch_size=1`, `gradient_accumulation_steps=4`)
- **Otimizador / Scheduler:** `adamw_8bit`, `cosine`, `learning_rate=5e-5`, `weight_decay=0.01`, `seed=42`
- **Duração:** `1` época completa (`351` steps)

### Resultados e Infraestrutura
- **GPU:** NVIDIA GeForce RTX 4070 Ti (12 GB VRAM)
- **Loss Final de Treino:** `0.6912`
- **Perplexidade de Treino:** `1.996`
- **Versão PEFT:** `0.21.2`
