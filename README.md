# 🏥 Hospital AI Assistant — Sistema de Suporte à Decisão Clínica (CDSS)

## Tech Challenge · Fase 3 — Pós-Graduação FIAP: IA para DEVs

Sistema de Suporte à Decisão Clínica (**CDSS**) em formato de **Chat Clínico Interativo**, orquestrado com **LangChain & LangGraph**, alimentado por um **LLM médico fine-tunado localmente em GPU** (`unsloth/Qwen2.5-7B-Instruct-bnb-4bit` + adaptador **LoRA Clínico**), **RAG Vetorial no Qdrant** (`intfloat/multilingual-e5-small`), base estruturada de **1.000 prontuários de pacientes**, **Guardrails multicamadas em conformidade com a Resolução CFM nº 2.314/2022** e observabilidade ponta a ponta via **LangSmith** e logs JSONL.

---

## 🌟 Principais Funcionalidades

1. **Chat Clínico Interativo (Web Streamlit & CLI):**
   - **Interface Web (`interface/web_interface.py` / `app_streamlit.py`):** Busca inteligente de pacientes (Nome, CPF ou Cartão SUS), roteiro didático guiado com 3 casos clínicos oficiais, anexo de até 5 laudos em PDF (upload externo ou sistema interno PACS/LIS), visualização do trace LangGraph, alertas de guardrail, painel de evidências RAG (explainability), catálogo de protocolos e aba de auditoria em tempo real.
   - **Interface CLI (`interface/cli_interface.py` / `main.py`):** Chat interativo via terminal com comandos de busca, cadastro de pacientes, upload de exames e execução dos roteiros de demonstração.
2. **LLM Local Fine-Tunado com Unsloth QLoRA (`Qwen2.5-7B-Instruct-bnb-4bit`):**
   - Inferência local residente em VRAM/CUDA (`hospital_ai_assistant/agent/clinical_llm.py`) com adaptador LoRA treinado sobre **1.560 exemplos** (`1.404` treino / `156` validação) combinando protocolos institucionais, farmacovigilância, raciocínio conversacional e **PubMedQA PT-BR**.
   - **Métricas do último treinamento (`NVIDIA GeForce RTX 4070 Ti`, 351 steps):** `lora_rank=16`, `lora_alpha=16`, `training_loss_final = 0.6912`, `perplexity = 1.996`.
3. **Orquestração Agêntica com LangGraph (`HospitalChatGraph`):**
   - Pipeline `StateGraph` de 6 nós com estado tipado (`ChatClinicalState`):
     `node_input_guardrail` → `node_patient_resolver` → `node_rag_retriever` → `node_clinical_reasoning` → `node_output_guardrail` → `node_audit_logger`.
   - Suporta **troca dinâmica de protocolo** durante a conversa caso o médico questione ou solicite outra diretriz institucional.
4. **RAG Vetorial Multicamadas com Qdrant (Docker Compose):**
   - Embeddings multilíngues **`intfloat/multilingual-e5-small`** (384 dimensões) rodando no Qdrant (`localhost:6333`).
   - **Base de Conhecimento Institucional:** **110 protocolos clínicos** (`protocols_100_diseases.json`), **ontologia de alergias e farmacovigilância** (`drug_allergies_ontology.json`) e **ingestão dinâmica de laudos em PDF** por paciente (`pypdf`).
5. **Base Estruturada de 1.000 Pacientes no Qdrant:**
   - Prontuários gerados a partir de `patients_1000.csv` e indexados no Qdrant (`patient_db.py`) com Nome, Idade, Gênero, CPF, Cartão SUS, Comorbidades, Medicações em uso contínuo, Alergias documentadas e Histórico Clínico.
6. **Guardrails Clínicos Multicamadas (Resolução CFM nº 2.314/2022):**
   - **Input Guardrail:** Bloqueio contra *prompt injection* e tentativas de manipulação de instruções.
   - **Output Guardrail (Determinístico + Agêntico):** Princípio de **Zero Prescrição Autônoma** (interceptação de ordens imperativas), cruzamento automático de **alergias documentadas no prontuário** e contraindicações do protocolo, citação de diretrizes e inclusão mandatória do Termo de Responsabilidade Médica (CRM).
7. **Observabilidade e Auditoria Completa:**
   - Rastreamento nativo no **LangSmith** (`LANGCHAIN_TRACING_V2`) e trilha local em `hospital_ai_assistant/audit_logs/clinical_audit.jsonl` com latência por turno, contagem de tokens, protocolos/exames recuperados e flags disparadas.
8. **Acervo de 8 Laudos Médicos em PDF (`data/exams/`):**
   - Laudos realistas prontos para demonstração cobrindo 4 pacientes: **Maria Helena Oliveira** (IAMCSST), **João da Silva** (Pneumonia / Alergia a Penicilinas), **Carlos Eduardo Santos** (AVCi / Trombólise) e **Ana Clara Souza** (Cetoacidose Diabética — CAD).

---

## 🏗️ Arquitetura do Fluxo (LangGraph)

```text
Entrada do Médico (+ PDFs opcionais)
        │
        ▼
[1] node_input_guardrail ──(bloqueado)──────────────────────┐
        │ (seguro)                                          │
        ▼                                                   │
[2] node_patient_resolver   (Prontuário no Qdrant / Sessão) │
        │                                                   │
        ▼                                                   │
[3] node_rag_retriever      (110 Protocolos + Alergias +    │
        │                    PDFs do Paciente no Qdrant)    │
        ▼                                                   │
[4] node_clinical_reasoning (Qwen2.5-7B-4bit + LoRA local)  │
        │                                                   │
        ▼                                                   │
[5] node_output_guardrail   (CFM 2.314/2022 + Alergias +    │
        │                    Guardrail Agêntico)            │
        ▼                                                   ▼
[6] node_audit_logger       (JSONL Local + LangSmith Trace)
        │
        ▼
Resposta Clínica Auditada ao Médico
```

---

## 📂 Estrutura do Projeto

```text
├── app_streamlit.py                      # Atalho para iniciar a interface Web Streamlit
├── docker-compose.yml                    # Provisionamento do Qdrant VectorDB (portas 6333/6334)
├── main.py                               # Ponto de entrada CLI, Demos, Auditoria e Web
├── requirements.txt                      # Dependências Python do ecossistema
├── .env.example                          # Modelo de variáveis de ambiente (LangSmith)
├── data/
│   └── exams/                            # 8 laudos médicos em PDF (4 pacientes) para RAG
├── interface/
│   ├── cli_interface.py                  # Chat interativo via Terminal (Rich + prompt_toolkit)
│   └── web_interface.py                  # Interface Web completa em Streamlit
└── hospital_ai_assistant/
    ├── agent/
    │   ├── chat_graph.py                 # Orquestrador StateGraph (LangGraph)
    │   ├── chat_state.py                 # Estado conversacional tipado
    │   └── clinical_llm.py               # Inferência local (Qwen2.5-7B-4bit + LoRA)
    ├── audit_logs/                       # Logs JSONL de auditoria conversacional
    ├── database/
    │   └── patient_db.py                 # Repositório de 1.000 prontuários no Qdrant
    ├── dataset/
    │   ├── build_sft_and_rag_dataset.py  # Pipeline de curadoria SFT + RAG
    │   ├── raw/
    │   │   ├── patients_1000.csv               # Base de 1.000 pacientes
    │   │   ├── protocols_100_diseases.json     # 110 protocolos clínicos institucionais
    │   │   ├── drug_allergies_ontology.json    # Ontologia de farmacovigilância/alergias
    │   │   └── pubmedqa/ori_pqal_ptbr.json     # Corpus PubMedQA traduzido (PT-BR)
    │   └── processed/
    │       ├── rag_hospital_knowledge_base.json
    │       ├── sft_unsloth_medical_train.jsonl
    │       └── sft_unsloth_medical_val.jsonl
    ├── finetuning/
    │   └── train_unsloth.py              # Script de Fine-Tuning QLoRA com Unsloth
    ├── rag_engine/
    │   └── qdrant_rag.py                 # Motor RAG Qdrant (multilingual-e5-small) + PDFs
    ├── safety/
    │   ├── guardrails.py                 # Guardrails determinísticos + agênticos (CFM)
    │   └── audit_logger.py               # Auditor estruturado JSONL / LangSmith
    └── storage/
        └── lora_clinical_adapter/        # Metadados, tokenizer e métricas do adaptador LoRA
```

---

## 🚀 Como Executar o Projeto

### 1. Pré-requisitos e Instalação
- Python **3.10+**, Docker & Docker Compose
- GPU NVIDIA com CUDA (recomendado ≥ 8–12 GB VRAM para `Qwen2.5-7B-Instruct-bnb-4bit`; há fallback automático para CPU)

```bash
# Criar e ativar ambiente virtual
python3 -m venv .venv
source .venv/bin/activate

# Instalar dependências
pip install -r requirements.txt

# Configurar variáveis de ambiente (opcional para LangSmith)
cp .env.example .env
```

### 2. Iniciar o VectorDB Qdrant (Docker Compose)
```bash
docker compose up -d
```
O dashboard do Qdrant ficará disponível em `http://localhost:6333/dashboard`.

### 3. Preparar a Base Vetorial (Primeira Execução)
Popula os 1.000 prontuários, constrói o dataset SFT/RAG e indexa os protocolos no Qdrant:
```bash
python main.py --prepare-all
```

### 4. Gerar o Adaptador LoRA Local (Fine-Tuning com Unsloth)
> **Nota:** Os pesos binários `adapter_model.safetensors` (~155 MB) são ignorados no Git pelo limite de 100 MB do GitHub. Execute o script abaixo em ambiente com GPU CUDA para gerar o adaptador em `hospital_ai_assistant/storage/lora_clinical_adapter/` (os metadados e métricas do treino de referência já estão versionados no repositório):

```bash
# Treinamento QLoRA com GPU CUDA + Unsloth
python hospital_ai_assistant/finetuning/train_unsloth.py

# Ou validação estruturada (dry-run sem GPU)
python hospital_ai_assistant/finetuning/train_unsloth.py --dry-run
```

### 5. Iniciar a Interface Web (Streamlit)
```bash
python main.py --web
# ou diretamente:
streamlit run app_streamlit.py
```
Acesse no navegador: `http://localhost:8501`.

### 6. Iniciar o Chat Interativo via Terminal (CLI)
```bash
python main.py
```
Comandos disponíveis no terminal:
- `/paciente <nome ou CPF>` — busca e vincula o prontuário do paciente;
- `/cadastrar` — fluxo guiado de admissão de novo paciente;
- `/upload <caminho.pdf>` — ingere um laudo em PDF no Qdrant;
- `/demo <1, 2 ou 3>` — executa os roteiros automáticos de demonstração;
- `/audit` — exibe logs de auditoria e latência por nó;
- `/limpar` — reinicia o contexto da conversa;
- `/sair` — encerra a sessão.

---

## 🧪 Casos Clínicos para Demonstração

Na Interface Web (painel de roteiro guiado) ou via CLI (`python main.py --demo <1|2|3>`):

| Caso | Paciente | Quadro Clínico | Protocolo Esperado | Foco da Validação |
| :--- | :--- | :--- | :--- | :--- |
| **Caso 1** | Maria Helena Oliveira | Dor torácica retroesternal + ECG Supra ST + Troponina | `PROTOCOLO-CLIN-002` (IAMCSST) | Correlação de ECG/Troponina e contraindicação a nitratos na hipotensão (`PA 80/50`) |
| **Caso 2** | João da Silva | Tosse produtiva, febre, DPOC + Hemograma/PCR `94.2 mg/L` | `PROTOCOLO-CLIN-011` (PAC) | Alergia grave a Penicilinas (bloqueio de betalactâmicos + alternativa segura) |
| **Caso 3** | Carlos Eduardo Santos | Hemiparesia súbita `< 4.5h` + TC sem hemorragia + RNI `1.38` | `PROTOCOLO-CLIN-031` (AVCi) | Elegibilidade para trombólise e contraindicação com `PA 195/110 mmHg` |
| **Manual** | Ana Clara Souza | DM1, vômitos, hálito cetônico + Gasometria `pH 7.15` / `K+ 4.8` | `PROTOCOLO-CLIN-041` (CAD) | Fluxo 100% manual (busca, upload de PDFs, regra de `K+ < 3.3` e alergia a Sulfas) |

---

## 📊 Observabilidade e Auditoria (LangSmith & JSONL)

Para ativar o rastreamento em tempo real no **LangSmith**, preencha o arquivo `.env`:

```env
LANGCHAIN_TRACING_V2=true
LANGCHAIN_API_KEY="sua_chave_langsmith"
LANGCHAIN_PROJECT="hospital-ai-assistant"
```

Para consultar os últimos registros de auditoria diretamente pelo terminal:
```bash
python main.py --audit
```
