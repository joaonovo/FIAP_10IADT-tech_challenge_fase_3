"""
Módulo de Construção do Dataset de Fine-Tuning e RAG (PubMedQA + 100 Doenças)
=============================================================================

Tech Challenge - Fase 3: IA para DEVs (FIAP)
Combina:
1. Base PubMedQA (Perguntas, Contextos e Respostas Científicas Baseadas em Evidências)
2. Base de 100 Doenças e Protocolos Institucionais do Hospital Universitário FIAP

Executa:
- Higienização e Pré-processamento textual;
- Anonimização conforme LGPD e HIPAA;
- Formatação em ChatML / LLaMA 3.2 Instruct;
- Split de Treino (90%) e Validação (10%);
- Exportação de Corpus para Ingestão no VectorDB (Qdrant).
"""

import os
import re
import json
import random
import logging
from typing import Dict, List, Any, Tuple

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

RAW_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "raw")
PROCESSED_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "processed")
PUBMED_PTBR_PATH = os.path.join(RAW_DIR, "pubmedqa", "ori_pqal_ptbr.json")
PUBMED_PATH = os.path.join(RAW_DIR, "pubmedqa", "ori_pqal.json")
PROTOCOLS_PATH = os.path.join(RAW_DIR, "protocols_100_diseases.json")
ALLERGY_PATH = os.path.join(RAW_DIR, "drug_allergies_ontology.json")

TRAIN_OUTPUT = os.path.join(PROCESSED_DIR, "sft_unsloth_medical_train.jsonl")
VAL_OUTPUT = os.path.join(PROCESSED_DIR, "sft_unsloth_medical_val.jsonl")
RAG_KNOWLEDGE_OUTPUT = os.path.join(PROCESSED_DIR, "rag_hospital_knowledge_base.json")

SYSTEM_PROMPT = (
    "Você é o Assistente Virtual Médico do Hospital Universitário FIAP (CDSS). "
    "Sua função é auxiliar o corpo clínico médico com raciocínio clínico baseado em evidências (CoT), "
    "sugestões de conduta conforme protocolos institucionais e literatura biomédica (PubMedQA). "
    "REGRAS BIOÉTICAS E DE AUDITORIA MANDATÓRIAS:\n"
    "1. NUNCA prescreva diretamente ou emita ordens médicas autônomas ao paciente.\n"
    "2. Todas as condutas farmacológicas são SUGESTÕES DIRETIVAS sujeitas à validação presencial humana com CRM ativo.\n"
    "3. Cite explicitamente a fonte institucional ou científica utilizada na fundamentação.\n"
    "4. Alerte sobre contraindicações graves e exames laboratoriais/imagem mandatórios."
)


def sanitize_text(text: str) -> str:
    """Higieniza ruídos, quebras irregulares e espaços repetidos."""
    if not text:
        return ""
    text = re.sub(r"<[^>]+>", "", text)  # Remove tags HTML se houver
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def anonymize_clinical_text(text: str) -> str:
    """Remove potenciais dados sensíveis (PII) segundo diretrizes LGPD/HIPAA."""
    # Anonimiza potenciais nomes de médicos/pesquisadores
    text = re.sub(r"\b(Dr\.|Dra\.|Prof\.|Professor)\s+[A-Z][a-z]+(\s+[A-Z][a-z]+)*", "[EQUIPE_MÉDICA_DESIDENTIFICADA]", text)
    # Anonimiza CPFs
    text = re.sub(r"\b\d{3}\.\d{3}\.\d{3}-\d{2}\b", "[CPF_DESIDENTIFICADO]", text)
    # Anonimiza Telefones
    text = re.sub(r"\(?\d{2}\)?\s?9?\d{4}-?\d{4}", "[CONTATO_DESIDENTIFICADO]", text)
    return text


def build_protocol_conversations(protocols: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Gera pares conversacionais instrucionais para cada protocolo de doença."""
    dataset = []
    
    question_templates = [
        "Qual o protocolo de conduta imediata para um paciente com quadro de {disease} ({icd10})?",
        "Paciente admitido com suspeita de {disease}. Quais são os sintomas típicos, exames recomendados e contraindicações no hospital?",
        "Poderia orientar a conduta clínica e o manejo terapêutico de primeira linha para {disease}?",
        "Quais exames devem ser solicitados e qual a conduta padronizada para {disease} ({icd10}) no FIAP?"
    ]

    for p in protocols:
        name = p["disease_name"]
        icd = p["icd10"]
        cat = p["category"]
        symptoms = p["symptoms"]
        exams = p["recommended_exams"]
        treatment = p["treatment_protocol"]
        warnings = p["contraindications_and_warnings"]
        proto_id = p["protocol_id"]

        for q_tpl in question_templates:
            user_msg = q_tpl.format(disease=name, icd10=icd)
            
            assistant_response = (
                f"### Raciocínio Clínico e Conduta Institucional\n\n"
                f"**1. Avaliação Clínica e Hipótese:**\n"
                f"- Patologia: {name} (CID-10: {icd}) - Especialidade: {cat}.\n"
                f"- Sintomas Típicos e Apresentação: {symptoms}\n\n"
                f"**2. Investigação e Exames Recomendados:**\n"
                f"{exams}\n\n"
                f"**3. Protocolo Terapêutico de Suporte (Sugestão de 1ª Linha):**\n"
                f"{treatment}\n\n"
                f"**4. Alertas e Contraindicações Críticas:**\n"
                f"{warnings}\n\n"
                f"**5. Fonte e Rastreabilidade Institucional:**\n"
                f"Diretriz: [{proto_id}] - Manual de Condutas Clínicas do Hospital Universitário FIAP (Nível de Evidência 1A).\n\n"
                f"⚠️ *Nota de Segurança / CFM: Este sistema de IA atua exclusivamente como suporte à decisão clínica. "
                f"A prescrição e conduta definitiva são de responsabilidade exclusiva e indelegável do médico assistente (CRM).*"
            )

            dataset.append({
                "source": "hospital_protocols",
                "protocol_id": proto_id,
                "disease": name,
                "icd10": icd,
                "category": cat,
                "system": SYSTEM_PROMPT,
                "instruction": user_msg,
                "output": assistant_response
            })

        # Exemplo conversacional extra: apresentação clínica por sintomas + contexto RAG (sem citar o nome da doença)
        rag_user_msg = (
            f"[DIRETRIZES INSTITUCIONAIS RECUPERADAS VIA RAG]\n"
            f"Diretriz HFIAP: {name} ({proto_id}) - CID-10: {icd}\n"
            f"Sintomas Clínicos Típicos: {symptoms}\n"
            f"Exames Recomendados: {exams}\n"
            f"Conduta Terapêutica: {treatment}\n"
            f"Contraindicações: {warnings}\n\n"
            f"Solicitação atual do Médico Assistente:\n"
            f"Paciente admitido apresentando {symptoms.lower()}. Qual a hipótese principal, exames prioritários e conduta segura segundo a diretriz?"
        )
        rag_assistant_resp = (
            f"Com base na apresentação clínica descrita e na diretriz institucional **{proto_id} ({name} — CID-10: {icd})**, "
            f"a principal hipótese diagnóstica é **{name}**.\n\n"
            f"- **Investigação Prioritária:** {exams}\n"
            f"- **Sugestão Terapêutica (sujeita à validação do CRM):** {treatment}\n"
            f"- **Pontos Críticos de Segurança e Contraindicações:** {warnings}\n\n"
            f"Caso existam alergias documentadas ou resultados de exames complementares, informe no chat para reavaliarmos a linha de cuidado."
        )
        dataset.append({
            "source": "hospital_protocols_conversational",
            "protocol_id": proto_id,
            "disease": name,
            "icd10": icd,
            "category": cat,
            "system": SYSTEM_PROMPT,
            "instruction": rag_user_msg,
            "output": rag_assistant_resp
        })

    return dataset


def build_allergy_conversations(allergy_file: str) -> List[Dict[str, Any]]:
    """Gera exemplos conversacionais de farmacovigilância e manejo seguro de alergias em PT-BR."""
    if not os.path.exists(allergy_file):
        return []
    with open(allergy_file, "r", encoding="utf-8") as f:
        data = json.load(f)
    classes = data.get("classes", {})
    dataset = []
    for ckey, cinfo in classes.items():
        cname = cinfo.get("name", ckey)
        drugs = cinfo.get("contraindicated_drugs", [])
        safe_alts = cinfo.get("safe_alternatives", {})
        drugs_str = ", ".join(drugs)
        alts_lines = "\n".join(f"- **{k}:** {v}" for k, v in safe_alts.items())
        user_msg = (
            f"Paciente com histórico de alergia grave (anafilaxia prévia) a {cname}. "
            f"Quais medicamentos estão contraindicados e quais alternativas seguras podemos considerar?"
        )
        assistant_resp = (
            f"🚨 **Alerta de Farmacovigilância (Diretriz ALERGIA-{ckey.upper()}):**\n\n"
            f"Em pacientes com hipersensibilidade grave a **{cname}**, estão **absolutamente contraindicados**: {drugs_str}.\n\n"
            f"**Alternativas Terapêuticas Seguras (sujeitas à validação presencial do médico assistente):**\n"
            f"{alts_lines}\n\n"
            f"⚠️ *Fonte: Comissão de Farmacovigilância do Hospital Universitário FIAP.*"
        )
        dataset.append({
            "source": "hospital_pharmacovigilance",
            "protocol_id": f"ALERGIA-{ckey.upper()}",
            "system": SYSTEM_PROMPT,
            "instruction": user_msg,
            "output": assistant_resp
        })
    return dataset


def build_pubmedqa_conversations(pubmed_file: str, max_samples: int = 1000) -> List[Dict[str, Any]]:
    """Carrega o PubMedQA (ori_pqal.json) e o transforma em diálogos clínicos de evidência científica."""
    if not os.path.exists(pubmed_file):
        logger.warning("Arquivo PubMedQA não encontrado em %s. Prosseguindo sem ele.", pubmed_file)
        return []

    with open(pubmed_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    logger.info("PubMedQA carregado com %d amostras brutas. Processando até %d...", len(data), max_samples)
    dataset = []

    for pmid, item in list(data.items())[:max_samples]:
        question = sanitize_text(item.get("QUESTION", ""))
        contexts = item.get("CONTEXTS", [])
        clean_context = sanitize_text(" ".join(contexts))
        decision = item.get("final_decision", "").upper()
        long_answer = sanitize_text(item.get("LONG_ANSWER", ""))

        if not question or not clean_context:
            continue

        clean_question = anonymize_clinical_text(question)
        clean_context = anonymize_clinical_text(clean_context)
        clean_answer = anonymize_clinical_text(long_answer)

        user_msg = (
            f"Com base na literatura biomédica indexada (PubMed/MEDLINE), avalie a seguinte questão clínica:\n"
            f"\"{clean_question}\"\n\n"
            f"Contexto do Estudo:\n{clean_context}"
        )

        decision_trans = {
            "YES": "SIM (Evidência favorável)",
            "NO": "NÃO (Evidência desfavorável)",
            "MAYBE": "INCONCLUSIVO / REQUER PARCIMÔNIA"
        }.get(decision, decision)

        assistant_response = (
            f"### Síntese de Evidência Científica (PubMedQA / PMID: {pmid})\n\n"
            f"**Decisão Clínica Baseada no Estudo:** {decision_trans}\n\n"
            f"**Análise Crítica dos Resultados:**\n{clean_answer}\n\n"
            f"**Fonte Científica:**\n"
            f"PubMed ID: {pmid} | Conforme base de dados MEDLINE/NLM.\n\n"
            f"⚠️ *Nota: A aplicação clínica de estudos requer correlação com a condição individual do paciente e avaliação do médico assistente.*"
        )

        dataset.append({
            "source": "pubmedqa",
            "pmid": pmid,
            "system": SYSTEM_PROMPT,
            "instruction": user_msg,
            "output": assistant_response
        })

    return dataset


def main():
    os.makedirs(PROCESSED_DIR, exist_ok=True)

    # 1. Carrega protocolos pré-gerados das 100 doenças (JSON consolidado)
    if not os.path.exists(PROTOCOLS_PATH):
        raise FileNotFoundError(f"Arquivo de protocolos não encontrado em {PROTOCOLS_PATH}.")

    with open(PROTOCOLS_PATH, "r", encoding="utf-8") as f:
        protocols = json.load(f)

    # 2. Constrói dataset dos protocolos + exemplos conversacionais RAG em PT-BR
    proto_samples = build_protocol_conversations(protocols)
    logger.info("Protocolos transformados em %d amostras de instrução em PT-BR.", len(proto_samples))

    # 2b. Constrói dataset de farmacovigilância / alergias em PT-BR
    allergy_samples = build_allergy_conversations(ALLERGY_PATH)
    logger.info("Farmacovigilância/Alergias transformadas em %d amostras em PT-BR.", len(allergy_samples))

    # 3. Constrói dataset do PubMedQA (prioriza versão traduzida em PT-BR quando disponível)
    if os.path.exists(PUBMED_PTBR_PATH):
        logger.info("Base PubMedQA traduzida em PT-BR detectada em %s!", PUBMED_PTBR_PATH)
        pubmed_samples = build_pubmedqa_conversations(PUBMED_PTBR_PATH, max_samples=1000)
    else:
        logger.info("Usando PubMedQA original (EN) limitado a 400 amostras para balancear com PT-BR.")
        pubmed_samples = build_pubmedqa_conversations(PUBMED_PATH, max_samples=400)
    logger.info("PubMedQA transformado em %d amostras de instrução.", len(pubmed_samples))

    # 4. Combina e mistura
    full_dataset = proto_samples + allergy_samples + pubmed_samples
    random.seed(42)
    random.shuffle(full_dataset)

    # 5. Split de Treino (90%) e Validação (10%)
    split_idx = int(0.9 * len(full_dataset))
    train_data = full_dataset[:split_idx]
    val_data = full_dataset[split_idx:]

    logger.info("Split definido: %d Treino | %d Validação | Total: %d", len(train_data), len(val_data), len(full_dataset))

    with open(TRAIN_OUTPUT, "w", encoding="utf-8") as f:
        for item in train_data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    with open(VAL_OUTPUT, "w", encoding="utf-8") as f:
        for item in val_data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    # 6. Salva base de RAG institucional (protocolos + metadados detalhados para indexação no Qdrant)
    rag_documents = []
    for p in protocols:
        content = (
            f"PROTOCOLO CLÍNICO INSTITUCIONAL: {p['protocol_id']}\n"
            f"Patologia: {p['disease_name']} (CID-10: {p['icd10']})\n"
            f"Especialidade: {p['category']}\n"
            f"Sintomas e Quadro Clínico:\n{p['symptoms']}\n\n"
            f"Exames Indicados:\n{p['recommended_exams']}\n\n"
            f"Protocolo de Tratamento de 1ª Linha:\n{p['treatment_protocol']}\n\n"
            f"Contraindicações e Alertas Críticos:\n{p['contraindications_and_warnings']}\n"
            f"Fonte Normativa: Manual de Práticas Médicas do Hospital Universitário FIAP"
        )
        rag_documents.append({
            "doc_id": p["protocol_id"],
            "title": f"Protocolo: {p['disease_name']}",
            "disease": p["disease_name"],
            "category": p["category"],
            "icd10": p["icd10"],
            "content": content
        })

    with open(RAG_KNOWLEDGE_OUTPUT, "w", encoding="utf-8") as f:
        json.dump(rag_documents, f, ensure_ascii=False, indent=2)

    print("======================================================================")
    print("✅ PROCESSAMENTO DE DADOS CONCLUÍDO COM SUCESSO!")
    print(f" • Dataset de Treino SFT (ChatML): {TRAIN_OUTPUT} ({len(train_data)} linhas)")
    print(f" • Dataset de Validação SFT:      {VAL_OUTPUT} ({len(val_data)} linhas)")
    print(f" • Base de RAG Institucional:     {RAG_KNOWLEDGE_OUTPUT} ({len(rag_documents)} docs)")
    print("======================================================================")


if __name__ == "__main__":
    main()

