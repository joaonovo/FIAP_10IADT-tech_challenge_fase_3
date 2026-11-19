"""
Módulo de Orquestração com LangGraph (Chat Clínico Interativo com Guardrails)
============================================================================

Tech Challenge - Fase 3: IA para DEVs (FIAP)
Arquitetura:
- LangChain Core + LangGraph StateGraph
- Guardrails Clínicos Multicamadas (Input + Output Guardrails)
- RAG Vetorial Qdrant (Protocolos Institucionais + Exames do Paciente)
- Prontuário Estruturado de 1.000 Pacientes (Qdrant VectorDB / RAG)
- Auditoria Completa e Observabilidade via LangSmith
"""

import os
import sys
import re
import time
import json
import logging
from typing import Dict, Any, List, Optional

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

# Garante resolução de pacotes raiz
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
root_dir = os.path.dirname(parent_dir)
for p in [current_dir, parent_dir, root_dir]:
    if p not in sys.path:
        sys.path.insert(0, p)

from langgraph.graph import StateGraph, END
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage, BaseMessage

# Configura LangSmith se chaves estiverem presentes no ambiente
LANGSMITH_TRACING = os.getenv("LANGCHAIN_TRACING_V2", "false").lower() == "true"
LANGSMITH_API_KEY = os.getenv("LANGCHAIN_API_KEY", "")
LANGSMITH_PROJECT = os.getenv("LANGCHAIN_PROJECT", "hospital-ai-assistant")

try:
    from langsmith import Client, traceable
    langsmith_client = Client() if LANGSMITH_API_KEY else None
except Exception:
    langsmith_client = None
    def traceable(*args, **kwargs):
        def decorator(func):
            return func
        return decorator

try:
    from hospital_ai_assistant.agent.chat_state import ChatClinicalState
    from hospital_ai_assistant.database.patient_db import patient_db
    from hospital_ai_assistant.rag_engine.qdrant_rag import rag_manager
    from hospital_ai_assistant.safety.audit_logger import ClinicalAuditLogger
    from hospital_ai_assistant.safety.guardrails import ClinicalGuardrails
    from hospital_ai_assistant.agent.clinical_llm import clinical_llm
except ImportError:
    from agent.chat_state import ChatClinicalState
    from database.patient_db import patient_db
    from rag_engine.qdrant_rag import rag_manager
    from safety.audit_logger import ClinicalAuditLogger
    from safety.guardrails import ClinicalGuardrails
    from agent.clinical_llm import clinical_llm

logger = logging.getLogger(__name__)


def extract_executive_exam_summary(content: str, filename: str) -> str:
    """
    Analisa e resume os laudos de exames clínicos de forma aberta e generalista,
    extraindo dinamicamente achados alterados/críticos e o parecer/laudo conclusivo oficial do exame.
    """
    if not content or not content.strip():
        return f"• **`{filename}`**: (Arquivo anexado sem conteúdo legível extraído)"

    # 1. Extração estruturada do Parecer Clínico / Conclusão Diagnóstica oficial
    conclusion_text = ""
    conclusion_match = re.search(
        r"(?:PARECER CL[ÍI]NICO\s*/\s*CONCLUS[ÃA]O DIAGN[ÓO]STICA|PARECER CL[ÍI]NICO|CONCLUS[ÃA]O DIAGN[ÓO]STICA|IMPRESS[ÃA]O DIAGN[ÓO]STICA)(.*?)(?:Dr(?:a)?\.|\Z)",
        content,
        re.DOTALL | re.IGNORECASE
    )
    if conclusion_match:
        c = conclusion_match.group(1).strip()
        c = re.sub(r"^[:\s]*(?:LAUDO|RESULTADO|CONCLUSÃO|PARECER)[:\s]*", "", c, flags=re.IGNORECASE).strip()
        c = re.sub(r"\s+", " ", c)
        if len(c) > 10:
            conclusion_text = c

    # 2. Extração dos marcadores com valores críticos ou alterados
    findings = []
    lines = [l.strip() for l in content.split("\n") if l.strip()]
    crit_regex = re.compile(
        r"(?:CR[ÍI]TICO|ALTERAD[OA]|ELEVAD[OA]|AUMENTAD[OA]|REDUZID[OA]|DIMINU[ÍI]D[OA]|POSITIVO|REAGENTE|ANORMAL|ISQUEMIA|HEMORRAGIA|INFARTO|SUPRADESN[ÍI]VEL|INFRADESN[ÍI]VEL|ACIDOSE)",
        re.IGNORECASE
    )
    for idx, line in enumerate(lines):
        if crit_regex.search(line) and len(line) < 160:
            if not any(header in line.lower() for header in ["referência", "valores de", "metodologia"]):
                prev = lines[idx - 1] if idx > 0 and len(lines[idx - 1]) < 60 and not any(h in lines[idx - 1].lower() for h in ["referência", "unidade", "resultado", "marcador", "laudo de"]) else ""
                clean_item = f"{prev}: {line}" if prev else line
                clean_item = re.sub(r"\s+", " ", clean_item).strip()
                if clean_item not in findings and len(clean_item) > 4:
                    findings.append(clean_item)

    parts = []
    if findings:
        parts.append(f"**Achados em Destaque:** {' | '.join(findings[:3])}")
    if conclusion_text:
        parts.append(f"**Laudo / Parecer Conclusivo:** {conclusion_text}")
    if not parts:
        informative = [l for l in lines if len(l) > 20 and not any(w in l.lower() for w in ["hospital", "página", "cpf", "cartão", "divisão", "complexo"])][:2]
        parts.append(" ".join(informative) if informative else "Exame anexado e processado no prontuário.")

    return f"• **`{filename}`**:\n  " + "\n  ".join(parts)


def format_clinical_bullets(text: str) -> str:
    """Quebra frases longas em tópicos limpos para rápida leitura médica."""
    if not text:
        return ""
    if "\n•" in text or "\n-" in text or "\n1." in text:
        return text.strip()
    sentences = [s.strip() for s in re.split(r"(?<=[.;])\s+", text) if len(s.strip()) > 5]
    if len(sentences) <= 1:
        return f"• {text.strip()}"
    return "\n".join(f"• {s}" for s in sentences)


class HospitalChatGraph:
    """Orquestrador do fluxo conversacional clínico em LangGraph."""

    def __init__(self):
        self.audit_logger = ClinicalAuditLogger()
        self.guardrails = ClinicalGuardrails()
        self.workflow = self._build_graph()
        self.app = self.workflow.compile()
        logger.info("LangGraph HospitalChatGraph compilado com sucesso!")

    def _build_graph(self) -> StateGraph:
        builder = StateGraph(ChatClinicalState)

        # Registra nós do grafo
        builder.add_node("node_input_guardrail", self.node_input_guardrail)
        builder.add_node("node_patient_resolver", self.node_patient_resolver)
        builder.add_node("node_rag_retriever", self.node_rag_retriever)
        builder.add_node("node_clinical_reasoning", self.node_clinical_reasoning)
        builder.add_node("node_output_guardrail", self.node_output_guardrail)
        builder.add_node("node_audit_logger", self.node_audit_logger)

        # Fluxo de execução
        builder.set_entry_point("node_input_guardrail")

        builder.add_conditional_edges(
            "node_input_guardrail",
            self.check_input_safety,
            {
                "safe": "node_patient_resolver",
                "blocked": "node_audit_logger"
            }
        )

        builder.add_edge("node_patient_resolver", "node_rag_retriever")
        builder.add_edge("node_rag_retriever", "node_clinical_reasoning")
        builder.add_edge("node_clinical_reasoning", "node_output_guardrail")
        builder.add_edge("node_output_guardrail", "node_audit_logger")
        builder.add_edge("node_audit_logger", END)

        return builder

    # -----------------------------------------------------------------------
    # Nós do Grafo (StateGraph Nodes)
    # -----------------------------------------------------------------------

    def node_input_guardrail(self, state: ChatClinicalState) -> Dict[str, Any]:
        """Nó 1: Validação de segurança de entrada e detecção de prompt injection."""
        user_input = state.get("current_user_input", "")
        flags = []
        is_safe = True

        malicious_patterns = [
            r"ignore\s+(all\s+)?previous\s+instructions",
            r"system\s*prompt",
            r"jailbreak",
            r"drop\s+database",
            r"modo\s+dan",
            r"esqueça\s+suas\s+regras"
        ]

        for pat in malicious_patterns:
            if re.search(pat, user_input, re.IGNORECASE):
                flags.append(f"INPUT_VIOLATION_PROMPT_INJECTION: {pat}")
                is_safe = False

        if not is_safe:
            return {
                "guardrail_input_safe": False,
                "guardrail_flags": flags,
                "final_response": "⚠️ Requisição bloqueada pelo Guardrail de Entrada: Tentativa de manipulação de instruções detectada.",
                "execution_trace": ["node_input_guardrail (BLOCKED)"]
            }

        return {
            "guardrail_input_safe": True,
            "guardrail_flags": [],
            "execution_trace": ["node_input_guardrail (OK)"]
        }

    def check_input_safety(self, state: ChatClinicalState) -> str:
        return "safe" if state.get("guardrail_input_safe", True) else "blocked"

    def node_patient_resolver(self, state: ChatClinicalState) -> Dict[str, Any]:
        """Nó 2: Identifica menção de pacientes na mensagem ou preserva o paciente ativo."""
        user_input = state.get("current_user_input", "").strip()
        active_patient = state.get("active_patient")
        trace_step = "node_patient_resolver"
        is_patient_entry_turn = False
        resolved_input = user_input

        # 0. Verifica se o usuário informou o nome de um paciente no início da mensagem
        # Ex: "Carlos Santos, informando muita dor...", "Carlos Santos: febre"
        prefix_cand = None
        cand_match = re.match(r"^([A-ZÀ-Ú][a-zà-ú]+(?:\s+[A-ZÀ-Ú][a-zà-ú]+)+)\s*[,:\-]\s*(.*)$", user_input.strip())
        if cand_match:
            candidate_word = cand_match.group(1).strip()
            first_word = candidate_word.split()[0].lower()
            if first_word not in ["paciente", "quadro", "caso", "atendimento", "dor", "inicio", "entrada", "admitido", "admitida"]:
                prefix_cand = candidate_word
                resolved_input = cand_match.group(2).strip()

        if prefix_cand and not active_patient:
            matches = patient_db.search_patient(prefix_cand)
            if matches:
                active_patient = matches[0]
                trace_step += f" (Paciente Vinculado via Entrada: {active_patient['nome']})"

        # 1. Procura por Cartão SUS (15 dígitos, com ou sem espaços/pontos)
        sus_match = re.search(r"\b(\d{3}\s?\d{4}\s?\d{4}\s?\d{4}|\d{15})\b", user_input)
        if sus_match:
            sus_clean = "".join(ch for ch in sus_match.group(0) if ch.isdigit())
            matches = patient_db.search_patient(sus_clean)
            if matches:
                active_patient = matches[0]
                trace_step += f" (Identificado Cartão SUS: {active_patient['nome']})"
                if len(user_input.split()) <= 6 or "cartao" in user_input.lower() or "sus" in user_input.lower():
                    is_patient_entry_turn = True

        # 2. Procura por CPF na mensagem
        if not active_patient or sus_match is not None:
            pass
        else:
            cpf_match = re.search(r"\b(\d{3}\.?\d{3}\.?\d{3}-?\d{2}|\d{11})\b", user_input)
            if cpf_match:
                cpf_str = cpf_match.group(0)
                matches = patient_db.search_patient(cpf_str)
                if matches:
                    active_patient = matches[0]
                    trace_step += f" (Identificado CPF: {active_patient['nome']})"
                    if len(user_input.split()) <= 6 or "cpf" in user_input.lower():
                        is_patient_entry_turn = True

        # 3. Procura menção por prefixos explícitos apenas se não houver paciente ativo: "paciente João da Silva"
        if not active_patient:
            pac_match = re.search(r"\b(?:paciente|atender|admitir)\s+([A-Za-zÀ-ÖØ-öø-ÿ\s]+)", user_input, re.IGNORECASE)
            if pac_match:
                term = pac_match.group(1).strip()
                # Apenas se parecer um nome com até 4 palavras
                if len(term.split()) <= 4:
                    matches = patient_db.search_patient(term)
                    if matches:
                        active_patient = matches[0]
                        trace_step += f" (Identificado Nome: {active_patient['nome']})"
                        if len(user_input.split()) <= 6:
                            is_patient_entry_turn = True

        # 4. Busca direta se o usuário apenas digitou o nome do paciente no chat (ex: "Carlos Eduardo Santos")
        if not active_patient and len(user_input.split()) <= 5:
            direct_matches = patient_db.search_patient(user_input.strip())
            if direct_matches:
                active_patient = direct_matches[0]
                trace_step += f" (Identificado Nome Direto: {active_patient['nome']})"
                is_patient_entry_turn = True

        return {
            "active_patient": active_patient,
            "current_user_input": resolved_input,
            "session_metadata": {
                **state.get("session_metadata", {}),
                "is_patient_entry_turn": is_patient_entry_turn
            },
            "execution_trace": [trace_step]
        }

    def node_rag_retriever(self, state: ChatClinicalState) -> Dict[str, Any]:
        """Nó 3: Busca semântica de protocolos institucionais e laudos de exames no Qdrant RAG."""
        user_input = state.get("current_user_input", "")
        active_patient = state.get("active_patient")
        patient_id = active_patient.get("id") if active_patient else None
        active_exam_files = state.get("session_metadata", {}).get("attached_exam_filenames", [])

        # 1. Recupera exames anexados e pertencentes ao paciente ativo
        exams = []
        if active_exam_files:
            all_pat_exams = rag_manager.get_patient_exams(patient_id=patient_id, filenames=active_exam_files, limit=5)
            if active_patient:
                pat_first_name = active_patient.get("nome", "").split()[0].lower()
                if pat_first_name:
                    filtered = [
                        e for e in all_pat_exams
                        if pat_first_name in e.get("filename", "").lower() or pat_first_name in e.get("content", "").lower()
                    ]
                    exams = filtered if filtered else all_pat_exams
                else:
                    exams = all_pat_exams
            else:
                exams = all_pat_exams

        # 2. Constrói a busca vetorial semântica + IDF de protocolos no Qdrant
        history_msgs = state.get("messages", [])
        doctor_inputs = []
        for m in history_msgs:
            if isinstance(m, HumanMessage):
                c = getattr(m, "content", "").strip()
                if c and c not in doctor_inputs:
                    doctor_inputs.append(c)
        if user_input.strip() and user_input.strip() not in doctor_inputs:
            doctor_inputs.append(user_input.strip())

        # Verifica se o médico mencionou explicitamente um protocolo na mensagem atual
        explicit_proto_ids = re.findall(r"PROTOCOLO-CLIN-\d+", user_input, re.IGNORECASE)
        explicit_proto_id = explicit_proto_ids[0].upper() if explicit_proto_ids else None

        # Busca primária: combina a mensagem atual com exames anexados; se for turno de seguimento curto,
        # inclui também o contexto clínico inicial para manter continuidade sem travar o diagnóstico
        query_elements = []
        if user_input.strip():
            query_elements.append(user_input.strip())
        if len(doctor_inputs) > 1 and not explicit_proto_id:
            # Inclui a queixa inicial como contexto complementar (sem duplicar)
            if doctor_inputs[0] not in query_elements:
                query_elements.append(doctor_inputs[0])
        for ex in exams:
            ex_summary = extract_executive_exam_summary(ex.get("content", ""), ex.get("filename", ""))
            if ex_summary:
                query_elements.append(ex_summary)

        protocol_query = " ".join(query_elements)
        protocols = rag_manager.search_protocols(query=protocol_query, limit=4, exclude_allergy=True)

        # Identifica qual era o protocolo ativo no ÚLTIMO turno do assistente (nunca trava no 1º turno!)
        previous_proto_id = None
        for m in reversed(history_msgs):
            if isinstance(m, AIMessage):
                found = re.findall(r"PROTOCOLO-CLIN-\d+", getattr(m, "content", ""))
                if found:
                    previous_proto_id = found[0].upper()
                    break

        protocol_switched = False
        if explicit_proto_id:
            # Médico redirecionou/confirmou explicitamente um protocolo -> promove para posição principal
            explicit_doc = rag_manager.get_protocol_by_id(explicit_proto_id)
            if explicit_doc:
                protocols = [explicit_doc] + [p for p in protocols if p.get("protocol_id", "").upper() != explicit_proto_id]
                protocol_switched = (previous_proto_id is not None and previous_proto_id != explicit_proto_id)
        else:
            # Verifica se o médico mencionou diretamente o nome de um dos protocolos recuperados na fala atual
            user_lower = user_input.lower()
            matched_idx = None
            for idx_p, p in enumerate(protocols):
                p_title = (p.get("title") or p.get("disease_name") or "").lower()
                # Match por nome substancial da patologia citado pelo médico
                if p_title and len(p_title) > 6 and p_title in user_lower:
                    matched_idx = idx_p
                    break
            if matched_idx is not None and matched_idx > 0:
                chosen = protocols.pop(matched_idx)
                protocols.insert(0, chosen)
                if previous_proto_id and chosen.get("protocol_id", "").upper() != previous_proto_id:
                    protocol_switched = True
            elif previous_proto_id and protocols:
                top_now_id = protocols[0].get("protocol_id", "").upper()
                if top_now_id != previous_proto_id:
                    # Só mantém o protocolo anterior se o turno atual for uma pergunta curta de seguimento
                    # sem novos exames e com score próximo; se a nova evidência for mais forte, permite trocar!
                    prev_in_results = next((p for p in protocols if p.get("protocol_id", "").upper() == previous_proto_id), None)
                    top_score = protocols[0].get("score", 0.0)
                    prev_score = prev_in_results.get("score", 0.0) if prev_in_results else 0.0
                    is_short_followup = len(user_input.split()) <= 25 and not exams
                    if is_short_followup and prev_in_results and (top_score - prev_score) < 0.08:
                        protocols = [prev_in_results] + [p for p in protocols if p.get("protocol_id", "").upper() != previous_proto_id]
                    else:
                        protocol_switched = True

        # 3. Recuperação dinâmica de diretrizes de farmacovigilância e alergia pertinentes ao prontuário
        if active_patient and active_patient.get("alergias"):
            allergy_docs = rag_manager.retrieve_allergy_guidelines(active_patient.get("alergias"), limit=2)
            for ad in allergy_docs:
                if not any(p.get("protocol_id") == ad.get("protocol_id") for p in protocols):
                    protocols.append(ad)

        trace_step = f"node_rag_retriever ({len(protocols)} protocolos/diretrizes, {len(exams)} exames" + (", protocolo atualizado)" if protocol_switched else ")")
        return {
            "retrieved_protocols": protocols,
            "retrieved_exams": exams,
            "session_metadata": {
                **state.get("session_metadata", {}),
                "protocol_switched": protocol_switched,
                "previous_protocol_id": previous_proto_id
            },
            "execution_trace": [trace_step]
        }

    def node_clinical_reasoning(self, state: ChatClinicalState) -> Dict[str, Any]:
        """
        Nó 4: Geração de Raciocínio Clínico (Chain-of-Thought) adaptativo em linguagem natural
        utilizando o modelo neural fine-tunado LLaMA 3.2 1B LoRA e contexto clínico do RAG.
        """
        user_input = state.get("current_user_input", "")
        patient = state.get("active_patient")
        protocols = state.get("retrieved_protocols", [])
        exams = state.get("retrieved_exams", [])
        session_meta = state.get("session_metadata", {})
        is_patient_entry_turn = session_meta.get("is_patient_entry_turn", False)
        critical_alerts = []

        # ---------------------------------------------------------------------
        # CASO 1: Acolhimento inicial na abertura do prontuário
        # ---------------------------------------------------------------------
        if is_patient_entry_turn and patient:
            alergias_txt = patient.get("alergias", "Negadas")
            entry_resp = (
                f"### 📋 Acolhimento no Hospital Universitário FIAP (HFIAP) - Prontuário Aberto\n"
                f"- **Nome do Paciente:** {patient.get('nome')} | **Idade:** {patient.get('idade')} anos | **Sexo:** {patient.get('genero')}\n"
                f"- **Cartão SUS:** `{patient.get('cartao_sus')}` | **CPF:** `{patient.get('cpf')}`\n"
                f"- **Comorbidades Conhecidas:** {patient.get('comorbidades') or 'Nenhuma registrada'}\n"
                f"- **Alergias Documentadas:** {alergias_txt}\n"
                f"- **Medicações Contínuas:** {patient.get('medicamentos_em_uso') or 'Nenhuma'}\n\n"
                f"---\n"
                f"**👨‍⚕️ Próximo Passo do Atendimento Clínico:**\n"
                f"Dr(a)., o prontuário do paciente foi carregado na sessão ativa.\n"
                f"Por favor, informe a **Queixa Principal, História da Doença Atual (HDA) e Sinais Vitais** "
                f"do paciente (ex: início dos sintomas, PA, FC, FR, SpO2, temperatura).\n\n"
                f"💡 *O assistente utilizará os protocolos institucionais do HFIAP para avaliar a suspeita diagnóstica, "
                f"orientar se exames complementares imediatos são realmente indicados e guiar a conduta terapêutica.*"
            )
            return {
                "clinical_reasoning": entry_resp,
                "critical_alerts": critical_alerts,
                "execution_trace": ["node_clinical_reasoning (Prontuário Aberto - Aguardando Queixa)"]
            }

        # ---------------------------------------------------------------------
        # CASO 2: Raciocínio Clínico Conversacional Baseado em Evidências (CDSS)
        # ---------------------------------------------------------------------
        top_proto = protocols[0] if protocols else {}
        proto_name = top_proto.get('title') or top_proto.get('disease_name', 'Avaliação Clínica de Emergência')
        proto_id = top_proto.get('protocol_id', 'PROTOCOLO-HFIAP')
        proto_icd = top_proto.get('icd10', '')
        icd_str = f" (CID-10: {proto_icd})" if proto_icd else ""
        recommended_exams_text = top_proto.get('recommended_exams', '')
        treatment_text = top_proto.get('treatment_protocol', '')
        warnings_text = top_proto.get('contraindications_and_warnings', '')
        symptoms_protocol = top_proto.get('symptoms', '')

        # Cruzamento estrito de alergias do prontuário com a conduta clínica
        patient_allergies = patient.get('alergias', '') if patient else ''
        has_allergy_risk = False
        allergy_warning_msg = ""
        if patient_allergies and not any(n in patient_allergies.lower() for n in ["nenhuma", "negada", "não relatada", "namc", "sem alergia"]):
            allergy_warning_msg = f"Alergia documentada no prontuário: **{patient_allergies}**. Atenção rigorosa para evitar fármacos desta classe biológica."
            critical_alerts.append(allergy_warning_msg)
            has_allergy_risk = True

        # Contexto clínico estruturado para apoio cognitivo
        context_parts = []
        if patient:
            context_parts.append(
                f"[DADOS DO PACIENTE]\n"
                f"Nome: {patient.get('nome')}, Idade: {patient.get('idade')} anos, Sexo: {patient.get('genero')}\n"
                f"Comorbidades: {patient.get('comorbidades') or 'Nenhuma relatada'}\n"
                f"Alergias Documentadas: {patient.get('alergias') or 'Nega'}\n"
                f"Medicamentos Contínuos: {patient.get('medicamentos_em_uso') or 'Nenhum'}"
            )

        if exams:
            exams_summary_lines = []
            for ex in exams:
                fn = ex.get('filename') or ex.get('title') or 'Laudo'
                summary = extract_executive_exam_summary(ex.get('content', ''), fn)
                exams_summary_lines.append(summary)
            context_parts.append("[LAUDOS DE EXAMES ANEXADOS NO PRONTUÁRIO]\n" + "\n".join(exams_summary_lines))

        relevant_allergy_proto = None
        allergy_safe_alternative = None
        if protocols:
            # Separa protocolos clínicos principais e diretrizes de farmacovigilância/alergia recuperadas via RAG
            clinical_protos = [
                p for p in protocols
                if not str(p.get("protocol_id", "")).startswith("ALERGIA-")
                and "farmacovigilância" not in str(p.get("category", "")).lower()
            ]
            allergy_protos = [
                p for p in protocols
                if str(p.get("protocol_id", "")).startswith("ALERGIA-")
                or "farmacovigilância" in str(p.get("category", "")).lower()
            ]

            # Detecta interseção real entre os fármacos da alergia e o protocolo ativo ou a fala do médico
            if allergy_protos:
                active_scope_text = f"{user_input}\n{treatment_text}\n{warnings_text}".lower()
                for ap in allergy_protos:
                    contra_ap = str(ap.get("contraindications_and_warnings", ""))
                    m_drugs = re.search(r"Uso de (.*?) em pacientes", contra_ap, re.IGNORECASE)
                    if m_drugs:
                        drug_terms = [t.strip().lower() for t in m_drugs.group(1).split(",") if len(t.strip()) > 3]
                        if any(re.search(r"\b" + re.escape(dt) + r"\b", active_scope_text, re.IGNORECASE) for dt in drug_terms):
                            relevant_allergy_proto = ap
                            break
                if relevant_allergy_proto:
                    allergy_safe_alternative = ClinicalGuardrails._extract_contextual_alternative(
                        str(relevant_allergy_proto.get("treatment_protocol", "")),
                        f"{proto_name} {symptoms_protocol} {user_input}"
                    )

            protos_text = []
            for p in clinical_protos[:1]:
                conduta_ctx = (
                    f"[BLOQUEADA POR ALERGIA DOCUMENTADA — USAR EXCLUSIVAMENTE A ALTERNATIVA SEGURA DE FARMACOVIGILÂNCIA: {allergy_safe_alternative}]"
                    if (relevant_allergy_proto and allergy_safe_alternative)
                    else p.get('treatment_protocol')
                )
                protos_text.append(
                    f"Diretriz HFIAP: {p.get('title')} ({p.get('protocol_id')}) - CID-10: {p.get('icd10')}\n"
                    f"Sintomas Clínicos Típicos: {p.get('symptoms')}\n"
                    f"Exames Recomendados: {p.get('recommended_exams')}\n"
                    f"Conduta Terapêutica: {conduta_ctx}\n"
                    f"Contraindicações Absolutas: {p.get('contraindications_and_warnings')}"
                )
            if protos_text:
                context_parts.append("[DIRETRIZ CLÍNICA ATIVA RECUPERADA VIA RAG]\n" + "\n\n".join(protos_text))

            if relevant_allergy_proto:
                context_parts.append(
                    "[DIRETRIZ DE FARMACOVIGILÂNCIA E ALERGIA APLICÁVEL À CONDUTA]\n"
                    f"Diretriz: {relevant_allergy_proto.get('title')} ({relevant_allergy_proto.get('protocol_id')})\n"
                    f"Contraindicações: {str(relevant_allergy_proto.get('contraindications_and_warnings', ''))[:300]}\n"
                    f"Alternativa Segura Prioritária para este Quadro: {allergy_safe_alternative}"
                )

            # Destaque objetivo das contraindicações da diretriz ativa (sem palavras-chave hardcoded)
            safety_highlights = []
            if warnings_text:
                safety_highlights.append(f"- Diretriz ativa ({proto_id}): {warnings_text}")
            if has_allergy_risk:
                safety_highlights.append(f"- Alergia documentada do paciente: {patient_allergies}")
            if safety_highlights:
                context_parts.append(
                    "[CRITÉRIOS DE SEGURANÇA E CONTRAINDICAÇÕES ABSOLUTAS]\n" + "\n".join(safety_highlights)
                )

        full_context_str = "\n\n".join(context_parts)

        # Identifica contexto temporal e tipo de turno ANTES de gerar o raciocínio do LLM
        doctor_turns = [
            getattr(m, "content", "").strip()
            for m in state.get("messages", [])
            if isinstance(m, HumanMessage) and len(getattr(m, "content", "").split()) >= 4
        ]
        initial_complaint = doctor_turns[0] if doctor_turns else user_input
        is_subsequent_turn = len(doctor_turns) > 1
        protocol_switched = session_meta.get("protocol_switched", False)
        previous_proto_id = session_meta.get("previous_protocol_id")

        has_already_correlated = any(
            isinstance(m, AIMessage) and "Correlação Clínico-Laboratorial" in getattr(m, "content", "")
            for m in state.get("messages", [])
        )

        if exams and not has_already_correlated:
            turn_mode = "exams_correlation"
            turn_instruction = (
                "MODO DO TURNO: CORRELAÇÃO DE EXAMES ANEXADOS.\n"
                "Correlacione diretamente os achados objetivos dos laudos em [LAUDOS DE EXAMES ANEXADOS NO PRONTUÁRIO] "
                "com o quadro da admissão, confirme a hipótese diagnóstica ou elegibilidade terapêutica e oriente a conduta "
                "segura sob a diretriz ativa. NÃO inicie com 'Não/CONTRAINDICADO' a menos que um exame anexado realmente "
                "contraindique a terapia principal."
            )
        elif is_subsequent_turn and (has_already_correlated or protocol_switched or ("?" in user_input and len(user_input.split()) < 60)):
            turn_mode = "clinical_question"
            turn_instruction = (
                "MODO DO TURNO: PARECER DIRETO A DÚVIDA CLÍNICA / INTERCORRÊNCIA.\n"
                "ATENÇÃO: NÃO repita o formato de admissão ('1. Hipótese Diagnóstica', '2. Exames', etc.).\n"
                "Responda DIRETAMENTE à pergunta atual do médico começando com **SIM** ou **NÃO / CONTRAINDICADO** para CADA fármaco ou conduta perguntada "
                "(se o médico perguntar sobre dois itens — ex: nitratos E betabloqueadores, ou trombolítico E AAS/heparina — responda explicitamente em tópicos separados para CADA UM).\n"
                "Compare rigorosamente os números da pergunta ATUAL (ex: PA atual, alergia, janela de tempo) com [CRITÉRIOS DE SEGURANÇA E CONTRAINDICAÇÕES ABSOLUTAS]. "
                "Se a PA ou alergia atual violar o protocolo, responda **NÃO / CONTRAINDICADO** de forma inequívoca, explique o risco e indique apenas a conduta segura do protocolo "
                "(usando [DIRETRIZ DE FARMACOVIGILÂNCIA E ALERGIA APLICÁVEL À CONDUTA] quando houver alergia)."
            )
        else:
            turn_mode = "initial_admission"
            turn_instruction = (
                "MODO DO TURNO: ADMISSÃO CLÍNICA INICIAL.\n"
                "Apresente de forma objetiva e fluida: (1) Hipótese diagnóstica principal e justificativa clínica, "
                "(2) Exames complementares imediatos necessários, (3) Conduta inicial segura sob a diretriz ativa e "
                "(4) Alertas críticos/alergias. NÃO comece com 'Não/CONTRAINDICADO' na admissão inicial."
            )

        # System Prompt conversacional em Português do Brasil para o CDSS
        system_prompt = (
            "Você é o Assistente Virtual Médico do Hospital Universitário FIAP (CDSS).\n"
            "Responda SEMPRE exclusivamente em português do Brasil técnico-médico fluente — NUNCA use palavras em inglês.\n"
            "Baseie seu raciocínio estritamente nos dados do paciente, exames anexados e diretrizes HFIAP fornecidas no contexto.\n"
            f"{turn_instruction}\n"
            "Se houver [DIRETRIZ DE FARMACOVIGILÂNCIA E ALERGIA APLICÁVEL À CONDUTA] no contexto, NUNCA recomende fármacos da classe "
            "proibida; indique apenas a Alternativa Segura Prioritária descrita.\n"
            "Foque exclusivamente na patologia e na pergunta atual do médico — não desvie a resposta para alergias do prontuário que não têm relação com os medicamentos discutidos.\n"
            "Formule SEMPRE todas as condutas farmacológicas em linguagem consultiva (ex: 'Sugere-se avaliar...', 'Sujeito à validação médica presencial pelo médico assistente/CRM'), "
            "nunca em primeira pessoa ou ordem direta.\n"
            "Se o médico questionar ou redirecionar o diagnóstico/protocolo, reavalie o caso e justifique clinicamente.\n"
            "NUNCA repita títulos como 'Raciocínio Clínico do Modelo' ou termos legais do CFM (eles já são adicionados pelo sistema) e finalize sempre frases completas."
        )

        # Constrói histórico conversacional recente limpo (remove banners de UI e termos legais para não poluir o LLM)
        raw_history = state.get("messages", [])
        chat_history_turns = []
        for m in raw_history[:-1]:
            content = (getattr(m, "content", "") or "").strip()
            if not content:
                continue
            if isinstance(m, HumanMessage):
                chat_history_turns.append({"role": "user", "content": content})
            elif isinstance(m, AIMessage):
                cleaned_hist = re.sub(r"^>\s*🧠\s*\*\*Modelo Neural Local:.*?\n+", "", content, flags=re.MULTILINE)
                cleaned_hist = re.sub(r"^>\s*🔄\s*\*\*Protocolo Clínico Atualizado:.*?\n+", "", cleaned_hist, flags=re.MULTILINE)
                cleaned_hist = re.sub(r"^###\s*🧠\s*(?:Raciocínio Clínico do Modelo|Análise Clínica do Assistente)[^\n]*\n+", "", cleaned_hist, flags=re.MULTILINE)
                cleaned_hist = cleaned_hist.split("⚖️ **TERMO DE RESPONSABILIDADE MÉDICA")[0].strip()
                if cleaned_hist:
                    chat_history_turns.append({"role": "assistant", "content": cleaned_hist})

        neural_reasoning = None
        if clinical_llm.is_available():
            try:
                raw_gen = clinical_llm.generate_clinical_reasoning(
                    system_prompt=system_prompt,
                    user_query=user_input,
                    context_str=full_context_str,
                    chat_history=chat_history_turns,
                    max_new_tokens=600
                )
                if raw_gen:
                    # Validação de qualidade: rejeita se contiver artefatos em inglês do PubMedQA
                    english_artifacts = [
                        "question:", "does surgery", "is there anything else", "what is the treatment of choice",
                        "about this case: do you think", "yes / no", "explanation:", "pubmed/medline",
                        "pmid:", "additional resources", "the following references"
                    ]
                    gen_lower = raw_gen.lower()
                    is_corrupted = any(art in gen_lower for art in english_artifacts)
                    common_en_words = len(re.findall(r"\b(the|and|for|with|this|that|from|have|are|treatment|which|other|after|should|been)\b", gen_lower))
                    if common_en_words > 4:
                        is_corrupted = True

                    if not is_corrupted and len(raw_gen.strip()) > 30:
                        cleaned = re.sub(r"^(?:Resposta do Assistente|Assistant|AI):\s*", "", raw_gen.strip(), flags=re.IGNORECASE)
                        # Remove cabeçalhos duplicados caso o LLM ecoe títulos da UI
                        cleaned = re.sub(r"^(?:###\s*🧠\s*(?:Raciocínio Clínico do Modelo|Análise Clínica do Assistente)[^\n]*\n+)+", "", cleaned, flags=re.IGNORECASE).strip()
                        # Remove qualquer vazamento ocasional de caracteres CJK (Qwen2.5 bilíngue)
                        cleaned = "\n".join(
                            ln for ln in cleaned.splitlines()
                            if not re.search(r"[\u4e00-\u9fff\u3000-\u303f]", ln)
                        ).strip()
                        if len(cleaned) > 30:
                            neural_reasoning = cleaned
            except Exception as e:
                logger.warning(f"Falha na inferência neural: {e}")

        # Prepara bloco de conduta terapêutica seguro (substitui automaticamente por alternativa do RAG
        # quando houver conflito direto com alergia documentada do paciente)
        if relevant_allergy_proto and allergy_safe_alternative:
            allergy_pid = relevant_allergy_proto.get("protocol_id", "ALERGIA")
            safe_treatment_display = (
                f"• ⚠️ **CONDUTA ADAPTADA POR FARMACOVIGILÂNCIA ({allergy_pid}):** "
                f"Esquema padrão da diretriz `{proto_id}` bloqueado por alergia documentada no prontuário (**{patient_allergies}**).\n"
                f"• ✅ **Alternativa Terapêutica Segura Indicada pelo RAG ({allergy_pid}):** {allergy_safe_alternative}"
            )
        else:
            safe_treatment_display = format_clinical_bullets(treatment_text) if treatment_text else ""

        # Montagem Estruturada e Conversacional da Resposta Clínica (LLM + Referência Auditável do RAG)
        response_sections = []
        device_str = "VRAM / CUDA (NVIDIA GPU)" if clinical_llm._device == "cuda" else "Memória RAM (CPU)"
        model_label = clinical_llm.get_model_label() if hasattr(clinical_llm, "get_model_label") else "LLaMA 3.2 1B LoRA"
        llm_status_tag = "Raciocínio Neural Ativo" if neural_reasoning else "Fallback Estruturado (saída neural filtrada)"
        response_sections.append(f"> 🧠 **Modelo Neural Local:** `{model_label}` | **{device_str}** | *{llm_status_tag}*\n")

        if protocol_switched and previous_proto_id:
            response_sections.append(
                f"> 🔄 **Protocolo Clínico Atualizado:** Linha de cuidado redirecionada de `{previous_proto_id}` para **`{proto_id}` — {proto_name}{icd_str}**.\n"
            )

        # =====================================================================
        # CENÁRIO A: Médico enviou exames anexados (Primeira vez que os laudos são correlacionados)
        # =====================================================================
        if turn_mode == "exams_correlation":
            if neural_reasoning:
                response_sections.append(f"### 🧠 Análise Clínica do Assistente (Raciocínio Conversacional)\n{neural_reasoning}\n")

            response_sections.append("### 🔬 1. Correlação Clínico-Laboratorial (Admissão x Laudos Anexados)")
            if is_subsequent_turn and initial_complaint != user_input:
                short_initial = initial_complaint if len(initial_complaint) <= 220 else initial_complaint[:220] + "..."
                response_sections.append(f"• **Quadro Clínico na Admissão:** *\"{short_initial}\"*\n")

            response_sections.append("**Achados nos Exames Anexados:**")
            for ex in exams:
                fn = ex.get('filename') or ex.get('title') or 'Laudo'
                response_sections.append(extract_executive_exam_summary(ex.get('content', ''), fn))

            response_sections.append(
                f"\n• **Conclusão da Correlação:** Os achados dos laudos correlacionam-se com a diretriz institucional "
                f"**{proto_name}**{icd_str} (**{proto_id}**)."
            )

            # 2. Conduta Terapêutica Definitiva e Intervenção Imediata
            if safe_treatment_display:
                response_sections.append(
                    f"\n### ⚡ 2. Conduta Terapêutica de Referência (Diretriz {proto_id})\n"
                    f"{safe_treatment_display}"
                )

            # 3. Alertas de Segurança, Contraindicações e Alergias
            contra_bullets = []
            if has_allergy_risk:
                contra_bullets.append(f"• 🚨 **ALERTA DE FARMACOVIGILÂNCIA:** {allergy_warning_msg}")
            if warnings_text:
                contra_bullets.append(format_clinical_bullets(warnings_text))
            if contra_bullets:
                response_sections.append(
                    f"\n### ⚠️ 3. Alertas de Segurança e Contraindicações Clínicas\n" +
                    "\n".join(contra_bullets)
                )

        # =====================================================================
        # CENÁRIO C: Dúvida Clínica / Redirecionamento de Protocolo / Evolução em Turno Posterior
        # =====================================================================
        elif turn_mode == "clinical_question":
            response_sections.append("### 🩺 1. Parecer Conversacional à Solicitação do Médico")
            response_sections.append(f"**Solicitação / Intercorrência:** *\"{user_input.strip()}\"*\n")

            if neural_reasoning:
                response_sections.append(f"{neural_reasoning}\n")

            # Avaliação de Alergias do Prontuário (Farmacovigilância)
            if has_allergy_risk:
                response_sections.append(
                    f"• 🚨 **ALERTA DE FARMACOVIGILÂNCIA DO PRONTUÁRIO:** {allergy_warning_msg}\n"
                )

            # Referência Estruturada da Diretriz Ativa Atual (RAG)
            response_sections.append(f"### 📋 2. Referência Institucional Ativa — {proto_id} ({proto_name}{icd_str})")
            if safe_treatment_display:
                response_sections.append(
                    f"**Manejo Terapêutico (Diretriz {proto_id}):**\n"
                    f"{safe_treatment_display}\n"
                )
            if warnings_text:
                response_sections.append(
                    f"**Contraindicações e Critérios de Segurança ({proto_id}):**\n"
                    f"{format_clinical_bullets(warnings_text)}\n"
                )
            if recommended_exams_text and protocol_switched:
                response_sections.append(
                    f"**Exames Recomendados nesta Diretriz ({proto_id}):**\n"
                    f"{format_clinical_bullets(recommended_exams_text)}\n"
                )

            # Diretrizes Complementares ou Alternativas (excluindo repetição da ativa)
            alt_clinical = [p for p in protocols[1:3] if p.get("protocol_id") != proto_id]
            if alt_clinical:
                alt_lines = []
                for alt_p in alt_clinical:
                    alt_title = alt_p.get("title") or alt_p.get("disease_name", "")
                    alt_id = alt_p.get("protocol_id", "")
                    if alt_title and alt_id:
                        alt_lines.append(f"• **{alt_id}** — {alt_title}")
                if alt_lines:
                    response_sections.append(
                        "### 📚 3. Diretrizes Relacionadas / Diagnósticos Diferenciais\n" + "\n".join(alt_lines)
                    )

        # =====================================================================
        # CENÁRIO B: Consulta Clínica Inicial / Avaliação de Queixa e Sinais Vitais
        # =====================================================================
        else:
            if neural_reasoning:
                response_sections.append(f"### 🧠 Raciocínio Clínico do Modelo (CDSS Conversacional)\n{neural_reasoning}\n")

            # 1. Hipótese Diagnóstica Primária & Suspeita Clínica
            response_sections.append(
                f"### 💡 1. Hipótese Diagnóstica Primária & Suspeita Clínica\n"
                f"Com base na apresentação clínica relatada e na correlação com os protocolos institucionais do HFIAP, "
                f"o quadro é altamente sugestivo de **{proto_name}**{icd_str}, "
                f"protocolado sob a diretriz institucional **{proto_id}**.\n"
                f"• **Apresentação Clínica:** {symptoms_protocol or 'Compatível com o quadro descrito.'}"
            )

            # 2. Estratificação de Risco e Sinais de Alarme (Red Flags)
            pat_age = patient.get('idade') if patient else 'N/I'
            pat_comorb = patient.get('comorbidades') or 'nenhuma registrada'
            response_sections.append(
                f"\n### 🚨 2. Estratificação de Risco & Sinais de Alarme (*Red Flags*)\n"
                f"• **Perfil de Risco:** Paciente com {pat_age} anos e fatores de risco documentados ({pat_comorb}).\n"
                f"• **Gravidade Imediata:** Quadro agudo com potencial instabilidade clínica. Requer leito de emergência com monitorização contínua de ritmo cardíaco, pressão arterial e oximetria de pulso.\n"
                f"• **Janela Crítica:** Atenção prioritária ao tempo de início dos sintomas para garantir intervenções dentro da janela terapêutica padronizada."
            )

            # 3. Investigação Complementar (Necessidade e Exames a Solicitar Imediatamente)
            if recommended_exams_text:
                response_sections.append(
                    f"\n### 🧪 3. Investigação Complementar (Exames Indicados Imediatamente)\n"
                    f"**Direcionamento Clínico:** É **mandatória a solicitação imediata** dos seguintes exames complementares para confirmação diagnóstica e planejamento terapêutico:\n"
                    f"{format_clinical_bullets(recommended_exams_text)}"
                )
            else:
                response_sections.append(
                    f"\n### 🧪 3. Investigação Complementar\n"
                    f"Quadro de evolução benigna e sem red flags: não há indicação de exames invasivos imediatos, orientando-se o uso racional de recursos diagnósticos."
                )

            # 4. Conduta Terapêutica Inicial Sugerida (Diretriz Institucional)
            if safe_treatment_display:
                response_sections.append(
                    f"\n### ⚡ 4. Conduta Terapêutica Inicial Sugerida (Diretriz {proto_id})\n"
                    f"{safe_treatment_display}"
                )

            # 5. Alertas de Segurança e Contraindicações Clínicas
            contra_bullets = []
            if has_allergy_risk:
                contra_bullets.append(f"• 🚨 **ALERTA DE FARMACOVIGILÂNCIA:** {allergy_warning_msg}")
            if warnings_text:
                contra_bullets.append(format_clinical_bullets(warnings_text))
            if contra_bullets:
                response_sections.append(
                    f"\n### ⚠️ 5. Alertas de Segurança e Contraindicações Clínicas\n" +
                    "\n".join(contra_bullets)
                )

            # 6. Instrução Conversacional de Próximos Passos
            response_sections.append(
                f"\n> 📎 **Próximo Passo do Atendimento:**\n"
                f"> Assim que os exames forem colhidos e os laudos emitidos, **anexe os arquivos em PDF utilizando o botão (+) ao lado da caixa de mensagens** (ou envie os resultados no chat). O assistente realizará a interpretação imediata para auxiliar na validação diagnóstica definitiva."
            )

        final_reasoning = "\n".join(response_sections)
        llm_trace = "LLM=OK" if neural_reasoning else "LLM=FALLBACK"
        trace_tag = f"node_clinical_reasoning (CoT {proto_id}, {llm_trace}, {len(protocols)} protos, {len(exams)} exames)"
        return {
            "clinical_reasoning": final_reasoning,
            "critical_alerts": critical_alerts,
            "execution_trace": [trace_tag]
        }


    def node_output_guardrail(self, state: ChatClinicalState) -> Dict[str, Any]:
        """
        Nó 5: Guardrail de Saída - Validação Estrita de Prescrição Autônoma, Alergias e Compliance CRM.
        Aplica o princípio de Zero Prescrição sem Validação Humana e Bloqueio Incondicional de Alergias Fatais.
        """
        reasoning = state.get("clinical_reasoning", "")
        flags = list(state.get("guardrail_flags", []))
        critical_alerts = list(state.get("critical_alerts", []))
        protocols = state.get("retrieved_protocols", [])
        exams = state.get("retrieved_exams", [])
        patient = state.get("active_patient")
        user_input = state.get("current_user_input", "")

        # 1. Executa validação multicamada através do módulo ClinicalGuardrails (incluindo alergias, exames e contraindicações do RAG)
        guard_res = self.guardrails.validate_and_sanitize(
            reasoning, protocols, patient=patient, retrieved_exams=exams, user_input=user_input
        )
        flags.extend(guard_res.flags_triggered)
        flags = list(dict.fromkeys(flags))

        # Adiciona alertas biológicos identificados pelo guardrail de saída
        new_alerts = guard_res.audit_summary.get("critical_alerts") or guard_res.audit_summary.get("allergy_alerts", [])
        for na in new_alerts:
            if na not in critical_alerts:
                critical_alerts.append(na)

        final_content = guard_res.sanitized_content

        # 2. Padrões adicionais que indicam prescrição direta e imperativa não autorizada
        imperative_prescription_patterns = [
            r"\b(administre|prescreva|injetar|aplique|dar|tomar)\s+\d+\s*(mg|g|ml|gotas|comprimidos|ui)\b",
            r"\bprescrição:\s*\d+\s*(mg|g|ml)",
            r"\btome\s+\d+\s*vezes\s+ao\s+dia\b"
        ]

        for pat in imperative_prescription_patterns:
            if re.search(pat, final_content, re.IGNORECASE):
                if "GUARDRAIL_TRIGGERED: ZERO_AUTONOMOUS_PRESCRIPTION" not in flags:
                    flags.append("GUARDRAIL_TRIGGERED: ZERO_AUTONOMOUS_PRESCRIPTION")
                final_content = re.sub(
                    r"\b(administre|prescreva|aplique)\b",
                    "sugere-se a consideração de",
                    final_content,
                    flags=re.IGNORECASE
                )

        # 3. Adiciona o Disclaimer Legal Obrigatório do Conselho Federal de Medicina (CFM)
        legal_disclaimer = (
            "\n\n---\n"
            "⚖️ **TERMO DE RESPONSABILIDADE MÉDICA (Resolução CFM nº 2.314/2022):**\n"
            "Este assistente opera exclusivamente como **Sistema de Suporte à Decisão Clínica (CDSS)**. "
            "É terminantemente **vedada qualquer prescrição medicamentosa autônoma** ou conduta sem avaliação clínica presencial. "
            "A validação diagnóstica e a prescrição formal são atribuições exclusivas e indelegáveis do **médico assistente devidamente registrado no CRM**."
        )

        if "TERMO DE RESPONSABILIDADE MÉDICA" not in final_content:
            final_content += legal_disclaimer

        return {
            "final_response": final_content,
            "guardrail_output_safe": guard_res.is_safe,
            "guardrail_flags": flags,
            "critical_alerts": critical_alerts,
            "execution_trace": ["node_output_guardrail (CFM Compliance Verificado)"]
        }

    def node_audit_logger(self, state: ChatClinicalState) -> Dict[str, Any]:
        """Nó 6: Auditoria Conversacional Estruturada e Registro no LangSmith."""
        user_input = state.get("current_user_input", "")
        final_resp = state.get("final_response", "")
        patient = state.get("active_patient")
        flags = state.get("guardrail_flags", [])
        trace = state.get("execution_trace", [])
        protocols = state.get("retrieved_protocols", [])
        exams = state.get("retrieved_exams", [])
        session_meta = state.get("session_metadata", {})
        start_t = session_meta.get("timestamp", time.time())
        latency_ms = round((time.time() - start_t) * 1000, 2)

        # Registra no logger de auditoria JSONL com total transparência
        audit_record = self.audit_logger.log_turn(
            doctor_input=user_input,
            assistant_output=final_resp,
            patient=patient,
            retrieved_protocols=protocols,
            retrieved_exams=exams,
            guardrail_flags=flags,
            execution_trace=trace,
            latency_ms=latency_ms
        )

        if LANGSMITH_TRACING and LANGSMITH_API_KEY:
            logger.info(f"Trace conversacional registrado no LangSmith (Run: {audit_record['transaction_id'][:8]})")

        return {
            "execution_trace": ["node_audit_logger (Audit Persisted)"],
            "session_metadata": {
                **session_meta,
                "audit_id": audit_record["transaction_id"],
                "total_tokens": audit_record["tokens"]["total_tokens"],
                "latency_ms": latency_ms
            }
        }

    def chat(
        self,
        user_message: str,
        active_patient: Optional[Dict[str, Any]] = None,
        message_history: Optional[List[BaseMessage]] = None,
        attached_exam_filenames: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """Executa um turno completo de conversa através do grafo LangGraph com rastreamento LangSmith."""
        history = list(message_history) if message_history else []
        history.append(HumanMessage(content=user_message))

        start_time = time.time()
        initial_state: ChatClinicalState = {
            "messages": history,
            "current_user_input": user_message,
            "active_patient": active_patient,
            "retrieved_protocols": [],
            "retrieved_exams": [],
            "guardrail_input_safe": True,
            "guardrail_output_safe": True,
            "guardrail_flags": [],
            "critical_alerts": [],
            "clinical_reasoning": "",
            "final_response": "",
            "execution_trace": [],
            "session_metadata": {
                "timestamp": start_time,
                "langsmith_active": LANGSMITH_TRACING,
                "attached_exam_filenames": list(attached_exam_filenames) if attached_exam_filenames else []
            }
        }

        # Invoca o grafo compilado com tags e metadados para observabilidade
        run_name = f"Turno-{active_patient['nome'] if active_patient else 'Consulta-Geral'}"
        config = {
            "run_name": run_name,
            "tags": ["Hospital-CDSS", "LangGraph", "CFM-2314-Guardrails"],
            "metadata": {
                "patient_name": active_patient.get("nome") if active_patient else "Geral",
                "patient_cpf": active_patient.get("cpf") if active_patient else "N/A"
            }
        }

        result = self.app.invoke(initial_state, config=config)

        # Adiciona resposta da IA ao histórico e calcula tempo
        result["messages"].append(AIMessage(content=result["final_response"]))
        result["latency_ms"] = round((time.time() - start_time) * 1000, 2)
        return result


# Instância global do grafo conversacional
hospital_chat_graph = HospitalChatGraph()


if __name__ == "__main__":
    print("\n🩺 Testando HospitalChatGraph (LangGraph)...")
    graph = HospitalChatGraph()
    res = graph.chat("Qual o protocolo para paciente João da Silva com dor no peito?")
    print("\nTrace percorrido:")
    print(" ➡️  ".join(res["execution_trace"]))
    print("\nResposta Final do Chat:")
    print(res["final_response"][:500] + "...")
