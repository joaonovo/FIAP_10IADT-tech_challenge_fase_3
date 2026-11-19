"""
Interface Web em Streamlit - Assistente Médico Hospitalar de IA (CDSS)
======================================================================

Tech Challenge - Fase 3: IA para DEVs (FIAP)
Suporte à Apresentação e Gravação do Vídeo de 15 Minutos:
1. Chat Clínico Interativo com visual moderno de IA;
2. Botão (+) Anexar Laudo (PDF) diretamente no chat (estilo ChatGPT/Claude);
3. Chip visual de documento anexado com opção de remover;
4. Ingestão no VectorDB Qdrant disparada explicitamente no envio da mensagem;
5. Botão 'Novo Atendimento / Limpar Sessão' para reiniciar o atendimento;
6. Roteiro guiado na tela com 3 Casos Oficiais para a gravação do vídeo.
"""

import os
import sys
import time
import random
import uuid
import unicodedata

current_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = current_dir
while root_dir and not os.path.exists(os.path.join(root_dir, 'hospital_ai_assistant')):
    parent = os.path.dirname(root_dir)
    if parent == root_dir:
        break
    root_dir = parent
if root_dir not in sys.path:
    sys.path.insert(0, root_dir)

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

import streamlit as st
from langchain_core.messages import HumanMessage, AIMessage
from hospital_ai_assistant.agent.chat_graph import hospital_chat_graph
from hospital_ai_assistant.database.patient_db import patient_db
from hospital_ai_assistant.rag_engine.qdrant_rag import rag_manager
from hospital_ai_assistant.safety.audit_logger import ClinicalAuditLogger

exams_dir = os.path.join(root_dir, "data", "exams")
available_exams = sorted([f for f in os.listdir(exams_dir) if f.endswith(".pdf")]) if os.path.exists(exams_dir) else []


st.set_page_config(
    page_title="HFIAP - Assistente Virtual Médico",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Estilização CSS para transformar o botão de anexo em um círculo moderno (+) ao lado do chat
st.markdown("""
<style>
/* Container do popover */
div[data-testid="stPopover"],
div[data-testid="stPopover"] > div {
    display: inline-flex !important;
    align-items: center !important;
    justify-content: center !important;
    width: 44px !important;
    height: 44px !important;
    min-width: 44px !important;
    min-height: 44px !important;
    padding: 0 !important;
    margin: 0 !important;
}

/* O botão em si: círculo perfeito e centralizado */
div[data-testid="stPopover"] button,
button[data-testid="stBaseButton-secondary"][aria-haspopup="dialog"] {
    border-radius: 50% !important;
    width: 44px !important;
    height: 44px !important;
    min-height: 44px !important;
    max-height: 44px !important;
    min-width: 44px !important;
    max-width: 44px !important;
    padding: 0 !important;
    margin: 0 !important;
    box-sizing: border-box !important;
    display: inline-flex !important;
    align-items: center !important;
    justify-content: center !important;
    background: #1e222b !important;
    border: 2px solid #0083B0 !important;
    box-shadow: 0 2px 8px rgba(0, 131, 176, 0.45) !important;
    cursor: pointer !important;
    transition: all 0.2s ease-in-out !important;
    overflow: hidden !important;
}

/* Ocultar chevron/seta do popover */
div[data-testid="stPopover"] button svg,
div[data-testid="stPopover"] svg,
div[data-testid="stPopover"] button span:last-child:not(:first-child),
div[data-testid="stPopover"] button [data-testid="stIconMaterial"],
div[data-testid="stPopover"] button [aria-hidden="true"],
div[data-testid="stPopover"] button i {
    display: none !important;
    width: 0 !important;
    height: 0 !important;
    opacity: 0 !important;
    visibility: hidden !important;
    position: absolute !important;
    pointer-events: none !important;
}

/* Centralização do símbolo (+) dentro do círculo */
div[data-testid="stPopover"] button div[data-testid="stMarkdownContainer"],
div[data-testid="stPopover"] button div,
div[data-testid="stPopover"] button p,
div[data-testid="stPopover"] button span {
    font-size: 26px !important;
    font-weight: 700 !important;
    line-height: 1 !important;
    margin: 0 !important;
    padding: 0 !important;
    color: #00d2ff !important;
    display: flex !important;
    align-items: center !important;
    justify-content: center !important;
    text-align: center !important;
}

/* Hover e Active */
div[data-testid="stPopover"] button:hover {
    background: #0083B0 !important;
    border-color: #00d2ff !important;
    transform: scale(1.08) !important;
    box-shadow: 0 4px 14px rgba(0, 210, 255, 0.65) !important;
}

div[data-testid="stPopover"] button:hover p,
div[data-testid="stPopover"] button:hover span {
    color: #ffffff !important;
}

div[data-testid="stPopover"] button:active {
    transform: scale(0.95) !important;
}

/* Alinhamento perfeito do container inferior com chat_input */
[data-testid="stBottom"] div[data-testid="stHorizontalBlock"] {
    align-items: center !important;
    gap: 10px !important;
}

[data-testid="stBottom"] div[data-testid="column"]:first-child {
    flex: 0 0 48px !important;
    min-width: 48px !important;
    max-width: 48px !important;
    display: flex !important;
    align-items: center !important;
    justify-content: center !important;
}
</style>
""", unsafe_allow_html=True)

# Inicializa estados de sessão
if "messages" not in st.session_state:
    st.session_state.messages = []
if "active_patient" not in st.session_state:
    st.session_state.active_patient = None
if "audit_logger" not in st.session_state:
    st.session_state.audit_logger = ClinicalAuditLogger()
if "last_ingested_exam" not in st.session_state:
    st.session_state.last_ingested_exam = None
if "attached_exams" not in st.session_state:
    st.session_state.attached_exams = []
if "attached_exam" not in st.session_state:
    st.session_state.attached_exam = None
if "suggested_prompt" not in st.session_state:
    st.session_state.suggested_prompt = ""
if "suggested_step_label" not in st.session_state:
    st.session_state.suggested_step_label = ""
if "active_demo_case" not in st.session_state:
    st.session_state.active_demo_case = None
if "active_demo_step" not in st.session_state:
    st.session_state.active_demo_step = 0
if "exam_widget_version" not in st.session_state:
    st.session_state.exam_widget_version = 0
if "pending_turn_msg" not in st.session_state:
    st.session_state.pending_turn_msg = None
if "pending_turn_attachments" not in st.session_state:
    st.session_state.pending_turn_attachments = []


# ==============================================================================
# DEFINIÇÃO DOS 3 CASOS CLÍNICOS DIDÁTICOS ESTRUTURADOS (ROTEIRO GUIADO)
# ==============================================================================
DEMO_CASES = {
    "caso1": {
        "title": "🔴 Caso 1: Dor Torácica / IAMCSST",
        "patient_name": "Maria Helena Oliveira",
        "badge": "🔴",
        "caption": "• **Paciente:** Maria Helena Oliveira (58 anos)\n• **Quadro:** Dor torácica retroesternal opressiva há 40min\n• **Diretriz:** PROTOCOLO-CLIN-002 (IAMCSST)",
        "steps": [
            {
                "step_num": 1,
                "name": "Sintomas Iniciais na Admissão",
                "desc": "Apresentação da queixa precordial e solicitação de exames de urgência.",
                "prompt": "Paciente Maria Helena Oliveira, 58 anos, admitida na sala de emergência com quadro agudo de dor torácica retroesternal opressiva de forte intensidade iniciada há 40 minutos, irradiada para membro superior esquerdo e mandíbula, acompanhada de sudorese fria e náuseas. Sinais vitais: PA 140/90 mmHg, FC 94 bpm, SpO2 97% em ar ambiente.",
                "exam_files": []
            },
            {
                "step_num": 2,
                "name": "Laudos de Exames Complementares",
                "desc": "ECG com Supra de ST e Troponina ultrassensível crítica (4.850 pg/mL).",
                "prompt": "Seguem os laudos de exames complementares anexados: ECG de 12 derivações com dosagem de Troponina ultrassensível e Hemograma com Coagulograma completo da paciente Maria Helena Oliveira. Solicito análise dos traçados de ECG e marcadores cardíacos e definição de conduta terapêutica imediata sob o protocolo institucional.",
                "exam_files": [
                    "MARIA_HELENA_OLIVEIRA_817293049182_ECG_TROPONINA.pdf",
                    "MARIA_HELENA_OLIVEIRA_817293049183_HEMOGRAMA_COAGULOGRAMA.pdf"
                ]
            },
            {
                "step_num": 3,
                "name": "Dúvida Clínica & Intercorrência Hemodinâmica",
                "desc": "Queda abrupta de PA (80/50 mmHg) e contraindicação formal a nitratos e betabloqueadores.",
                "prompt": "A paciente acabou de evoluir com queda de pressão: PA 80/50 mmHg. Podemos manter nitratos venosos ou betabloqueadores nesse momento?",
                "exam_files": []
            }
        ]
    },
    "caso2": {
        "title": "🛡️ Caso 2: Pneumonia & Guardrails CFM",
        "patient_name": "João da Silva",
        "badge": "🛡️",
        "caption": "• **Paciente:** João da Silva (67 anos)\n• **Quadro:** Tosse produtiva, febre alta e dispneia (PAC)\n• **Destaque:** Alergia a Penicilinas (Bloqueio CFM)",
        "steps": [
            {
                "step_num": 1,
                "name": "Sintomas Iniciais na Admissão",
                "desc": "Apresentação de quadro infeccioso respiratório e sinais de alerta.",
                "prompt": "Paciente João da Silva, 67 anos, admitido no pronto-socorro com tosse produtiva com escarro purulento, febre alta há 3 dias (Tax: 38.4°C), calafrios e piora progressiva da falta de ar aos esforços. Sinais vitais: FR 24 irpm, FC 102 bpm, SpO2 91% em ar ambiente. Prontuário destaca histórico de DPOC.",
                "exam_files": []
            },
            {
                "step_num": 2,
                "name": "Laudos de Exames & Confirmação de PAC",
                "desc": "Hemograma com Leucocitose e PCR 94.2 mg/L confirmando infecção pulmonar bacteriana.",
                "prompt": "Seguem os laudos de exames complementares anexados: Hemograma completo com contagem leucocitária e dosagem de Proteína C Reativa (PCR) do paciente João da Silva. Solicito a correlação clínico-laboratorial e confirmação diagnóstica.",
                "exam_files": [
                    "JOAO_DA_SILVA_948201839201_HEMOGRAMA_PCR.pdf"
                ]
            },
            {
                "step_num": 3,
                "name": "Dúvida Clínica & Guardrail de Alergia",
                "desc": "Tentativa de prescrição de betalactâmicos, bloqueio de segurança e indicação de Levofloxacino.",
                "prompt": "Podemos prescrever Amoxicilina com Clavulanato 875mg ou Ceftriaxona venosa para tratar a pneumonia do João da Silva? Qual a alternativa antimicrobiana segura indicada pelo protocolo para este paciente alérgico a penicilinas?",
                "exam_files": []
            }
        ]
    },
    "caso3": {
        "title": "🧠 Caso 3: AVC Isquêmico & Trombólise",
        "patient_name": "Carlos Eduardo Santos",
        "badge": "🧠",
        "caption": "• **Paciente:** Carlos Eduardo Santos (72 anos)\n• **Quadro:** Hemiparesia súbita e disartria (< 4.5h)\n• **Diretriz:** PROTOCOLO-CLIN-031 (Trombólise)",
        "steps": [
            {
                "step_num": 1,
                "name": "Sintomas Iniciais na Admissão",
                "desc": "Chegada pelo SAMU com hemiparesia súbita há 80 minutos.",
                "prompt": "Paciente Carlos Eduardo Santos, 72 anos, hipertenso e em uso crônico de Varfarina, admitido na emergência com déficit neurológico agudo caracterizado por hemiparesia braquiocrural direita e disartria súbitas iniciadas há 80 minutos (delta T < 4.5h). Sinais vitais: PA 160/95 mmHg, FC 82 bpm, Glicemia capilar 118 mg/dL.",
                "exam_files": []
            },
            {
                "step_num": 2,
                "name": "Laudos de Exames (TC de Crânio & RNI)",
                "desc": "TC sem hemorragia e RNI 1.38 (< 1.7) confirmando elegibilidade para Alteplase.",
                "prompt": "Seguem os laudos de exames complementares anexados: Tomografia Computadorizada de Crânio sem contraste e Coagulograma com RNI do paciente Carlos Eduardo Santos. Solicito avaliação de elegibilidade imediata para trombólise venosa.",
                "exam_files": [
                    "CARLOS_EDUARDO_SANTOS_736192840192_TC_CRANIO.pdf",
                    "CARLOS_EDUARDO_SANTOS_736192840193_COAGULOGRAMA_RNI.pdf"
                ]
            },
            {
                "step_num": 3,
                "name": "Dúvida Clínica & Intercorrência Pressórica",
                "desc": "Pico de PA (195/110 mmHg), contraindicação a trombolítico e bloqueio de AAS/heparina em 24h.",
                "prompt": "O paciente apresentou pico hipertensivo com PA 195/110 mmHg e a equipe quer administrar AAS e heparina agora. Podemos iniciar o trombolítico com essa PA e administrar o AAS nas primeiras 24 horas?",
                "exam_files": []
            }
        ]
    }
}


def reset_session():
    """Limpa todo o atendimento atual permitindo um novo atendimento limpo."""
    if st.session_state.get("active_patient"):
        try:
            rag_manager.clear_patient_exams(st.session_state.active_patient.get("id"))
        except Exception:
            pass
    st.session_state.messages = []
    st.session_state.active_patient = None
    st.session_state.last_ingested_exam = None
    st.session_state.attached_exams = []
    st.session_state.attached_exam = None
    st.session_state.session_attached_filenames = []
    st.session_state.suggested_prompt = ""
    st.session_state.suggested_step_label = ""
    st.session_state.active_demo_case = None
    st.session_state.active_demo_step = 0
    st.session_state.exam_widget_version = 0
    st.session_state.pending_turn_msg = None
    st.session_state.pending_turn_attachments = []
    st.rerun()


def prepare_demo_step(case_key: str, step_num: int):
    """Configura o caso clínico e a etapa selecionada para simulação didática guiada."""
    if case_key not in DEMO_CASES:
        return
    c_info = DEMO_CASES[case_key]
    matches = patient_db.search_patient(c_info["patient_name"])
    if not matches:
        return

    pat = matches[0]
    st.session_state.active_patient = pat
    st.session_state.active_demo_case = case_key
    st.session_state.active_demo_step = step_num

    # Se for passo 1, reinicia o histórico de atendimento do paciente
    if step_num == 1:
        try:
            rag_manager.clear_patient_exams(pat.get("id"))
        except Exception:
            pass
        st.session_state.messages = []
        st.session_state.session_attached_filenames = []

    step_info = c_info["steps"][step_num - 1]
    st.session_state.suggested_prompt = step_info["prompt"]
    st.session_state.suggested_step_label = f"Passo {step_num}/3 ({step_info['name']})"

    # Anexa os arquivos PDF correspondentes à etapa (se houver)
    st.session_state.attached_exams = []
    st.session_state.attached_exam = None
    if step_info.get("exam_files"):
        exams_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "exams")
        for ef in step_info["exam_files"]:
            fpath = os.path.join(exams_dir, ef)
            if os.path.exists(fpath):
                st.session_state.attached_exams.append({
                    "filename": ef,
                    "path": fpath,
                    "origem": "Interno HFIAP"
                })
        if st.session_state.attached_exams:
            st.session_state.attached_exam = st.session_state.attached_exams[0]

    st.rerun()


def advance_demo_step_if_active():
    """Avança de forma fluida para a próxima etapa do atendimento guiado após a resposta do assistente."""
    case_key = st.session_state.get("active_demo_case")
    curr_step = st.session_state.get("active_demo_step", 0)

    if not case_key or case_key not in DEMO_CASES:
        return

    c_info = DEMO_CASES[case_key]
    if curr_step == 1:
        next_step = 2
        step_info = c_info["steps"][next_step - 1]
        st.session_state.active_demo_step = next_step
        st.session_state.suggested_prompt = step_info["prompt"]
        st.session_state.suggested_step_label = f"Passo {next_step}/3 ({step_info['name']})"

        # Carrega automaticamente os exames correspondentes da etapa 2
        exams_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "exams")
        st.session_state.attached_exams = []
        for ef in step_info.get("exam_files", []):
            fpath = os.path.join(exams_dir, ef)
            if os.path.exists(fpath):
                st.session_state.attached_exams.append({
                    "filename": ef,
                    "path": fpath,
                    "origem": "Interno HFIAP"
                })
        st.session_state.attached_exam = st.session_state.attached_exams[0] if st.session_state.attached_exams else None

    elif curr_step == 2:
        next_step = 3
        step_info = c_info["steps"][next_step - 1]
        st.session_state.active_demo_step = next_step
        st.session_state.suggested_prompt = step_info["prompt"]
        st.session_state.suggested_step_label = f"Passo {next_step}/3 ({step_info['name']})"
        st.session_state.attached_exams = []
        st.session_state.attached_exam = None

    elif curr_step == 3:
        st.session_state.active_demo_step = 4
        st.session_state.suggested_prompt = ""
        st.session_state.suggested_step_label = "✅ Atendimento Guiado Concluído!"
        st.session_state.attached_exams = []
        st.session_state.attached_exam = None


def prepare_demo_case(patient_name: str, exam_filename: str, prompt_text: str):
    """Compatibilidade legada para preparar casos de demonstração."""
    for k, v in DEMO_CASES.items():
        if v["patient_name"].lower() in patient_name.lower() or patient_name.lower() in v["patient_name"].lower():
            prepare_demo_step(k, 1)
            return
    matches = patient_db.search_patient(patient_name)
    if matches:
        st.session_state.active_patient = matches[0]
    st.session_state.suggested_prompt = prompt_text
    st.rerun()


def queue_turn(user_text: str):
    """Enfileira a mensagem para renderização imediata com loader animado no corpo do chat."""
    attached_list = list(st.session_state.get("attached_exams", []))
    st.session_state.attached_exams = []  # consome os anexos
    st.session_state.attached_exam = None

    clean_text = user_text.strip()
    if not clean_text and attached_list:
        clean_text = "Seguem os laudos de exames complementares realizados para avaliação diagnóstica e conduta médica."
    elif not clean_text:
        return

    # 0. Suporte ao comando /paciente <termo> direto no chat
    if clean_text.lower().startswith("/paciente "):
        termo = clean_text[10:].strip()
        if len(termo) >= 4:
            found = patient_db.search_patient(termo)
            if found:
                st.session_state.active_patient = found[0]
                st.session_state.messages.append({
                    "role": "user",
                    "content": user_text
                })
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": f"✅ **Paciente Selecionado via Auto-Sugestão:** `{found[0]['nome']}` (ID: `HFIAP-{found[0]['id']:04d}`)\n\n"
                               f"• **CPF:** `{found[0]['cpf']}` | **Idade:** {found[0]['idade']} anos | **Sexo:** {found[0]['genero']}\n"
                               f"• **Alergias Documentadas:** `{found[0]['alergias']}`\n"
                               f"• **Comorbidades:** {found[0]['comorbidades']}\n\n"
                               f"Prontuário vinculado com sucesso para suporte à decisão clínica.",
                    "trace": ["node_patient_resolver (Auto-Sugestão Ativada)"]
                })
                st.rerun()
                return

    attached_badge = ""
    if attached_list:
        nomes_exames = ", ".join(f"`{a['filename']}`" for a in attached_list)
        attached_badge = f"📎 **[Laudo(s) Anexado(s) ({len(attached_list)}): {nomes_exames}]**\n\n"

    final_user_msg = attached_badge + clean_text

    st.session_state.messages.append({
        "role": "user",
        "content": final_user_msg,
        "attached_filenames": [a["filename"] for a in attached_list]
    })

    st.session_state.pending_turn_msg = final_user_msg
    st.session_state.pending_turn_attachments = attached_list
    st.rerun()


execute_turn = queue_turn


# ==============================================================================
# SIDEBAR - Prontuário Exclusivo do Paciente em Tratamento
# ==============================================================================
with st.sidebar:
    st.header("🏥 Prontuário do Paciente")
    st.caption("Exibição estruturada dos dados do paciente em atendimento ativo")

    if st.session_state.active_patient:
        p = st.session_state.active_patient
        st.success("🟢 **Atendimento Clínico Ativo**")
        st.markdown(f"### 📋 `{p['nome']}`")
        st.markdown(f"• **ID Prontuário:** `HFIAP-{p['id']:04d}`")
        st.markdown(f"• **Idade / Sexo:** {p['idade']} anos | {p['genero']}")
        st.markdown(f"• **CPF:** `{p['cpf']}`")
        st.markdown(f"• **Cartão SUS:** `{p['cartao_sus']}`")

        # Alergias com destaque visual
        alergias_txt = p.get('alergias', 'Negadas')
        is_allergy_free = any(neg in alergias_txt.lower() for neg in ["nenhuma", "negada", "não relatada", "sem alergia"])
        if is_allergy_free:
            st.info(f"🛡️ **Alergias:** {alergias_txt}")
        else:
            st.error(f"🚨 **Alergias Documentadas:** {alergias_txt}")

        st.markdown(f"• **Comorbidades:**\n{p.get('comorbidades') or 'Nenhuma'}")
        st.markdown(f"• **Medicações Contínuas:**\n{p.get('medicamentos_em_uso') or 'Nenhuma'}")
        if p.get('endereco'):
            st.caption(f"📍 **Endereço:** {p['endereco']}")
        if p.get('historico_clinico'):
            st.caption(f"📝 **Histórico Admissional:** {p['historico_clinico'][:150]}...")

        st.divider()
        col_side1, col_side2 = st.columns(2)
        with col_side1:
            if st.button("🔄 Trocar Paciente", use_container_width=True, help="Desvincular prontuário atual"):
                st.session_state.active_patient = None
                st.session_state.messages = []
                st.session_state.suggested_prompt = ""
                st.session_state.attached_exams = []
                st.session_state.attached_exam = None
                st.rerun()
        with col_side2:
            if st.button("🧹 Limpar Chat", use_container_width=True, help="Limpar mensagens do chat"):
                st.session_state.messages = []
                st.session_state.attached_exams = []
                st.session_state.attached_exam = None
                st.session_state.suggested_prompt = ""
                st.rerun()

    else:
        st.info(
            "🔒 **Nenhum paciente selecionado**\n\n"
            "A barra lateral exibirá o prontuário completo assim que você:\n"
            "1. Localizar o paciente na **barra de busca central acima do chat** (entre os 1.000 prontuários da base);\n"
            "2. Ou selecionar um dos casos de teste rápido no painel superior."
        )

    st.divider()
    if st.button("🔄 Reiniciar Todo o Atendimento", type="secondary", use_container_width=True):
        reset_session()


# ==============================================================================
# CABEÇALHO PRINCIPAL E CENTRAL DE DEMONSTRAÇÃO DO VÍDEO
# ==============================================================================
st.title("🩺 Hospital Universitário FIAP - CDSS")
st.caption("Sistema de Suporte à Decisão Clínica | LangChain + LangGraph + Qdrant RAG + Unsloth LoRA + Guardrails CFM nº 2.314/2022")

# Central de Demonstrações Rápidas (3 Casos Clínicos Estruturados)
with st.expander("🎬 **Roteiro de Apresentação** - Clique para abrir", expanded=False):
    st.markdown("Selecione um dos **3 Casos Clínicos Estruturados** para simular um atendimento hospitalar completo passo a passo:")
    
    col_d1, col_d2, col_d3 = st.columns(3)

    for col, (ckey, cdata) in zip([col_d1, col_d2, col_d3], DEMO_CASES.items()):
        with col:
            st.markdown(f"#### {cdata['title']}")
            st.caption(cdata["caption"])
            
            st.markdown("""
            **Etapas do Atendimento:**
            1. 📋 *Sintomas Iniciais & Admissão*
            2. 🔬 *Laudo(s) de Exame Complementar*
            3. 🧠 *Dúvida Clínica & Decisão Médica*
            """)
            
            is_active_case = (st.session_state.get("active_demo_case") == ckey)
            if is_active_case:
                st.info(f"🟡 **Caso em Andamento:** {st.session_state.get('suggested_step_label', 'Em atendimento')}")

            if st.button(f"▶️ Iniciar Caso no Chat", use_container_width=True, key=f"btn_start_{ckey}"):
                prepare_demo_step(ckey, 1)

            c_s1, c_s2, c_s3 = st.columns(3)
            with c_s1:
                if st.button("Passo 1", use_container_width=True, key=f"btn_s1_{ckey}", help="Passo 1: Sintomas Iniciais"):
                    prepare_demo_step(ckey, 1)
            with c_s2:
                if st.button("Passo 2", use_container_width=True, key=f"btn_s2_{ckey}", help="Passo 2: Envio de Exames"):
                    prepare_demo_step(ckey, 2)
            with c_s3:
                if st.button("Passo 3", use_container_width=True, key=f"btn_s3_{ckey}", help="Passo 3: Dúvida Clínica"):
                    prepare_demo_step(ckey, 3)


# ==============================================================================
# ABAS PRINCIPAIS DE NAVEGAÇÃO
# ==============================================================================
tab_chat, tab_protocols, tab_audit = st.tabs([
    "💬 Chat Clínico & Decisão", 
    "📚 Base de Protocolos (100 Doenças)", 
    "🛡️ Logs & Auditoria (LangSmith)"
])

# ------------------------------------------------------------------------------
# ABA 1: CHAT CLÍNICO
# ------------------------------------------------------------------------------
with tab_chat:
    # --------------------------------------------------------------------------
    # BARRA CENTRAL DE IDENTIFICAÇÃO E ACOLHIMENTO DO PACIENTE (ACIMA DO CHAT)
    # --------------------------------------------------------------------------
    if st.session_state.active_patient is None:
        st.markdown("### 🔍 Identificação do Paciente no Qdrant VectorDB (RAG)")
        st.caption("Pesquise na base de 1.000 prontuários vetoriais por **Nome, CPF ou Cartão SUS** usando o Qdrant RAG como Fonte Única da Verdade:")

        search_pac = st.text_input(
            "Pesquisar Prontuário no Qdrant VectorDB (a partir de 4 caracteres):",
            placeholder="Digite o Nome, CPF ou Cartão SUS (Ex: Maria, João, Carlos, 9876, 890...)...",
            key="central_patient_search_input"
        )

        # Resultados da busca
        if search_pac:
            clean_term = search_pac.strip()
            if len(clean_term) >= 4:
                found_list = patient_db.search_patient(clean_term)
                if found_list:
                    st.caption(f"💡 **Sugestões recuperadas no Qdrant VectorDB ({len(found_list)}):** Clique para carregar o prontuário:")
                    cols_res = st.columns(min(len(found_list), 3))
                    for i, pt in enumerate(found_list[:3]):
                        with cols_res[i]:
                            btn_text = f"👤 **{pt['nome']}**\n\nCPF: `{pt['cpf']}`\nSUS: `{pt['cartao_sus']}`"
                            if st.button(btn_text, key=f"btn_choose_patient_{pt['id']}", use_container_width=True):
                                st.session_state.active_patient = pt
                                st.session_state.messages = [{
                                    "role": "assistant",
                                    "content": (
                                        f"### 📋 Prontuário Aberto — Hospital Universitário FIAP (HFIAP)\n"
                                        f"- **Paciente:** {pt['nome']} (ID: `HFIAP-{pt['id']:04d}`) | **Idade:** {pt['idade']} anos | **Sexo:** {pt['genero']}\n"
                                        f"- **Cartão SUS:** `{pt['cartao_sus']}` | **CPF:** `{pt['cpf']}`\n"
                                        f"- **Comorbidades:** {pt.get('comorbidades') or 'Nenhuma registrada'}\n"
                                        f"- **Alergias Documentadas:** {pt.get('alergias') or 'Negadas'}\n"
                                        f"- **Medicações Contínuas:** {pt.get('medicamentos_em_uso') or 'Nenhuma'}\n\n"
                                        f"---\n"
                                        f"**👨‍⚕️ Atendimento Liberado:** Dr(a)., o prontuário foi vinculado ao atendimento.\n"
                                        f"Por favor, **digite a Queixa Principal e os Sinais Vitais** do paciente no chat abaixo.\n\n"
                                        f"💡 *O assistente verificará as diretrizes institucionais para avaliar se exames complementares imediatos são realmente necessários ou se a conduta clínica é suficiente.*"
                                    ),
                                    "trace": ["node_patient_resolver (Prontuário Aberto via Busca Central)"]
                                }]
                                st.rerun()
                else:
                    st.warning(f"⚠️ Nenhum paciente cadastrado encontrado para '{clean_term}'. Por favor, verifique o Nome, CPF ou Cartão SUS entre os 1.000 prontuários da base.")
            else:
                st.caption(f"ℹ️ Digite mais {4 - len(clean_term)} caractere(s) para carregar sugestões.")

    else:
        # Modo: Paciente Ativo Já Selecionado
        p = st.session_state.active_patient
        col_bar1, col_bar2, col_bar3 = st.columns([5, 1, 1], vertical_alignment="center")
        with col_bar1:
            st.success(f"👤 **Paciente em Atendimento:** {p['nome']} (ID: `HFIAP-{p['id']:04d}`) | Idade: {p['idade']} anos | SUS: `{p['cartao_sus']}` | Alergias: **{p['alergias']}**")
        with col_bar2:
            if st.button("🔄 Trocar Paciente", use_container_width=True, help="Desvincular paciente atual e buscar outro"):
                st.session_state.active_patient = None
                st.session_state.messages = []
                st.session_state.suggested_prompt = ""
                st.session_state.attached_exams = []
                st.session_state.attached_exam = None
                st.rerun()
        with col_bar3:
            if st.button("🧹 Limpar Chat", use_container_width=True, help="Limpar histórico de mensagens"):
                st.session_state.messages = []
                st.session_state.attached_exams = []
                st.session_state.attached_exam = None
                st.session_state.suggested_prompt = ""
                st.rerun()

    st.markdown("---")

    # Se nenhum paciente vinculado, exibe card informativo de chat bloqueado
    if st.session_state.active_patient is None:
        st.markdown("""
        <div style="background-color: #1e222b; border: 1.5px solid #3b4252; border-radius: 10px; padding: 22px; margin: 15px 0;">
            <h3 style="color: #00d2ff; margin-top: 0;">🏥 Pronto-Atendimento Virtual — Hospital Universitário FIAP</h3>
            <h4 style="color: #ebcb8b; margin-top: 8px;">🔒 Chat Clínico Inativo (Segurança do Paciente & Resolução CFM nº 2.314/2022)</h4>
            <p style="color: #e5e9f0; line-height: 1.6;">
                Por conformidade ética médica e segurança clínica, <b>o chat permanece inativo para envio de dados</b> até que o prontuário seja devidamente vinculado.
            </p>
            <div style="background: rgba(0, 131, 176, 0.15); border-left: 4px solid #00d2ff; padding: 12px 16px; margin: 14px 0; border-radius: 4px;">
                <p style="color: #ffffff; margin: 0; font-weight: 500;">
                    👉 <b>Como liberar o chat clínico:</b><br>
                    1. <b>Busca Central:</b> Utilize a barra de pesquisa acima para localizar o paciente por <b>Nome, CPF ou Cartão SUS</b> (a partir de 4 caracteres entre os 1.000 prontuários da base de dados);<br>
                    2. <b>Demonstração Rápida:</b> Para testes rápidos, selecione um dos 3 Casos Clínicos no painel superior.
                </p>
            </div>
        </div>
        """, unsafe_allow_html=True)
    else:
        # Renderização do Histórico de Mensagens
        for msg in st.session_state.messages:
            if msg["role"] == "user":
                with st.chat_message("user", avatar="👨‍⚕️"):
                    st.markdown(f"**Médico Assistente:**\n\n{msg['content']}")
            else:
                with st.chat_message("assistant", avatar="🤖"):
                    # 1. Pipeline Visual do LangGraph
                    trace_nodes = msg.get("trace", [])
                    if trace_nodes:
                        st.caption(f"⚡ **Pipeline LangGraph:** {' ➡️ '.join(trace_nodes)}")

                    # 2. Interceptações de Guardrail (Destaque em Vermelho/Âmbar)
                    guardrail_flags = msg.get("guardrails", [])
                    critical_alerts = msg.get("critical_alerts", [])
                    if guardrail_flags or critical_alerts:
                        with st.container():
                            st.error("🚨 **INTERCEPTAÇÃO POR GUARDRAILS CLÍNICOS (CFM nº 2.314/2022)**")
                            for gf in guardrail_flags:
                                st.markdown(f"• **Alerta de Segurança:** `{gf}`")
                            for ca in critical_alerts:
                                st.markdown(f"• **Contraindicação:** {ca}")

                    # 3. Conteúdo Principal da Resposta
                    st.markdown(msg["content"])

                    # 4. Detalhes de Evidências RAG (Explainability)
                    col_exp1, col_exp2 = st.columns(2)
                    with col_exp1:
                        protocols_list = msg.get("protocols", [])
                        if protocols_list:
                            with st.expander(f"📚 {len(protocols_list)} Protocolos HFIAP Recuperados no Qdrant"):
                                for proto in protocols_list:
                                    st.markdown(f"**{proto.get('title')}** (Score Cosseno: `{proto.get('score', 0):.4f}`)")
                                    st.caption(proto.get('content', '')[:250] + "...")
                    with col_exp2:
                        exams_list = msg.get("exams", [])
                        if exams_list:
                            with st.expander(f"🔬 {len(exams_list)} Fragmentos de Laudos em PDF Recuperados"):
                                for exam in exams_list:
                                    st.markdown(f"**Fonte:** `{exam.get('source')}`")
                                    st.caption(exam.get('content', '')[:250] + "...")

        # Processa turno pendente com feedback visual de carregamento (st.status no chat)
        if st.session_state.get("pending_turn_msg"):
            pending_msg = st.session_state.pop("pending_turn_msg")
            pending_attachments = st.session_state.pop("pending_turn_attachments", [])

            with st.chat_message("assistant", avatar="🤖"):
                with st.status("🔬 **Processando Atendimento Clínico (HFIAP CDSS)...**", expanded=True) as status_box:
                    if pending_attachments:
                        st.write(f"📄 **Vetorizando {len(pending_attachments)} laudo(s) no VectorDB Qdrant...**")
                        pat_id = st.session_state.active_patient["id"] if st.session_state.active_patient else None
                        for att in pending_attachments:
                            if os.path.exists(att.get("path", "")):
                                res = rag_manager.ingest_pdf(att["path"], patient_id=pat_id)
                                st.session_state.last_ingested_exam = res

                    st.write("🏥 **Recuperando protocolos clínicos e prontuário no Qdrant...**")
                    if "session_attached_filenames" not in st.session_state:
                        st.session_state.session_attached_filenames = []
                    if pending_attachments:
                        for att in pending_attachments:
                            fn = att.get("filename")
                            if fn and fn not in st.session_state.session_attached_filenames:
                                st.session_state.session_attached_filenames.append(fn)

                    chat_history = []
                    for m in st.session_state.messages[:-1]:
                        if m.get("role") == "user":
                            chat_history.append(HumanMessage(content=m.get("content", "")))
                        elif m.get("role") == "assistant":
                            chat_history.append(AIMessage(content=m.get("content", "")))

                    result = hospital_chat_graph.chat(
                        user_message=pending_msg,
                        active_patient=st.session_state.active_patient,
                        message_history=chat_history,
                        attached_exam_filenames=st.session_state.session_attached_filenames
                    )

                    old_pat_id = st.session_state.active_patient.get("id") if st.session_state.active_patient else None
                    new_pat = result.get("active_patient")
                    if new_pat:
                        new_pat_id = new_pat.get("id")
                        if old_pat_id and new_pat_id != old_pat_id:
                            st.session_state.session_attached_filenames = []
                            st.session_state.attached_exams = []
                            st.session_state.attached_exam = None
                        st.session_state.active_patient = new_pat

                    st.write("🧠 **Raciocínio clínico formulado e validado sob a Resolução CFM nº 2.314/2022!**")
                    status_box.update(label="✅ **Análise e Raciocínio Clínico Concluídos!**", state="complete", expanded=False)

                st.session_state.messages.append({
                    "role": "assistant",
                    "content": result.get("final_response", ""),
                    "trace": result.get("execution_trace", []),
                    "guardrails": result.get("guardrail_flags", []),
                    "critical_alerts": result.get("critical_alerts", []),
                    "protocols": result.get("retrieved_protocols", []),
                    "exams": result.get("retrieved_exams", []),
                    "latency_ms": result.get("latency_ms", 35.0)
                })

                # Se estiver executando um caso demonstrativo guiado, avança para a próxima etapa
                advance_demo_step_if_active()
                st.rerun()

    # --------------------------------------------------------------------------
    # BARRA FIXA INFERIOR: BOTÃO CIRCULAR (+) LADO A LADO COM O CHAT INPUT
    # --------------------------------------------------------------------------
    with st.bottom:
        is_chat_enabled = st.session_state.active_patient is not None

        # Pergunta sugerida do roteiro guiado (se houver e paciente ativo)
        if st.session_state.suggested_prompt and is_chat_enabled:
            col_sugg_txt, col_sugg_btn = st.columns([5, 1], vertical_alignment="center")
            with col_sugg_txt:
                step_lbl = st.session_state.get("suggested_step_label", "💡 Roteiro Guiado")
                st.success(f"**{step_lbl}:** {st.session_state.suggested_prompt}")
            with col_sugg_btn:
                btn_lbl = "📤 Enviar Exames" if (st.session_state.get("active_demo_step") == 2 and st.session_state.attached_exams) else "📤 Enviar"
                if st.button(btn_lbl, type="primary", use_container_width=True, key="btn_send_suggested_prompt"):
                    text_to_send = st.session_state.suggested_prompt
                    st.session_state.suggested_prompt = ""
                    queue_turn(text_to_send)

        # Chip visual dos documentos anexados
        if st.session_state.attached_exams and is_chat_enabled:
            total_att = len(st.session_state.attached_exams)
            lista_str = " • ".join(f"`{a['filename']}` ({a.get('origem', 'Laudo')})" for a in st.session_state.attached_exams)

            if st.session_state.suggested_prompt:
                # Se há mensagem guiada pronta, o botão de envio da sugestão já a despacha com os anexos
                c_chip, c_del = st.columns([10, 2], vertical_alignment="center")
                with c_chip:
                    st.info(f"📎 **{total_att}/5 Laudo(s) Anexado(s) para este Passo:** {lista_str}")
                with c_del:
                    if st.button("❌ Limpar Anexos", key="btn_remove_att", use_container_width=True):
                        st.session_state.attached_exams = []
                        st.session_state.attached_exam = None
                        st.rerun()
            else:
                c_chip, c_send_att, c_del = st.columns([7, 3, 2], vertical_alignment="center")
                with c_chip:
                    st.info(f"📎 **{total_att}/5 Laudo(s) Anexado(s):** {lista_str}")
                with c_send_att:
                    if st.button("📤 Enviar Exames para Análise", type="primary", use_container_width=True, key="btn_send_exams_direct"):
                        queue_turn("Seguem os laudos de exames complementares realizados para avaliação diagnóstica e conduta médica.")
                with c_del:
                    if st.button("❌ Limpar", key="btn_remove_att", use_container_width=True):
                        st.session_state.attached_exams = []
                        st.session_state.attached_exam = None
                        st.rerun()

        # Layout lado a lado na mesma linha horizontal: [ + ] [ Caixa de Texto Chat Input... ]
        col_circle, col_input = st.columns([1.2, 22], vertical_alignment="center")

        with col_circle:
            if is_chat_enabled:
                with st.popover("+", help="Clique para anexar até 5 laudos de exames em PDF (externos ou internos do hospital)"):
                    st.markdown("##### 📎 Anexar Laudos de Exames ao Atendimento")
                    st.caption("Você pode anexar **até 5 exames** por envio. Eles serão vetorizados no Qdrant RAG para análise clínica cruzada.")

                    # ---------------------------------------------------------
                    # 1. Upload Externo de Laudos em PDF (trazidos de fora)
                    # ---------------------------------------------------------
                    st.markdown("###### 📤 1. Upload de Laudos Externos (PDF)")
                    st.caption("Envie arquivos PDF do seu dispositivo (selecione até 5 arquivos):")
                    uploaded_pdfs = st.file_uploader(
                        "Selecionar PDFs externos:",
                        type=["pdf"],
                        accept_multiple_files=True,
                        key=f"uploader_chat_popover_{st.session_state.exam_widget_version}"
                    )

                    # ---------------------------------------------------------
                    # 2. Exames Internos do Hospital Universitário FIAP (PACS/LIS)
                    # ---------------------------------------------------------
                    st.markdown("---")
                    st.markdown("###### 🏥 2. Sistema PACS / LIS Interno do Hospital FIAP")
                    st.caption("Laudos laboratoriais e de imagem realizados nas dependências do Hospital Universitário FIAP (rede hospitalar interna):")

                    col_filter_ex, col_refresh_ex = st.columns([5, 1], vertical_alignment="bottom")
                    with col_filter_ex:
                        exam_search_query = st.text_input(
                            "Filtrar laudos internos:",
                            placeholder="Buscar por paciente, exame ou N/S...",
                            key=f"search_exams_popover_{st.session_state.exam_widget_version}"
                        )
                    with col_refresh_ex:
                        if st.button("🔄", help="Atualizar e sincronizar laudos da base interna do hospital", key="btn_refresh_hospital_exams"):
                            st.toast("Base interna de exames do Hospital FIAP atualizada com sucesso!", icon="🏥")
                            st.rerun()

                    fresh_exams = sorted([f for f in os.listdir(exams_dir) if f.endswith(".pdf")]) if os.path.exists(exams_dir) else []

                    if exam_search_query.strip():
                        norm_q = unicodedata.normalize("NFD", exam_search_query.strip().lower()).encode("ascii", "ignore").decode("utf-8")
                        filtered_exams = [
                            e for e in fresh_exams
                            if norm_q in unicodedata.normalize("NFD", e.lower()).encode("ascii", "ignore").decode("utf-8")
                        ]
                    else:
                        filtered_exams = fresh_exams

                    if not filtered_exams:
                        st.warning("⚠️ Nenhum exame interno localizado com o filtro pesquisado.")
                        selected_internal = []
                    else:
                        st.caption(f"Disponíveis {len(filtered_exams)} exame(s) no sistema hospitalar interno:")
                        selected_internal = st.multiselect(
                            "Selecione os exames realizados no hospital (até 5):",
                            options=filtered_exams,
                            max_selections=5,
                            placeholder="Clique para escolher exames internos...",
                            key=f"multiselect_internal_exams_{st.session_state.exam_widget_version}"
                        )

                    # ---------------------------------------------------------
                    # Botão de Confirmação & Validação de Limite
                    # ---------------------------------------------------------
                    st.markdown("---")
                    col_act1, col_act2 = st.columns(2)
                    with col_act1:
                        btn_just_attach = st.button("📎 Apenas Anexar", use_container_width=True, key="btn_confirm_popover")
                    with col_act2:
                        btn_attach_and_send = st.button("🚀 Enviar para Análise Agora", type="primary", use_container_width=True, key="btn_send_immediate_popover")

                    if btn_just_attach or btn_attach_and_send:
                        new_items = []
                        # 1. Processa PDFs enviados por upload
                        if uploaded_pdfs:
                            for up_f in uploaded_pdfs:
                                save_path = os.path.join(exams_dir, up_f.name)
                                with open(save_path, "wb") as f:
                                    f.write(up_f.getbuffer())
                                new_items.append({
                                    "filename": up_f.name,
                                    "path": save_path,
                                    "origem": "Upload Externo"
                                })

                        # 2. Processa exames selecionados da base interna do hospital
                        if selected_internal:
                            for ex_f in selected_internal:
                                if not any(item["filename"] == ex_f for item in new_items):
                                    new_items.append({
                                        "filename": ex_f,
                                        "path": os.path.join(exams_dir, ex_f),
                                        "origem": "Interno HFIAP"
                                    })

                        if not new_items:
                            st.warning("Nenhum exame selecionado ou enviado por upload.")
                        elif len(new_items) > 5:
                            st.error(f"⚠️ Limite excedido: Você selecionou {len(new_items)} exames. O limite máximo é de 5 exames por envio.")
                        else:
                            # Adiciona à lista de anexos acumulados respeitando o limite de 5
                            combined = list(st.session_state.attached_exams)
                            for it in new_items:
                                if not any(c["filename"] == it["filename"] for c in combined):
                                    combined.append(it)
                            if len(combined) > 5:
                                st.error(f"⚠️ Limite excedido: O total acumulado seria de {len(combined)} exames. O limite máximo permitido é 5 exames.")
                            else:
                                st.session_state.attached_exams = combined
                                st.session_state.attached_exam = combined[0] if combined else None
                                st.session_state.exam_widget_version += 1
                                if btn_attach_and_send:
                                    queue_turn("Seguem os laudos de exames complementares realizados para avaliação diagnóstica e conduta médica.")
                                else:
                                    st.toast(f"{len(combined)} exame(s) anexado(s) com sucesso!", icon="📎")
                                    st.rerun()

            else:
                with st.popover("+", help="Chat inativo: Selecione um paciente primeiro"):
                    st.warning("🔒 Selecione um paciente na barra central acima para liberar o anexo de exames ao prontuário.")

        with col_input:
            if is_chat_enabled:
                user_prompt = st.chat_input("Digite a queixa principal, sinais vitais ou orientação para o paciente...")
                if user_prompt:
                    st.session_state.suggested_prompt = ""
                    queue_turn(user_prompt)
            else:
                st.chat_input("🔒 Chat inativo: Localize um paciente na barra central acima para liberar o chat...", disabled=True)


# ------------------------------------------------------------------------------
# ABA 2: BASE DE PROTOCOLOS (100 DOENÇAS)
# ------------------------------------------------------------------------------
with tab_protocols:
    st.subheader("Catálogo de Diretrizes Institucionais Indexadas no Qdrant")
    proto_file = os.path.join(root_dir, "hospital_ai_assistant", "dataset", "processed", "protocols_100_diseases.json")
    if os.path.exists(proto_file):
        import json
        with open(proto_file, "r", encoding="utf-8") as f:
            protocols = json.load(f)
        st.write(f"Total de protocolos cadastrados: **{len(protocols)} diretrizes clínicas oficiais**")
        st.dataframe(protocols, use_container_width=True)
    else:
        st.info("Arquivo de protocolos não encontrado em dataset/processed/.")


# ------------------------------------------------------------------------------
# ABA 3: AUDITORIA E LOGS (LANGSMITH / JSONL)
# ------------------------------------------------------------------------------
with tab_audit:
    st.subheader("🛡️ Observabilidade, Auditoria Conversacional e LangSmith")
    st.markdown(
        "Rastreabilidade ponta a ponta em conformidade com as diretrizes hospitalares e o CFM. "
        "Cada turno conversacional registra o input, output, tokens consumidos, protocolos recuperados e latências."
    )

    # 1. Cards de Métricas em Tempo Real
    summary = st.session_state.audit_logger.get_audit_summary()
    kpi1, kpi2, kpi3, kpi4 = st.columns(4)
    with kpi1:
        st.metric("Consultas Auditadas", f"{summary['total_consultas']} turnos")
    with kpi2:
        st.metric("Latência Média", f"{summary['media_latencia_ms']} ms")
    with kpi3:
        st.metric("Total de Tokens", f"{summary['total_tokens']:,}")
    with kpi4:
        st.metric("Alertas de Guardrail", f"{summary['alertas_guardrails']} intervenções")

    st.markdown("---")

    # 2. Conexão Oficial com o LangSmith
    col_ls1, col_ls2 = st.columns([3, 1])
    with col_ls1:
        st.markdown("""
        **☁️ LangSmith Tracing em Tempo Real (LangChain / LangGraph)**  
        * **Projeto:** `hospital-ai-assistant`  
        * **Status:** 🟢 **Conectado e Rastreando Turnos**  
        * **Endpoint:** `https://api.smith.langchain.com`
        """)
    with col_ls2:
        st.link_button("🌐 Abrir no LangSmith", "https://smith.langchain.com/", use_container_width=True)

    st.markdown("---")

    # 3. Lista Estruturada de Transações Auditadas
    st.markdown("#### 📋 Trilha de Auditoria Detalhada (Últimas Transações)")
    audits = st.session_state.audit_logger.read_recent_audits(limit=15)
    
    if not audits:
        st.info("Nenhuma transação auditada ainda. Realize uma consulta no Chat Clínico para gerar os registros.")
    else:
        for idx, rec in enumerate(reversed(audits)):
            pat = rec.get("patient") or {}
            pat_str = f"{pat.get('nome', 'Consulta Geral')} (ID: {pat.get('id', 'N/A')})"
            status_badge = "🟢 APROVADO" if rec.get("safety_approved") else "🚨 INTERCEPTADO_GUARDRAIL"
            tok = rec.get("tokens", {})
            
            with st.expander(f"Turno #{len(audits) - idx} - {rec.get('timestamp_utc', '')[:19]} | Paciente: {pat_str} | {status_badge}", expanded=(idx == 0)):
                # Métricas rápidas da transação
                c1, c2, c3, c4 = st.columns(4)
                with c1:
                    st.markdown(f"**⏱️ Latência:** `{rec.get('total_latency_ms', 0)} ms`")
                with c2:
                    st.markdown(f"**🔢 Tokens:** Entrada: `{tok.get('prompt_tokens', 0)}` | Saída: `{tok.get('completion_tokens', 0)}` (Total: `{tok.get('total_tokens', 0)}`)")
                with c3:
                    st.markdown(f"**🛡️ Status:** `{rec.get('safety_status')}`")
                with c4:
                    st.markdown(f"**🆔 LangSmith Run:** `{rec.get('transaction_id', '')[:8]}`")

                st.markdown("##### 📥 O Que Entrou (Input do Médico):")
                st.info(rec.get("doctor_input", ""))

                st.markdown("##### 📤 O Que Saiu (Resposta do Assistente):")
                st.markdown(rec.get("assistant_output", ""))

                # RAG e Guardrails
                flags = rec.get("guardrail_flags", [])
                if flags:
                    st.warning(f"⚠️ **Flags de Guardrail Disparadas:** {', '.join(flags)}")

                protos = rec.get("retrieved_protocols", [])
                exams = rec.get("retrieved_exams", [])
                if protos or exams:
                    st.caption(f"📚 **Protocolos:** {', '.join(protos) if protos else 'Nenhum'} | 🔬 **Laudos:** {', '.join(exams) if exams else 'Nenhum'}")

                trace = rec.get("execution_trace", [])
                if trace:
                    st.caption(f"⚡ **Grafo percorrido:** {' ➡️ '.join(trace)}")

                with st.expander("💾 Visualizar Registro JSONL Bruto", expanded=False):
                    st.json(rec)
