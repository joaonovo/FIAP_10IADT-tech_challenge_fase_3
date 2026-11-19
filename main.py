#!/usr/bin/env python3
"""
Ponto de Entrada Principal - Assistente Clínico Hospitalar de IA
================================================================

Tech Challenge - Fase 3: IA para DEVs (Pós-Graduação FIAP)
Arquitetura Geral:
- Chat Interativo Médico (Entrada manual por escrita)
- Fine-Tuning LLaMA 3.2 1B com Unsloth (PubMedQA + 100 Doenças)
- RAG Vetorial com VectorDB Qdrant (Provisionado via Docker Compose)
- Base Estruturada de 1.000 Pacientes / Prontuários no Qdrant VectorDB (RAG)
- Orquestração em Grafo via LangChain & LangGraph
- Camada de Guardrails Clínicos e Auditoria Conversacional (LangSmith)
"""

import sys
import os
import argparse

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

# Garante resolução de imports
root_dir = os.path.dirname(os.path.abspath(__file__))
if root_dir not in sys.path:
    sys.path.insert(0, root_dir)

from interface.cli_interface import MedicalChatSession
from hospital_ai_assistant.safety.audit_logger import ClinicalAuditLogger


def parse_args():
    parser = argparse.ArgumentParser(
        description="Sistema de Suporte à Decisão Clínica Hospitalar (CDSS) - Chat Interativo com LangGraph"
    )
    parser.add_argument(
        "--demo",
        type=str,
        choices=["1", "2", "3"],
        help="Executa diretamente um dos 3 roteiros de demonstração para o vídeo de 15 minutos (1: IAMCSST, 2: Guardrail Alergia, 3: AVC Agudo)"
    )
    parser.add_argument(
        "--web",
        action="store_true",
        help="Inicia a interface Web com Streamlit na porta 8501"
    )
    parser.add_argument(
        "--audit",
        action="store_true",
        help="Exibe os últimos registros de auditoria persistidos em JSONL"
    )
    parser.add_argument(
        "--prepare-all",
        action="store_true",
        help="Executa o pipeline completo: gera pacientes, 100 doenças, dataset SFT, PDFs de exames e indexa no Qdrant"
    )
    return parser.parse_args()


def main():
    args = parse_args()

    if args.prepare_all:
        print("\n⚙️  Executando preparação completa de dados e ambiente...")
        from hospital_ai_assistant.database.patient_db import patient_db
        from hospital_ai_assistant.dataset.build_sft_and_rag_dataset import main as build_sft
        from hospital_ai_assistant.rag_engine.qdrant_rag import rag_manager

        patient_db.load_patients_from_csv(patient_db.seed_csv_path)
        build_sft()
        proto_file = os.path.join(root_dir, "hospital_ai_assistant", "dataset", "processed", "protocols_100_diseases.json")
        rag_manager.initialize_protocols(proto_file)
        print("✅ Todo o ecossistema hospitalar foi preparado com sucesso!\n")
        return

    if args.audit:
        logger = ClinicalAuditLogger()
        records = logger.read_recent_audits(limit=10)
        print("\n" + "=" * 80)
        print("  REGISTROS DE AUDITORIA CONVERSACIONAL (JSONL / LANGSMITH)")
        print("=" * 80)
        for r in records:
            print(f"Transação: {r.get('transaction_id')} | Data: {r.get('timestamp_utc')} | Paciente: {r.get('patient_id')}")
            print(f"  Latência Total: {r.get('total_latency_ms')} ms | Aprovado: {r.get('safety_approved')}")
            print(f"  Guardrails: {r.get('guardrail_flags')}")
            print(f"  Trace: {r.get('metadata', {}).get('execution_trace')}")
            print("-" * 80)
        return

    if args.web:
        print("\n🌐 Iniciando servidor Streamlit do Assistente Hospitalar...")
        import subprocess
        streamlit_app = os.path.join(root_dir, "interface", "web_interface.py")
        subprocess.run([sys.executable, "-m", "streamlit", "run", streamlit_app, "--server.port", "8501"])
        return

    # Inicia a sessão de chat interativo do médico assistente
    session = MedicalChatSession()
    if args.demo:
        session.print_welcome_banner()
        session.run_demo_scenario(args.demo)
    else:
        session.run_interactive_loop()


if __name__ == "__main__":
    main()
