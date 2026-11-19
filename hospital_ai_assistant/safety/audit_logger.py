"""
Módulo de Auditoria e Logging Estruturado Clínico
=================================================

Tech Challenge - Fase 3: IA para DEVs (FIAP)
Garante rastreabilidade total (Audit Trail) e observabilidade completa:
1. O que entrou: Pergunta exata do médico assistente e contexto do paciente;
2. O que saiu: Resposta clínica gerada, recomendações e disclaimers;
3. Consumo de Tokens: Tokens de entrada, saída e total via tiktoken;
4. Latência: Tempo de execução total e detalhamento nó a nó do LangGraph;
5. Segurança: Flags disparadas por Guardrails (CFM nº 2.314/2022 e Alergias);
6. RAG: Protocolos institucionais e laudos de exames em PDF consultados;
7. LangSmith: Integração nativa com rastreamento na nuvem.
"""

import os
import sys
import time
import json
import uuid
import logging
from datetime import datetime, timezone
from typing import Dict, Any, Optional, List

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

try:
    import tiktoken
    _TOKEN_ENCODER = tiktoken.get_encoding("cl100k_base")
except Exception:
    _TOKEN_ENCODER = None


def estimate_tokens(text: str) -> int:
    """Calcula ou estima o número de tokens de uma string."""
    if not text:
        return 0
    if _TOKEN_ENCODER:
        try:
            return len(_TOKEN_ENCODER.encode(text))
        except Exception:
            pass
    return max(1, len(text.split()) * 4 // 3)


logger = logging.getLogger("HospitalAIAudit")


class AuditTransaction:
    """Estrutura de compatibilidade para transações ativas de auditoria."""

    def __init__(self, patient_id: str = "PACIENTE_GERAL", case_id: Optional[str] = None):
        self.transaction_id = str(uuid.uuid4())
        self.patient_id = patient_id
        self.case_id = case_id
        self.guardrail_flags: List[str] = []
        self.safety_approved: bool = True
        self.metadata: Dict[str, Any] = {}
        self.retrieved_protocols: List[str] = []
        self.retrieved_exams: List[str] = []
        self.is_critical: bool = False
        self.start_time = time.time()
        self.start_timestamp = datetime.now(timezone.utc).isoformat()
        self.node_latencies: Dict[str, float] = {}

    def record_node_latency(self, node_name: str, duration_ms: float):
        self.node_latencies[node_name] = round(duration_ms, 2)


class ClinicalAuditLogger:
    """Gerenciador de auditoria e persistência de registros clínicos em JSONL."""

    DEFAULT_LOG_FILE = "clinical_audit.jsonl"

    def __init__(self, log_dir: Optional[str] = None):
        self.log_dir = log_dir or self._resolve_default_log_dir()
        os.makedirs(self.log_dir, exist_ok=True)
        self.log_file_path = os.path.join(self.log_dir, self.DEFAULT_LOG_FILE)

    @staticmethod
    def _resolve_default_log_dir() -> str:
        """Resolve o diretório padrão de audit_logs."""
        current_dir = os.path.dirname(os.path.abspath(__file__))
        target = os.path.join(current_dir, "..", "audit_logs")
        return os.path.normpath(target)

    def log_turn(
        self,
        doctor_input: str,
        assistant_output: str,
        patient: Optional[Dict[str, Any]] = None,
        retrieved_protocols: Optional[List[Dict[str, Any]]] = None,
        retrieved_exams: Optional[List[Dict[str, Any]]] = None,
        guardrail_flags: Optional[List[str]] = None,
        execution_trace: Optional[List[str]] = None,
        latency_ms: float = 0.0,
        node_latencies: Optional[Dict[str, float]] = None,
        langsmith_run_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Registra com total transparência o ciclo completo de um atendimento.
        Captura input, output, tokens, tempo, guardrails e dados de RAG.
        """
        transaction_id = str(uuid.uuid4())
        timestamp_utc = datetime.now(timezone.utc).isoformat()

        # Cálculo preciso de tokens
        input_tokens = estimate_tokens(doctor_input)
        output_tokens = estimate_tokens(assistant_output)
        total_tokens = input_tokens + output_tokens

        # Extração de protocolos e exames simplificados
        protocols_summary = []
        if retrieved_protocols:
            for p in retrieved_protocols:
                protocols_summary.append(p.get("title") or p.get("protocol_id", "Protocolo HFIAP"))

        exams_summary = []
        if retrieved_exams:
            for e in retrieved_exams:
                exams_summary.append(e.get("source", "Laudo PDF"))

        flags = guardrail_flags or []
        safety_approved = len(flags) == 0

        patient_info = None
        if patient:
            patient_info = {
                "id": f"HFIAP-{patient.get('id', 0):04d}" if isinstance(patient.get("id"), int) else str(patient.get("id")),
                "nome": patient.get("nome", "Desconhecido"),
                "cpf": patient.get("cpf", "N/A"),
                "alergias": patient.get("alergias", "Nenhuma"),
                "comorbidades": patient.get("comorbidades", "Nenhuma")
            }

        record = {
            "transaction_id": transaction_id,
            "timestamp_utc": timestamp_utc,
            "patient": patient_info,
            "doctor_input": doctor_input,
            "assistant_output": assistant_output,
            "tokens": {
                "prompt_tokens": input_tokens,
                "completion_tokens": output_tokens,
                "total_tokens": total_tokens
            },
            "retrieved_protocols": protocols_summary,
            "retrieved_exams": exams_summary,
            "guardrail_flags": flags,
            "safety_status": "APROVADO" if safety_approved else "INTERCEPTADO_GUARDRAIL",
            "safety_approved": safety_approved,
            "total_latency_ms": round(latency_ms, 2),
            "node_latencies": node_latencies or {},
            "execution_trace": execution_trace or [],
            "langsmith": {
                "project": os.getenv("LANGCHAIN_PROJECT", "hospital-ai-assistant"),
                "run_id": langsmith_run_id or transaction_id,
                "traced": bool(os.getenv("LANGCHAIN_TRACING_V2", "false").lower() == "true")
            }
        }

        # Persistência em JSONL
        try:
            with open(self.log_file_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
        except Exception as e:
            logger.error(f"Falha ao persistir log de auditoria: {e}")

        logger.info(
            f"[AUDIT] ID={transaction_id[:8]} | Paciente={patient_info['nome'] if patient_info else 'Geral'} | "
            f"Tokens={total_tokens} | Latência={record['total_latency_ms']}ms | Status={record['safety_status']}"
        )
        return record

    def log_transaction(
        self,
        patient_id: str,
        input_text: str,
        output_text: str,
        guardrail_flags: Optional[List[str]] = None,
        execution_trace: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Any]] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """Compatibilidade reversa com chamadas antigas."""
        return self.log_turn(
            doctor_input=input_text,
            assistant_output=output_text,
            patient={"id": patient_id, "nome": f"Paciente {patient_id}", "cpf": "N/A", "alergias": "N/A"},
            guardrail_flags=guardrail_flags,
            execution_trace=execution_trace,
            latency_ms=metadata.get("latency_ms", 30.0) if metadata else 30.0
        )

    def read_recent_audits(self, limit: int = 15) -> List[Dict[str, Any]]:
        """Lê os últimos registros do arquivo de auditoria JSONL."""
        if not os.path.exists(self.log_file_path):
            return []

        records = []
        try:
            with open(self.log_file_path, "r", encoding="utf-8") as f:
                for line in f:
                    line_str = line.strip()
                    if line_str:
                        try:
                            records.append(json.loads(line_str))
                        except Exception:
                            pass
        except Exception as e:
            logger.error(f"Erro ao ler registros de auditoria: {e}")

        return records[-limit:]

    def get_audit_summary(self) -> Dict[str, Any]:
        """Calcula métricas agregadas dos atendimentos auditados."""
        records = self.read_recent_audits(limit=100)
        if not records:
            return {
                "total_consultas": 0,
                "media_latencia_ms": 0.0,
                "total_tokens": 0,
                "alertas_guardrails": 0
            }

        total = len(records)
        total_lat = sum(r.get("total_latency_ms", 0) for r in records)
        total_tok = sum(r.get("tokens", {}).get("total_tokens", 0) for r in records)
        total_alerts = sum(len(r.get("guardrail_flags", [])) for r in records)

        return {
            "total_consultas": total,
            "media_latencia_ms": round(total_lat / max(1, total), 2),
            "total_tokens": total_tok,
            "alertas_guardrails": total_alerts
        }
