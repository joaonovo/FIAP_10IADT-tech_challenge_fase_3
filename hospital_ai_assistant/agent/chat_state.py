"""
Módulo de Estado Conversacional do LangGraph (Chat State)
=========================================================

Tech Challenge - Fase 3: IA para DEVs (FIAP)
Define o estado mutável do Grafo durante o diálogo interativo do médico.
"""

from typing import List, Dict, Any, Optional, TypedDict, Annotated
import operator
from langchain_core.messages import BaseMessage


class ChatClinicalState(TypedDict):
    """Estado do fluxo conversacional clínico."""
    messages: Annotated[List[BaseMessage], operator.add]
    current_user_input: str
    active_patient: Optional[Dict[str, Any]]
    retrieved_protocols: List[Dict[str, Any]]
    retrieved_exams: List[Dict[str, Any]]
    guardrail_input_safe: bool
    guardrail_output_safe: bool
    guardrail_flags: List[str]
    critical_alerts: List[str]
    clinical_reasoning: str
    final_response: str
    execution_trace: Annotated[List[str], operator.add]
    session_metadata: Dict[str, Any]

