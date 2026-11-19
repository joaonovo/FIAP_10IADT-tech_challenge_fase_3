"""
Módulo de Acesso e Gerenciamento de Prontuários de Pacientes (Qdrant VectorDB / RAG)
=====================================================================================

Tech Challenge - Fase 3: IA para DEVs (Pós-Graduação FIAP)
Finalidade:
Camada de acesso a dados (DAO/Repository) para busca, contextualização e cadastro
de pacientes na base vetorial Qdrant (RAG) do Hospital Universitário FIAP.

Fonte da Verdade:
O Qdrant VectorDB substitui o SQLite e centraliza 100% dos 1.000 prontuários sintéticos,
permitindo busca semântica em linguagem natural, busca por CPF, Cartão SUS e Nome.
"""

import os
import logging
from typing import Dict, List, Any, Optional
from hospital_ai_assistant.rag_engine.qdrant_rag import rag_manager

logger = logging.getLogger(__name__)

DEFAULT_CSV_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "dataset",
    "raw",
    "patients_1000.csv"
)


class PatientDatabaseManager:
    """
    Gerenciador de acesso ao banco de dados vetorial (Qdrant RAG) de pacientes e prontuários.
    Garante que o VectorDB seja a ÚNICA Fonte da Verdade do sistema hospitalar.
    """

    def __init__(self, seed_csv_path: str = DEFAULT_CSV_PATH):
        self.seed_csv_path = seed_csv_path
        self._init_qdrant_patients()

    def _init_qdrant_patients(self):
        """Inicializa e popula os 1.000 pacientes no Qdrant RAG se ainda não estiverem indexados."""
        total = rag_manager.count_patients()
        if total < 1000 and os.path.exists(self.seed_csv_path):
            logger.info("Indexando 1.000 pacientes sintéticos no Qdrant VectorDB a partir de %s...", self.seed_csv_path)
            rag_manager.initialize_patients(self.seed_csv_path)

    def count_patients(self) -> int:
        """Retorna o total de prontuários de pacientes cadastrados no Qdrant RAG."""
        return rag_manager.count_patients()

    def load_patients_from_csv(self, csv_filepath: str):
        """Ingere os 1.000 registros do arquivo CSV diretamente para o Qdrant VectorDB."""
        rag_manager.initialize_patients(csv_filepath)

    def search_patient(self, query: str, limit: int = 5) -> List[Dict[str, Any]]:
        """
        Busca pacientes no Qdrant VectorDB via RAG:
        - Busca por CPF (com ou sem formatação)
        - Busca por Cartão SUS
        - Busca por termos compostos no Nome
        - Busca Semântica contextual no Prontuário
        """
        return rag_manager.search_patients(query, limit=limit)

    def get_patient_by_id(self, patient_id: Any) -> Optional[Dict[str, Any]]:
        """Recupera prontuário de paciente no Qdrant RAG por ID."""
        return rag_manager.get_patient_by_id(patient_id)


# Instância global do DAO de pacientes baseado no Qdrant RAG
patient_db = PatientDatabaseManager()


if __name__ == "__main__":
    db = PatientDatabaseManager()
    total = db.count_patients()
    print(f"Total de pacientes no Qdrant RAG: {total}")
    sample = db.search_patient("Maria Helena")
    if sample:
        p = sample[0]
        print(f"Amostra localizada no Qdrant: [{p.get('id')}] {p.get('nome')} | CPF: {p.get('cpf')} | SUS: {p.get('cartao_sus')}")
