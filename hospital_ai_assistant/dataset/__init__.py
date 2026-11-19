"""Módulo dataset e curadoria de dados clínicos (PubMedQA + 100 Doenças)."""
from .build_sft_and_rag_dataset import main as build_sft_and_rag_dataset

__all__ = ["build_sft_and_rag_dataset"]

