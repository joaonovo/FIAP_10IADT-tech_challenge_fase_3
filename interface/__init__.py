"""
Pacote de Interfaces do Assistente Hospitalar (CDSS)
- cli_interface: Terminal interativo rico (Rich + prompt_toolkit)
- web_interface: Aplicação Web gráfica interativa (Streamlit)
"""

from interface.cli_interface import MedicalChatSession

__all__ = ["MedicalChatSession"]

