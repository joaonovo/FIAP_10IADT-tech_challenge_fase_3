"""
Redirecionador para a interface Web Streamlit em interface/web_interface.py
Tech Challenge - Fase 3: IA para DEVs (FIAP)
"""
import runpy
import os

target = os.path.join(os.path.dirname(__file__), "interface", "web_interface.py")
runpy.run_path(target, run_name="__main__")

