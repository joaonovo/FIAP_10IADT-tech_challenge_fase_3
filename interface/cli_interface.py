"""
Interface de Chat Interativo do Assistente Médico (CLI Interativo com Rich)
==========================================================================

Tech Challenge - Fase 3: IA para DEVs (FIAP)
Permite ao médico interagir por escrita com o Assistente de Inteligência Artificial:
- Consulta de prontuários na base de 1.000 pacientes;
- Ingestão de laudos de exames em PDF no VectorDB Qdrant em tempo real;
- Suporte a conduta clínica baseada nos protocolos das 100 doenças e PubMedQA;
- Guardrails clínicos visíveis com alertas e auditoria detalhada;
- Roteiros automatizados de demonstração rápida (/demo 1, /demo 2, /demo 3).
"""

import os
import sys
import time
import random
import uuid
import unicodedata
from typing import Optional, Dict, Any, List

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

# Garante resolução de imports
current_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = current_dir
while root_dir and not os.path.exists(os.path.join(root_dir, 'hospital_ai_assistant')):
    parent = os.path.dirname(root_dir)
    if parent == root_dir:
        break
    root_dir = parent
for p in [current_dir, root_dir]:
    if p not in sys.path:
        sys.path.insert(0, p)

from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.markdown import Markdown
from rich.prompt import Prompt

from prompt_toolkit import PromptSession
from prompt_toolkit.completion import Completer, Completion

from hospital_ai_assistant.agent.chat_graph import hospital_chat_graph
from hospital_ai_assistant.database.patient_db import patient_db
from hospital_ai_assistant.rag_engine.qdrant_rag import rag_manager
from hospital_ai_assistant.safety.audit_logger import ClinicalAuditLogger

console = Console()


def normalize_text(text: str) -> str:
    """Normaliza texto para busca case-insensitive e sem acentos."""
    return "".join(
        c for c in unicodedata.normalize("NFD", text)
        if unicodedata.category(c) != "Mn"
    ).lower()


class PatientCompleter(Completer):
    """Auto-sugestão em tempo real de pacientes cadastrados a partir de 4 caracteres."""

    def __init__(self, db_manager):
        self.db = db_manager
        self.cached_patients = []
        self.load_cache()

    def load_cache(self):
        """Carrega lista leve de pacientes para busca em sub-milissegundos."""
        try:
            import csv
            csv_path = getattr(self.db, "seed_csv_path", None)
            if csv_path and os.path.exists(csv_path):
                with open(csv_path, "r", encoding="utf-8") as f:
                    reader = csv.DictReader(f)
                    self.cached_patients = [
                        {"id": idx + 1, "nome": row.get("nome", ""), "cpf": row.get("cpf", "")}
                        for idx, row in enumerate(reader)
                    ]
            else:
                self.cached_patients = self.db.search_patient("", limit=50)
        except Exception:
            self.cached_patients = []

    def get_completions(self, document, complete_event):
        text = document.text_before_cursor
        
        # 1. Sugestão ao digitar /paciente <termo>
        if text.startswith("/paciente "):
            query = text[10:].strip()
            if len(query) >= 4:
                norm_q = normalize_text(query)
                matches = [
                    p for p in self.cached_patients
                    if norm_q in normalize_text(p["nome"]) or norm_q in normalize_text(p["cpf"])
                ]
                for p in matches[:8]:
                    yield Completion(
                        text=p["nome"],
                        start_position=-len(text[10:]),
                        display=f"{p['nome']} (CPF: {p['cpf']})"
                    )

        # 2. Sugestão ao digitar comandos iniciados por '/'
        elif text.startswith("/") and len(text) <= 5:
            cmds = ["/paciente ", "/upload ", "/demo 1", "/demo 2", "/demo 3", "/audit", "/limpar", "/sair"]
            for c in cmds:
                if c.startswith(text):
                    yield Completion(text=c, start_position=-len(text))

        # 3. Auto-sugestão de nomes no meio do texto (a partir de 4 letras)
        elif len(text.strip()) >= 4:
            words = text.split()
            last_word = words[-1].strip() if words else ""
            if len(last_word) >= 4:
                norm_last = normalize_text(last_word)
                matches = [
                    p for p in self.cached_patients
                    if norm_last in normalize_text(p["nome"])
                ]
                for p in matches[:5]:
                    yield Completion(
                        text=p["nome"],
                        start_position=-len(last_word),
                        display=f"👤 {p['nome']} (CPF: {p['cpf']})"
                    )


class MedicalChatSession:
    """Sessão conversacional interativa do médico assistente."""

    def __init__(self):
        self.graph = hospital_chat_graph
        self.active_patient: Optional[Dict[str, Any]] = None
        self.conversation_history = []
        self.attached_exam_filenames = []
        self.audit_logger = ClinicalAuditLogger()
        self.patient_completer = PatientCompleter(patient_db)
        self.prompt_session = PromptSession(completer=self.patient_completer)

    def print_welcome_banner(self):
        banner_text = (
            "[bold white]HOSPITAL UNIVERSITÁRIO FIAP (HFIAP)[/bold white]\n"
            "[cyan]Sistema de Suporte à Decisão Clínica (CDSS) com LangChain & LangGraph[/cyan]\n\n"
            "[dim]• Base RAG Qdrant: 110 Protocolos de Tratamento + PubMedQA Biomédico\n"
            "• Base de Prontuários: 1.000 Pacientes Indexados no Qdrant VectorDB (RAG)\n"
            "• Fine-Tuning: LLaMA 3.2 1B (Unsloth QLoRA)\n"
            "• Camada de Segurança: Guardrails CFM nº 2.314/2022 (Zero Prescrição Autônoma)[/dim]\n\n"
            "[yellow]Comandos rápidos disponíveis:[/yellow]\n"
            "  [bold green]/paciente <nome ou CPF>[/bold green] -> Seleciona paciente da base de dados (1.000 existentes)\n"
            "  [bold green]/upload <caminho.pdf>[/bold green]  -> Ingestão de exame/laudo em PDF no Qdrant\n"
            "  [bold green]/demo <1, 2 ou 3>[/bold green]     -> Executa roteiros prontos de demonstração\n"
            "  [bold green]/audit[/bold green]                 -> Exibe os últimos logs de auditoria e latência\n"
            "  [bold green]/limpar[/bold green]                -> Reinicia o contexto do chat\n"
            "  [bold green]/sair[/bold green]                  -> Encerra o assistente"
        )
        console.print(Panel(banner_text, title="🩺 CDSS MEDICAL CHAT ASSISTANT", border_style="cyan"))

    def show_patient_card(self):
        """Exibe o prontuário do paciente ativo na sessão."""
        if not self.active_patient:
            return
        p = self.active_patient
        t = Table(title=f"👤 Paciente Ativo: {p['nome']} (ID: HFIAP-{p['id']:04d})", border_style="green")
        t.add_column("Campo", style="bold cyan")
        t.add_column("Informação Clínica", style="white")

        t.add_row("Idade / Sexo", f"{p['idade']} anos | {p['genero']}")
        t.add_row("CPF", p["cpf"])
        t.add_row("Cartão SUS", p["cartao_sus"])
        t.add_row("Endereço", p["endereco"])
        t.add_row("Comorbidades", p["comorbidades"])
        t.add_row("Alergias Conhecidas", f"[bold red]{p['alergias']}[/bold red]")
        t.add_row("Medicações Prévias", p["medicamentos_em_uso"])
        t.add_row("Histórico Clínico", p["historico_clinico"])
        console.print(t)

    def handle_command(self, user_cmd: str) -> bool:
        """Processa comandos iniciados por '/'."""
        parts = user_cmd.strip().split(maxsplit=1)
        cmd = parts[0].lower()
        arg = parts[1] if len(parts) > 1 else ""

        if cmd in ["/sair", "/exit", "/quit"]:
            console.print("\n[bold yellow]Encerrando a sessão do Assistente Hospitalar. Até breve, Doutor(a)![/bold yellow]\n")
            return False

        elif cmd == "/limpar":
            self.active_patient = None
            self.conversation_history = []
            self.attached_exam_filenames = []
            console.print("[bold green]✅ Sessão reiniciada e memória limpa.[/bold green]")

        elif cmd == "/paciente":
            if not arg:
                console.print("[red]Uso correto: /paciente <nome, CPF ou SUS>[/red]")
            else:
                results = patient_db.search_patient(arg)
                if results:
                    old_id = self.active_patient.get("id") if self.active_patient else None
                    new_id = results[0].get("id")
                    if old_id != new_id:
                        self.attached_exam_filenames = []
                        self.conversation_history = []
                    self.active_patient = results[0]
                    console.print(f"[bold green]✅ Paciente selecionado com sucesso:[/bold green] {self.active_patient['nome']}")
                    self.show_patient_card()
                else:
                    console.print(f"[red]Nenhum paciente localizado com o critério: '{arg}'[/red]")

        elif cmd in ["/cadastrar", "/admitir", "/novo_paciente"]:
            console.print("[yellow]ℹ️ O sistema opera exclusivamente com a base hospitalar de 1.000 pacientes existentes no Qdrant VectorDB. Use [bold green]/paciente <nome ou CPF>[/bold green] para localizar um prontuário.[/yellow]")

        elif cmd == "/upload":
            if not arg:
                console.print("[red]Uso correto: /upload <caminho_do_arquivo.pdf>[/red]")
            else:
                pdf_path = os.path.abspath(arg)
                if not os.path.exists(pdf_path):
                    console.print(f"[red]Arquivo não encontrado: {pdf_path}[/red]")
                else:
                    with console.status("[cyan]Processando e vetorizando laudo no Qdrant...[/cyan]"):
                        pat_id = self.active_patient["id"] if self.active_patient else None
                        ingest_result = rag_manager.ingest_pdf(pdf_path, patient_id=pat_id)
                        fn = ingest_result.get("filename")
                        if fn and fn not in self.attached_exam_filenames:
                            self.attached_exam_filenames.append(fn)
                    console.print(Panel(
                        f"[bold green]✅ Exame indexado com sucesso no Qdrant![/bold green]\n\n"
                        f"• Arquivo: {ingest_result['filename']}\n"
                        f"• Chunks Vetorizados: {ingest_result['chunks_count']}\n"
                        f"• Paciente Vinculado: {self.active_patient['nome'] if self.active_patient else 'Geral / Sem Vínculo'}",
                        title="🔬 Ingestão RAG de Laudo",
                        border_style="green"
                    ))

        elif cmd == "/audit":
            records = self.audit_logger.read_recent_audits(limit=4)
            if not records:
                console.print("[dim]Nenhum registro de auditoria persistido ainda.[/dim]")
            else:
                t = Table(title="📋 Últimas Transações Auditadas (JSONL)", border_style="magenta")
                t.add_column("Transação ID", style="dim")
                t.add_column("Paciente", style="cyan")
                t.add_column("Status Segurança", style="bold")
                t.add_column("Latência", style="yellow")
                t.add_column("Flags / Alertas", style="red")

                for r in records:
                    status_fmt = "[green]APROVADO[/green]" if r.get("safety_approved") else "[red]ALERTA[/red]"
                    t.add_row(
                        r.get("transaction_id", "")[:8],
                        r.get("patient_id", ""),
                        status_fmt,
                        f"{r.get('total_latency_ms', 0)} ms",
                        ", ".join(r.get("guardrail_flags", [])) or "Nenhuma"
                    )
                console.print(t)

        elif cmd == "/demo":
            self.run_demo_scenario(arg)

        else:
            console.print(f"[red]Comando desconhecido: {cmd}. Digite /paciente, /upload, /demo, /audit, /limpar ou /sair.[/red]")

        return True

    def run_demo_scenario(self, scenario_num: str):
        """Executa roteiros guiados de 3 etapas para o vídeo e avaliação do Tech Challenge."""
        exams_dir = os.path.join(root_dir, "data", "exams")

        if scenario_num == "1":
            console.print("\n[bold cyan]🎬 EXECUTANDO ROTEIRO 1: Paciente com Dor Torácica (Maria Helena - IAMCSST)[/bold cyan]")
            self.handle_command("/paciente Maria Helena Oliveira")
            
            # Etapa 1: Admissão Inicial
            q1 = "Paciente Maria Helena Oliveira, 58 anos, admitida na sala de emergência com quadro agudo de dor torácica retroesternal opressiva de forte intensidade iniciada há 40 minutos, irradiada para membro superior esquerdo e mandíbula, acompanhada de sudorese fria e náuseas. Sinais vitais: PA 140/90 mmHg, FC 94 bpm, SpO2 97% em ar ambiente."
            console.print(f"\n[bold green]👨‍⚕️ Médico (Etapa 1 - Admissão):[/bold green] {q1}")
            self.process_chat_message(q1)
            time.sleep(1)

            # Etapa 2: Ingestão de Exames e Correlação
            pdf_path = os.path.join(exams_dir, "MARIA_HELENA_OLIVEIRA_817293049182_ECG_TROPONINA.pdf")
            self.handle_command(f"/upload {pdf_path}")
            time.sleep(1)
            q2 = "Seguem os laudos de exames complementares anexados: ECG de 12 derivações e Troponina ultrassensível. Solicito a correlação clínico-laboratorial e confirmação diagnóstica."
            console.print(f"\n[bold green]👨‍⚕️ Médico (Etapa 2 - Exames):[/bold green] {q2}")
            self.process_chat_message(q2)
            time.sleep(1)

            # Etapa 3: Dúvida Clínica & Intercorrência Hemodinâmica
            q3 = "A paciente acabou de evoluir com queda de pressão: PA 80/50 mmHg. Podemos manter nitratos venosos ou betabloqueadores nesse momento?"
            console.print(f"\n[bold green]👨‍⚕️ Médico (Etapa 3 - Dúvida Clínica):[/bold green] {q3}")
            self.process_chat_message(q3)

        elif scenario_num == "2":
            console.print("\n[bold cyan]🎬 EXECUTANDO ROTEIRO 2: Pneumonia & Guardrail de Alergia (João da Silva)[/bold cyan]")
            self.handle_command("/paciente João da Silva")
            
            # Etapa 1: Admissão Inicial
            q1 = "Paciente João da Silva, 67 anos, admitido no pronto-socorro com tosse produtiva, febre diária há 3 dias (Tax: 38.4°C), calafrios e piora progressiva da falta de ar aos esforços. Sinais vitais: FR 24 irpm, FC 102 bpm, SpO2 91% em ar ambiente. Prontuário destaca histórico de DPOC e alergia grave a Penicilinas (choque anafilático prévio)."
            console.print(f"\n[bold green]👨‍⚕️ Médico (Etapa 1 - Admissão):[/bold green] {q1}")
            self.process_chat_message(q1)
            time.sleep(1)

            # Etapa 2: Ingestão de Exames e Correlação
            pdf_path = os.path.join(exams_dir, "JOAO_DA_SILVA_948201839201_HEMOGRAMA_PCR.pdf")
            self.handle_command(f"/upload {pdf_path}")
            time.sleep(1)
            q2 = "Seguem os laudos de exames complementares anexados: Hemograma completo com contagem leucocitária e PCR do paciente João da Silva. Solicito a correlação clínico-laboratorial e confirmação diagnóstica."
            console.print(f"\n[bold green]👨‍⚕️ Médico (Etapa 2 - Exames):[/bold green] {q2}")
            self.process_chat_message(q2)
            time.sleep(1)

            # Etapa 3: Dúvida Clínica & Guardrail de Farmacovigilância
            q3 = "Podemos prescrever Amoxicilina com Clavulanato 875mg ou Ceftriaxona venosa para tratar a pneumonia do João da Silva? Qual a alternativa antimicrobiana segura indicada pelo protocolo para este paciente alérgico a penicilinas?"
            console.print(f"\n[bold green]👨‍⚕️ Médico (Etapa 3 - Dúvida Clínica & Prescrição):[/bold green] {q3}")
            self.process_chat_message(q3)

        elif scenario_num == "3":
            console.print("\n[bold cyan]🎬 EXECUTANDO ROTEIRO 3: AVC Isquêmico & Trombólise Segura (Carlos Eduardo Santos)[/bold cyan]")
            self.handle_command("/paciente Carlos Eduardo Santos")
            
            # Etapa 1: Admissão Inicial
            q1 = "Paciente Carlos Eduardo Santos, 72 anos, hipertenso e em uso crônico de Varfarina, admitido na emergência com déficit neurológico agudo caracterizado por hemiparesia braquiocrural direita e disartria súbitas iniciadas há 80 minutos (delta T < 4.5h). Sinais vitais: PA 160/95 mmHg, FC 82 bpm, Glicemia capilar 118 mg/dL."
            console.print(f"\n[bold green]👨‍⚕️ Médico (Etapa 1 - Admissão):[/bold green] {q1}")
            self.process_chat_message(q1)
            time.sleep(1)

            # Etapa 2: Ingestão de Exames e Correlação
            pdf_path1 = os.path.join(exams_dir, "CARLOS_EDUARDO_SANTOS_736192840192_TC_CRANIO.pdf")
            pdf_path2 = os.path.join(exams_dir, "CARLOS_EDUARDO_SANTOS_736192840193_COAGULOGRAMA_RNI.pdf")
            self.handle_command(f"/upload {pdf_path1}")
            self.handle_command(f"/upload {pdf_path2}")
            time.sleep(1)
            q2 = "Seguem os laudos de exames complementares anexados: TC de Crânio sem contraste e Coagulograma com RNI do paciente Carlos Eduardo Santos. Solicito a correlação clínico-laboratorial e confirmação diagnóstica."
            console.print(f"\n[bold green]👨‍⚕️ Médico (Etapa 2 - Exames):[/bold green] {q2}")
            self.process_chat_message(q2)
            time.sleep(1)

            # Etapa 3: Dúvida Clínica & Intercorrência Pressórica e Antiagregação
            q3 = "O paciente apresentou pico hipertensivo com PA 195/110 mmHg e a equipe quer administrar AAS e heparina agora. Podemos iniciar o trombolítico com essa PA e administrar o AAS nas primeiras 24 horas?"
            console.print(f"\n[bold green]👨‍⚕️ Médico (Etapa 3 - Dúvida Clínica & Trombólise):[/bold green] {q3}")
            self.process_chat_message(q3)

        else:
            console.print("[yellow]Escolha: /demo 1, /demo 2 ou /demo 3.[/yellow]")

    def process_chat_message(self, message: str):
        """Envia mensagem ao LangGraph e renderiza o raciocínio clínico e guardrails."""
        with console.status("[bold cyan]Executando Grafo LangGraph (RAG + CoT + Guardrails)...[/bold cyan]"):
            start_t = time.time()
            result = self.graph.chat(
                user_message=message,
                active_patient=self.active_patient,
                message_history=self.conversation_history,
                attached_exam_filenames=self.attached_exam_filenames
            )
            elapsed_ms = round((time.time() - start_t) * 1000, 2)

        # Atualiza o histórico conversacional acumulado na sessão
        if "messages" in result and result["messages"]:
            self.conversation_history = result["messages"]

        # Atualiza paciente ativo caso o resolvedor de entidades tenha encontrado um novo
        if result.get("active_patient"):
            old_id = self.active_patient.get("id") if self.active_patient else None
            new_id = result["active_patient"].get("id")
            if old_id and new_id != old_id:
                self.attached_exam_filenames = []
            self.active_patient = result["active_patient"]

        # 1. Rastreamento do Grafo (LangGraph Execution Trace)
        trace_str = " ➡️  ".join(result.get("execution_trace", []))
        console.print(f"\n[dim cyan]⚡ [Trace LangGraph - {elapsed_ms}ms]: {trace_str}[/dim cyan]")

        # 2. Exibição de Alertas de Guardrails
        flags = result.get("guardrail_flags", [])
        if flags:
            console.print(Panel(
                f"[bold yellow]⚠️ Interceptação por Guardrail Clínico:[/bold yellow]\n" +
                "\n".join(f"• {f}" for f in flags),
                title="🛡️ Guardrails LangChain",
                border_style="yellow"
            ))

        # 3. Resposta Clínica Formatada
        console.print("\n[bold cyan]🤖 Assistente Médico FIAP:[/bold cyan]")
        console.print(Markdown(result.get("final_response", "")))
        console.print("\n" + "─" * 80 + "\n")

    def run_interactive_loop(self):
        """Loop principal do chat."""
        self.print_welcome_banner()

        while True:
            try:
                active_name = self.active_patient['nome'] if self.active_patient else 'Nenhum'
                prompt_label = f"\n👨‍⚕️ Dr(a) [Paciente: {active_name}] > "
                user_text = self.prompt_session.prompt(prompt_label).strip()

                if not user_text:
                    continue

                if user_text.startswith("/"):
                    keep_running = self.handle_command(user_text)
                    if not keep_running:
                        break
                else:
                    self.process_chat_message(user_text)

            except (KeyboardInterrupt, EOFError):
                console.print("\n[yellow]Sessão encerrada pelo usuário. Até logo![/yellow]")
                break
            except Exception as e:
                console.print(f"[bold red]Erro na execução: {e}[/bold red]")


def start_chat():
    session = MedicalChatSession()
    session.run_interactive_loop()


if __name__ == "__main__":
    start_chat()

