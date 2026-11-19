"""
Módulo de Inferência em Memória do Modelo Fine-Tunado (LLaMA 3.2 1B com LoRA)
Carrega o modelo base + adaptadores LoRA diretamente na VRAM da GPU (NVIDIA RTX)
com fallback inteligente para CPU caso GPU não esteja disponível.
"""

import os
import re
import json
import time
import torch
from typing import Optional, Dict, Any, List
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

current_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.dirname(os.path.dirname(current_dir))
DEFAULT_ADAPTER_DIR = os.path.join(root_dir, "hospital_ai_assistant", "storage", "lora_clinical_adapter")


def _resolve_default_base_model(adapter_dir: str) -> str:
    env_model = os.environ.get("HFIAP_BASE_MODEL")
    if env_model:
        return env_model
    adapter_cfg = os.path.join(adapter_dir, "adapter_config.json")
    if os.path.exists(adapter_cfg):
        try:
            with open(adapter_cfg, "r", encoding="utf-8") as f:
                cfg = json.load(f)
            base_name = cfg.get("base_model_name_or_path")
            if base_name:
                return base_name
        except Exception:
            pass
    return "unsloth/Qwen2.5-7B-Instruct-bnb-4bit"


DEFAULT_BASE_MODEL = _resolve_default_base_model(DEFAULT_ADAPTER_DIR)


class ClinicalFineTunedLLM:
    """Gerenciador Singleton do Modelo Fine-Tunado em Memória."""
    _instance = None
    _model = None
    _tokenizer = None
    _device = None

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super(ClinicalFineTunedLLM, cls).__new__(cls)
        return cls._instance

    def __init__(self, adapter_dir: str = DEFAULT_ADAPTER_DIR, base_model_id: str = DEFAULT_BASE_MODEL):
        if self._model is not None:
            return

        self.adapter_dir = adapter_dir
        self.base_model_id = base_model_id
        self._load_model()

    def _load_model(self):
        """Carrega o Tokenizer, Modelo Base e Adaptadores LoRA na Memória."""
        self._device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"🧠 [ClinicalFineTunedLLM] Carregando modelo em memória ({self._device.upper()})...")
        t0 = time.time()

        try:
            # 1. Carrega Tokenizer com tokens especiais do LLaMA 3.2
            self._tokenizer = AutoTokenizer.from_pretrained(
                self.adapter_dir if os.path.exists(os.path.join(self.adapter_dir, "tokenizer.json")) else self.base_model_id,
                trust_remote_code=True
            )
            if self._tokenizer.pad_token is None:
                self._tokenizer.pad_token = self._tokenizer.eos_token

            # 2. Carrega Modelo Base
            if self._device == "cuda":
                compute_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
                base_model = AutoModelForCausalLM.from_pretrained(
                    self.base_model_id,
                    device_map="auto",
                    dtype=compute_dtype,
                    trust_remote_code=True
                )
            else:
                base_model = AutoModelForCausalLM.from_pretrained(
                    self.base_model_id,
                    dtype=torch.float32,
                    low_cpu_mem_usage=True,
                    trust_remote_code=True
                )

            # 3. Anexa os Adaptadores LoRA Clínicos Fine-Tunados
            if os.path.exists(os.path.join(self.adapter_dir, "adapter_model.safetensors")):
                self._model = PeftModel.from_pretrained(base_model, self.adapter_dir)
                print(f"✅ [ClinicalFineTunedLLM] Adaptador LoRA anexado com sucesso!")
            else:
                self._model = base_model
                print(f"⚠️ [ClinicalFineTunedLLM] Adaptador LoRA não encontrado. Usando modelo base.")

            self._model.eval()
            print(f"🚀 [ClinicalFineTunedLLM] Modelo residente em VRAM carregado em {time.time() - t0:.2f}s!")

        except Exception as e:
            print(f"⚠️ [ClinicalFineTunedLLM] Aviso ao carregar modelo neural: {e}")
            self._model = None
            self._tokenizer = None

    def is_available(self) -> bool:
        return self._model is not None and self._tokenizer is not None

    def get_model_label(self) -> str:
        short_name = self.base_model_id.split("/")[-1]
        has_lora = os.path.exists(os.path.join(self.adapter_dir, "adapter_model.safetensors"))
        return f"{short_name} + LoRA Clínico" if has_lora else short_name

    def generate_clinical_reasoning(
        self,
        system_prompt: str,
        user_query: str,
        context_str: str = "",
        chat_history: Optional[List[Dict[str, str]]] = None,
        max_new_tokens: int = 350
    ) -> Optional[str]:
        """Gera resposta neural conversacional do modelo fine-tunado usando o Chat Template."""
        if not self.is_available():
            return None

        # Monta mensagens estruturadas com histórico recente (multi-turno) + contexto RAG
        messages: List[Dict[str, str]] = [{"role": "system", "content": system_prompt}]

        if chat_history:
            for turn in chat_history[-2:]:
                role = turn.get("role", "user")
                content = (turn.get("content") or "").strip()
                if content and role in ("user", "assistant"):
                    # Mantém apenas resumo curto de turnos anteriores para não competir com o contexto RAG atual
                    messages.append({"role": role, "content": content[:350]})

        full_user_input = user_query
        if context_str:
            full_user_input = f"{context_str}\n\nSolicitação atual do Médico Assistente:\n{user_query}"
        messages.append({"role": "user", "content": full_user_input})

        # Usa apply_chat_template se o tokenizer suportar (Qwen2.5, Llama-3-Instruct, etc.); fallback LLaMA 3.2
        if hasattr(self._tokenizer, "apply_chat_template") and getattr(self._tokenizer, "chat_template", None):
            prompt = self._tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )
        else:
            prompt_parts = ["<|begin_of_text|>"]
            for m in messages:
                prompt_parts.append(
                    f"<|start_header_id|>{m['role']}<|end_header_id|>\n{m['content']}<|eot_id|>\n"
                )
            prompt_parts.append("<|start_header_id|>assistant<|end_header_id|>\n")
            prompt = "".join(prompt_parts)

        try:
            inputs = self._tokenizer(prompt, return_tensors="pt", truncation=True, max_length=6144).to(self._device)
            input_len = inputs.input_ids.shape[1]

            eos_ids = []
            if self._tokenizer.eos_token_id is not None:
                eos_ids.append(self._tokenizer.eos_token_id)
            for tok in ("<|eot_id|>", "<|im_end|>"):
                tid = self._tokenizer.convert_tokens_to_ids(tok)
                if isinstance(tid, int) and tid >= 0 and tid not in eos_ids:
                    eos_ids.append(tid)

            with torch.no_grad():
                outputs = self._model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    temperature=0.2,
                    top_p=0.9,
                    repetition_penalty=1.1,
                    do_sample=True,
                    pad_token_id=self._tokenizer.pad_token_id,
                    eos_token_id=eos_ids if len(eos_ids) > 1 else (eos_ids[0] if eos_ids else None)
                )

            # Decodifica somente os novos tokens gerados pela resposta
            generated_tokens = outputs[0][input_len:]
            response_text = self._tokenizer.decode(generated_tokens, skip_special_tokens=True).strip()

            # Corta eventuais vazamentos de role/template no final da geração
            for stop_marker in ["\nassistant\n", "\nuser\n", "\nsystem\n", "<|start_header_id|>", "The following references were cited", "PubMed ID:"]:
                if stop_marker in response_text:
                    response_text = response_text.split(stop_marker)[0].strip()

            # Sanitização contra loops repetitivos residuais em modelos pequenos
            lines = [line.strip() for line in response_text.split("\n") if line.strip()]
            unique_lines = []
            seen_prefixes = set()
            for l in lines:
                prefix = l[:30].lower()
                if prefix not in seen_prefixes:
                    seen_prefixes.add(prefix)
                    unique_lines.append(l)
            if len(unique_lines) < len(lines):
                response_text = "\n".join(unique_lines)

            return response_text

        except Exception as e:
            print(f"⚠️ [ClinicalFineTunedLLM] Erro durante a geração neural: {e}")
            return None


# Instância global Singleton
clinical_llm = ClinicalFineTunedLLM()

