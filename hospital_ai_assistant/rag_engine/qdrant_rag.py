"""
Módulo RAG Engine com VectorDB Qdrant (Docker Compose) e Ingestão de Exames/PDFs
================================================================================

Tech Challenge - Fase 3: IA para DEVs (FIAP)
Finalidade:
- Provisiona e gerencia a coleção vetorial no Qdrant (`http://localhost:6333`);
- Indexa o catálogo das 100 doenças e protocolos institucionais do HFIAP;
- Ingestão dinâmica de laudos e exames clínicos em PDF (via PyPDF) vinculados a pacientes;
- Busca semântica e contextualizada para o Grafo do LangChain/LangGraph.
"""

import os
import re
import json
import logging
from datetime import time
from typing import List, Dict, Any, Optional

from qdrant_client import QdrantClient
from qdrant_client.http import models as qmodels
from pypdf import PdfReader

logger = logging.getLogger(__name__)

COLLECTION_NAME = "hospital_knowledge_and_exams"
QDRANT_URL = os.environ.get("QDRANT_URL", "http://localhost:6333")
EMBEDDING_DIM = 384  # Padrão sentence-transformers all-MiniLM-L6-v2 ou projeção determinística


EMBEDDING_MODEL_NAME = os.environ.get(
    "HFIAP_EMBEDDING_MODEL",
    "intfloat/multilingual-e5-small"
)
EMBEDDING_DEVICE = os.environ.get("HFIAP_EMBEDDING_DEVICE", "cpu")


class ClinicalEmbeddingModel:
    """
    Gerenciador de embeddings multilíngues para RAG assimétrico (PT-BR) com fallback local.
    Utiliza `intfloat/multilingual-e5-small` (384 dims, 512 tokens) com prefixos nativos
    `passage:` (indexação) e `query:` (busca), executando em CPU para reservar 100% da VRAM ao LLM.
    """

    def __init__(self):
        self._st_model = None
        self._is_e5 = "e5" in EMBEDDING_MODEL_NAME.lower()
        try:
            from sentence_transformers import SentenceTransformer
            logger.info("Carregando modelo SentenceTransformer %s (device=%s)...", EMBEDDING_MODEL_NAME, EMBEDDING_DEVICE)
            self._st_model = SentenceTransformer(EMBEDDING_MODEL_NAME, device=EMBEDDING_DEVICE)
            logger.info("SentenceTransformer carregado com sucesso!")
        except Exception as e:
            logger.warning("SentenceTransformer não carregou (%s). Utilizando projeção semântica rápida.", e)

    def _apply_prefix(self, texts: List[str], mode: str) -> List[str]:
        if not self._is_e5:
            return texts
        prefix = "query: " if mode == "query" else "passage: "
        return [t if t.startswith(("query: ", "passage: ")) else f"{prefix}{t}" for t in texts]

    def encode(self, texts: List[str], mode: str = "passage") -> List[List[float]]:
        if self._st_model is not None:
            try:
                prepared = self._apply_prefix(texts, mode=mode)
                embeddings = self._st_model.encode(prepared, normalize_embeddings=True)
                return [emb.tolist() for emb in embeddings]
            except Exception as e:
                logger.warning("Falha na inferência neural: %s. Utilizando fallback vetorial.", e)

        # Projeção determinística de dimensionalidade fixa 384
        import hashlib
        import math
        encoded_vectors = []
        for text in texts:
            words = re.findall(r"\w+", text.lower())
            vector = [0.0] * EMBEDDING_DIM
            for i, word in enumerate(words):
                h = int(hashlib.md5(word.encode("utf-8")).hexdigest(), 16)
                idx = h % EMBEDDING_DIM
                weight = 1.0 / math.sqrt(i + 1)
                vector[idx] += weight
            # Normalização L2
            norm = math.sqrt(sum(x * x for x in vector)) or 1.0
            vector = [x / norm for x in vector]
            encoded_vectors.append(vector)
        return encoded_vectors

    def encode_one(self, text: str, mode: str = "query") -> List[float]:
        return self.encode([text], mode=mode)[0]


class QdrantHospitalRAG:
    """Gerenciador do VectorDB Qdrant para protocolos e prontuários/exames."""

    def __init__(self, qdrant_url: str = QDRANT_URL):
        self.qdrant_url = qdrant_url
        self.client = None
        self.embedding_model = ClinicalEmbeddingModel()
        self.is_connected = False
        self._memory_backup: List[Dict[str, Any]] = []

        self._init_connection()

    def expand_clinical_query(self, query: str) -> str:
        """
        Retorna a consulta clínica de maneira direta e sem viés de termos ou hipóteses pré-fixadas,
        garantindo que a busca semântica no Qdrant seja feita de forma aberta e generalista.
        """
        return query.strip() if query else ""

    def _init_connection(self):
        """Conecta ao Qdrant e cria a coleção se necessário."""
        try:
            self.client = QdrantClient(url=self.qdrant_url, timeout=5.0)
            collections = self.client.get_collections().collections
            names = [c.name for c in collections]

            if COLLECTION_NAME not in names:
                logger.info("Criando coleção '%s' no Qdrant (dimensão %d)...", COLLECTION_NAME, EMBEDDING_DIM)
                self.client.create_collection(
                    collection_name=COLLECTION_NAME,
                    vectors_config=qmodels.VectorParams(
                        size=EMBEDDING_DIM,
                        distance=qmodels.Distance.COSINE
                    )
                )
            self.is_connected = True
            logger.info("Conexão com VectorDB Qdrant estabelecida com sucesso!")
        except Exception as e:
            self.is_connected = False
            logger.warning("Não foi possível conectar ao Qdrant em %s (%s). Ativando memória local RAG.", self.qdrant_url, e)

    def initialize_protocols(self, protocols_json_path: str):
        """Indexa todos os 100+ protocolos médicos na base vetorial."""
        if not os.path.exists(protocols_json_path):
            logger.warning("Arquivo de protocolos não encontrado em: %s", protocols_json_path)
            return

        with open(protocols_json_path, "r", encoding="utf-8") as f:
            protocols = json.load(f)

        logger.info("Indexando %d protocolos hospitalares no Qdrant...", len(protocols))
        texts_to_embed = []
        payloads = []
        ids = []

        for idx, p in enumerate(protocols):
            doc_text = (
                f"Protocolo Institucional: {p.get('protocol_id', '')} - {p.get('disease_name', '')}\n"
                f"CID-10: {p.get('icd10', '')} | Especialidade: {p.get('category', '')}\n"
                f"Sintomas Clínicos:\n{p.get('symptoms', '')}\n\n"
                f"Exames Complementares Indicados:\n{p.get('recommended_exams', '')}\n\n"
                f"Conduta e Tratamento de 1ª Linha:\n{p.get('treatment_protocol', '')}\n\n"
                f"Contraindicações e Alertas Críticos:\n{p.get('contraindications_and_warnings', '')}\n"
                f"Fonte: Diretriz do Hospital Universitário FIAP"
            )
            texts_to_embed.append(doc_text)
            payloads.append({
                "doc_type": "protocol",
                "protocol_id": p.get("protocol_id", f"PROTO-{idx}"),
                "disease_name": p.get("disease_name", ""),
                "icd10": p.get("icd10", ""),
                "category": p.get("category", ""),
                "symptoms": p.get("symptoms", ""),
                "recommended_exams": p.get("recommended_exams", ""),
                "treatment_protocol": p.get("treatment_protocol", ""),
                "contraindications_and_warnings": p.get("contraindications_and_warnings", ""),
                "content": doc_text
            })
            ids.append(idx + 1)

        vectors = self.embedding_model.encode(texts_to_embed)

        if self.is_connected and self.client:
            points = [
                qmodels.PointStruct(id=ids[i], vector=vectors[i], payload=payloads[i])
                for i in range(len(ids))
            ]
            # Inserção em lotes de 50
            batch_size = 50
            for i in range(0, len(points), batch_size):
                self.client.upsert(
                    collection_name=COLLECTION_NAME,
                    points=points[i:i + batch_size]
                )
            logger.info("Sucesso: %d protocolos inseridos e indexados no Qdrant!", len(protocols))
        else:
            # Backup em memória caso Docker/Qdrant esteja indisponível
            for i in range(len(ids)):
                self._memory_backup.append({
                    "id": ids[i],
                    "vector": vectors[i],
                    "payload": payloads[i]
                })

    def initialize_patients(self, csv_path: str):
        """
        Indexa os 1.000 pacientes sintéticos no Qdrant VectorDB como prontuários estruturados.
        Transforma o Qdrant RAG na Fonte Única da Verdade para prontuários clínicos.
        """
        if not os.path.exists(csv_path):
            logger.warning("Arquivo de pacientes não encontrado em: %s", csv_path)
            return

        import csv
        logger.info("Indexando 1.000 pacientes de %s no Qdrant RAG...", csv_path)
        texts_to_embed = []
        payloads = []
        ids = []

        start_id = 20000  # Offset seguro para ids de pacientes no Qdrant
        with open(csv_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for idx, row in enumerate(reader):
                pid = idx + 1
                cpf_raw = row.get("cpf", "").strip()
                cpf_clean = re.sub(r"\D", "", cpf_raw)
                sus_raw = row.get("cartao_sus", "").strip()
                sus_clean = re.sub(r"\D", "", sus_raw)
                nome = row.get("nome", "").strip()

                doc_text = (
                    f"Prontuário Hospitalar: {nome} (ID: {pid})\n"
                    f"CPF: {cpf_raw} | Cartão SUS: {sus_raw}\n"
                    f"Idade: {row.get('idade')} anos | Gênero: {row.get('genero')}\n"
                    f"Endereço: {row.get('endereco')} | Telefone: {row.get('telefone')}\n"
                    f"Comorbidades: {row.get('comorbidades')}\n"
                    f"Alergias Documentadas: {row.get('alergias')}\n"
                    f"Medicamentos em Uso Contínuo: {row.get('medicamentos_em_uso')}\n"
                    f"Histórico Clínico Pregresso: {row.get('historico_clinico')}"
                )
                texts_to_embed.append(doc_text)
                payloads.append({
                    "doc_type": "patient_record",
                    "id": pid,
                    "patient_id": str(pid),
                    "nome": nome,
                    "idade": int(row.get("idade", 0)) if str(row.get("idade", "")).isdigit() else row.get("idade"),
                    "genero": row.get("genero", ""),
                    "cpf": cpf_raw,
                    "cpf_clean": cpf_clean,
                    "cartao_sus": sus_raw,
                    "sus_clean": sus_clean,
                    "endereco": row.get("endereco", ""),
                    "telefone": row.get("telefone", ""),
                    "comorbidades": row.get("comorbidades", ""),
                    "alergias": row.get("alergias", ""),
                    "medicamentos_em_uso": row.get("medicamentos_em_uso", ""),
                    "historico_clinico": row.get("historico_clinico", ""),
                    "content": doc_text
                })
                ids.append(start_id + pid)

        # Inserção em lotes de 50 no Qdrant
        batch_size = 50
        for i in range(0, len(texts_to_embed), batch_size):
            batch_texts = texts_to_embed[i:i + batch_size]
            batch_vectors = self.embedding_model.encode(batch_texts)
            if self.is_connected and self.client:
                points = [
                    qmodels.PointStruct(id=ids[i + j], vector=batch_vectors[j], payload=payloads[i + j])
                    for j in range(len(batch_texts))
                ]
                self.client.upsert(collection_name=COLLECTION_NAME, points=points)
            else:
                for j in range(len(batch_texts)):
                    self._memory_backup.append({
                        "id": ids[i + j],
                        "vector": batch_vectors[j],
                        "payload": payloads[i + j]
                    })

        logger.info("Sucesso: %d prontuários de pacientes indexados no Qdrant RAG!", len(payloads))

    def search_patients(self, query: str, limit: int = 5) -> List[Dict[str, Any]]:
        """
        Busca pacientes no Qdrant RAG por CPF, Cartão SUS, Nome ou busca semântica em prontuário.
        Substitui integralmente o SQLite pelo banco vetorial Qdrant.
        """
        clean_q = query.strip()
        digits = re.sub(r"\D", "", clean_q)

        # 1. Busca por CPF ou Cartão SUS (numérico)
        if len(digits) >= 9:
            must_conds = [
                qmodels.FieldCondition(key="doc_type", match=qmodels.MatchValue(value="patient_record"))
            ]
            if len(digits) == 11:
                must_conds.append(qmodels.FieldCondition(key="cpf_clean", match=qmodels.MatchValue(value=digits)))
            else:
                must_conds.append(qmodels.FieldCondition(key="sus_clean", match=qmodels.MatchValue(value=digits)))

            if self.is_connected and self.client:
                try:
                    scroll_res, _ = self.client.scroll(
                        collection_name=COLLECTION_NAME,
                        scroll_filter=qmodels.Filter(must=must_conds),
                        limit=limit,
                        with_payload=True
                    )
                    if scroll_res:
                        return [p.payload for p in scroll_res]
                except Exception as e:
                    logger.warning("Falha ao buscar paciente por documento no Qdrant: %s", e)

        # 2. Busca por Scroll com filtro de termos no Nome (case-insensitive)
        matched = []
        seen_ids = set()
        terms = [t.lower() for t in clean_q.split() if len(t) > 1]

        if self.is_connected and self.client:
            try:
                # Scroll nos prontuários para match textual direto de nome
                scroll_res, _ = self.client.scroll(
                    collection_name=COLLECTION_NAME,
                    scroll_filter=qmodels.Filter(
                        must=[
                            qmodels.FieldCondition(key="doc_type", match=qmodels.MatchValue(value="patient_record"))
                        ]
                    ),
                    limit=1000,
                    with_payload=True
                )
                for point in scroll_res:
                    p = point.payload or {}
                    p_name = p.get("nome", "").lower()
                    p_cpf = p.get("cpf", "")
                    p_sus = p.get("cartao_sus", "")
                    # Verifica se todos os termos digitados estão presentes no nome ou documento
                    if (terms and all(t in p_name for t in terms)) or clean_q.lower() in p_name or clean_q in p_cpf or clean_q in p_sus:
                        pid = p.get("id")
                        if pid not in seen_ids:
                            seen_ids.add(pid)
                            matched.append(p)
                            if len(matched) >= limit:
                                return matched
            except Exception as e:
                logger.warning("Falha ao pesquisar pacientes no Qdrant: %s", e)

        # 3. Busca Vetorial Semântica (fallback / busca por sintomas ou contexto)
        if len(matched) < limit and self.is_connected and self.client:
            try:
                query_vec = self.embedding_model.encode_one(clean_q)
                resp = self.client.query_points(
                    collection_name=COLLECTION_NAME,
                    query=query_vec,
                    query_filter=qmodels.Filter(
                        must=[
                            qmodels.FieldCondition(key="doc_type", match=qmodels.MatchValue(value="patient_record"))
                        ]
                    ),
                    limit=limit
                )
                hits = resp.points if hasattr(resp, "points") else resp
                for h in hits:
                    p = h.payload or {}
                    pid = p.get("id")
                    if pid not in seen_ids:
                        seen_ids.add(pid)
                        matched.append(p)
                        if len(matched) >= limit:
                            break
            except Exception as e:
                logger.warning("Falha na busca vetorial de pacientes no Qdrant: %s", e)

        # Fallback memória local
        if not matched and self._memory_backup:
            for item in self._memory_backup:
                p = item.get("payload", {})
                if p.get("doc_type") == "patient_record":
                    p_name = p.get("nome", "").lower()
                    if all(t in p_name for t in terms):
                        matched.append(p)
                        if len(matched) >= limit:
                            break

        return matched

    def get_patient_by_id(self, patient_id: Any) -> Optional[Dict[str, Any]]:
        """Recupera prontuário de paciente no Qdrant RAG por ID."""
        pid_str = str(patient_id).strip()
        if self.is_connected and self.client:
            try:
                scroll_res, _ = self.client.scroll(
                    collection_name=COLLECTION_NAME,
                    scroll_filter=qmodels.Filter(
                        must=[
                            qmodels.FieldCondition(key="doc_type", match=qmodels.MatchValue(value="patient_record")),
                            qmodels.FieldCondition(key="patient_id", match=qmodels.MatchValue(value=pid_str))
                        ]
                    ),
                    limit=1,
                    with_payload=True
                )
                if scroll_res:
                    return scroll_res[0].payload
            except Exception as e:
                logger.warning("Falha ao buscar paciente por id no Qdrant: %s", e)

        # Fallback memória
        for item in self._memory_backup:
            p = item.get("payload", {})
            if p.get("doc_type") == "patient_record" and str(p.get("patient_id")) == pid_str:
                return p
        return None

    def count_patients(self) -> int:
        """Conta quantos prontuários de pacientes estão indexados no Qdrant."""
        if self.is_connected and self.client:
            try:
                scroll_res, _ = self.client.scroll(
                    collection_name=COLLECTION_NAME,
                    scroll_filter=qmodels.Filter(
                        must=[
                            qmodels.FieldCondition(key="doc_type", match=qmodels.MatchValue(value="patient_record"))
                        ]
                    ),
                    limit=2000,
                    with_payload=False
                )
                return len(scroll_res)
            except Exception:
                return 0
        return sum(1 for item in self._memory_backup if item.get("payload", {}).get("doc_type") == "patient_record")

    def initialize_allergy_knowledge(self, allergy_json_path: str):
        """Indexa a ontologia institucional de alergias e farmacovigilância no Qdrant (RAG)."""
        if not os.path.exists(allergy_json_path):
            logger.warning("Arquivo de ontologia de alergias não encontrado em: %s", allergy_json_path)
            return

        with open(allergy_json_path, "r", encoding="utf-8") as f:
            data = json.load(f)
            classes = data.get("classes", {})

        logger.info("Indexando %d diretrizes de alergia e farmacovigilância no Qdrant...", len(classes))
        texts_to_embed = []
        payloads = []
        ids = []

        start_id = 10000  # Offset seguro para ids no VectorDB
        for idx, (ckey, cinfo) in enumerate(classes.items()):
            drugs_str = ", ".join(cinfo.get("contraindicated_drugs", []))
            triggers_str = ", ".join(cinfo.get("allergy_terms", []))
            safe_alts = cinfo.get("safe_alternatives", {})
            alts_str = " | ".join(f"{k}: {v}" for k, v in safe_alts.items())

            doc_text = (
                f"Protocolo Institucional de Farmacovigilância: ALERGIA-{ckey.upper()} - {cinfo.get('name', '')}\n"
                f"Termos e Gatilhos Alérgicos: {triggers_str}\n"
                f"Medicamentos e Princípios Ativos Contraindicados (Risco Biológico):\n{drugs_str}\n\n"
                f"Esquemas Terapêuticos Alternativos Seguros:\n{alts_str}\n"
                f"Fonte: Comissão de Controle de Infecção Hospitalar e Farmacovigilância HFIAP"
            )
            texts_to_embed.append(doc_text)
            payloads.append({
                "doc_type": "protocol",
                "protocol_id": f"ALERGIA-{ckey.upper()}",
                "disease_name": f"Diretriz de Farmacovigilância: {cinfo.get('name')}",
                "icd10": "T78.4",
                "category": "Farmacovigilância e Segurança",
                "symptoms": f"Reação de hipersensibilidade ou anafilaxia a {cinfo.get('name')}",
                "recommended_exams": "Monitoramento de sinais vitais, oximetria e IgE específica se indicado.",
                "treatment_protocol": alts_str,
                "contraindications_and_warnings": f"Contraindicações Absolutas: Uso de {drugs_str} em pacientes com histórico de hipersensibilidade.",
                "content": doc_text
            })
            ids.append(start_id + idx + 1)

        vectors = self.embedding_model.encode(texts_to_embed)
        if self.is_connected and self.client:
            points = [
                qmodels.PointStruct(id=ids[i], vector=vectors[i], payload=payloads[i])
                for i in range(len(ids))
            ]
            self.client.upsert(collection_name=COLLECTION_NAME, points=points)
            logger.info("Sucesso: %d diretrizes de alergia indexadas no Qdrant!", len(classes))
        else:
            for i in range(len(ids)):
                self._memory_backup.append({
                    "id": ids[i],
                    "vector": vectors[i],
                    "payload": payloads[i]
                })

    def ingest_pdf(self, pdf_path: str, patient_id: Optional[str] = None) -> Dict[str, Any]:
        """
        Lê e indexa um arquivo PDF de laudo médico no VectorDB vinculado ao paciente.
        """
        if not os.path.exists(pdf_path):
            raise FileNotFoundError(f"Arquivo PDF não encontrado: {pdf_path}")

        reader = PdfReader(pdf_path)
        extracted_text = ""
        for page_idx, page in enumerate(reader.pages):
            text = page.extract_text() or ""
            extracted_text += f"\n--- Página {page_idx + 1} ---\n" + text

        clean_text = extracted_text.strip()
        filename = os.path.basename(pdf_path)

        # Chunking: quebra se for extenso
        chunks = []
        if len(clean_text) > 1200:
            words = clean_text.split()
            chunk_size = 200
            for i in range(0, len(words), chunk_size - 30):
                chunk = " ".join(words[i:i + chunk_size])
                chunks.append(chunk)
        else:
            chunks.append(clean_text)

        logger.info("Ingerindo PDF '%s' (%d chunks) para paciente [%s]...", filename, len(chunks), patient_id)
        vectors = self.embedding_model.encode(chunks)

        import random
        base_id = random.randint(100000, 999999)

        inserted_points = []
        for i, chunk in enumerate(chunks):
            point_id = base_id + i
            payload = {
                "doc_type": "patient_exam",
                "patient_id": patient_id or "GERAL",
                "filename": filename,
                "chunk_index": i,
                "content": chunk,
                "source": f"Laudo Diagnóstico PDF ({filename})"
            }

            if self.is_connected and self.client:
                self.client.upsert(
                    collection_name=COLLECTION_NAME,
                    points=[
                        qmodels.PointStruct(id=point_id, vector=vectors[i], payload=payload)
                    ]
                )
            else:
                self._memory_backup.append({
                    "id": point_id,
                    "vector": vectors[i],
                    "payload": payload
                })

            inserted_points.append(point_id)

        return {
            "status": "INGESTED",
            "filename": filename,
            "patient_id": patient_id,
            "chunks_count": len(chunks),
            "total_chars": len(clean_text)
        }

    def clear_patient_exams(self, patient_id: Optional[Any] = None):
        """Remove exames de pacientes do Qdrant para garantir isolamento de sessões e evitar diagnósticos precoces."""
        target_pid = str(patient_id).strip() if patient_id else None
        if self.is_connected and self.client:
            try:
                conds = [qmodels.FieldCondition(key="doc_type", match=qmodels.MatchValue(value="patient_exam"))]
                if target_pid:
                    conds.append(qmodels.FieldCondition(key="patient_id", match=qmodels.MatchValue(value=target_pid)))
                self.client.delete(
                    collection_name=COLLECTION_NAME,
                    points_selector=qmodels.Filter(must=conds)
                )
                logger.info("Exames de pacientes limpos no Qdrant (filtro: %s).", target_pid or "TODOS")
            except Exception as e:
                logger.warning("Erro ao limpar exames no Qdrant: %s", e)

        # Limpa na memória local também
        self._memory_backup = [
            item for item in self._memory_backup
            if item.get("payload", {}).get("doc_type") != "patient_exam" or
            (target_pid and str(item.get("payload", {}).get("patient_id")) != target_pid)
        ]

    def get_patient_exams(self, patient_id: Any = None, filenames: Optional[List[str]] = None, limit: int = 5) -> List[Dict[str, Any]]:
        """
        Recupera diretamente os exames/laudos indexados para um paciente específico no Qdrant.
        Garante que cada arquivo PDF seja retornado com conteúdo íntegro sem duplicações de chunks.
        """
        target_pid = str(patient_id).strip() if patient_id is not None else None
        target_fns = [f.lower().strip() for f in filenames] if filenames else []
        exams_by_file: Dict[str, Dict[str, Any]] = {}

        if self.is_connected and self.client:
            try:
                scroll_res, _ = self.client.scroll(
                    collection_name=COLLECTION_NAME,
                    scroll_filter=qmodels.Filter(
                        must=[
                            qmodels.FieldCondition(
                                key="doc_type",
                                match=qmodels.MatchValue(value="patient_exam")
                            )
                        ]
                    ),
                    limit=100,
                    with_payload=True
                )
                for point in scroll_res:
                    payload = point.payload or {}
                    doc_pid = str(payload.get("patient_id", "")).strip()
                    doc_fname = payload.get("filename", "Laudo_Clinico.pdf")

                    is_match = False
                    if target_fns:
                        is_match = any(tfn in doc_fname.lower() for tfn in target_fns)
                    elif target_pid:
                        is_match = (doc_pid == target_pid or doc_pid == "GERAL" or f"hfiap-pac-000{target_pid}" in doc_pid.lower() or f"hfiap-000{target_pid}" in doc_pid.lower())

                    if is_match:
                        chunk_text = payload.get("content", "")
                        chunk_idx = payload.get("chunk_index", 0)

                        if doc_fname not in exams_by_file:
                            exams_by_file[doc_fname] = {
                                "score": 1.0,
                                "doc_type": "patient_exam",
                                "title": doc_fname,
                                "filename": doc_fname,
                                "protocol_id": "EXAME-ANEXADO",
                                "chunks": {},
                                "source": payload.get("source", f"Laudo Diagnóstico ({doc_fname})")
                            }
                        exams_by_file[doc_fname]["chunks"][chunk_idx] = chunk_text

            except Exception as e:
                logger.warning("Falha ao recuperar exames do paciente no Qdrant: %s. Utilizando fallback.", e)

        # Fallback de busca em memória
        if not exams_by_file:
            for item in self._memory_backup:
                payload = item.get("payload", {})
                if payload.get("doc_type") == "patient_exam":
                    doc_pid = str(payload.get("patient_id", "")).strip()
                    doc_fname = payload.get("filename", "Laudo_Clinico.pdf")

                    is_match = False
                    if target_fns:
                        is_match = any(tfn in doc_fname.lower() for tfn in target_fns)
                    elif target_pid:
                        is_match = (doc_pid == target_pid or doc_pid == "GERAL" or f"hfiap-pac-000{target_pid}" in doc_pid.lower())

                    if is_match:
                        chunk_text = payload.get("content", "")
                        chunk_idx = payload.get("chunk_index", 0)

                        if doc_fname not in exams_by_file:
                            exams_by_file[doc_fname] = {
                                "score": 1.0,
                                "doc_type": "patient_exam",
                                "title": doc_fname,
                                "filename": doc_fname,
                                "protocol_id": "EXAME-ANEXADO",
                                "chunks": {},
                                "source": payload.get("source", f"Laudo Diagnóstico ({doc_fname})")
                            }
                        exams_by_file[doc_fname]["chunks"][chunk_idx] = chunk_text

        # Junta chunks ordenados para formar o laudo completo
        final_exams = []
        for fname, ex_data in exams_by_file.items():
            sorted_chunks = [ex_data["chunks"][k] for k in sorted(ex_data["chunks"].keys())]
            full_content = "\n\n".join(sorted_chunks)
            final_exams.append({
                "score": 1.0,
                "doc_type": "patient_exam",
                "title": fname,
                "filename": fname,
                "protocol_id": "EXAME-ANEXADO",
                "content": full_content,
                "source": ex_data["source"]
            })
            if len(final_exams) >= limit:
                break

        return final_exams

    def search_protocols(self, query: str, limit: int = 4, exclude_allergy: bool = False) -> List[Dict[str, Any]]:
        """
        Busca semântica híbrida (Dense Vector Multilíngue + Re-ranking Léxico IDF/BM25)
        de protocolos institucionais do HFIAP.
        O IDF dinâmico valoriza automaticamente termos clinicamente específicos e
        reduz o peso de adjetivos genéricos (ex: 'grave', 'aguda', 'crônica') sem
        nenhuma lista hardcoded de palavras-chave clínicas.
        """
        import math

        clean_query = query.strip() if query else ""
        query_vector = self.embedding_model.encode_one(clean_query)

        # Extração agnóstica de tokens significativos (removendo stopwords gramaticais PT-BR)
        raw_tokens = re.findall(r"\b[a-zA-ZÀ-ÿ]{3,}\b", clean_query.lower())
        pt_stops = {
            "paciente", "anos", "admitida", "admitido", "sala", "emergência", "emergencia", "quadro",
            "iniciada", "iniciado", "minutos", "horas", "dias", "sinais", "vitais", "ambiente",
            "forte", "fraco", "apresenta", "refere", "queixa", "historia", "histórico", "historico",
            "prontuário", "prontuario", "destaca", "para", "com", "sem", "uma", "uns", "umas",
            "este", "esta", "esse", "essa", "qual", "quais", "onde", "como", "qualquer", "mais",
            "muito", "pelo", "pela", "sobre", "entre", "quando", "mesmo", "outro", "outra", "pode",
            "ser", "dos", "das", "nos", "nas", "que", "por", "pronto", "socorro", "esforcos",
            "esforços", "progressiva", "piora", "prévio", "previo", "diária", "diaria"
        }
        # Mantém tokens únicos preservando ordem para evitar inflação por repetição
        q_tokens = list(dict.fromkeys(t for t in raw_tokens if t not in pt_stops))

        candidates = []

        if self.is_connected and self.client:
            try:
                # 1. Recupera candidatos na coleção hospitalar (avalia todos os protocolos cadastrados)
                response = self.client.query_points(
                    collection_name=COLLECTION_NAME,
                    query=query_vector,
                    query_filter=qmodels.Filter(
                        must=[
                            qmodels.FieldCondition(
                                key="doc_type",
                                match=qmodels.MatchValue(value="protocol")
                            )
                        ]
                    ),
                    limit=130
                )
                hits = response.points if hasattr(response, "points") else response
                for hit in hits:
                    payload = hit.payload or {}
                    cos_sim = float(getattr(hit, "score", 0.5))
                    candidates.append((cos_sim, payload))
            except Exception as e:
                logger.error("Erro na busca vetorial de protocolos Qdrant: %s. Utilizando fallback.", e)

        # Fallback para memória se o Qdrant não retornar candidatos
        if not candidates and self._memory_backup:
            import numpy as np
            for item in self._memory_backup:
                pay = item.get("payload", {})
                if pay.get("doc_type") == "protocol":
                    vec = item.get("vector", [])
                    if vec:
                        cos_sim = float(np.dot(query_vector, vec) / (np.linalg.norm(query_vector) * np.linalg.norm(vec)))
                    else:
                        cos_sim = 0.5
                    candidates.append((cos_sim, pay))

        # Filtra diretrizes de farmacovigilância/alergia se busca for diagnóstica principal
        if exclude_allergy:
            candidates = [
                (cs, p) for cs, p in candidates
                if not str(p.get("protocol_id", "")).startswith("ALERGIA-")
                and "farmacovigilância" not in str(p.get("category", "")).lower()
            ]

        # 2. Cálculo dinâmico de IDF sobre os protocolos candidatos
        # Termos que aparecem em muitos títulos/protocolos (ex: 'grave', 'aguda', 'síndrome')
        # recebem IDF naturalmente baixo; termos específicos (ex: 'tosse', 'calafrios', 'produtiva') recebem IDF alto.
        num_docs = max(len(candidates), 1)
        doc_freq: Dict[str, int] = {}
        name_doc_freq: Dict[str, int] = {}
        doc_token_sets = []
        for _, payload in candidates:
            sym_set = set(re.findall(r"\b[a-zA-ZÀ-ÿ]{3,}\b", payload.get("symptoms", "").lower()))
            name_set = set(re.findall(r"\b[a-zA-ZÀ-ÿ]{3,}\b", payload.get("disease_name", "").lower()))
            exam_set = set(re.findall(r"\b[a-zA-ZÀ-ÿ]{3,}\b", payload.get("recommended_exams", "").lower()))
            all_set = sym_set | name_set
            doc_token_sets.append((sym_set, name_set, exam_set))
            for tok in all_set:
                doc_freq[tok] = doc_freq.get(tok, 0) + 1
            for tok in name_set:
                name_doc_freq[tok] = name_doc_freq.get(tok, 0) + 1

        # 3. Re-ranking Híbrido (Dense Cosine Similarity Multilíngue + Overlap Léxico ponderado por IDF)
        scored_protocols = []
        for idx_c, (cos_sim, payload) in enumerate(candidates):
            sym_set, name_set, exam_set = doc_token_sets[idx_c]

            if q_tokens:
                raw_lex = 0.0
                for t in q_tokens:
                    df = doc_freq.get(t, 0)
                    idf = math.log((num_docs + 1.0) / (df + 1.0)) + 1.0
                    # Penaliza adjetivos/substantivos muito frequentes em títulos (ex: 'grave', 'aguda', 'síndrome')
                    ndf = name_doc_freq.get(t, 0)
                    name_idf = math.log((num_docs + 1.0) / (ndf + 1.0)) + 0.5
                    if t in sym_set:
                        raw_lex += 1.6 * idf
                    if t in name_set:
                        # Só dá peso forte no nome se o termo for específico no catálogo de títulos
                        weight = 1.4 if ndf <= 3 else 0.35
                        raw_lex += weight * name_idf
                    if t in exam_set and t not in sym_set and t not in name_set:
                        raw_lex += 0.4 * idf
                # Saturação estilo BM25 (independente do comprimento total da query com dados de prontuário)
                lex_score = raw_lex / (raw_lex + 8.0) if raw_lex > 0 else 0.0
            else:
                lex_score = 0.0

            # Fusão ponderada equilibrada (45% semântica multilíngue + 55% especificidade clínica IDF)
            hybrid_score = (0.45 * max(cos_sim, 0.0)) + (0.55 * lex_score)

            content_text = payload.get("content", "")
            treatment = payload.get("treatment_protocol", "")
            contra = payload.get("contraindications_and_warnings", "")
            exams_rec = payload.get("recommended_exams", "")

            if not treatment and content_text:
                m = re.search(r"Conduta e Tratamento de 1ª Linha:?\s*\n?(.*?)(?=(?:Contraindicações|Fonte:)|\Z)", content_text, re.DOTALL | re.IGNORECASE)
                if m:
                    treatment = m.group(1).strip()
            if not contra and content_text:
                m = re.search(r"Contraindicações e Alertas Críticos:?\s*\n?(.*?)(?=(?:Fonte:)|\Z)", content_text, re.DOTALL | re.IGNORECASE)
                if m:
                    contra = m.group(1).strip()
            if not exams_rec and content_text:
                m = re.search(r"Exames Complementares Indicados:?\s*\n?(.*?)(?=(?:Conduta e Tratamento)|\Z)", content_text, re.DOTALL | re.IGNORECASE)
                if m:
                    exams_rec = m.group(1).strip()

            scored_protocols.append((hybrid_score, {
                "score": round(hybrid_score, 4),
                "dense_score": round(cos_sim, 4),
                "lex_score": round(lex_score, 4),
                "doc_type": "protocol",
                "title": payload.get("disease_name", "Diretriz Clínica"),
                "disease_name": payload.get("disease_name", ""),
                "protocol_id": payload.get("protocol_id", "PROTOCOLO-HFIAP"),
                "icd10": payload.get("icd10", ""),
                "category": payload.get("category", ""),
                "symptoms": payload.get("symptoms", ""),
                "recommended_exams": exams_rec,
                "treatment_protocol": treatment,
                "contraindications_and_warnings": contra,
                "content": content_text,
                "source": f"Protocolo HFIAP ({payload.get('protocol_id', '')})"
            }))

        scored_protocols.sort(key=lambda x: x[0], reverse=True)
        return [p for _, p in scored_protocols[:limit]]

    def retrieve_allergy_guidelines(self, allergy_query: str, limit: int = 4) -> List[Dict[str, Any]]:
        """
        Recupera dinamicamente diretrizes de farmacovigilância e alergias indexadas no Qdrant RAG.
        Permite que o Guardrail Agêntico opere de forma desacoplada, consultando o VectorDB
        em vez de arquivos estáticos ou regras hardcoded no código Python.
        """
        if not allergy_query or any(neg in allergy_query.lower() for neg in ["nenhuma", "negada", "não relatada", "namc", "sem alergia"]):
            return []

        search_q = f"Diretriz de Farmacovigilância Alergia Hipersensibilidade {allergy_query}"
        candidates = self.search_protocols(search_q, limit=limit * 2)
        
        # Prioriza documentos que são diretrizes de farmacovigilância e alergia
        allergy_docs = [
            c for c in candidates
            if "ALERGIA-" in c.get("protocol_id", "") or "Farmacovigilância" in c.get("title", "") or "Farmacovigilância" in c.get("category", "")
        ]
        return allergy_docs[:limit] if allergy_docs else candidates[:limit]

    def search(self, query: str, patient_id: Optional[str] = None, limit: int = 4) -> List[Dict[str, Any]]:
        """
        Busca híbrida integrada: combina protocolos institucionais e exames do paciente sem concorrência de scores.
        """
        results = []
        # 1. Recupera exames do paciente se especificado
        if patient_id:
            exams = self.get_patient_exams(patient_id=patient_id, limit=limit)
            results.extend(exams)

        # 2. Recupera protocolos
        protocols = self.search_protocols(query=query, limit=limit)
        results.extend(protocols)

        return results

    def get_protocol_by_id(self, protocol_id: str) -> Optional[Dict[str, Any]]:
        """Recupera diretamente o protocolo pelo seu código oficial (ex: PROTOCOLO-CLIN-002)."""
        if not protocol_id:
            return None
        proto_file = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "dataset", "raw", "protocols_100_diseases.json"
        )
        if not os.path.exists(proto_file):
            proto_file = os.path.join(
                os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                "dataset", "processed", "protocols_100_diseases.json"
            )
        if os.path.exists(proto_file):
            with open(proto_file, "r", encoding="utf-8") as f:
                protos = json.load(f)
                for p in protos:
                    if p.get("protocol_id", "").strip().upper() == protocol_id.strip().upper():
                        return {
                            "title": p.get("disease_name"),
                            "protocol_id": p.get("protocol_id"),
                            "icd10": p.get("icd10"),
                            "category": p.get("category"),
                            "symptoms": p.get("symptoms"),
                            "recommended_exams": p.get("recommended_exams"),
                            "treatment_protocol": p.get("treatment_protocol"),
                            "contraindications_and_warnings": p.get("contraindications_and_warnings"),
                            "score": 1.0,
                            "source": f"Diretriz HFIAP ({p.get('protocol_id')})"
                        }
        return None


# Instância global do RAG Manager
rag_manager = QdrantHospitalRAG()


if __name__ == "__main__":
    proto_file = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "dataset", "processed", "protocols_100_diseases.json"
    )
    rag_manager.initialize_protocols(proto_file)
    print("\n🔍 Testando busca no Qdrant RAG:")
    res = rag_manager.search("conduta para dor no peito e infarto")
    for r in res[:2]:
        print(f"[{r['protocol_id']}] {r['title']} (Score: {r['score']:.4f})")
