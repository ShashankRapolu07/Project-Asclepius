import datetime
import json
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from typing import Any, Union, Optional, Sequence
from pydantic import SecretStr
import time
from langchain_core.documents import Document
from qdrant_client import QdrantClient, models
from langchain_qdrant import FastEmbedSparse
# import os
import ast
import re
import numpy as np
from qdrant_client.models import Filter, FieldCondition, MatchValue, Range
import matplotlib.pyplot as plt
import os
import tempfile
import uuid
from urllib.parse import urlparse

class IntelligentInsightNLUParser:
    """
    The IntelligentInsightNLUParser is a specialized biomedical natural language understanding (NLU) component
    for structuring user queries to drive downstream literature retrieval and synthesis.

    Purpose:
      - Analyze and parse a user's free-form natural language query related to biomedical or healthcare topics.
      - Produce a structured Python dictionary string, guiding all subsequent pipeline stages.

    Core Features:
      - Role Classification: Identifies the most appropriate user role ("research", "clinical", "strategic") 
        to tailor retrieval strategies.
      - Enhanced Queries: Decomposes or paraphrases the original query into multiple focused sub-questions 
        for broader or deeper coverage.
      - Keyword Extraction: Identifies biomedical entities, technical terms, and key concepts relevant to the query.
      - Visualization Detection: Determines if the user’s intent likely requires visual output (charts, trends, comparisons).
      - Filter Extraction: Parses explicit or strongly implied filters (e.g., publication date ranges) and resolves 
        relative dates using the current date.
      - Ambiguity Handling: For vague, short, or nonsensical queries, returns a safe default structure for system robustness.

    Key Parameters:
      - max_enhanced_queries: Maximum number of enhanced (paraphrased or decomposed) queries generated.
      - user_query_limit: Maximum allowed user query length.
      - model, temperature, max_llm_tokens: Underlying LLM settings.

    Usage:
      - Pass a user query string to `parse(user_query)`.
      - Returns a Python dictionary with role, enhanced queries, keywords, visualization need, and filters.

    All parsing logic is designed to be robust, deterministic, and to output a structurally valid dictionary 
    even in the presence of ambiguous or malformed user input.
    """

    ALLOWED_ROLES: list[str] = ["research", "clinical", "strategic"]
        
    PROMPT_TEMPLATE_STR = """You are an advanced Natural Language Understanding (NLU) Parser AI, specialized in bio-medical and healthcare queries.
Your primary function is to meticulously analyze a user's natural language query (NLQ) and transform it into a precise Python dictionary string. This dictionary will be used to improve information retrieval and downstream processing.
Adhere STRICTLY to the output format and all instructions.

**User's Natural Language Query (NLQ):**
{user_query}

**Your Task:**
Generate a Python dictionary string with the following content rules:
"user_role": "string",
"enhanced_queries": ["string"],
"keywords": ["string"],
"need_visualization": True_or_False,
"filters": dict

**Detailed Instructions for Each Key:**

1.  **`user_role` (String):**
    * Classify the user's likely role based on the NLQ. This informs retrieval strategy.
    * Allowed values:
        * "research": Queries focused on scientific discovery, mechanisms, methodologies, in-depth data, experimental results, literature reviews. (e.g., "What are the latest advancements in mRNA vaccine delivery systems?", "Compare CRISPR-Cas9 and base editing for sickle cell anemia.")
        * "clinical": Queries related to patient care, diagnostics, treatment protocols, clinical trials, drug interactions, patient outcomes. (e.g., "Standard of care for early-stage lung cancer?", "Side effects of metformin in patients with renal impairment?", "Ongoing phase 3 trials for Alzheimer's drug X?")
        * "strategic": Queries focused on market trends, competitive landscape, investment opportunities, healthcare policy, company information, broader industry impact. (e.g., "Market size for GLP-1 agonists?", "Key patent filings by BioNTech in the last year?", "Impact of new FDA regulations on gene therapy development?")
    * If the role is ambiguous or the query is too generic/nonsensical, default to "research".

2.  **`enhanced_queries` (List of Strings):**
    * Generate a list of refined natural language questions to improve information retrieval quality and coverage.
    * **Format & Purpose:**
        * Each item in the list MUST be a complete, natural-language question suitable for database querying (not a statement or search fragment).
        * These questions should paraphrase, decompose complex queries into focused sub-questions, expand on the original query to clarify implied topics, or explore relevant facets using alternate phrasings.
    * **Context:** All enhanced queries MUST remain strictly within the context of the original NLQ.
    * **Quantity:** Generate **at most {max_enhanced_queries}** such questions.
    * **Handling Simple/Nonsensical Queries:** For very short, nonsensical, or unprocessable queries, this list should be empty.

3.  **`keywords` (List of Strings):**
    * Extract bio-medical keywords, terms, entities (e.g., genes, proteins, drugs, diseases, procedures), and technical jargon from the NLQ.
    * **Handling Non-Bio-medical/Gibberish Queries:** If the query is completely non-bio-medical or gibberish, return an empty list . Also, return [] if no such specific bio-medical keywords meeting the relevance criteria are identifiable.

4.  **`need_visualization` (Boolean):**
    * Set to `True` (default value) if the NLQ implies a desire for or would benefit significantly from visual information (e.g., plots, charts, graphs).
    * Infer this if the query asks for comparisons, trends over time, distributions, correlations, statistical summaries, or data that is typically visualized. (e.g., "Show me the trend of X over the last 5 years", "Compare the efficacy of drug A vs drug B", "What is the distribution of Y in population Z?").
    * Set to `False` if the query primarily seeks textual information, facts, definitions, or if visualization is not beneficial.
    * For nonsensical queries, default to `False`.
    * must be a Python boolean object.

5.  **`filters` (Dictionary):**
    * Extract any explicit or strongly implied filters from the NLQ that can be applied during database searches.
    * Only include a filter field if the NLQ provides clear information for it. If no filters are found, output an empty dictionary.
    * **Supported Filter Fields & Structure:**
        * **`published_date`**: For filtering by publication date. The value must be a dictionary.
            * Allowed keys: "$gte" (greater than or equal to) and/or "$lte" (less than or equal to).
            * Date format MUST be ISO YYYY-MM-DD.
    * **Important:** Only extract filters if they are clearly indicated. Do not infer aggressively.
    
**Handling Ambiguous, Nonsensical, or Very Short Queries:**
If the "{user_query}" is highly ambiguous, nonsensical (e.g., ".", "sadas.dsdad34a", "-z.`1"), or too short to derive meaningful intent:
* `user_role`: "research" (as the default)
* `enhanced_queries`: If the query is truly empty, whitespace, or gibberish with no question structure, return an empty list. Otherwise, return a list containing only the original "{user_query}" itself (attempt to phrase as a question if possible but not critical for gibberish).
* `keywords`: []
* `need_visualization`: False
* `filters`: *empty dictionary
Your goal is to always return a structurally valid dictionary string, even if its content reflects low understanding of the query.

**FINAL CRITICAL INSTRUCTION:**
Your entire response MUST be ONLY the Python dictionary string as described and exemplified above. Do NOT include any introductory phrases, explanations, apologies, markdown formatting (like \`\`\`python ... \`\`\`), or any other text outside this single dictionary string. Ensure all strings within the dictionary are properly quoted (use double quotes for keys and string values as per standard Python dict string representation for `ast.literal_eval`).
"""
    
    def __init__(
        self,
        google_api_key: Optional[Union[SecretStr, str]] = None,
        model: str = "gemini-2.0-flash-lite",
        user_query_limit: int = 1024,
        max_enhanced_queries: int = 5,
        max_llm_tokens: int = 2048,
        temperature: float = 0
    ):  
        self.google_api_key = (
            SecretStr(google_api_key) if isinstance(google_api_key, str) else google_api_key
        )
        
        self.user_query_limit = user_query_limit
        self.max_enhanced_queries = max_enhanced_queries
        
        self.prompt_template = PromptTemplate(
            input_variables=["user_query", "max_enhanced_queries"],
            template=self.PROMPT_TEMPLATE_STR
        )
        self.nlu_agent = ChatGoogleGenerativeAI(
            model=model,
            api_key=self.google_api_key,
            max_tokens=max_llm_tokens,
            temperature=temperature
        )
        self.nlu_chain = self.prompt_template | self.nlu_agent | StrOutputParser()
        
    def parse(self, user_query: str) -> dict[str, Any]:
        print("[IntelligentInsightEngine] Understanding and enhancing user query...")
        
        if not user_query:
            raise ValueError("user_query must not be empty")
        elif len(user_query) > self.user_query_limit:
            raise ValueError(f"user_query length {len(user_query)} exceeds limit {self.user_query_limit}")
        
        try:
            raw_output = self._invoke_with_retry({
                "user_query": user_query,
                "max_enhanced_queries": self.max_enhanced_queries
            })
            
            cleaned = self._clean_tuple_str(raw_output)
            
            result = ast.literal_eval(cleaned)
            self._validate(result)
            result["need_visualization"] = True
            result["user_query"] = user_query
        except Exception as e:
            print(f"[NLUParser] WARNING: {type(e).__name__}: {e}")
            result = {
                "user_role": self.ALLOWED_ROLES[0],
                "user_query": user_query,
                "enhanced_queries": [],
                "keywords": [],
                "need_visualization": False,
                "filters": {}
            }
        
        print()
        print("Identified Role: ", result.get("user_role", "research"))
        print("Enhanced Queries: ", result.get("enhanced_queries", []))
        print("Keywords: ", result.get("keywords", []))
        print("Filters: ", result.get("filters", {}))
        print()
        
        return result
        
    def _invoke_with_retry(self, inputs: dict[str, Any]) -> str:
        MAX_RETRIES = 3
        BACKOFF = 5
        
        attempt = 1
        while attempt <= MAX_RETRIES:
            try:
                return self.nlu_chain.invoke(inputs)
            except Exception as e:
                if attempt >= MAX_RETRIES:
                    raise # bubble up to its parent handler
                wait = BACKOFF * 2 ** (attempt - 1)
                print(f"[NLUParser] Retry {attempt}/{MAX_RETRIES} after {wait:.1f}s — {e}")
                time.sleep(wait)
                attempt += 1
                
    def _validate(self, d: dict[str, Any]):
        mandatory = (
            "user_role",
            "enhanced_queries",
            "keywords",
            "need_visualization",
            "filters"
        )
        if not all(k in d for k in mandatory):
            raise ValueError("Missing required keys")
            
        if d["user_role"] not in self.ALLOWED_ROLES:
            raise ValueError(f"Illegal user_role '{d['user_role']}'")
            
        if not isinstance(d["enhanced_queries"], list):
            raise ValueError("enhanced_queries must be list")
        if not all(isinstance(q, str) for q in d["enhanced_queries"]):
            raise ValueError("enhanced_queries must be list[str]")
            
        if not isinstance(d["keywords"], list) or not all(isinstance(k, str) for k in d["keywords"]):
            raise ValueError("keywords must be list[str]")
            
        if not isinstance(d["need_visualization"], bool):
            raise ValueError("need_visualization must be bool")
            
        if not isinstance(d["filters"], dict):
            raise ValueError("filters must be dict")
        if "published_date" in d["filters"]:
            pd = d["filters"]["published_date"]
            if not isinstance(pd, dict):
                raise ValueError("published_date filter must be dict")
            for op in ("$gte", "$lte"):
                if op in pd and not self._is_iso_date(pd[op]):
                    raise ValueError(f"{op} must be ISO YYYY-MM-DD")
        
    @staticmethod
    def _is_iso_date(s: str) -> bool:
        try:
            datetime.date.fromisoformat(s)
            return True
        except Exception:
            return False
        
    @staticmethod
    def _clean_tuple_str(s: str) -> str:
        text = s.strip()
        if text.startswith("```"):
            lines = text.splitlines()
            if len(lines) >= 2 and lines[-1].strip() == "```":
                text = "\n".join(lines[1:-1]).strip()
        if text.startswith("`") and text.endswith("`"):
            text = text[1:-1].strip()
        text = re.sub(r'\btrue\b',  'True',  text, flags=re.IGNORECASE)
        text = re.sub(r'\bfalse\b', 'False', text, flags=re.IGNORECASE)
        text = re.sub(r'\bnull\b',  'None',  text, flags=re.IGNORECASE)
        return text
    
class IntelligentInsightRetriever:
    """
    The IntelligentInsightRetriever is a hybrid dense + sparse biomedical retrieval engine 
    designed for context-aware literature discovery in R&D, clinical, or strategic workflows.

    Purpose:
      - Retrieve the most relevant document chunks (e.g., research papers, clinical trials, news)
        from a Qdrant vector store, grounded on a parsed natural language user query.
      - Supports multi-query (“enhanced queries”) expansion and robust filter support (e.g., publication date).
      - Implements Reciprocal Rank Fusion (RRF) to merge results across multiple queries and modalities.

    Core Features:
      - Hybrid embedding-based (dense) and keyword/BM25 (sparse) retrieval with failover logic.
      - Role-adaptive: uses user role ("research", "clinical", "strategic") to prioritize document types, e.g.:
          - Research: More papers, fewer news.
          - Clinical: More trials.
          - Strategic: More news and trend signals.
      - Robust, idempotent scoring and deduplication of results across queries.

    Key Parameters:
      - retrieval_k: Number of candidates retrieved per query (before fusion/deduplication).
      - final_k: Final number of top-ranked unique chunks returned.
      - smooth_factor: RRF fusion parameter (controls rank smoothing).
      - collection_name, dense_model, sparse_model: Qdrant and embedding settings.

    Usage:
      - Pass the output of the NLU parser (parsed query dict) to `retrieve(parsed)`.
      - Returns a dict with user role, query, visualization intent, and a ranked list of top Document objects.

    All retrieval steps are designed to be robust to model/API/network failures (with retries), 
    and filters are strictly enforced at the vector store level.
    """

    ROLE_FRACTIONS: dict[str, dict[str, float]] = {
        "research": {"research_paper": 0.6, "clinical_trial": 0.3, "news_article": 0.1},
        "clinical": {"research_paper": 0.4, "clinical_trial": 0.5, "news_article": 0.1},
        "strategic": {"research_paper": 0.45, "clinical_trial": 0.15, "news_article": 0.40},
    }
        
    def __init__(
        self,
        qdrant_client: QdrantClient,
        google_api_key: Optional[Union[SecretStr, str]] = None,
        collection_name: str = "project_asclepius",
        dense_model: str = "models/embedding-001",
        sparse_model: str = "Qdrant/bm25",
        retrieval_k: int = 200,
        final_k: int = 200,
        smooth_factor: int = 60
    ):
        self.google_api_key = (
            SecretStr(google_api_key) if isinstance(google_api_key, str) else google_api_key
        )
        
        self.client = qdrant_client
        self.collection_name = collection_name
        self.retrieval_k = retrieval_k
        self.final_k = final_k
        self.smooth_factor = smooth_factor
        
        self.embedding_model = GoogleGenerativeAIEmbeddings(
            model=dense_model,
            google_api_key=self.google_api_key
        )
        self.sparse_model = FastEmbedSparse(model_name=sparse_model)
        
    def retrieve(self, parsed: dict[str, Any]) -> dict[str, Any]:
        print("[IntelligentInsightEngine] Starting retrieval process...")
        print()
        
        role = parsed["user_role"]
#         fractions = self.ROLE_FRACTIONS.get(role, self.ROLE_FRACTIONS["research"])
        query_texts: list[str] = [parsed["user_query"]] + parsed.get("enhanced_queries", [])
        qdrant_filter = self._build_filter(parsed.get("filters", {}))
        
        print(f"[Retriever] User role: {role}, Queries to process: {len(query_texts)} (1 original + {len(parsed.get("enhanced_queries", []))} enhancements)")
        
        scores: dict[tuple[str, int], tuple[Document, float]] = {}
        for q_idx, q in enumerate(query_texts, 1):
            print(f"[Retriever] Query {q_idx}/{len(query_texts)}: '{q}'")
            
            dense = self._embed_with_retry(q)
            sparse = self.sparse_model.embed_query(q)
            
            if dense is not None:
                    pts = self._hybrid_search_once(dense, sparse, self.retrieval_k, doc_type=None, base_filter=qdrant_filter)
            else:
                print("[Retriever] WARNING: Dense embedding failed, falling back to keyword-only search.")
                pts = self._sparse_search_once(sparse, self.retrieval_k, doc_type=None, base_filter=qdrant_filter)
            
            print(f"[Retriever] Successfully retrieved {len(pts)} docs for Query {q_idx}/{len(query_texts)}.")
            
            for rank, pt in enumerate(pts, 1):
                doc = self._point_to_doc(pt)
                doc_id = doc.metadata.get("source_doc_id") or ""
                chunk_idx = doc.metadata.get("chunk_index", -1)
                if not doc_id or chunk_idx < 0:
                        continue
                key = (doc_id, chunk_idx)
                rrf_score = 1.0 / (self.smooth_factor + rank)
                if key in scores:
                    doc_prev, score_prev = scores[key]
                    scores[key] = (doc, score_prev + rrf_score)
                else:
                    scores[key] = (doc, rrf_score)
                        
        print(f"[Retriever] Retrieved total of {len(scores)} unique document chunks. Performing RRF...")
        ranked = sorted(scores.values(), key=lambda x: x[1], reverse=True)
        top_docs: list[Document] = [doc for doc, _ in ranked[: self.final_k]]    
            
#             for doc_type, frac in fractions.items():
#                 k_type = max(1, math.ceil(self.retrieval_k * frac))
                
#                 if dense is not None:
#                     pts = self._hybrid_search_once(dense, sparse, k_type, doc_type, base_filter=qdrant_filter)
#                 else:
#                     print("[Retriever] WARNING: Dense embedding failed, falling back to keyword-only search.")
#                     pts = self._sparse_search_once(sparse, k_type, doc_type, base_filter=qdrant_filter)
                    
#                 print(f"[Retriever] Retrieved {len(pts)} points for type '{doc_type}'")
#                 for rank, pt in enumerate(pts, 1):
#                     doc = self._point_to_doc(pt)
#                     doc_id = doc.metadata.get("source_doc_id") or ""
#                     chunk_idx = doc.metadata.get("chunk_index", -1)
#                     if not doc_id or chunk_idx < 0:
#                         continue
#                     key = (doc_id, chunk_idx)
#                     rrf_score = 1.0 / (self.smooth_factor + rank)
#                     if key in scores:
#                         doc_prev, score_prev = scores[key]
#                         scores[key] = (doc, score_prev + rrf_score)
#                     else:
#                         scores[key] = (doc, rrf_score)
        
#         print(f"[Retriever] Aggregated {len(scores)} unique document chunks. Performing RRF...")
#         ranked = sorted(scores.values(), key=lambda x: x[1], reverse=True)
#         top_docs: list[Document] = [doc for doc, _ in ranked[: self.final_k]]
#         print(f"[Retriever] Returning top {len(top_docs)} documents based on RRF.")
        
        print()
        print(f"[IntelligentInsightEngine] Retrieved top {len(top_docs)} documents based on RRF.")
        print()
        
        return {
            "user_role": role,
            "user_query": parsed.get("user_query") or "",
            "need_visualization": parsed.get("need_visualization") or False,
            "retrieved_documents": top_docs
        }
    
    def _embed_with_retry(self, text: str) -> Optional[list[float]]:
        MAX_RETRIES = 3
        BACKOFF = 4
        
        for attempt in range(1, MAX_RETRIES + 1):
            try:
                return self.embedding_model.embed_query(text)
            except Exception as e:
                if attempt == MAX_RETRIES:
                    print(f"[Retriever] WARN: dense embed_query failed after {attempt} attempts – {e}")
                    return None
                wait = BACKOFF * 2 ** (attempt - 1)
                print(f"[Retriever] WARN: dense embed_query attempt {attempt} failed – retrying in {wait} seconds...")
                time.sleep(wait)
    
    def _hybrid_search_once(
        self,
        dense_vec: list[float],
        sparse_vec: list[float],
        limit: int,
        doc_type: Optional[str] = None,
        base_filter: Optional[Filter] = None
    ) -> list[Any]:
        return self._qdrant_search(dense_vec, sparse_vec, limit, doc_type, base_filter)
    
    def _sparse_search_once(
        self,
        sparse_vec: list[float],
        limit: int,
        doc_type: Optional[str] = None,
        base_filter: Optional[Filter] = None
    ) -> list[Any]:
        return self._qdrant_search(None, sparse_vec, limit, doc_type, base_filter)
    
    def _qdrant_search(
        self,
        dense_vec: Optional[list[float]],
        sparse_vec: list[float],
        limit: int,
        doc_type: Optional[str] = None,
        base_filter: Optional[Filter] = None
    ) -> list[Any]:
        MAX_RETRIES = 3
        BACKOFF = 4
        
        must = [
            FieldCondition(key="metadata.content_type", match=MatchValue(value="chunk"))
        ]
        if doc_type:
            must.extend([FieldCondition(key="metadata.original_doc_type", match=MatchValue(value=doc_type))])
        if base_filter:
            must.extend(base_filter.must)
        search_filter = Filter(must=must)
        
        if isinstance(sparse_vec, dict) and 'indices' in sparse_vec and 'values' in sparse_vec:
            sparse_indices = sparse_vec['indices']
            sparse_values = sparse_vec['values']
        elif hasattr(sparse_vec, 'indices') and hasattr(sparse_vec, 'values'):
            sparse_indices = sparse_vec.indices
            sparse_values = sparse_vec.values
        else:
            sparse_indices = [i for i, val in enumerate(sparse_vec) if val != 0.0]
            sparse_values = [val for val in sparse_vec if val != 0.0]
        
        for attempt in range(1, MAX_RETRIES + 1):
            try:     
                dense_results = []
                if dense_vec:
                    dense_results = self.client.query_points(
                        collection_name=self.collection_name,
                        query=dense_vec,
                        using="dense",
                        query_filter=search_filter,
                        limit=limit
                    ).points
                    
                sparse_results = self.client.query_points(
                    collection_name=self.collection_name,
                    query=models.SparseVector(
                        indices=sparse_indices,
                        values=sparse_values
                    ),
                    using="langchain-sparse",
                    query_filter=search_filter,
                    limit=limit
                ).points
                    
                seen_keys: set[tuple[str, int]] = set()
                merged: list[Any] = []
                for pt in dense_results + sparse_results:
                    md = pt.payload.get("metadata", {})
                    key = (md.get("source_doc_id"), md.get("chunk_index"))
                    if key in seen_keys:
                        continue
                    seen_keys.add(key)
                    merged.append(pt)
                
                return merged
            except Exception as e:
                if attempt >= MAX_RETRIES:
                    print(f"[Retriever] ERROR: search failed after {attempt} attempts – {e}")
                    return []
                wait = BACKOFF * 2 ** (attempt - 1)
                print(f"[Retriever] WARN: search attempt {attempt} failed – retrying in {wait} seconds... ({e})")
                time.sleep(wait)
        
    @staticmethod
    def _build_filter(filters: dict[str, Any]) -> Union[Filter, None]:
        must: list[Any] = []
        if "published_date" in filters:
            rng = filters["published_date"]
            range_params = {}
            
            if "$gte" in rng:
                v = rng["$gte"]
                if isinstance(v, str):
                    try:
                        dt = datetime.datetime.fromisoformat(v)
                        range_params["gte"] = dt.timestamp()
                    except Exception as e:
                        pass
            if "$lte" in rng:
                v = rng["$lte"]
                if isinstance(v, str):
                    try:
                        dt = datetime.datetime.fromisoformat(v)
                        range_params["lte"] = dt.timestamp()
                    except Exception as e:
                        pass
                
            if range_params:
                date_range = Range(**range_params)
                must.append(FieldCondition(key="metadata.published_date_ts", range=date_range))
        
        return Filter(must=must) if must else None
    
    @staticmethod
    def _point_to_doc(pt) -> Document:
        payload = pt.payload
        return Document(page_content=payload["page_content"], metadata=payload["metadata"])
    
class VisualizationTools:
    """
    A collection of static methods for generating common biomedical data visualizations using matplotlib.

    Provides simple interfaces for creating standard plots—line charts, bar charts, scatter plots, histograms, pie charts, and heatmaps—given preprocessed data arrays and labels.

    Each method returns a matplotlib Figure object for further manipulation, saving, or embedding elsewhere.

    Methods raise ValueError for invalid or inconsistent input data.

    Example:
        >>> viz = VisualizationTools()
        >>> fig = viz.line_chart(
        ...     x=["Jan", "Feb", "Mar"], 
        ...     y=[10, 15, 8], 
        ...     title="Patients Over Time",
        ...     x_label="Month",
        ...     y_label="Count"
        ... )
        >>> fig.savefig("patients_line.png")
    """

    def line_chart(
        self,
        x: Sequence[Union[int, float, str]],
        y: Sequence[Union[int, float]],
        title: str,
        x_label: str,
        y_label: str
    ) -> str:
        if len(x) != len(y) or not x:
            raise ValueError("line_chart: x and y must be same-length, non-empty lists")
        
        fig, ax = plt.subplots()
        ax.plot(x, y)
        ax.set_title(title)
        ax.set_xlabel(x_label)
        ax.set_ylabel(y_label)

        for lbl in ax.get_xticklabels():
            lbl.set_rotation(45)
            lbl.set_ha("right")

        return fig
    
    def bar_chart(
        self,
        x: Sequence[str],
        y: Sequence[Union[int, float]],
        title: str,
        x_label: str,
        y_label: str
    ) -> str:
        if len(x) != len(y) or not x:
            raise ValueError("bar_chart: x and y must be same-length, non-empty lists")
        
        fig, ax = plt.subplots()
        ax.bar(x, y)
        ax.set_title(title)
        ax.set_xlabel(x_label)
        ax.set_ylabel(y_label)

        for lbl in ax.get_xticklabels():
            lbl.set_rotation(45)
            lbl.set_ha("right")

        return fig
    
    def scatter_plot(
        self,
        x: Sequence[Union[int, float]],
        y: Sequence[Union[int, float]],
        title: str,
        x_label: str,
        y_label: str
    ) -> str:
        if len(x) != len(y) or not x:
            raise ValueError("scatter_plot: x and y must be same-length, non-empty lists")
            
        fig, ax = plt.subplots()
        ax.scatter(x, y)
        ax.set_title(title)
        ax.set_xlabel(x_label)
        ax.set_ylabel(y_label)

        for lbl in ax.get_xticklabels():
            lbl.set_rotation(45)
            lbl.set_ha("right")

        return fig
    
    def histogram(
        self,
        values: Sequence[Union[int, float]],
        bins: int,
        title: str,
        x_label: str,
        y_label: str
    ) -> str:
        if not values:
            raise ValueError("histogram: values must be non-empty")
        
        fig, ax = plt.subplots()
        ax.hist(values, bins=bins)
        ax.set_title(title)
        ax.set_xlabel(x_label)
        ax.set_ylabel(y_label)

        return fig
    
    def pie_chart(
        self,
        labels: Sequence[str],
        sizes: Sequence[Union[int, float]],
        title: str
    ) -> str:
        if len(labels) != len(sizes) or not labels:
            raise ValueError("pie_chart: labels and sizes must match and be non-empty")
            
        fig, ax = plt.subplots()
        ax.pie(sizes, labels=labels, autopct="%1.1f%%")
        ax.set_title(title)

        return fig
    
    def heatmap(
        self,
        matrix: list[list[Union[int, float]]],
        x_labels: Sequence[str],
        y_labels: Sequence[str],
        title: str
    ) -> str:
        mat = np.asarray(matrix, dtype=float)
        
        if mat.ndim != 2 or mat.shape[0] != len(y_labels) or mat.shape[1] != len(x_labels):
            raise ValueError("heatmap: matrix shape must match label dimensions")
            
        fig, ax = plt.subplots()
        im = ax.imshow(mat, aspect="auto")
        ax.set_xticks(np.arange(len(x_labels)), x_labels, rotation=45, ha="right")
        ax.set_yticks(np.arange(len(y_labels)), y_labels)
        ax.set_title(title)
        fig.colorbar(im, ax=ax)

        return fig
    
    def _tmp_png_filename(self) -> str:
        return os.path.join(
            tempfile.gettempdir(), f"insight_plot_{uuid.uuid4().hex[:16]}.png"
        )

class IntelligentInsightVisualizer:
    """
    The IntelligentInsightVisualizer is a biomedical data visualization orchestrator designed 
    to generate contextually relevant, role-specific plot specifications and images to enhance 
    user insights during literature exploration.

    Purpose:
      - Analyze the user’s parsed query, role, and the retrieved document context to identify 
        and specify up to `max_plots` essential visualizations.
      - Synthesize all data for plots *exclusively* from the retrieved context—no external data 
        or hallucination permitted.
      - Output plot specification dictionaries (and generated figures) in a consistent, 
        machine-readable format for downstream consumption.

    Core Features:
      - Supports a diverse set of biomedical-relevant plot types (line, bar, scatter, histogram, 
        pie, heatmap), each with strict data and citation requirements.
      - Role-aware plot selection:
          - Research: Data trends, experimental comparisons, mechanistic plots.
          - Clinical: Outcome comparisons, epidemiological trends, clinical trial results.
          - Strategic: Market, publication, or technology adoption trends.
      - Each plot’s data is fully attributable—specifies supporting context chunk citations.
      - Robust input validation for every plot specification, ensuring all generated plots are 
        meaningful and reproducible.
      - Returns both plot specifications and ready-to-render figure objects.

    Key Parameters:
      - max_plots: Maximum number of plot specifications to generate per query.
      - max_context_chars, max_chunk_chars: Controls for context truncation.
      - model, llm_max_tokens, temperature: LLM prompt and generation settings.

    Usage:
      - Pass the output of the Retriever (parsed, with documents) to `visualize(retrieval_output)`.
      - Returns an updated dictionary containing plot specifications, figures, citation mapping, 
        and a stringified version of the plotted context.

    All visualization generation is robust to missing or malformed data, and only contextually 
    valid, evidence-grounded plots are produced.
    """

    PLOT_TOOL_DESCRIPTIONS = """Below are the available plot types and the exact data keys they require in the JSON specification you generate. Ensure all data provided matches the expected types (e.g., string, number, list of strings, list of numbers, 2D list of numbers for matrix) and that list lengths are consistent where required.

- `plot_type: "line_chart"`: For showing trends over a continuous range or time.
  - `title` (string): Plot title.
  - `x` (list of strings, numbers, or floats): Data for the x-axis (e.g., dates, time points, categories).
  - `y` (list of numbers or floats): Data for the y-axis. Must be the same length as `x` and non-empty.
  - `x_label` (string): Label for the x-axis.
  - `y_label` (string): Label for the y-axis.
  - `citations` (list of integers): List of supporting citation indices from the context (positive integers only).

- `plot_type: "bar_chart"`: For comparing values across discrete categories.
  - `title` (string): Plot title.
  - `x` (list of strings): Categories for the x-axis.
  - `y` (list of numbers or floats): Values for the y-axis corresponding to each category. Must be the same length as `x` and non-empty.
  - `x_label` (string): Label for the x-axis.
  - `y_label` (string): Label for the y-axis.
  - `citations` (list of integers): List of supporting citation indices from the context.

- `plot_type: "scatter_plot"`: For showing relationships or correlations between two numerical variables.
  - `title` (string): Plot title.
  - `x` (list of numbers or floats): Data for the x-axis.
  - `y` (list of numbers or floats): Data for the y-axis. Must be the same length as `x` and non-empty.
  - `x_label` (string): Label for the x-axis.
  - `y_label` (string): Label for the y-axis.
  - `citations` (list of integers): List of supporting citation indices from the context.

- `plot_type: "histogram"`: For showing the distribution of a numerical dataset.
  - `title` (string): Plot title.
  - `values` (list of numbers or floats): The dataset for which to plot the distribution. Must be non-empty.
  - `bins` (integer): Number of bins for the histogram (must be a positive integer).
  - `x_label` (string): Label for the x-axis (often the name of the variable in `values`).
  - `y_label` (string): Label for the y-axis (usually "Frequency" or "Count").
  - `citations` (list of integers): List of supporting citation indices from the context.

- `plot_type: "pie_chart"`: For showing proportions of a whole.
  - `title` (string): Plot title.
  - `labels` (list of strings): Labels for each slice of the pie.
  - `sizes` (list of numbers or floats): The relative size or value of each slice. Must be the same length as `labels` and non-empty.
  - `citations` (list of integers): List of supporting citation indices from the context.

- `plot_type: "heatmap"`: For visualizing magnitude of a phenomenon in 2D (matrix format).
  - `title` (string): Plot title.
  - `matrix` (list of lists of numbers or floats): The 2D data matrix.
  - `x_labels` (list of strings): Labels for the x-axis (columns). Length must match the number of columns in the `matrix`.
  - `y_labels` (list of strings): Labels for the y-axis (rows). Length must match the number of rows in the `matrix`.
  - `citations` (list of integers): List of supporting citation indices from the context.
"""
    
    PROMPT_TEMPLATE_STR = """You are an expert Bio-Medical Data Synthesis and Visualization AI.
Your primary mission is to analyze the user's query, their role, the current date, and the provided retrieved context to identify and specify at most {max_plots} essential and insightful visualizations that directly help answer the user's query in a role-appropriate manner.
You must synthesize or extract all necessary data for these plots *exclusively* from the **Retrieved Context**. All synthesized data must be attributable to specific context chunks via citations.
Your output MUST be a JSON string representing a list of plot specification dictionaries. Adhere meticulously to all instructions regarding data grounding, citation, and output format.

**User Query:**
`{user_query}`

**User Role:** `{user_role}` *(This role (e.g., "research", "clinical", "strategic") should significantly influence your choice and prioritization of plots.)*

**Retrieved Context (string representation of a Python list of tuples: `(content_chunk, publication_date, citation_index)`):**
`CONTEXT_DATA = {retrieved_context_str}`
*(Note: In the `CONTEXT_DATA` above, each tuple represents a data snippet. The first element is the text content, the second is its publication date (string, may be empty), and the third is an integer citation index. A `citation_index` of -1 means the chunk is unciteable.)*

**Available Plotting Tools and Their Data Requirements:**
{plot_tool_descriptions}
---

**CRITICAL INSTRUCTIONS FOR GENERATING PLOT SPECIFICATIONS:**

1.  **Understand the Goal:** Your primary goal is "Intelligent Insight Delivery" tailored to the user. This means you must *think*, analyze the `CONTEXT_DATA` in relation to the `user_query`, `user_role`, and decide which (if any) visualizations would provide the most valuable insights. Do not just plot raw data if a synthesized view is more informative (e.g., trends, comparisons, distributions).

2.  **Plot Selection & Quantity (Guided by User Role & Current Date):**
    * Based on the `user_query`, the `{user_role}`, and available `CONTEXT_DATA`, determine if visualizations are necessary and what type of plots would be most effective and insightful.
    * **Prioritize plots relevant to the `{user_role}`:**
        * For `"research"`: Focus on plots that reveal underlying data patterns, experimental results, scientific mechanisms, detailed comparisons, or synthesize evidence from multiple studies.
        * For `"clinical"`: Prioritize plots illustrating treatment efficacy, patient outcomes, diagnostic performance, epidemiological trends, or comparisons relevant to clinical decision-making.
        * For `"strategic"`: Favor plots showing market trends, research activity over time (e.g., publication velocity), adoption rates, competitive landscapes, or the broader impact of innovations.
    * Generate specifications for **at most {max_plots}** distinct plots. If no plots are relevant or possible from the context, output an empty list `[]`.
    * Consider complex plots (time-series, comparative, distributions, heatmaps of correlations if data allows) if they provide significant insight for the query and user role.

3.  **Data Synthesis and Grounding (MANDATORY):**
    * All data (`x`, `y`, `values`, `sizes`, `matrix`, etc.) for each plot specification MUST be derived or synthesized *exclusively* from the `content_chunk` and `publication_date` elements within the **`CONTEXT_DATA`**.
    * You can perform calculations (e.g., counting occurrences, grouping by date as a reference, extracting numerical values, calculating frequencies/percentages) on the context to create NEW data for plotting, but this new data must be entirely traceable to the provided context.
    * **NO HALLUCINATION:** Never invent data or use external knowledge. The authenticity of the data is paramount.
    * Utilize `publication_date` (from context)` for relevant time-series analysis or chronological ordering.

4.  **Citations (MANDATORY):**
    * For each plot specification, you MUST provide a `citations` field. This field must be a list of *integer* `citation_index` values that directly support or were used to derive the data presented in that plot.
    * These indices refer to the 3rd element of the tuples in the `CONTEXT_DATA`.
    * **Crucially, only include positive integer `citation_index` values in your output `citations` list.** Do NOT include `-1` (which indicates an unciteable source chunk) in this list.
    * If data is synthesized from multiple chunks, include all relevant, citable indices.

5.  **Plot Specification Details (For Each Plot):**
    * `plot_type` (string): Must be one of the types listed in **Available Plotting Tools**.
    * `title` (string): A clear, descriptive title for the plot that explains what it shows in relation to the user query and user role.
    * **Data Fields** (e.g., `x`, `y`, `values`, `bins`, `labels`, `sizes`, `matrix`, `x_labels`, `y_labels`): Provide the specific data required by the chosen `plot_type`, ensuring types match the tool descriptions.
    * `x_label` (string): Label for the x-axis.
    * `y_label` (string): Label for the y-axis.

6.  **Output Format (MANDATORY):**
    * Your entire response MUST be a single JSON string representing a list of plot specification dictionaries.
    * Each dictionary in the list must conform to the structure outlined in Instruction 5 and the **Available Plotting Tools**.
    * Ensure the JSON is perfectly valid.

**Example of a PERFECT Plot Specification Dictionary (within the output list):**
```json
{{
    "plot_type": "line_chart",
    "title": "Trend of Keyword 'XYZ' Mentions Over Time (Relevant to Strategic Analysis)",
    "x": ["2023-01", "2023-02", "2023-03", "2023-04"],
    "y": [5, 8, 12, 10],
    "x_label": "Month-Year",
    "y_label": "Number of Mentions",
    "citations": [1, 3, 5, 8, 12]
}}
```

*(The final output will be a list of such dictionaries, e.g., `[ {{"plot_type": "line_chart", ...plot1_details...}}, {{"plot_type": "bar_chart", ...plot2_details...}} ]` or `[]` if no plots are suitable.)*

**FINAL CRITICAL INSTRUCTION:**
Your entire response MUST be ONLY the valid JSON string as described. Do NOT include any introductory phrases, explanations, apologies, markdown formatting (like \`\`\`json ... \`\`\`), or any other text outside this JSON list string. If no plots are generated, output an empty JSON list `[]`.
---

Begin generating plot specifications now:
"""
    
    def __init__(
        self,
        google_api_key: Optional[Union[str, SecretStr]] = None,
        model: str = "gemini-2.0-flash",
        max_plots: int = 3,
        max_context_chars: int = 300000,
        max_chunk_chars: int = 2000,
        llm_max_tokens: int = 4096,
        temperature: float = 0
    ):
        self.google_api_key = (
            SecretStr(google_api_key) if isinstance(google_api_key, str) else google_api_key
        )
        
        self.max_plots = max_plots
        self.max_context_chars = max_context_chars
        self.max_chunk_chars = max_chunk_chars
        
        self.visualization_tools = VisualizationTools()
        self.PLOT_FUNC_MAP = {
            "line_chart": self.visualization_tools.line_chart,
            "bar_chart": self.visualization_tools.bar_chart,
            "scatter_plot": self.visualization_tools.scatter_plot,
            "histogram": self.visualization_tools.histogram,
            "pie_chart": self.visualization_tools.pie_chart,
            "heatmap": self.visualization_tools.heatmap
        }
        
        self.prompt_template = PromptTemplate(
            input_variables=[
                "user_role",
                "user_query",
                "retrieved_context_str",
                "max_plots",
                "plot_tool_descriptions"
            ],
            template=self.PROMPT_TEMPLATE_STR
        )
        self.visualization_agent = ChatGoogleGenerativeAI(
            model=model,
            api_key=self.google_api_key,
            max_tokens=llm_max_tokens,
            temperature=temperature
        )
        self.visualization_chain = self.prompt_template | self.visualization_agent | StrOutputParser()
        
    def visualize(self, retrieval_output: dict[str, Any]) -> dict[str, Any]:
        print("[Visualizer] Generating visual insights if possible...")
        
        docs: list[Document] = retrieval_output.get("retrieved_documents") or []
        if not docs:
            retrieval_output["plots"] = []
            retrieval_output["retrieved_context_str"] = ""
            retrieval_output["citations_map"] = {}
            return retrieval_output
        
        context_items: list[str] = []
        id_map: dict[str, int] = {}
        citations_map: dict[int, str] = {}
        context_limit = self.max_context_chars
        chunk_limit = self.max_chunk_chars
        char_count = 0
        
        for d in docs:
            chunk = (d.metadata["original_content"] or d.page_content or "").strip()[: chunk_limit]
            url = d.metadata.get("url") or ""
            publication_date = d.metadata.get("published_date") or ""
            doc_id = d.metadata.get("source_doc_id") or ""
            
            is_sane_url = self._is_sane_url(url)

            if is_sane_url and doc_id and doc_id not in id_map:
                id_map[doc_id] = len(id_map)
                
            citation_idx = id_map.get(doc_id, -1)
            if citation_idx != -1:
                citations_map[citation_idx] = url
            
            tup = (chunk, publication_date, citation_idx)
            item_str = repr(tup) + ", "
            
            if char_count + len(item_str) > context_limit:
#                 print("[Visualizer] Context length limit reached; truncating further context entries.")
                break
            
            context_items.append(item_str)
            char_count += len(item_str)
            
        retrieved_context_str = '[' + "".join(context_items).rstrip(", ") + "]"
        
        if not retrieval_output.get("need_visualization", False):
            print("[Visualizer] Visualization not required for this retrieval output.")
            retrieval_output["retrieved_context_str"] = retrieved_context_str
            retrieval_output["plots"] = []
            retrieval_output["citations_map"] = citations_map
            return retrieval_output
        
        plots_spec: list[dict[str, Any]] = self._invoke_agent_with_retry(
            retrieval_output["user_role"],
            retrieval_output["user_query"],
            retrieved_context_str,
            self.max_plots,
            self.PLOT_TOOL_DESCRIPTIONS
        )
            
        final_plots: list[dict[str, Any]] = []
        for spec in plots_spec[: self.max_plots]:
            try:
                self._validate_spec(spec)
                plot_dict = self._execute_spec(spec, citations_map)
                final_plots.append(plot_dict)
            except Exception as e:
                print(f"[Visualizer] Skipping invalid spec: {e}")
                
        print(f"[Visualizer] Synthesized {len(final_plots)} plots.")
                
        retrieval_output["plots"] = final_plots
        retrieval_output["retrieved_context_str"] = retrieved_context_str
        retrieval_output["citations_map"] = citations_map
        return retrieval_output
            
    def _invoke_agent_with_retry(
        self,
        user_role: str,
        user_query: str,
        context_str: str,
        max_plots: int,
        plot_tool_descriptions: str
    ) -> list[dict[str, Any]]:
        MAX_RETRIES = 3
        BACKOFF = 4
        
        for attempt in range(1, MAX_RETRIES + 1):
            try:
                raw = self.visualization_chain.invoke({
                    "user_role": user_role,
                    "user_query": user_query,
                    "retrieved_context_str": context_str,
                    "max_plots": max_plots,
                    "plot_tool_descriptions": plot_tool_descriptions
                }).strip()
                cleaned = self._clean_json_str(raw)
                return json.loads(cleaned) if cleaned else []
            except Exception as e:
                if attempt >= MAX_RETRIES:
                    print(f"[Visualizer] Agent failed after {attempt} tries – {e}")
                    return []
                wait = BACKOFF * 2 ** (attempt - 1)
                print(f"[Visualizer] Attempt {attempt} failed: {e}. Retrying in {wait} seconds... ({e})")
                time.sleep(wait)
         
    def _validate_spec(self, plot_spec):
        plot_type = plot_spec.get("plot_type")
        if not self.PLOT_FUNC_MAP.get(plot_type, None):
            raise ValueError(f"Invalid plot_type '{plot_type}'")
            
        citations = plot_spec.get("citations")
        if citations is None or not isinstance(citations, list) or not all(isinstance(i, int) for i in citations):
            raise ValueError("'citations' must be a list of integers")
            
        if plot_type in ("line_chart", "bar_chart", "scatter_plot"):
            x = plot_spec.get("x")
            y = plot_spec.get("y")
            if not isinstance(x, list) or not isinstance(y, list) or len(x) != len(y) or len(x) == 0:
                raise ValueError(f"'{plot_type}' requires non-empty 'x' and 'y' lists of equal length")
        elif plot_type == "histogram":
            values = plot_spec.get("values")
            bins = plot_spec.get("bins")
            if not isinstance(values, list) or len(values) == 0:
                raise ValueError("'histogram' requires non-empty 'values' list")
            if not isinstance(bins, int) or bins <= 0:
                raise ValueError("'histogram' requires a positive integer 'bins'")
        elif plot_type == "pie_chart":
            labels = plot_spec.get("labels")
            sizes = plot_spec.get("sizes")
            if not isinstance(labels, list) or not isinstance(sizes, list) or len(labels) != len(sizes) or len(labels) == 0:
                raise ValueError("'pie_chart' requires non-empty 'labels' and 'sizes' lists of equal length")
        elif plot_type == "heatmap":
            matrix = plot_spec.get("matrix")
            x_labels = plot_spec.get("x_labels")
            y_labels = plot_spec.get("y_labels")
            if not isinstance(matrix, list) or not all(isinstance(row, list) for row in matrix):
                raise ValueError("'heatmap' requires 'matrix' as a 2D list of numbers")
            if len(matrix) == 0 or len(matrix) != len(y_labels) or any(len(row) != len(x_labels) for row in matrix):
                raise ValueError("'heatmap' matrix dimensions must match length of 'x_labels' and 'y_labels'")
        else:
            raise ValueError(f"Unhandled plot_type '{plot_type}' in validation")
            
        for label_field in ("title", "x_label", "y_label", "x_labels", "y_labels"):
            if label_field in plot_spec and plot_spec[label_field] is not None and not isinstance(plot_spec[label_field], str):
                raise ValueError(f"'{label_field}' must be a string if provided")
        
    def _execute_spec(self, plot_spec: dict[str, Any], citations_map: dict[int, str]) -> dict[str, Any]:
        plot_type = plot_spec.get("plot_type")
        citations_in = plot_spec.get("citations")
        citations_out = [citations_map[idx] for idx in citations_in if idx in citations_map]
        
        func = self.PLOT_FUNC_MAP[plot_type]
        try:
            if plot_type in ("line_chart", "bar_chart", "scatter_plot"):
                fig = func(
                    plot_spec["x"],
                    plot_spec["y"],
                    plot_spec.get("title") or "",
                    plot_spec.get("x_label") or "",
                    plot_spec.get("y_label") or ""
                )
            elif plot_type == "histogram":
                fig = func(
                    plot_spec["values"],
                    plot_spec["bins"],
                    plot_spec.get("title") or "",
                    plot_spec.get("x_label") or "",
                    plot_spec.get("y_label") or ""
                )
            elif plot_type == "pie_chart":
                fig = func(
                    plot_spec["labels"],
                    plot_spec["sizes"],
                    plot_spec.get("title") or ""
                )
            elif plot_type == "heatmap":
                fig = func(
                    plot_spec["matrix"],
                    plot_spec["x_labels"],
                    plot_spec["y_labels"],
                    plot_spec["title"]
                )
            else:
                raise ValueError("unreachable")
        except KeyError as ke:
            raise ValueError(f"Missing field {ke} for {plot_type}")
        except Exception as e:
            raise ValueError(f"Matplotlib error: {e}")
            
        spec_clean = dict(plot_spec)
        spec_clean["figure"] = fig
#         spec_clean["image_path"] = image_path
        spec_clean["citations"] = citations_out
        return spec_clean
    
    def _is_sane_url(self, url: str) -> bool:
        if not url or not isinstance(url, str):
            return False
        
        try:
            parsed = urlparse(url)
            if parsed.scheme not in ("http", "https"):
                return False
            if not parsed.netloc:
                return False
            if re.search(r"[\s]", url):
                return False
            return True
        except Exception:
            return False
        
    @staticmethod
    def _clean_json_str(raw: str) -> str:
        text = raw.strip()
        if text.startswith("```"):
            lines = text.splitlines()
            if lines[-1].strip().startswith("```"):
                text = "\n".join(lines[1:-1]).strip()
        if text.startswith("`") and text.endswith("`"):
            text = text[1:-1].strip()
        return text
    
class IntelligentInsightGenerator:
    """
    The IntelligentInsightGenerator is a biomedical narrative synthesis engine engineered
    to deliver high-impact, role-tailored insight summaries grounded strictly in contextually
    retrieved literature and visual evidence.

    Purpose:
      - Synthesize a comprehensive, structured Markdown narrative that directly answers
        the user’s query, leveraging both extracted document context and generated plot data.
      - Ensure every assertion is evidence-based: all facts, claims, and interpretations are
        strictly tied to provided context chunks or plots—no hallucination or external knowledge.
      - Produce insight strings that are richly cited, logically organized, and precisely adapted
        to research, clinical, or strategic user roles.

    Core Features:
      - Deep context synthesis: goes far beyond summarization, connecting evidence from
        multiple sources to identify trends, gaps, and nuanced patterns.
      - Integrated plot interpretation: references and interprets role-relevant plots
        alongside textual evidence, enhancing depth and clarity of the insight.
      - Enforces meticulous, in-line citation with chunk index mapping (e.g., <<3>>),
        ensuring transparent and auditable traceability for every insight.
      - Markdown output is strictly formatted, word-limited, and tuned to user role,
        ensuring clarity and actionable value for domain experts.
      - All output is sanitized to remove references to unciteable data.

    Key Parameters:
      - max_words: Word limit for synthesized insight.
      - model, llm_max_tokens, temperature: LLM prompt and generation controls.

    Usage:
      - Pass the output of the Visualizer (including context, plots, and citations) to `generate(viz_output)`.
      - Returns a dict containing the final insight markdown, plots, and context-to-citation mapping.

    All synthesis steps are robust to context or plot serialization issues, and every generated
    insight is fully grounded, role-appropriate, and reproducible—making this engine an essential
    component for reliable biomedical literature intelligence.
    """

    CITE_RE = re.compile(r"<<(\d+)>>")
    
    PROMPT_TEMPLATE_STR = """You are an expert Bio-Medical Insight Synthesis AI.
Your mission is to generate a comprehensive, deeply insightful, and impeccably structured narrative that directly answers the **User Query**. This narrative must be tailored to the **User Role** and synthesize information *exclusively* from the **Retrieved Contextual Data** and the provided **Plot Specifications**.
Your response must be a rich tapestry of information, demonstrating sophisticated understanding and analytical skill. Authenticity through meticulous citation is paramount. The insights should be so compelling and well-supported that they astound the user with their reliability and depth.

**User Query:**
`{user_query}`

**User Role:** `{user_role}` *(This role (e.g., "research", "clinical", "strategic") must significantly shape the focus, depth, and language of your generated insights.)*

**Current Date:** `{current_date}` *(Use this as a reference for the timeliness of information and for contextualizing trends relative to this date.)*

**Retrieved Contextual Data (string representation of a Python list of tuples: `(content_chunk, publication_date, citation_index)`):**
`CONTEXT_DATA = {retrieved_context_str}`
*(Note: In `CONTEXT_DATA`, each tuple is `(content_chunk, publication_date_str, citation_index_int)`. A `citation_index` of -1 means the chunk is unciteable by you with a `<<N>>` marker.)*

**Plot Specifications (JSON string representing a list of plot objects generated by the Visualization Agent):**
`PLOTS_DATA = {plots_json_str}`
*(Note: `PLOTS_DATA` is a JSON string. If not empty, it's a list of plot objects, each with keys like "plot_type", "title", "x", "y", and "citations" (which are citation indexes for plot data, distinct from the `<<N>>` context citations you will use), etc. You should refer to plots by their "title" (e.g., "As shown in the plot titled 'Growth of Biomarker Publications'...") or sequentially (e.g., "Plot 1: [Title]"). Interpret the plot's data to enrich your insights.)*
---

**CRITICAL INSTRUCTIONS FOR GENERATING TEXT INSIGHTS:**

1.  **Primary Goal - Deep Synthesis & "Intelligent Insight Delivery":**
    * Go far beyond mere summarization. Your task is to *synthesize* information from diverse `content_chunk` values and the provided `PLOTS_DATA`.
    * Identify patterns, draw connections, highlight discrepancies or convergences, and extract nuanced meanings relevant to the `user_query` and `user_role`.
    * The insights should demonstrate a sophisticated understanding of the bio-medical domain as reflected in the provided materials.

2.  **Strict Grounding & Authenticity (MANDATORY):**
    * **ALL factual statements and assertions in your response MUST be derived *exclusively* from the `CONTEXT_DATA` or the information presented in `PLOTS_DATA` (title, data).**
    * **NO HALLUCINATION / NO EXTERNAL KNOWLEDGE:** Do not invent information, speculate beyond what the data supports, or use any knowledge outside of the provided inputs. Authenticity is absolutely crucial.

3.  **Citation Requirements (MANDATORY):**
    * Any statement or piece of information you derive from a `content_chunk` in `CONTEXT_DATA` MUST be meticulously cited using the format `<<N>>`, where `N` is the positive integer `citation_index` (the 3rd element) from the corresponding tuple in `CONTEXT_DATA`.
    * If a `citation_index` is `-1`, that chunk is unciteable; DO NOT generate a `<<N>>` marker for it.
    * If a synthesized insight draws from multiple citable chunks, cite all relevant indices (e.g., "This trend is supported by several studies <<1, 5, 12>>.").
    * Citations for plot interpretations should reference the plot (e.g., "as seen in Plot 1: 'X Trend'") and, if the plot's underlying data or a direct interpretation comes from specific context chunks, cite those context chunks too.

4.  **Integration of Plot Information:**
    * If `PLOTS_DATA` is provided (i.e., not an empty list `[]`), you MUST intelligently weave insights from these visualizations into your narrative.
    * Refer to plots by their `title` or sequentially (e.g., "The bar chart titled 'Comparative Efficacy' (Plot 1) clearly illustrates...").
    * Explain what the plots show and what conclusions can be drawn from them in the context of the `user_query` and other textual evidence. Use the plot `caption` to guide your interpretation.

5.  **Tailoring to User Role & Query:**
    * **`user_role: "research"`**: Focus on detailed scientific findings, methodologies, data interpretation, nuances in research, gaps in knowledge, and potential future research directions based on the context.
    * **`user_role: "clinical"`**: Emphasize clinical relevance, diagnostic/therapeutic implications, patient outcomes, comparative effectiveness, and evidence supporting clinical practice, as found in the data.
    * **`user_role: "strategic"`**: Highlight market trends, competitive insights (if derivable from context), innovation trajectories, potential for disruption, and broader impact on healthcare or the bio-tech industry, using the `{current_date}` to frame recency and trends.

6.  **Organization, Structure & Tone (MANDATORY for "Baffling" Quality):**
    * The insights MUST be "neatly organized" and "beautifully framed." The structure should be logical, engaging, and easy for the user to digest.
    * **Use Markdown for clear structuring** depending on the `user_role` and `user_query`.
    * Maintain a professional, analytical, and insightful tone appropriate for the target `user_role`.

7.  **Word Count:**
    * The *entire* generated text insights (including all Markdown elements) MUST NOT exceed **{max_words} words**.

8.  **Output:**
    * Your response MUST be ONLY the Markdown formatted text insight string.
    * Do NOT include any preamble, apologies, or any text outside of this string. If no meaningful insights can be drawn (e.g., context is entirely irrelevant or empty), provide a polite message stating that, still within the word limit.

---
Begin generating the Text-Based Insights now, adhering strictly to all instructions to produce an exceptionally reliable, insightful, and well-structured response:
"""
    
    def __init__(
        self,
        google_api_key: Optional[Union[SecretStr, str]] = None,
        model: str = "gemini-2.0-flash",
        max_words: int = 600,
        llm_max_tokens: int = 4096,
        temperature: float = 0.7
    ):
        self.google_api_key = (
            SecretStr(google_api_key) if isinstance(google_api_key, str) else google_api_key
        )
        
        self.max_words = max_words
        
        self.prompt_template = PromptTemplate(
            input_variables=[
                "user_role",
                "user_query",
                "retrieved_context_str",
                "plots_json_str",
                "current_date",
                "max_words"
            ],
            template=self.PROMPT_TEMPLATE_STR
        )
        self.insight_agent = ChatGoogleGenerativeAI(
            model=model,
            api_key=self.google_api_key,
            max_tokens=llm_max_tokens,
            temperature=temperature
        )
        self.insight_chain = self.prompt_template | self.insight_agent | StrOutputParser()
        
    def generate(self, viz_output: dict[str, Any]) -> dict[str, Any]:
        print()
        print("[IntelligentInsightEngine] Gathering insights...")
        
        user_query = viz_output.get("user_query") or ""
        user_role = viz_output.get("user_role") or "research"
        retrieved_context_str = viz_output.get("retrieved_context_str") or "[]"
        citations_map = viz_output.get("citations_map") or {}
        plots = viz_output.get("plots") or []
        
        serializable_plots = [
            {k: v for k, v in p.items() if k != "figure"}
            for p in plots
        ]
        
        try:
            plots_json_str = json.dumps(serializable_plots, ensure_ascii=False)
        except Exception as e:
            print(f"[InsightGen] Warning: Could not serialize plots to JSON and plots will not be included. Error: {e}")
            plots_json_str = ""
        
        text_insights = self._invoke_agent_with_retry({
            "user_role": user_role,
            "user_query": user_query,
            "retrieved_context_str": retrieved_context_str,
            "plots_json_str": plots_json_str,
            "current_date": datetime.date.today().isoformat(),
            "max_words": self.max_words
        })
        
        citation_keys = set(citations_map)
        text_insights_sanitized = self._strip_unknown_cites(text_insights, citation_keys)

        print()
        print("[IntelligentInsightEngine] Successfully gathered insights.")

        return {
            "user_role": user_role,
            "user_query": user_query,
            "text_insights": text_insights_sanitized,
            "plots": plots,
            "citations_map": citations_map
        }
        
    def _invoke_agent_with_retry(self, vars: dict[str, Any]) -> str:
        MAX_RETRIES = 3
        BACKOFF = 4
        
        for attempt in range(1, MAX_RETRIES + 1):
            try:
                return self.insight_chain.invoke(vars)
            except Exception as e:
                if attempt >= MAX_RETRIES:
                    print(f"[InsightGen] LLM failed after {attempt} tries – {e}")
                    return "_Insight generation failed – please retry later._"
                wait = BACKOFF * 2 ** (attempt - 1)
                time.sleep(wait)
                
    def _strip_unknown_cites(self, text_insights: str, citation_keys: set[int]) -> str:
        return self.CITE_RE.sub(
            lambda m: f"<<{m.group(1)}>>" if int(m.group(1)) in citation_keys else "",
            text_insights
        )
    
class IntelligentInsightEngine:
    """
    The IntelligentInsightEngine is an advanced, modular pipeline for biomedical question answering—
    beyond a simple search engine.

    Key Highlights:
      - Sophisticated orchestration of multiple specialized modules (NLU, retrieval, visualization, synthesis).
      - Capable of deep query understanding, context-aware evidence retrieval, and multi-modal (text + visual) insight generation.
      - Role-adaptive: Customizes both what is retrieved and how answers are explained, based on whether the user is a researcher,
        clinician, or strategic stakeholder.
      - Strictly grounded: All outputs are evidence-based and meticulously cited, never hallucinated.
      - Exceptionally robust: Handles ambiguous input, retrieval failures, and complex multi-turn interactions gracefully.
      - Production-grade design: Intended for deployment in demanding biomedical intelligence and decision-support workflows.

    How it Works:
      1. **NLU Parsing:** Analyzes and structurizes user queries, extracting biomedical concepts, roles, filters, and enhanced queries.
      2. **Retrieval:** Finds the most relevant, high-quality evidence (papers, trials, news) using hybrid dense/sparse semantic search.
      3. **Visualization:** Synthesizes meaningful plots from the retrieved content—always cited and relevant to the query and user role.
      4. **Insight Generation:** Produces a deeply analytical, cited, and Markdown-formatted report that integrates both text and visual evidence.

    Usage:
      - Call `run(user_query)` with a biomedical or health-related question.
      - Returns a structured dictionary with: 
        - 'user_role', 'user_query', 'text_insights' (Markdown), 'plots' (list), 'citations_map' (dict).

    This engine exemplifies the frontier of applied AI for healthcare and biomedical research—delivering answers that are not just 
    responsive, but deeply reliable, insightful, and actionable.
    """

    def __init__(
        self,
        qdrant_client: QdrantClient,
        google_api_key: Optional[Union[SecretStr, str]] = None,
        nlu_kwargs: Optional[dict[str, Any]] = None,
        retriever_kwargs: Optional[dict[str, Any]] = None,
        visualizer_kwargs: Optional[dict[str, Any]] = None,
        insight_kwargs: Optional[dict[str, Any]] = None
    ):
        self.google_api_key = (
            SecretStr(google_api_key) if isinstance(google_api_key, str) else google_api_key
        )
        
        self.nlu_module = IntelligentInsightNLUParser(
            google_api_key=self.google_api_key,
            **(nlu_kwargs or {})
        )
        self.retrieval_module = IntelligentInsightRetriever(
            qdrant_client=qdrant_client,
            google_api_key=self.google_api_key,
            **(retriever_kwargs or {})
        )
        self.visualizer_module = IntelligentInsightVisualizer(
            google_api_key=self.google_api_key,
            **(visualizer_kwargs or {})
        )
        self.insight_gen_module = IntelligentInsightGenerator(
            google_api_key=self.google_api_key,
            **(insight_kwargs or {})
        )
        
    def run(self, user_query: str) -> dict[str, Any]:
        try:
            parsed = self.nlu_module.parse(user_query)
            retrieval_output = self.retrieval_module.retrieve(parsed)
            viz_out = self.visualizer_module.visualize(retrieval_output)
            insight = self.insight_gen_module.generate(viz_out)
            return insight
        except Exception as e:
            print("[InsightEngine] ERROR:", e)
            return {
                "user_query": user_query,
                "user_role": "unknown",
                "text_insights": (
                    "_An internal error occurred while generating insights. "
                    "Please try again later._"
                ),
                "plots": [],
                "citations_map": {},
            }
