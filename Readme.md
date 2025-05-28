# Project Asclepius: Technical Blueprint (Prototype Phase)

> Try directly at [Project Asclepius Streamlit App](https://project-asclepius-hjkzznzohu2b2yq6u9apks.streamlit.app/)

---

## 🚀 Setup Instructions

Follow these steps to set up and run the **Project Asclepius** Streamlit app locally:

### 1. Clone the Repository

```bash
git clone https://github.com/your-username/Project-Asclepius.git
cd Project-Asclepius
```

### 2. Create & Activate a Virtual Environment (Recommended)

```bash
# Create virtual environment (replace 'venv' with your preferred name)
python -m venv venv

# Activate the environment
# On Windows:
venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Set Up Environment Variables

Create a `.env` file in the root directory and add your own keys:

```bash
QDRANT_URL=# visit https://qdrant.tech/documentation/cloud-quickstart/
QDRANT_API_KEY=# visit https://qdrant.tech/documentation/cloud-quickstart/
GOOGLE_API_KEY=# visit https://aistudio.google.com/app/apikey
```

### 5. Run the Streamlit App

```bash
streamlit run app.py
```

(*or, if above command is not working)

```bash
python -m streamlit run app.py
```

**Note:** Ensure you are using **Python 3.8+**.

### 📂 Project Structure

```bash
Project-Asclepius/
├── engines/           # Core engines and modules
├── .env               # Environment variables (not tracked by git)
├── Readme.md          # Project documentation
├── app.py             # Main Streamlit app
├── requirements.txt   # Python dependencies
```

---

Below outlines the technical architecture, design choices, and methodologies employed in the prototype of Project Asclepius, focusing on the **Bio-Innovation Horizon Scanning & Synthesis Engine** and the **Intelligent Insight Delivery & Visualization Interface**. The goal is to provide 2070 Health with unparalleled foresight into the future of medicine and patient care.

---

## 1. Data Extraction and Ingestion Pipeline

The foundation of Project Asclepius lies in its ability to ingest, process, and manage a diverse corpus of bio-medical data. This pipeline is designed to be robust, scalable, and adaptable to new data sources. It consists of three main components: `DataExtractionModule`, `DataIngestionModule`, and the orchestrating `DataIngestPipeline`.

### 1.1. `DataExtractionModule`: Gathering Diverse Bio-medical Intelligence

* **Objective**: To fetch data from a variety of relevant bio-medical sources.
* **Data Sources**: For this prototype, the module targets:
    * **PubMed Central (PMC)**: For open-access research papers.
    * **arXiv**: For pre-print research articles, often indicating cutting-edge developments.
    * **ClinicalTrials.gov**: For information on ongoing and completed clinical trials.
    * **Biotech News Feeds**: Aggregated from sources like Labiotech, BioPharma Dive, STAT News, GEN News, and Nature Biotechnology to capture industry trends and announcements.
    * While not exhaustive (e.g., patents, FDA data are future considerations), this set provides a rich, multi-faceted view of the bio-innovation landscape for a prototype.
* **Robustness and Efficiency**:
    * **Rate Limiting**: Custom rate-limiting mechanisms are implemented for each source to comply with their respective API usage policies and ensure stable, long-term data access.
    * **Retry Mechanisms**: A resilient retry strategy with exponential backoff is in place to handle transient network issues or API errors, maximizing data retrieval success.
* **Normalization**:
    * A critical step is the normalization of data from disparate sources into a standardized internal representation. This simplifies downstream processing and analysis.
    * Key fields in this normalized structure include:
        * `doc_id`: A unique identifier for the source document.
        * `source`: The originating platform (e.g., "PubMed Central", "arXiv").
        * `doc_type`: Categorization of the document ("research\_paper", "clinical\_trial", "news\_article").
        * `title`: Original title of the document.
        * `abstract`: Abstract of the document (crucial for efficient topic detection in Horizon Scanning).
        * `body_text`: The main content of the document.
        * `published_date`: Standardized publication date.
        * `published_date_ts`: Timestamp of the publication date for efficient time-based filtering.
        * `url`: Link to the original source.
* **Output**: The module outputs a list of these normalized document dictionaries, ready for ingestion.

### 1.2. `DataIngestionModule`: Processing and Storing Data for Intelligent Retrieval

* **Objective**: To process the extracted documents, generate summaries, create embeddings, and store them efficiently in a vector database for hybrid search.
* **Agentic Summarization**:
    * For each document's `body_text`, an LLM agent (Gemini 1.5 Flash via `ChatGoogleGenerativeAI`) generates a concise summary (strictly <= 200 words).
    * This summary aims to capture the most critical bio-medical information and key findings.
    * The agent also extracts up to 10 relevant bio-medical keywords/keyphrases.
    * **Rationale**: Summaries serve two purposes:
        1.  Provide a condensed, noise-reduced version of document chunks for more precise context in downstream RAG tasks (especially Horizon Scanning).
        2.  The extracted keywords, while not currently used for direct sparse vector querying in this prototype, offer a pathway to enhance keyword-based search quality in future iterations (e.g., by using them as the basis for sparse vectors instead of full chunk text).
* **Vector Database - Qdrant**:
    * **Choice Rationale**: Qdrant was selected due to its native support for **hybrid search** (dense + sparse vectors), advanced filtering capabilities, and robust scalability. Hybrid search is particularly vital in the bio-medical domain to handle specific jargon and technical terms effectively, combining semantic understanding with keyword precision.
    * **Embeddings**:
        * **Dense Vectors**: Generated using `GoogleGenerativeAIEmbeddings` ("models/embedding-001"), producing 768-dimensional embeddings for semantic search.
        * **Sparse Vectors**: Generated using `FastEmbedSparse` ("Qdrant/bm25") for keyword-based search.
    * **Qdrant Cloud**: The prototype utilizes a Qdrant Cloud deployment, ensuring high availability, low-latency queries, and managed scalability.
* **Chunking Strategy**:
    * `RecursiveCharacterTextSplitter` from LangChain is used for chunking the `body_text`.
    * **Parameters**: A `chunk_size` of 2000 characters and `chunk_overlap` of 200 characters were chosen.
    * **Rationale**: Larger chunk sizes were preferred over smaller ones to reduce the total number of documents stored, enhancing storage efficiency and potentially preserving more contextual integrity within each chunk. The summaries then condense these larger chunks.
* **Stored Document Types**:
    * Within Qdrant, documents are categorized by `content_type`:
        * `"abstract"`: Stores the original abstract of a research paper or the main summary of news/clinical trials. These are used as representative samples for the Topic Detection module in Horizon Scanning.
        * `"chunk"`: Stores the LLM-generated summary of a body text chunk. The original chunk text is also stored in the metadata (`original_content`) for full-text access if needed. These summarized chunks form the primary context for synthesis tasks.
* **Payload Indexing**: To optimize query performance, payload indexes are created in Qdrant for frequently filtered metadata fields like `content_type`, `published_date_ts`, `original_doc_type`, `source`, `source_doc_id`, and `keywords`.

### 1.3. `DataIngestPipeline`: Orchestrating the Flow

* **Objective**: To manage the end-to-end process of data extraction and ingestion.
* **De-duplication**:
    * The pipeline maintains an `ingested.json` log file which stores the `doc_id` of every successfully ingested source document.
    * Before processing, the `DataExtractionModule` checks this log, ensuring that documents already present in the vector store are not re-fetched or re-processed, thus preventing data duplication and saving computational resources.
* **Workflow**:
    1.  Loads the set of already ingested document IDs.
    2.  Invokes `DataExtractionModule` to fetch new documents (excluding already ingested ones).
    3.  Passes the newly extracted documents to `DataIngestionModule` for summarization, embedding, and storage in Qdrant.
    4.  The `DataIngestionModule` uses a callback (`_eagerly_update_log`) to update the `ingested.json` log file in batches as documents are successfully stored, ensuring atomicity and resilience.

### 1.4. Scalability and Future Enhancements

* **Scalability**:
    * **Qdrant Cloud**: As a distributed vector database, Qdrant offers horizontal scalability for handling growing data volumes and query loads.
    * **Modular Design**: The separation of extraction, ingestion, and pipeline logic allows for independent scaling and optimization of each component.
* **Key Future Enhancement: Dual Vector Store Strategy for Dynamic Data**:
    * **Concept**: A significant improvement would be to utilize two distinct Qdrant collections (or instances):
        1.  **Static Data Store**: For documents like research papers and news articles, whose content generally doesn't change post-publication. The current de-duplication mechanism is suitable here.
        2.  **Dynamic Data Store**: For sources like clinical trial databases or FDA regulatory updates, where information for a given `doc_id` can change over time.
    * **Mechanism for Dynamic Data**: When the `DataExtractionModule` fetches data from dynamic sources, instead of merely skipping already logged `doc_id`s, the `DataIngestionModule` would:
        * Check if the document exists in the dynamic store.
        * If it exists, compare the newly fetched content/metadata with the stored version.
        * If changes are detected, update the existing record in Qdrant (including re-summarization and re-embedding if necessary).
    * **Benefits**: This approach is crucial for maintaining the currency and accuracy of rapidly evolving information, which is vital for reliable horizon scanning and insight generation. It ensures Project Asclepius doesn't operate on stale data from dynamic sources.
* **Other Enhancements**:
    * **Fine-tuned Embeddings**: Consider training or fine-tuning embedding models specifically on bio-medical corpora to capture domain-specific nuances better than general-purpose models.
    * **Advanced Keyword Utilization**: Leverage the LLM-extracted keywords from the summarization step as the direct input for sparse vector generation, potentially improving the precision of keyword searches.
    * **Expanded Data Sources**: Incorporate patent databases, FDA announcements, and other relevant data streams.

---

## 2. Bio-Innovation Horizon Scanning & Synthesis Engine

This engine is the proactive core of Project Asclepius, designed to identify emerging bio-medical trends and synthesize comprehensive intelligence reports. It operates through a `HorizonScanningEngine` class, which encapsulates three main modules: Topic Detection, Document Retrieval, and Synthesis.

### 2.1. `HorizonScanningEngine` Overview

* **Objective**: To analyze recent data (within a configurable `lookback_days` window, e.g., 30 days for the prototype) to detect trending topics, retrieve relevant comprehensive documentation for these topics, and then synthesize detailed intelligence briefs, risk/opportunity assessments, and predicted timelines for clinical relevance.
* **Vector Store Document Structure**: This engine primarily interacts with documents stored in Qdrant. The key fields utilized are:
    * `page_content`: For "abstract" type documents, this is the actual abstract. For "chunk" type documents, this is the LLM-generated summary.
    * `metadata.source_doc_id`: Unique ID of the original document.
    * `metadata.published_date_ts`: Publication timestamp for time-based filtering (lookup window).
    * `metadata.content_type`: "abstract" or "chunk".
    * `metadata.original_doc_type`: "research\_paper", "clinical\_trial", "news\_article".
    * `metadata.title`: Document title.
    * `metadata.keywords`: LLM-extracted keywords from the summary.
    * `metadata.original_content`: For "chunk" type, this stores the original, unsummarized text of the chunk.

### 2.2. Topic Detection Module

* **Goal**: To automatically identify the most significant and trending bio-medical topics from the abstracts of documents published within the specified `lookback_days`.
* **Methodology**:
    1.  **Data Retrieval**: Fetches all documents from Qdrant where `metadata.content_type` is "abstract" and `metadata.published_date_ts` is within the `lookback_days` window (e.g., last 30 days). Up to `max_documents` (default 10,000) are considered. Their dense vectors are also retrieved.
    2.  **Clustering with HDBSCAN**:
        * The dense vectors of these abstracts are clustered using HDBSCAN.
        * **Rationale for HDBSCAN**:
            * It does not require pre-specification of the number of clusters, allowing the data structure to define the topics.
            * It is effective at identifying and separating noise points, which is crucial given the diverse and potentially irrelevant abstracts that might be retrieved.
        * **Parameters**: `min_cluster_size` is set to 5 (`TopicDetectionConfig`). `min_samples` is left for HDBSCAN to determine.
    3.  **Cluster Scoring and Selection**:
        * For each valid cluster (excluding noise), a **topic score** is computed.
        * **Topic Score Formula**: `score = cluster_size * (1 + (num_distinct_sources_in_cluster / total_distinct_sources_in_dataset))`
        * This score balances:
            * **Cluster Size**: Larger clusters indicate more discussion around a potential topic.
            * **Cluster Diversity**: A higher number of distinct sources (e.g., different journals, institutions if source metadata is detailed enough) discussing the same theme suggests broader interest and corroboration, making the topic more significant. The prototype uses the high-level `source` field (e.g., "PubMed Central", "arXiv").
        * The clusters are ranked by this score, and the top `n_topics` (default 3, from `TopicDetectionConfig`) are selected.
    4.  **Topic Definition via LLM Agent**:
        * For each selected top cluster, a representative sample of abstracts (titles and abstract text) is chosen, up to `max_representatives` (default 30).
        * **Scalability**: This sampling ensures that even for very large clusters (which could occur with longer lookback windows), the input to the LLM remains manageable, controlling cost and latency.
        * This sample is provided to an LLM agent (`ChatGoogleGenerativeAI` with "gemini-2.0-flash-lite") tasked with identifying the single, most representative overarching bio-medical topic. [cite: 12] The agent is guided by specific instructions to focus on themes like emerging therapeutic targets, novel diagnostic technologies, and breakthroughs in areas like gene editing, regenerative medicine, neurotechnology, and synthetic biology.
        * **Agent Output**: The agent returns a Python tuple string: `('<topic_name_str>', '<topic_description_str>', ['<keyword1_str>', '<keyword2_str>', ...])`. This structured output includes a concise topic name, a 50-120 word description, and up to 8 relevant keywords.
* **Validation & Performance (HDBSCAN)**:
    * The choice of `min_cluster_size` (5) and `min_samples` (HDBSCAN determined) was based on empirical testing with the prototype's dataset. Higher values resulted in too many abstracts being classified as noise, leading to fewer or no topics detected. These parameters control the confidence level of detected topics; more rigorous settings would yield fewer, potentially more robust, topics.

### 2.3. Document Retrieval Module

* **Goal**: For each identified topic, retrieve a comprehensive and relevant set of document chunks from Qdrant to serve as context for the synthesis module.
* **Strategy**:
    1.  **Query Formulation**: For each topic `(t_name, t_desc, t_keywords)`, a query string `f"Topic:{t_name}\nDescription:{t_desc}"` is generated.
    2.  **Hybrid Search**:
        * The query string is embedded to get dense and sparse vectors.
        * The retrieval aims to fetch `RetrievalConfig.k` (default 300 for the prototype) document chunks (where `metadata.content_type` is "chunk").
        * **Fractional Retrieval by Document Type**: To ensure a balanced perspective relevant to innovation and foresight, the `k` documents are retrieved proportionally from different `metadata.original_doc_type` categories. The current prototype uses `type_fractions`:
            * `research_paper`: 0.55 (focus on innovation)
            * `clinical_trial`: 0.40 (foresight, feasibility)
            * `news_article`: 0.05 (broader impact, risks/opportunities)
            * This distribution can be tuned. More document types (e.g., patent databases, social media for sentiment) could be added.
        * For each document type, a hybrid search is performed: `max(k_type // 2, 1)` documents are fetched via dense vector search, and `max(k_type // 2, 1)` via sparse vector search. The results are then merged and de-duplicated.
* **Noise Reduction and Context Quality**:
    * **Pre-Summarization**: The primary noise reduction technique employed is the use of LLM-generated summaries of document chunks (stored as `page_content` for "chunk" types during ingestion). This condenses ~2000 character chunks into ~200-word summaries focused on essential bio-medical details. This is crucial for managing context length for the synthesis LLMs.
    * **Potential Enhancements for Noise Reduction**:
        * **Multi-Query Expansion & RRF**: Generate multiple query variations for each topic and use Reciprocal Rank Fusion to re-rank results.
        * **Attention-Based Re-ranking**: Employ a fine-tuned transformer model to assess the relevance of each retrieved document to the topic and discard irrelevant ones. This can significantly improve precision but adds computational overhead and latency.
    * **Importance**: High context precision is vital for Horizon Scanning. Given LLMs' finite context windows and the need for rich information for foresight, minimizing noise maximizes the utility of the provided context.
* **Output**: The module returns a dictionary: `Dict[str_topic_name, List[Document]]`, where each list contains the retrieved (summarized) document chunks for that topic.

### 2.4. Synthesis Module

* **Goal**: To generate in-depth intelligence reports for each topic, including a foresight analysis, risk/opportunity assessment, and predicted timelines.
* **Multi-Agentic System**: A sequence of three specialized LLM agents (`ChatGoogleGenerativeAI` with "gemini-1.5-pro" for brief/assessments, "gemini-1.5-pro" for timelines) is used.
    * **Rationale for Multiple Agents**:
        1.  **Focused Task Performance**: LLMs tend to produce higher quality outputs when assigned specific, narrowly defined tasks rather than a single, complex multi-part task. This allows for more detailed and nuanced generation for each section of the report.
        2.  **Modularity for Enhancements**: This design facilitates future enhancements, such as incorporating Chain-of-Thought (CoT) reasoning or tool use for individual agents (e.g., an agent accessing a specific database for its sub-task).
* **Sequential Agent Workflow**:
    1.  **Intelligence Brief Agent**:
        * **Task**: Analyzes the topic name/description and the retrieved contextual document chunks to produce an "Intelligence Brief". This brief includes an executive summary, a review of the current landscape and key developments, and a qualitative foresight analysis on the topic's potential evolution over a defined horizon (e.g., "next 3 years" from `SynthesisConfig`).
        * **Prompting**: Guided by `INTELLIGENCE_BRIEF_TEMPLATE_STR`, which emphasizes objectivity and grounding in provided data.
        * **Citations**: The agent is instructed to provide in-text citations `<<N>>` referencing the `citation_index` of the supporting context chunks.
        * **Potential Enhancement**: Implement CoT reasoning where the agent can formulate sub-queries to retrieve additional specific information from the vector store or even web search if needed to elaborate on certain aspects of the foresight analysis.
    2.  **Assessments Agent**:
        * **Task**: Takes the generated Intelligence Brief, the original topic, and the retrieved context to identify and articulate potential opportunities and risks associated with the topic over the defined horizon.
        * **Prompting**: Guided by `ASSESSMENTS_TEMPLATE_STR`.
        * **Citations**: Also provides citations to support its assessments.
        * **Potential Enhancement**: For more robust feasibility assessment of opportunities/risks, this agent could be augmented with CoT reasoning and access to specialized databases (e.g., detailed clinical trial outcome databases, market analysis data).
    3.  **Timeline Prediction Agent**:
        * **Task**: Synthesizes information from the Intelligence Brief, the Assessments, and the original context to predict a sequence of key future milestones for the topic, particularly those indicating clinical relevance or significant impact.
        * **Prompting**: Guided by `TIMELINE_PREDICTION_TEMPLATE_STR`.
        * **Citations**: Provides rationale and supporting evidence with citations.
        * **Potential Enhancement**: To improve the practicality of timeline predictions, this agent could leverage CoT and access to regulatory databases (e.g., FDA approval timelines, patent expiry data).
* **Citation Handling**:
    * Before being passed to the synthesis agents, the list of retrieved `Document` objects for a topic is formatted into a string representation of a Python list of tuples: `[(content_chunk_summary, publication_date_str, citation_index_int), ...]`.
    * An internal mapping (`id_map`) assigns a unique integer `citation_index` (starting from 0) to each unique `source_doc_id` encountered in the retrieved documents. Unciteable chunks (e.g., those without a valid URL or `source_doc_id`) are assigned an index of -1.
    * The agents are instructed to use these positive integer indices in `<<index>>` format for citations.
    * A final `citations_map` (`Dict[int_index, str_url]`) is generated for each topic, linking the integer citation indices back to the source URLs for the final report.
* **Output**: The synthesis module produces a dictionary for each topic, containing the markdown-formatted strings for the "intelligence\_brief", "assessments", and "predicted\_timelines". The overall output of the `HorizonScanningEngine` is a tuple: `(report_dict, citations_map_dict)`.

### 2.5. Scalability and Validation

* **Scalability**:
    * **Topic Detection**: Representative sampling helps manage LLM load irrespective of cluster size.
    * **Retrieval**: Relies on Qdrant's scalability. The use of summaries as context also helps manage the amount of text processed by synthesis agents.
    * **Synthesis**: LLM throughput is a factor. The multi-agent approach, while improving quality, adds sequential processing time. Batch processing of topics is employed.
* **Validation**:
    * **Topic Detection**: Evaluate cluster quality using metrics like silhouette score (if applicable, though HDBSCAN is density-based) or by human assessment of topic coherence and relevance. The `min_cluster_size` parameter is key for tuning.
    * **Retrieval**: Standard IR metrics such as Precision@K, Recall@K can be used if ground truth relevance judgments are available for specific topics. Human evaluation of retrieved document relevance is also crucial.
    * **Synthesis**: This is more qualitative. Validation methods include:
        * Human expert review of the generated briefs, assessments, and timelines for accuracy, coherence, and insightfulness.
        * Cross-referencing predictions with actual future developments (long-term validation).
        * Checking the consistency and correctness of citations.

---

## 3. Intelligent Insight Delivery & Visualization Interface

This engine provides an interactive way for users (research, clinical, strategic teams) to query Project Asclepius using natural language and receive evidence-backed textual insights, often complemented by visualizations. It is orchestrated by the `IntelligentInsightEngine` class.

### 3.1. `IntelligentInsightEngine` Overview

* **Objective**: To understand a user's natural language query (NLQ), retrieve the most relevant information from the Qdrant vector store, synthesize textual insights, and generate appropriate visualizations to aid comprehension.
* **Core Modules**:
    1.  `IntelligentInsightNLUParser`: Understands and refines the user's query.
    2.  `IntelligentInsightRetriever`: Fetches relevant documents.
    3.  `IntelligentInsightVisualizer`: Determines and specifies data for plots.
    4.  `IntelligentInsightGenerator`: Synthesizes textual insights.

### 3.2. NLU Parsing Module (`IntelligentInsightNLUParser`)

* **Goal**: To analyze and transform the user's raw NLQ into a structured format that enhances downstream retrieval and synthesis.
* **LLM Agent**: Uses `ChatGoogleGenerativeAI` ("gemini-2.0-flash-lite").
* **Key Functions & Outputs**:
    * **User Role Identification**: Classifies the query's intent into one of three roles:
        * `"research"`: Focus on scientific discovery, data, mechanisms.
        * `"clinical"`: Related to patient care, diagnostics, trials.
        * `"strategic"`: Market trends, policy, industry impact.
        This role guides retrieval and the tailoring of insights. Defaults to "research" if ambiguous.
    * **Enhanced Queries**: (`max_enhanced_queries` is 5 in prototype) The capability exists to generate paraphrased or decomposed sub-questions from the original NLQ to improve retrieval coverage.
    * **Keywords**: Extracts key bio-medical terms and jargons from the NLQ. While not directly used for sparse vector search in the current retrieval module of this engine, these are available for future targeted search enhancements.
    * **Need Visualization**: A boolean flag (defaults to `True` in the code after LLM processing, but the LLM is instructed to determine if visualization is truly beneficial). The LLM sets it to `False` if the query seeks purely textual info or if visualization isn't appropriate.
    * **Filters**: Extracts explicit or strongly implied filters, such as date ranges (`published_date: {"$gte": "YYYY-MM-DD", "$lte": "YYYY-MM-DD"}`). Dates are resolved relative to the current date.
* **Output Structure**: A dictionary, e.g.:
    ```python
    {
        "user_role": "research",
        "user_query": "Original user query string",
        "enhanced_queries": ["enhancement1", "enhancement2", ...],
        "keywords": ["keyword1", "keyword2", ...],
        "need_visualization": True,
        "filters": {"published_date": {"$gte": "2024-01-01"}}
    }
    ```

### 3.3. Retrieval Module (`IntelligentInsightRetriever`)

* **Goal**: To fetch the most relevant document chunks from Qdrant based on the parsed and (potentially enhanced) user query.
* **Methodology**:
    * **Input**: The structured output from the `IntelligentInsightNLUParser`.
    * **Query Texts**: Uses the original `user_query` and any `enhanced_queries` (if generated).
    * **Hybrid Search + RRF**:
        1.  For each query text, dense and sparse embeddings are generated.
        2.  A hybrid search is performed against Qdrant (fetching "chunk" content type) to retrieve an initial set of `retrieval_k` (default 200) documents. This search incorporates any filters (e.g., date ranges) extracted by the NLU module.
        3.  The scores from searches for each query (original + enhancements) are combined using **Reciprocal Rank Fusion (RRF)**. Each document receives an RRF score based on its rank in the results for each query variation.
        4.  The documents are then globally re-ranked by their aggregated RRF scores.
        5.  The top `final_k` (default 200) documents after RRF are selected as the context.
* **Role-Based Retrieval (Potential Enhancement)**:
    * Similar to the Horizon Scanning engine, fractional retrieval based on `original_doc_type` and the identified `user_role` could be implemented. For instance:
        ```python
        ROLE_FRACTIONS = {
            "research": {"research_paper": 0.6, "clinical_trial": 0.3, "news_article": 0.1},
            "clinical": {"research_paper": 0.4, "clinical_trial": 0.5, "news_article": 0.1},
            "strategic": {"research_paper": 0.45, "clinical_trial": 0.15, "news_article": 0.40},
        }
        ```
    * This was not implemented in the prototype for this engine to prioritize lower query latency, as it would involve multiple distinct retrieval operations per query. However, it's a valuable enhancement for tailoring context more precisely.
* **Output**: A dictionary containing the `user_role`, `user_query`, `need_visualization` flag, and a list of the top `final_k` `retrieved_documents` (LangChain `Document` objects).

### 3.4. Visualizer Module (`IntelligentInsightVisualizer`)

* **Goal**: If `need_visualization` is true (will be most of the times), this module analyzes the retrieved documents in context of the user query and role, and instructs an LLM agent to specify data for relevant plots.
* **Agent-Based Plot Specification**:
    * **LLM Agent**: Uses `ChatGoogleGenerativeAI` ("gemini-2.0-flash").
    * **Input**: User query, user role, current date, the retrieved context (formatted as a string of tuples `(original_content_chunk, publication_date, citation_index)`, similar to Horizon Scanning), and descriptions of available plot types (`PLOT_TOOL_DESCRIPTIONS`).
    * **Task**: The agent is instructed to identify and specify up to `max_plots` (default 3) visualizations that directly help answer the user's query. It must synthesize or extract all necessary data for these plots *exclusively* from the retrieved context and provide citations (indices from the context string).
    * **Output**: A JSON string representing a list of plot specification dictionaries. Example structure for one plot:
        ```json
        {
            "plot_type": "line_chart",
            "title": "Trend of Keyword 'XYZ' Mentions Over Time",
            "x": ["2023-01", "2023-02", "2023-03"],
            "y": [5, 8, 12],
            "x_label": "Month-Year",
            "y_label": "Number of Mentions",
            "citations": [1, 3, 8] // Indices from the input context_str
        }
        ```
* **Actual Plot Generation**:
    * The `IntelligentInsightVisualizer` parses the LLM's JSON output.
    * For each valid plot specification, it uses a mapping (`PLOT_FUNC_MAP`) to call the appropriate plotting function in `VisualizationTools` (which uses `matplotlib`).
    * The generated `matplotlib.figure` object and the list of citation URLs (resolved from the indices) are added to the plot specification.
* **Challenges and Limitations**:
    * LLMs currently demonstrate better proficiency in specifying data for simpler plots like line and bar charts. Complex plots (e.g., heatmaps, complex scatter plots requiring intricate data transformation) are more challenging for current models to specify accurately from unstructured text.
    * The quality of visualization heavily depends on the "plottability" of information within the retrieved text chunks.
* **Potential Improvement**:
    * **Ingestion-Phase Plottable Data Extraction**: An agent during the data ingestion phase could specifically extract and structure potentially plottable data (e.g., numerical values, relationships, tables) from each document. This structured data could then be more easily utilized by the visualization agent.
* **Verification**: Each plot specification includes a `citations` list (later converted to URLs), allowing users to trace the data back to its source chunks.

### 3.5. Insight Generation Module (`IntelligentInsightGenerator`)

* **Goal**: To synthesize comprehensive, evidence-backed textual insights that directly answer the user query, tailored to their role, and integrating information from both the retrieved textual context and the generated plot specifications.
* **LLM Agent**: Uses `ChatGoogleGenerativeAI` ("gemini-2.0-flash").
* **Input**:
    * User query and role.
    * Retrieved context string (same format as used by Visualizer).
    * JSON string of **serializable plot specifications** (excluding the matplotlib figure objects, but including title, data, labels, and plot-specific citations). This allows the LLM to "understand" what the plots represent.
    * Current date and a `max_words` constraint (default 600).
* **Methodology**:
    * The agent is prompted to go beyond summarization, aiming for deep synthesis: identifying patterns, drawing connections, and extracting nuanced meanings.
    * It MUST ground all factual statements in the provided context or plot data, using `<<N>>` citations for context chunks.
    * It is instructed to intelligently weave insights from the visualizations into the narrative, referring to plots by their titles or sequentially.
* **Citation Handling**: Same as in the Horizon Scanning engine. A final step (`_strip_unknown_cites`) ensures that only valid citations (those present in the `citations_map`) appear in the output.
* **Output**: A dictionary containing:
    * `user_query`, `user_role`.
    * `text_insights`: The markdown-formatted textual answer.
    * `plots`: The list of plot dictionaries, now including the generated `matplotlib.figure` objects.
    * `citations_map`: Mapping integer citation indices to source URLs.

### 3.6. Scalability and Validation

* **Scalability**:
    * **NLU, Visualization (Spec Gen), Insight Gen**: LLM agent calls are the main factor. Can be parallelized if multiple independent user queries are processed.
    * **Retrieval**: Qdrant is scalable. RRF adds a small overhead but is manageable for `k` up to a few hundreds/thousands.
    * **Plot Rendering**: Local `matplotlib` rendering is fast for a small number of plots.
* **Validation**:
    * **NLU Parser**: Measure accuracy of role classification, keyword extraction, and filter detection against a manually annotated dataset of queries.
    * **Retriever**: Standard IR metrics (Precision@K, Recall@K) on a benchmark dataset. User feedback on relevance.
    * **Visualizer**:
        * Correctness of plot specifications generated by LLM (does the data match the plot type? are arrays of correct length?).
        * Relevance of chosen plots to the query.
        * Accuracy of data extraction/synthesis for plotting.
        * User feedback on clarity and usefulness of visualizations.
    * **Insight Generator**:
        * Factual accuracy and grounding (verifiability through citations).
        * Coherence, relevance to the query and role.
        * Completeness in addressing the query.
        * Quality of integration of visual insights into the text.
        * Human evaluation is paramount for assessing the quality of generated insights.
