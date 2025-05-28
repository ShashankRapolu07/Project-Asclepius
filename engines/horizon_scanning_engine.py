import datetime
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from typing import Any, Union, Optional, Literal
from pydantic import SecretStr
import time
from langchain_core.documents import Document
from qdrant_client import QdrantClient, models
from langchain_qdrant import FastEmbedSparse
import ast
from dataclasses import dataclass, field
import re
import math
from langchain_core.runnables import RunnableLambda
import random
import hdbscan
from qdrant_client.models import Filter, FieldCondition, MatchValue, Range
from urllib.parse import urlparse
import numpy as np


@dataclass
class TopicDetectionConfig:
    """
    Configuration for horizon scanning topic detection and clustering.

    Attributes:
        lookback_days (int): Number of days in the past to consider for detecting emerging topics.
        max_documents (int): Maximum number of documents (abstracts) to fetch for clustering.
        min_cluster_size (int): Minimum size (number of samples) for a valid topic cluster (HDBSCAN).
        min_samples (Optional[int]): Minimum samples per cluster for HDBSCAN (controls noise sensitivity).
        num_unique_sources (int): Minimum unique sources to consider a cluster "diverse" (for scoring).
        n_topics (int): Number of top topics to detect and output for further synthesis.
        max_representatives (int): Max number of representative samples from each cluster for topic labeling.
    """

    # random_seed: int = 42
    lookback_days: int = 30
    max_documents: int = 100000
    min_cluster_size: int = 2
    min_samples: Optional[int] = None
    num_unique_sources: int = 7 # change this if num sources changes
    n_topics: int = 3
    max_representatives: int = 30
        
@dataclass
class RetrievalConfig:
    """
    Configuration for document retrieval following topic detection.

    Attributes:
        k (int): Total number of documents (chunks) to retrieve per topic for synthesis.
        type_fractions (dict[str, float]): Proportion of each document type to retrieve.
            - Keys: 'research_paper', 'clinical_trial', 'news_article'
            - Values: Fractions (must sum to 1.0).
    """

    k: int = 300
    type_fractions: dict[str, float] = field(
        default_factory=lambda: {
            "research_paper": 0.55,
            "clinical_trial": 0.4,
            "news_article":   0.05,
        }
    )
        
@dataclass
class SynthesisConfig:
    """
    Configuration for the downstream synthesis and report generation.

    Attributes:
        brief_words_max (int): Max word count for the "Intelligence Brief" section.
        max_assessment_points (int): Max number of opportunity/risk bullets in assessment.
        assessments_words_max (int): Max word count for the opportunity/risk assessment.
        max_context_length (int): Max total character count of context provided to LLM.
        max_chunk_length (int): Max length (characters) of any single content chunk.
        timeline_words_max (int): Max word count for the timeline prediction section.
        horizon_unit (Literal): Time unit ('years', 'months', or 'days') for horizon scanning.
        horizon_value (int): Number of units (years/months/days) for future outlook in synthesis.
    """

    brief_words_max: int = 1000
    max_assessment_points: int = 5
    assessments_words_max: int = 500
    max_context_length: int = 300000
    max_chunk_length: int = 500
    timeline_words_max: int = 500
    horizon_unit: Literal["years", "months", "days"] = "years"
    horizon_value: int = 3

class HorizonScanningEngine:
    """
    The HorizonScanningEngine is an advanced pipeline for **bio-medical horizon scanning**, designed to:
      1. Detect trending/emerging topics in recent scientific literature.
      2. Retrieve, cluster, and synthesize the most relevant documents for each topic.
      3. Produce a structured, role-ready intelligence report for each detected topic, including: 
         - Executive brief
         - Foresight analysis
         - Risk and opportunity assessment
         - Predicted future milestones/timeline

    This engine leverages state-of-the-art LLMs, dense and sparse retrieval, and custom synthesis prompts, with all critical steps governed by dedicated configuration dataclasses.

    Key Configuration Objects:
      - TopicDetectionConfig: Controls how clusters/topics are formed and what counts as a “trending” topic.
      - RetrievalConfig: Controls how many, and what type, of documents to retrieve for each topic.
      - SynthesisConfig: Controls the structure, length, and forward-looking horizon for all generated reports.

    **Pipeline stages:**
      - Topic detection and clustering via HDBSCAN and LLM summarization
      - Retrieval of topic-relevant literature with hybrid dense/sparse semantic search
      - Synthesis of comprehensive reports using multi-stage LLM prompting and evidence citation

    All report outputs are citation-anchored, evidence-grounded, and formatted for downstream use in strategic, clinical, or research contexts.
    """

    CITE_RE = re.compile(r"<<(\d+)>>")
    
    def __init__(
        self,
        qdrant_client: QdrantClient,
        topic_cfg: Optional[TopicDetectionConfig] = None,
        retrieval_cfg: Optional[RetrievalConfig] = None,
        synthesis_cfg: Optional[SynthesisConfig] = None,
        collection_name: str = "project_asclepius",
        topic_detection_model: str = "gemini-2.0-flash-lite",
        intelligence_brief_model: str = "gemini-2.0-flash",
        assessments_model: str = "gemini-2.0-flash",
        timeline_prediction_model: str = "gemini-2.0-flash",
        google_api_key: Optional[Union[SecretStr, str]] = None,
        embed_model: str = "models/embedding-001"
    ):  
        if isinstance(google_api_key, str):
            google_api_key = SecretStr(google_api_key)
            
        self.google_api_key = google_api_key
        
        self.collection_name = collection_name
        self.client = qdrant_client
        
        self.topic_cfg = topic_cfg or TopicDetectionConfig()
        self.retrieval_cfg = retrieval_cfg or RetrievalConfig()
        self.synthesis_cfg = synthesis_cfg or SynthesisConfig()
        
        self._validate_config()
        
        self.embedding_model = GoogleGenerativeAIEmbeddings(google_api_key=self.google_api_key, model=embed_model)
        self.sparse_model = FastEmbedSparse(model_name="Qdrant/bm25")
        
        self.CORE_CAPABILITIES_GUIDANCE = """The Bio-Innovation Horizon Scanning & Synthesis Engine focuses on identifying and synthesizing information from various sources. When determining the topic, prioritize themes related to:
- Emerging therapeutic targets and drug discovery pathways.
- Novel diagnostic technologies and biomarkers.
- Breakthroughs in areas like gene editing, regenerative medicine, neurotechnology, and synthetic biology.
- Potential disruptive innovations with high impact potential for future healthcare.

While the topic MUST be grounded in the provided document samples, if it aligns with or can be framed through these lenses, such framing is preferred. The ultimate goal is to achieve unparalleled foresight into the future of medicine and patient care.
"""
        
        self.topic_detection_model =  topic_detection_model
        self.intelligence_brief_model = intelligence_brief_model
        self.assessments_model = assessments_model
        self.timeline_prediction_model = timeline_prediction_model
        
        self._build_prompts_and_chains()
        
    def _validate_config(self):
        if not (1 <= self.topic_cfg.n_topics <= 100):
            raise ValueError("`n_topics` must be between 1 and 100.")
        if not (7 <= self.topic_cfg.lookback_days <= 3650):
            raise ValueError("`lookback_days` must be between 1 week and 10 years")
        if self.retrieval_cfg.k <= 0:
            raise ValueError("`k` must be positive")
        if not math.isclose(sum(self.retrieval_cfg.type_fractions.values()), 1.0, rel_tol=1e-6):
            raise ValueError("type_fractions must sum to 1.0")
    
    def _build_prompts_and_chains(self):
        TOPIC_DETECTION_TEMPLATE_STRING = """You are an exceptionally astute Bio-Medical Research Analyst and Taxonomist AI.
Your mission is to analyze a collection of document titles and abstracts (representative samples from a research cluster) and identify the single, most representative overarching bio-medical topic majority of the documents collectively point to, enabling to anticipate and shape the future of human wellbeing.

**Key Instructions (Follow PRECISELY):**

1.  **Topic Identification Process:**
    * First, carefully analyze each provided document sample (title and abstract) to understand its specific bio-medical theme, key findings, and scope.
    * Then, based on your understanding of all individual document themes, identify **ONE overarching bio-medical topic** that most accurately represents:
        * A specific, shared focus if all documents converge on it, OR
        * The predominant theme or most significant common thread if documents explore different facets of a broader area.
    * **Crucially, avoid forcing an overly narrow intersectional topic** (e.g., "Specific Technique X for Specific Disease Y") unless multiple documents explicitly and strongly support that precise combined focus. If documents are more loosely related (e.g., one about a general technique, another about a disease, a third about a diagnostic approach, all potentially linkable but not explicitly linked as a singular research thrust *within these specific documents*), identify a topic that reflects the most validated common ground or the most significant shared high-level concept.
    * The identified topic should reflect emerging trends and breakthrough innovations as indicated by the documents.

2.  **Output Format:** Your response MUST be a single string literal representing a Python tuple.
    The tuple structure is EXACTLY: `('<topic_name_str>', '<topic_description_str>', ['<keyword1_str>', '<keyword2_str>', ...])`

3.  **Content for `<topic_name_str>` (0th index):**
    * A concise, highly specific, and descriptive name for the identified overarching bio-medical topic, guided by the principles in Instruction 1.
    * It should accurately reflect the core theme derived from the collective analysis of the provided documents.
    * Example: 'Advanced Nanocarriers for Targeted mRNA Cancer Immunotherapy' or 'Novel Approaches in Neurodegenerative Disease Biomarker Discovery' (This latter example is broader if documents covered different biomarkers for different neurodegenerative diseases without a single unifying theme beyond that).

4.  **Content for `<topic_description_str>` (1st index):**
    * A clear, insightful, and informative description of the identified overarching topic.
    * Length: approximately 50-120 words.
    * It should explain the topic's essence, its potential significance in advancing proactive, predictive, and personalized healthcare, and how the provided documents (collectively or through their predominant themes) contribute to understanding this area.

5.  **Content for `['<keyword1_str>', ...]` (2nd index):**
    * A Python list containing atmost 8 unique, highly relevant bio-medical keywords or keyphrases.
    * These keywords must be central to the identified overarching topic and prominently featured or implied in the document samples.
    * Each keyword in the list must be a string.

6.  **Bio-Medical Focus & Core Capabilities Guidance:**
    * The identified topic, description, and keywords MUST be strictly bio-medical and forward-looking.
    * Critically consider the following "Core Bio-Medical Capabilities" guidance derived from Project Asclepius's Bio-Innovation Horizon Scanning engine. If the identified topic aligns with or can be framed through one of these lenses, prioritize that framing, ensuring it remains truthful to the document content.
    * Core Capabilities Guidance:
        ```
        {core_capabilities_guidance}
        ```
7.  **Example of PERFECT output format (given hypothetical input documents):**
    `('Neuroprotective Peptides for Alzheimer\\'s Disease Modulation', 'This cluster of research explores novel neuroprotective peptides and their mechanisms in modulating Alzheimer\\'s disease progression. The documents highlight potential therapeutic targets within amyloid and tau pathways, suggesting new avenues for early intervention and cognitive decline mitigation, aligning with advancements in neurotechnology.', ['Alzheimer\\'s disease', 'neuroprotection', 'peptides', 'amyloid beta', 'tau protein', 'neurotechnology', 'drug discovery'])`

8.  **CRITICAL - Adherence to Format (DO NOT DEVIATE):**
    * Your ENTIRE response MUST be ONLY the Python tuple string.
    * Do NOT include ANY introductory phrases (e.g., "Here is the topic:"), explanations, apologies, or any text outside the tuple string.
    * Do NOT use markdown (e.g., ```python ... ``` or ```json ... ```) to wrap your output.
    * Ensure all strings within the tuple (topic name, description, and each keyword) are properly enclosed in single quotes. Any single quotes *within* the string values themselves (e.g., in Alzheimer's) MUST be escaped with a backslash (e.g., `'Alzheimer\\'s disease'`). The list of keywords should be enclosed in square brackets `[]`. The entire output must be a valid Python tuple string literal.
    * Double-check that the output starts with `('` and ends with `')`.

**Document Samples to Analyze:**

{document_samples_str}

**Your Python tuple string output:**
"""
        self.topic_detection_prompt_template = PromptTemplate(
            input_variables=["document_samples_str", "core_capabilities_guidance"],
            template=TOPIC_DETECTION_TEMPLATE_STRING
        )
        self.topic_detection_agent = ChatGoogleGenerativeAI(
            model=self.topic_detection_model, max_tokens=2048, temperature=0, api_key=self.google_api_key
        )
        self.topic_detection_chain = (
            self.topic_detection_prompt_template
            | self.topic_detection_agent
            | StrOutputParser()
        )

        INTELLIGENCE_BRIEF_TEMPLATE_STR = """You are an expert Bio-Medical Intelligence Analyst AI.
Your mission is to generate a concise, objective, and professionally formatted "Intelligence Brief" focused *exclusively* on the provided bio-medical **Topic**.
This brief must be based *solely* on the **Provided Contextual Data**, which will be given as a string representation of a Python list of tuples.
The brief must include in-text citations, a qualitative foresight analysis on the topic's potential evolution, and adhere strictly to all specified constraints.
**Your role is to present factual summaries and analytical projections based on the data, NOT to provide strategic advice, risk/opportunity assessments, or discuss implications; those are handled by other specialized agents.**
The overarching goal is to provide foundational intelligence to anticipate future bio-medical developments.

**Topic for this Brief:** {topic_name}

**Provided Contextual Data:**
The following is a string representation of a Python list of tuples. Each tuple provides a piece of contextual information and is structured as `(content_chunk, publication_date, citation_index)`:
* `content_chunk` (string): The actual text content from a document.
* `publication_date` (string): The publication date, which may be an empty string if not available.
* `citation_index` (integer): An integer index for citation. **A `citation_index` of -1 means the `content_chunk` is UNCITABLE and MUST NOT be cited with a `<<N>>` marker.** For all other positive integer values `N`, this index should be used in `<<N>>` citation markers.

The data is provided in the following format:
`CONTEXT_DATA = {context_data_str}`
---

**CRITICAL INSTRUCTIONS FOR GENERATING THE INTELLIGENCE BRIEF:**

1.  **Exclusive Focus & Objectivity:**
    * The *entire* brief must be about the specified **Topic: {topic_name}**.
    * All information synthesized MUST originate from the `content_chunk` elements within the **Provided Contextual Data**. Do NOT introduce external knowledge.
    * Maintain an objective tone. **Do NOT include strategic recommendations, risk assessments, opportunity analysis, or discuss broader implications.**

2.  **Citation Requirements:**
    * Interpret the **Provided Contextual Data** as a list of `(content_chunk, publication_date, citation_index)` tuples.
    * Cite relevant information within the brief using the format `<<N>>`, where `N` is the integer value from the `citation_index` (the 3rd element) of the relevant tuple.
    * **VERY IMPORTANT:** If the `citation_index` (3rd element of a tuple) is `-1`, that specific `content_chunk` is UNCITABLE. You may use its information if relevant, but you MUST NOT include a `<<N>>` citation marker for it.
    * Place citations directly after the sentence or phrase they support.

3.  **Handling Context Quality:**
    * The `content_chunk` strings may be incomplete or fragmented. Your role is to intelligently synthesize the available information into a coherent and professional brief.
    * **DO NOT** comment on, mention, or apologize for any perceived incompleteness, inconsistencies, or quality issues in the `content_chunk` values. Your output should be polished.
    * Ignore `publication_date` if it's an empty string or not relevant to the narrative.

4.  **Structure & Formatting (MANDATORY - 3 Sections Only):**
    * Organize the brief using the following Markdown structure. Be professional and engaging.
    * The brief should be "beautifully framed" for readability and impact.

    ```markdown
    ### **1. Executive Summary**
    (A very concise 2-3 sentence objective overview of the topic's current definition and the key aspects covered in this brief, based on the provided data.)

    ### **2. Current Landscape & Key Developments**
    (Objectively synthesize the most critical information, breakthroughs, and current understanding of **{topic_name}** based *only* on the `content_chunk` values in the provided data. Integrate citations `<<N>>` meticulously where information is drawn from `content_chunk` values associated with a citable `citation_index`.)

    ### **3. Foresight Analysis: Projected Evolution for the Next {horizon_details_str}**
    (This section is for objective horizon scanning. Based *only* on the trends, data, and emerging signals within the `content_chunk` values, provide a **qualitative foresight analysis** regarding how the **Topic: {topic_name}** itself might evolve over the next **{horizon_details_str}**.
    Your analysis should objectively discuss:
    * Potential evolutionary paths of the topic, significant shifts in its scientific understanding, or new technical applications.
    * Key questions, areas of uncertainty, or new avenues of inquiry that may arise concerning the topic.
    * The general trajectory, momentum, or the changing nature of the topic as suggested by the evidence in the provided data.
    **Crucially, do NOT list specific, dated milestones or predict discrete future events with timelines.** Also, **do NOT include any analysis of risks, opportunities, strategic implications, or broader societal/commercial impacts here**, as those assessments are explicitly handled by other specialized agents. Your focus is strictly on the analytical projection of how the topic itself might change or develop based purely on the provided data. Synthesize this into a coherent narrative. Cite supporting evidence `<<N>>` from the data where appropriate.)
    ```

5.  **Word Count:**
    * The *entire* generated Intelligence Brief (all three sections) MUST NOT exceed **{brief_words_max} words**. Be concise yet comprehensive.

6.  **Output:**
    * Your response MUST be ONLY the Markdown formatted intelligence brief string, starting with `### **1. Executive Summary...` and ending after the "Foresight Analysis" section.
    * Do NOT include any preamble, apologies, or any text outside of this formatted brief.

---
Begin generating the Intelligence Brief for **Topic: {topic_name}** now, using the `CONTEXT_DATA` provided above and adhering strictly to all instructions, especially the exclusion of strategic implications and event-specific timeline predictions:
"""
        ASSESSMENTS_TEMPLATE_STR = """You are a Specialist Bio-Medical Opportunity & Risk Assessment AI for my bio-tech company.
Your mission is to conduct a focused assessment of potential opportunities and risks associated with the given **Topic**, considering its projected evolution over the specified **Horizon**.
You must base your assessment on the **Provided Contextual Data**, the accompanying **Intelligence Brief** and your knowledge.
The goal is to identify factors that could significantly impact the development, adoption, or consequences of innovations related to the topic to achieve unparalleled foresight.

**Topic for Assessment:** {topic_name}
**Assessment Horizon:** Next {horizon_details_str}

**Provided Intelligence Brief (summarizes current state and projects topic evolution):**
{intelligence_brief_str}

**Provided Contextual Data (original evidence - Python-like list of tuples: (content_chunk, publication_date, citation_index)):**
`CONTEXT_DATA = {context_data_str}`
---

**CRITICAL INSTRUCTIONS FOR GENERATING THE ASSESSMENT:**

1.  **Primary Focus:** Identify and articulate potential **Opportunities** and **Risks** specifically related to the **Topic: {topic_name}** within the **Assessment Horizon: {horizon_details_str}**.
    * Your assessment should be deeply informed by the "Foresight Analysis" section of the **Provided Intelligence Brief**.
    * Ground all points in evidence from the **Provided Contextual Data** where possible, using `<<N>>` citations.

2.  **Defining Opportunities:**
    * Consider avenues for significant scientific or technological breakthroughs.
    * Identify potential for novel applications, improved patient outcomes, or addressing unmet medical needs.
    * Highlight possibilities for disruptive innovations with high impact potential, especially those aligning with proactive, predictive, and personalized healthcare.

3.  **Defining Risks:**
    * Consider potential scientific, technical, or developmental challenges and hurdles.
    * Identify possible limitations, negative side effects, or unintended consequences of the innovation/topic itself.
    * **Scope of Risks:** Focus on risks inherent to the bio-medical topic/innovation (e.g., efficacy, safety, scalability, adoption barriers). Do NOT assess the ethics of AI systems or broad societal/data biases here, as those are handled by other specialized Project Asclepius frameworks. Only mention ethical considerations if they are a direct consequence of the bio-medical innovation itself as evidenced in the context.

4.  **Citation Requirements:**
    * When an opportunity or risk is directly supported by specific information in the **Provided Contextual Data**, cite it using `<<N>>`, where `N` is the integer value from the `citation_index` (the 3rd element) of the relevant tuple in `CONTEXT_DATA`.
    * If the `citation_index` is `-1`, the corresponding `content_chunk` is UNCITABLE; do not use a `<<N>>` marker for it, even if you refer to its content.
    * You are assessing based on the brief and context; direct citations should primarily point to the original `CONTEXT_DATA`.

5.  **Structure & Formatting (MANDATORY):**
    * Organize the assessment using the following Markdown structure. Be analytical, balanced, and clear.
    * **Each bullet point under "Potential Opportunities" and "Potential Risks" MUST follow this specific format:**
        `* **Concise Title in Bold Title Case:** Detailed explanation text. <<N>>`

    ```markdown
    ### **Risk & Opportunity Assessment (Outlook: Next {horizon_details_str})**

    #### **1. Potential Opportunities**
    (Identify and detail atmost **{max_assessment_points}** significant opportunities using the specified bullet point structure. For each opportunity, its title should be a concise summary, and the description should explain its nature and how it relates to the topic's projected evolution (as per the Intelligence Brief) or foundational evidence (from Contextual Data). Clearly articulate why it's an opportunity. Cite specific evidence from Contextual Data `<<N>>` if applicable in the description.)
    * **Opportunity Title 1:** [Detailed description of Opportunity 1, explaining its nature, link to foresight/context, and why it's an opportunity. Ensure this text follows the colon directly.] <<N>>
    * **Opportunity Title 2:** [Detailed description of Opportunity 2...] <<N>>
    * ...

    #### **2. Potential Risks**
    (Identify and detail atmost **{max_assessment_points}** significant risks using the specified bullet point structure. For each risk, its title should be a concise summary, and the description should explain its nature and how it relates to the topic's projected evolution or foundational evidence. Clearly articulate why it's a risk. Cite specific evidence from Contextual Data `<<N>>` if applicable in the description.)
    * **Risk Title 1:** [Detailed description of Risk 1, explaining its nature, link to foresight/context, and why it's a risk. Ensure this text follows the colon directly.] <<N>>
    * **Risk Title 2:** [Detailed description of Risk 2...] <<N>>
    * ...
    ```

6.  **Word Count & Conciseness:**
    * The *entire* generated Assessment (all three sections) MUST NOT exceed **{assessments_words_max} words**.
    * Be direct and impactful. Aim for atmost **{max_assessment_points}** distinct bullet points in both the Opportunities and Risks sections, unless fewer are strongly evident.

7.  **Output:**
    * Your response MUST be ONLY the Markdown formatted assessment string, starting with `### **Risk & Opportunity Assessment...`.
    * Do NOT include any preamble, apologies, or any text outside of this formatted assessment.

---
Begin generating the Risk & Opportunity Assessment for **Topic: {topic_name}** now, adhering strictly to all instructions:
"""
        TIMELINE_PREDICTION_TEMPLATE_STR = """You are a Specialist Bio-Medical Futurist and Timeline Prediction AI.
Your mission is to predict a sequence of key future milestones, particularly those indicating clinical relevance or significant impact, for the given **Topic**.
Your predictions MUST be logically derived from a synthesis of the **Provided Contextual Data**, the **Provided Intelligence Brief** (which outlines the topic's current state and projected evolution), and the **Provided Risk & Opportunity Assessment**.

**Topic for Timeline Prediction:** {topic_name}

**Provided Intelligence Brief (summarizes current state and projects topic evolution):**
{intelligence_brief_str}

**Provided Risk & Opportunity Assessment (highlights factors that may accelerate or decelerate progress):**
{assessments_str}

**Provided Contextual Data (original evidence - Python-like list of tuples: (content_chunk, publication_date, citation_index)):**
`CONTEXT_DATA = {context_data_str}`
---

**CRITICAL INSTRUCTIONS FOR GENERATING THE TIMELINE PREDICTION:**

1.  **Primary Task:** Identify and describe distinct, significant future milestones for the **Topic: {topic_name}** that are likely to occur.
    * Milestones should represent critical advancements, research breakthroughs, developmental stages (e.g., preclinical to Phase I, Phase I to Phase II, readiness for broader clinical application), or points of significant technological maturation relevant to achieving clinical impact or other disruptive potential.

2.  **Evidence-Based Prediction:**
    * Your predictions and rationale MUST be grounded in the synthesis of all provided inputs: the current R&D stage evident in `CONTEXT_DATA`, the qualitative evolution described in the `INTELLIGENCE_BRIEF`, and the accelerating/decelerating factors identified in the `ASSESSMENT`.
    * Do NOT introduce external knowledge or speculate beyond what can be reasonably inferred from the provided materials.

3.  **Timeframe Estimation:**
    * For each milestone, provide an estimated timeframe for its achievement (e.g., "within 6-12 months," "1-2 years from now," "by QX YYYY").
    * Timeframes should be realistic ranges, reflecting the uncertainties inherent in R&D.

4.  **Rationale Requirement:**
    * For each predicted milestone and its timeframe, provide a concise rationale. This rationale should explain *why* this milestone is predicted and *why* that specific timeframe is estimated, referencing specific insights from the `INTELLIGENCE_BRIEF`, `ASSESSMENT` (e.g., identified opportunities speeding things up, or risks causing delays), or supporting data points/development stage from `CONTEXT_DATA`.

5.  **Citation Requirements:**
    * If your rationale for a milestone or its timing is directly supported by specific information in the **Provided Contextual Data**, cite it using `<<N>>`, where `N` is the integer value from the `citation_index` (the 3rd element) of the relevant tuple in `CONTEXT_DATA`.
    * If the `citation_index` is `-1`, that `content_chunk` is UNCITABLE; do not use a `<<N>>` marker for it.

6.  **Structure & Formatting (MANDATORY):**
    * Organize the timeline prediction using the following Markdown structure. Be clear, concise, and analytical.

    ```markdown
    ### **Predicted Timeline & Key Milestones**

    Based on an integrated analysis of the provided intelligence brief, risk/opportunity assessment, and contextual data, the following key milestones are predicted for **{topic_name}**:

    1.  **Milestone:** [Clear description of the first predicted milestone (e.g., "Completion of Phase II Clinical Trials for X Application," "Demonstration of Y Technology in Large Animal Models," "Initial Regulatory Submissions for Z Diagnostic").]
        * **Estimated Timeframe:** [e.g., 9-11 months from now / H2 202X - H1 202Y]
        * **Rationale & Supporting Evidence:** [Concise explanation for predicting this milestone and its timeframe. Refer to insights from the Intelligence Brief (e.g., "aligns with the projected evolution towards clinical testing"), the Assessment (e.g., "accelerated by opportunity O1, but potential delay due to risk R1"), or specific Contextual Data points `<<N>>` (e.g., "current research is at stage S `<<N>>`").]

    2.  **Milestone:** [Description of the second predicted milestone.]
        * **Estimated Timeframe:** [...]
        * **Rationale & Supporting Evidence:** [...]

    * (Ensuring each is distinct and significant.)
    ```

8.  **Word Count & Conciseness:**
    * The *entire* generated Timeline Prediction (all sections) should ideally not exceed **{timeline_words_max} words** (if you choose to implement this constraint). Focus on the quality and justification of key milestones.

9.  **Output:**
    * Your response MUST be ONLY the Markdown formatted timeline prediction string, starting with `### **Predicted Timeline & Key Milestones...`.
    * Do NOT include any preamble, apologies, or any text outside of this formatted prediction.

---
Begin generating the Predicted Timeline for **Topic: {topic_name}** now, adhering strictly to all instructions:
"""

        self.intelligence_brief_prompt_template = PromptTemplate(
            input_variables=["topic_name", "context_data_str", "horizon_details_str", "brief_words_max"],
            template=INTELLIGENCE_BRIEF_TEMPLATE_STR
        )
        self.assessments_prompt_template = PromptTemplate(
            input_variables=[
                "topic_name",
                "horizon_details_str",
                "intelligence_brief_str",
                "context_data_str",
                "max_assessment_points",
                "assessments_words_max"
            ],
            template=ASSESSMENTS_TEMPLATE_STR
        )
        self.timeline_prediction_prompt_template = PromptTemplate(
            input_variables=[
                "topic_name",
                "intelligence_brief_str",
                "assessments_str",
                "context_data_str",
                "timeline_words_max"
            ],
            template=TIMELINE_PREDICTION_TEMPLATE_STR
        )
        
        self.intelligence_brief_agent = ChatGoogleGenerativeAI(
            model=self.intelligence_brief_model, max_tokens=4096, api_key=self.google_api_key
        )
        self.assessments_agent = ChatGoogleGenerativeAI(
            model=self.assessments_model, max_tokens=4096, api_key=self.google_api_key
        )
        self.timeline_prediction_agent = ChatGoogleGenerativeAI(
            model=self.timeline_prediction_model, max_tokens=4096, api_key=self.google_api_key
        )
        
        self.synthesis_chain = (
            RunnableLambda(
                lambda x: {
                    "topic_name": x["topic_name"],
                    "context_data_str": x["context_data_str"],
                    "horizon_details_str": x["horizon_details_str"],
                    "max_assessment_points": self.synthesis_cfg.max_assessment_points,
                    "assessments_words_max": self.synthesis_cfg.assessments_words_max,
                    "intelligence_brief_str": (
                        self.intelligence_brief_prompt_template
                        | self.intelligence_brief_agent
                        | StrOutputParser()
                        | RunnableLambda(lambda y: self._clean_markdown(y))
                    ).invoke({
                        "topic_name": x["topic_name"],
                        "context_data_str": x["context_data_str"],
                        "horizon_details_str": x["horizon_details_str"],
                        "brief_words_max": self.synthesis_cfg.brief_words_max
                    })
                }
            )
            | RunnableLambda(
                lambda x: {
                    "topic_name": x["topic_name"],
                    "context_data_str": x["context_data_str"],
                    "timeline_words_max": self.synthesis_cfg.timeline_words_max,
                    "intelligence_brief_str": x["intelligence_brief_str"],
                    "assessments_str": (
                        self.assessments_prompt_template
                        | self.assessments_agent
                        | StrOutputParser()
                        | RunnableLambda(lambda y: self._clean_markdown(y))
                    ).invoke({
                        "topic_name": x["topic_name"],
                        "horizon_details_str": x["horizon_details_str"],
                        "intelligence_brief_str": x["intelligence_brief_str"],
                        "context_data_str": x["context_data_str"],
                        "max_assessment_points": self.synthesis_cfg.max_assessment_points,
                        "assessments_words_max": self.synthesis_cfg.assessments_words_max
                    })
                }
            )
            | RunnableLambda(
                lambda x: {
                    "intelligence_brief": x["intelligence_brief_str"],
                    "assessments": x["assessments_str"],
                    "predicted_timelines": (
                        self.timeline_prediction_prompt_template
                        | self.timeline_prediction_agent
                        | StrOutputParser()
                        | RunnableLambda(lambda y: self._clean_markdown(y))
                    ).invoke({
                        "topic_name": x["topic_name"],
                        "intelligence_brief_str": x["intelligence_brief_str"],
                        "assessments_str": x["assessments_str"],
                        "context_data_str": x["context_data_str"],
                        "timeline_words_max": self.synthesis_cfg.timeline_words_max
                    })
                }
            )
        )
        
    def run(self) -> tuple[dict[str, dict[str, str]], dict[str, dict[int, str]]]:
        try:
            topics = self._detect_topics()
            if not topics:
                raise RuntimeError("No topics detected.")

            topics_docs = self._retrieve_documents(topics)
            if not topics_docs:
                raise RuntimeError("No documents retrieved for any topic.")

            report, citations = self._synthesise(topics_docs)
            if not report:
                raise RuntimeError("Synthesis failed for all topics.")

            return report, citations
        except Exception as e:
            print(f"[ERROR] HorizonScanningEngine pipeline failed: {e}")
            raise
    
    def _detect_topics(self) -> list[tuple[str, str, list[str]]]:
        cfg = self.topic_cfg
        # random.seed(cfg.random_seed)
        
        print(f"[INFO] Fetching abstracts from last {cfg.lookback_days} days…")
        abstracts, vectors = self._fetch_recent_abstract_vectors(cfg.lookback_days, cfg.max_documents)
        if not abstracts:
            raise RuntimeError("No abstract documents found in the specified window.")
            
        print(f"[INFO] Clustering {len(abstracts):,} abstracts with HDBSCAN…")
        clusterer = hdbscan.HDBSCAN(
            min_cluster_size=cfg.min_cluster_size,
            min_samples=cfg.min_samples,
            metric="euclidean",
            gen_min_span_tree=False,
            prediction_data=False
        )
        try:
            labels = clusterer.fit_predict(vectors)
        except Exception as e:
            raise RuntimeError(f"HDBSCAN clustering failed: {e}")
        
        label_indices: dict[int, list[int]] = {}
        for idx, lbl in enumerate(labels):
            if lbl == -1:
                continue # noise
            label_indices.setdefault(lbl, []).append(idx)
        if not label_indices:
            raise RuntimeError("All points classified as noise. Adjust HDBSCAN params.")
            
        scored_clusters: list[tuple[int, float]] = []
        for lbl, idxs in label_indices.items():
            cluster_size = len(idxs)
            num_sources = len({abstracts[i][0]["source"] for i in idxs if abstracts[i][0].get("source")})
            cluster_score = self._compute_cluster_score(cluster_size, num_sources, cfg.num_unique_sources)
            scored_clusters.append((lbl, cluster_score))
            
        scored_clusters.sort(key=lambda x: x[1], reverse=True)
        top_labels = [lbl for lbl, _ in scored_clusters[: cfg.n_topics]]
        
        print(f"[INFO] Identifying top {len(top_labels):,} trending topics in the last {cfg.lookback_days} days…")
        batch_inputs: list[list[tuple[str, str]]] = []
        for lbl in top_labels:
            idxs = label_indices[lbl]
            representative_samples: list[int] = random.sample(idxs, min(cfg.max_representatives, len(idxs)))
            representative_abstracts: list[tuple[str, str]] = [
                (abstracts[i][0]["title"] or "", abstracts[i][1] or "") for i in representative_samples
            ]
            batch_inputs.append(representative_abstracts)
        
        result = self._call_topic_detection_agent(batch_inputs)
        
        if len(result):
            print(f"[INFO] Successfully identified {len(result)} topics.")
        
        return result
    
    def _fetch_recent_abstract_vectors(
        self, lookback_days: int, max_documents: int = 10000
    ) -> tuple[list[tuple[dict[str, Any], str]], np.ndarray]:
        cutoff_dt = datetime.datetime.now(datetime.UTC) - datetime.timedelta(days=lookback_days)
        cutoff = cutoff_dt.timestamp()
        flt = Filter(
            must=[
                FieldCondition(key="metadata.content_type", match=MatchValue(value="abstract")),
                FieldCondition(key="metadata.published_date_ts", range=Range(gte=cutoff))
            ]
        )
        abstracts: list[tuple[dict[str, Any], str]] = []
        vectors: list[list[float]] = []
            
        total_count = 0
        scroll_off = None
        while True:
            if len(abstracts) >= max_documents:
                break
            res, scroll_off = self.client.scroll( # paginated way of fetching (suitable for large number of fetches)
                collection_name=self.collection_name,
                limit=min(2048, max_documents - len(abstracts)),
                with_vectors=True,
                offset=scroll_off,
                scroll_filter=flt
            )
            total_count += len(res)
            if not res:
                break
            for pt in res:
                abstracts.append((pt.payload["metadata"], pt.payload["page_content"]))
                vectors.append(pt.vector["dense"])
                if len(abstracts) >= max_documents:
                    break
            if scroll_off is None:
                break
        print(f"[INFO] Fetched {total_count} docs for Topic Detection.")
        return abstracts, np.asarray(vectors, dtype="float32")
    
    def _call_topic_detection_agent(self, batch_inputs: list[list[tuple[str, str]]]) -> list[tuple[str, str, list[str]]]:
        MAX_RETRIES = 3
        RETRY_BACKOFF = 5
        MAX_TITLE_LENGTH = 300
        MAX_ABSTRACT_LENGTH = 2000
        
        payloads: list[dict[str, Any]] = []
        for abstracts in batch_inputs:
            samples_str = "\n\n".join(
                f"Title: {t[: MAX_TITLE_LENGTH]}\nAbstract: {a[: MAX_ABSTRACT_LENGTH]}" for t, a in abstracts
            ).strip()
            payloads.append({
                "document_samples_str": samples_str,
                "core_capabilities_guidance": self.CORE_CAPABILITIES_GUIDANCE
            })
            
        if not payloads:
            raise RuntimeError("No clusters to score — batch_inputs to TopicDetectionAgent was empty.")
            
        attempt = 1
        raw_outputs = []
        while attempt <= MAX_RETRIES:
            try:
                raw_outputs = self.topic_detection_chain.batch(payloads)
                if len(raw_outputs) != len(payloads):
                    raise RuntimeError(f"Expected {len(payloads)} outputs, got {len(raw_outputs)}")
                break
            except Exception as e:
                if attempt == MAX_RETRIES:
                    raise RuntimeError(f"Topic detection agent failed after {MAX_RETRIES} attempts: {e}")
                wait = RETRY_BACKOFF * (2 ** (attempt - 1))
                print(f"[WARN] Topic detection batch attempt {attempt} failed: {e}. Retrying in {wait}s…")
                time.sleep(wait)
                attempt += 1
                
        parsed_results = []
        for idx, output_str in enumerate(raw_outputs):
            clean = self._clean_tuple_str(output_str)
            try:
                result = ast.literal_eval(clean)
                if (
                    isinstance(result, tuple)
                    and len(result) == 3
                    and isinstance(result[0], str)
                    and isinstance(result[1], str)
                    and isinstance(result[2], list)
                    and all(isinstance(k, str) for k in result[2])
                ):
                    parsed_results.append(result)
                else:
                    raise ValueError("Parsed structure mismatch")
            except Exception as e:
                print(f"[WARN] Failed to parse topic detection output for batch {idx}: {e}")
        
        return parsed_results
        
    def _compute_cluster_score(self, cluster_size: int, num_sources: int, unique_sources: int):
        if unique_sources == 0:
            return cluster_size
        diversity_factor = 1 + (num_sources / unique_sources)
        score = cluster_size * diversity_factor
        return score
    
    def _retrieve_documents(self, topics: list[tuple[str, str, list[str]]]) -> dict[str, list[Document]]:
        MAX_RETRIES = 3
        RETRY_BACKOFF = 5
        
        if not topics:
            print("[WARN] No topics provided for document retrieval")
            return {}
        
        cfg = self.retrieval_cfg
        
        results: dict[str, list[Document]] = {}
        
        for t_name, t_desc, _ in topics:
            topic_failed = False
            query = f"Topic:{t_name}\nDescription:{t_desc}".strip()
            q_dense_vec = self._embed_with_retry(query)
            q_sparse_vec = self.sparse_model.embed_query(query)
            
            use_hybrid = q_dense_vec is not None and len(q_dense_vec) > 0
            if not use_hybrid:
                print(f"[INFO] Dense embedding failed for '{t_name}', falling back to sparse-only search")
            
            if isinstance(q_sparse_vec, dict) and 'indices' in q_sparse_vec and 'values' in q_sparse_vec:
                sparse_indices = q_sparse_vec['indices']
                sparse_values = q_sparse_vec['values']
            elif hasattr(q_sparse_vec, 'indices') and hasattr(q_sparse_vec, 'values'):
                sparse_indices = q_sparse_vec.indices
                sparse_values = q_sparse_vec.values
            else:
                sparse_indices = [i for i, val in enumerate(q_sparse_vec) if val != 0.0]
                sparse_values = [val for val in q_sparse_vec if val != 0.0]
            
            per_type   = {dt: math.ceil(cfg.k * frac) for dt, frac in cfg.type_fractions.items()}
            collected: dict[str, list[Document]] = {dt: [] for dt in per_type}
            seen_keys: set[tuple[str, int | None]] = set()
            
            for doc_type, want in per_type.items():
                if want == 0:
                    continue
                
                query_filter = Filter(
                    must=[
                        FieldCondition(key="metadata.content_type", match=MatchValue(value="chunk")),
                        FieldCondition(key="metadata.original_doc_type", match=MatchValue(value=doc_type))
                    ]
                )
                
                attempt, docs = 1, []
                while attempt <= MAX_RETRIES:
                    try:
                        dense_results = []
                        if use_hybrid:
                            dense_results = self.client.query_points(
                                collection_name=self.collection_name,
                                query=q_dense_vec,
                                using="dense",
                                query_filter=query_filter,
                                limit=max(want // 2, 1)
                            ).points

                        sparse_results = self.client.query_points(
                            collection_name=self.collection_name,
                            query=models.SparseVector(
                                indices=sparse_indices,
                                values=sparse_values
                            ),
                            using="langchain-sparse",
                            query_filter=query_filter,
                            limit=max(want // 2, 1)
                        ).points or []

                        docs = dense_results + sparse_results
                        break
                    except Exception as e:
                        if attempt >= MAX_RETRIES:
                            topic_failed=True
                            print(f"[WARN] Hybrid Search failed for '{t_name}' ({doc_type}): {e}")
                            break
                        wait = RETRY_BACKOFF * 2 ** (attempt - 1)
                        print(f"[WARN] retry {attempt}/{MAX_RETRIES} "
                              f"({doc_type}) in {wait}s… ({e})")
                        time.sleep(wait)
                        attempt += 1
                        
                if topic_failed:
                    break
                
                for d in docs:
                    page_content, metadata = d.payload["page_content"], d.payload["metadata"]
                    key = (metadata.get("source_doc_id"), metadata.get("chunk_index"))
                    if key in seen_keys:
                        continue
                    seen_keys.add(key)
                    output_doc = Document(page_content=page_content, metadata=metadata)
                    collected[doc_type].append(output_doc)
                    
            if topic_failed:
                print(f"Skipping '{t_name}' topic...")
                continue
                        
            flat: list[Document] = []
            for v in collected.values():
                flat.extend(v)
            results[t_name] = flat
            
            summary = ", ".join(f"{dt}:{len(v)}" for dt, v in collected.items())
            print(f"[INFO] '{t_name}' → retrieved {len(results[t_name]):3d} docs ({summary})")
        
        return results
    
    def _synthesise(self, topics_docs: dict[str, list[Document]]) -> tuple[dict[str, dict[str, str]], dict[str, dict[int, str]]]:
        print(f"[INFO] Synthesizing reports for {len(topics_docs)} topics...")
        
        cfg = self.synthesis_cfg
        horizon_str = f"{cfg.horizon_value} {cfg.horizon_unit}"
        
        batch_inputs: list[dict[str, Any]] = []
        topic_order: list[str] = []
        citation_maps: dict[str, dict[int, str]] = {}
            
        for t_name, docs in topics_docs.items():
            if not docs:
                print(f"[WARN] Topic '{t_name}' skipped due to no context present...")
                continue
            
            id_map: dict[str, int] = {}
            url_map: dict[int, str] = {}
                
            context_items: list[str] = []
            context_limit = cfg.max_context_length
            chunk_limit = cfg.max_chunk_length
            char_count = 0
            
            for doc in docs:
                chunk = (doc.page_content or "").strip()[: chunk_limit]
                doc_id = doc.metadata.get("source_doc_id") or ""
                url = doc.metadata.get("url") or ""
                published_date = doc.metadata.get("published_date") or ""
                
                is_sane_url = self._is_sane_url(url)
                if is_sane_url and doc_id and doc_id not in id_map:
                    id_map[doc_id] = len(id_map)
                    
                idx = id_map.get(doc_id, -1)
                if idx != -1:
                    url_map[idx] = url
                
                tup = (chunk, published_date, idx)
                item_str = repr(tup) + ", "
                if char_count + len(item_str) > context_limit:
                    print(f"[INFO] Reached max context character limit ({context_limit}). Stopping context aggregation.")
                    break
                context_items.append(item_str)
                char_count += len(item_str)
                
            context_data_str = "[" + "".join(context_items).rstrip(", ") + "]"
            citation_maps[t_name] = url_map
            
            batch_inputs.append(
                {
                    "topic_name": t_name,
                    "context_data_str": context_data_str,
                    "horizon_details_str": horizon_str,
                    "brief_words_max": cfg.brief_words_max
                }
            )
            topic_order.append(t_name)
            
        MAX_RETRIES, BACKOFF = 3, 4
        batch_results = []
        for attempt in range(1, MAX_RETRIES + 1):
            try:
                batch_results = self.synthesis_chain.batch(batch_inputs)
                if len(batch_results) != len(batch_inputs):
                    raise RuntimeError(f"Synthesis returned {len(batch_results)} / {len(batch_inputs)} results")
                break
            except Exception as e:
                if attempt == MAX_RETRIES:
                    print(f"[WARN] Synthesis failed for all topics: {e}")
                    break
                wait = BACKOFF * 2 ** (attempt - 1)
                print(f"[WARN] Synthesis batch attempt {attempt} failed: {e}. Retrying in {wait} seconds…")
                time.sleep(wait)
                
        report: dict[str, dict[str, str]] = {}
        for t_name, result in zip(topic_order, batch_results):
            citation_map = citation_maps.get(t_name, {})
            citation_keys = set(citation_map)
            sanitized_result = {}
            for section, markdown_str in result.items():
                sanitized_result[section] = self._strip_unknown_cites(markdown_str, citation_keys)
            report[t_name] = sanitized_result
        
        if not report:
            raise RuntimeError("Synthesis failed for all topics.")
            
        print(f"[HorizonScanningEngine] Successfully synthesized for {len(report)} topics.")
            
        return report, citation_maps
    
    def _embed_with_retry(self, text: str) -> Optional[list[float]]:
        MAX_RETRIES = 3
        BACKOFF = 4
        
        attempt = 1
        while attempt <= MAX_RETRIES:
            try:
                return self.embedding_model.embed_query(text)
            except Exception as e:
                if attempt >= MAX_RETRIES:
                    print(f"[Retriever] ERROR: embed_query failed after {attempt} attempts – {e}")
                    return None
                wait = BACKOFF * 2**(attempt - 1)
                print(f"[HorizonScanner] embed_query retry {attempt}/{MAX_RETRIES} in {wait} seconds…")
                time.sleep(wait)
                attempt += 1
     
    @staticmethod
    def _clean_tuple_str(s: str) -> str:
        text = s.strip()
        if text.startswith("```"):
            lines = text.splitlines()
            if len(lines) >= 2 and lines[-1].strip() == "```":
                text = "\n".join(lines[1:-1]).strip()
        if text.startswith("`") and text.endswith("`"):
            text = text[1:-1].strip()
        return text
    
    @staticmethod
    def _clean_markdown(text: str) -> str:
        text = re.sub(r"^```(?:markdown)?\s*\n?", "", text)
        text = re.sub(r"\n?```$", "", text)
        if text.startswith("`") and text.endswith("`"):
            text = text[1:-1]
        return text.strip()
    
    def _strip_unknown_cites(self, text: str, citation_keys: set[int]) -> str:
        return self.CITE_RE.sub(
            lambda m: f"<<{m.group(1)}>>" if int(m.group(1)) in citation_keys else "",
            text
        )
    
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