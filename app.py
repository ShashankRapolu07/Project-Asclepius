import streamlit as st
import os
from dotenv import load_dotenv
from qdrant_client import QdrantClient
import sys
import re

from engines.wrappers import (
    LOOKBACK_UI,
    HORIZON_UI,
    get_horizon_engine,
    convert_to_days
)
from engines.horizon_scanning_engine import (
    TopicDetectionConfig,
    SynthesisConfig
)

from engines.intelligent_insight_engine import IntelligentInsightEngine

load_dotenv()

class UIStdout:
    def __init__(self, placeholder):
        self.placeholder = placeholder
        self._buf = ""

    def write(self, txt):
        self._buf += txt
        self.placeholder.text(self._buf)

    def flush(self):
        pass

def replace_citations(text: str, citation_map: dict[int, str]) -> str:
    text = text.replace("`<<", "<<").replace(">>`", ">>")
    pattern_double = re.compile(r"<<\s*(.*?)\s*>>")

    def repl_double(m: re.Match[str]) -> str:
        raw = m.group(1)
        cleaned = raw.replace("<", "").replace(">", "")
        parts = [p.strip() for p in cleaned.split(",") if p.strip()]

        seen: list[int] = []
        links: list[str] = []
        for token in parts:
            if not token.isdigit():
                continue
            n = int(token)
            if n == -1 or n in seen or n not in citation_map:
                continue
            seen.append(n)
            display = n + 1
            url = citation_map[n]
            links.append(f"[[{display}]]({url})")

        return " ".join(links)

    text = pattern_double.sub(repl_double, text)
    pattern_single = re.compile(r"(?<!\[)\[(\d+)\](?!\()")

    def repl_single(m: re.Match[str]) -> str:
        n = int(m.group(1))
        if n not in citation_map:
            return ""
        display, url = n + 1, citation_map[n]
        return f"[[{display}]]({url})"
    
    text = pattern_single.sub(repl_single, text)
    collapse = re.compile(r"(\[\[\d+\]\]\(\S+?\))(?:\s*\1)+")
    text = collapse.sub(r"\1", text)

    return text

def try_connect_qdrant():
    url = os.getenv("QDRANT_URL")
    key = os.getenv("QDRANT_API_KEY")
    try:
        client = QdrantClient(url=url, api_key=key)
        client.get_collections()
        st.session_state.update(qdrant_connected=True, client=client)
    except Exception as e:
        st.session_state.update(qdrant_connected=False, client=None)
        st.session_state["connection_error"] = str(e)


st.session_state.setdefault("qdrant_connected", False)
st.session_state.setdefault("client", None)
st.session_state.setdefault("qdrant_tried", False)

st.session_state.setdefault("stage", None)
st.session_state.setdefault("topics", None)
st.session_state.setdefault("docs", None)
st.session_state.setdefault("report", None)
st.session_state.setdefault("citations", None)

st.session_state.setdefault("intel_stage", None)
st.session_state.setdefault("intel_query", "")
st.session_state.setdefault("intel_parsed", None)
st.session_state.setdefault("intel_retrieval", None)
st.session_state.setdefault("intel_viz", None)
st.session_state.setdefault("intel_results", None)


if "horiz_logger" not in st.session_state:
    st.session_state.horiz_logger = UIStdout(None)
if "intel_logger" not in st.session_state:
    st.session_state.intel_logger = UIStdout(None)

if not st.session_state.qdrant_tried:
    with st.spinner("🔄 Connecting to Qdrant database…"):
        try_connect_qdrant()
    st.session_state.qdrant_tried = True

running_horiz = st.session_state.stage in ("detect","retrieve","synth")
running_intel = st.session_state.intel_stage in ("parse","retrieve","visualize","generate")

engine_choice = st.sidebar.radio(
    "Choose engine",
    ["Bio-Innovation Horizon Scanning", "Intelligent Insight Analysis"],
    disabled= running_horiz or running_intel
)

if st.session_state.qdrant_connected:
    st.sidebar.success("🟢 Qdrant connected")
else:
    st.sidebar.error("🔴 Qdrant not connected")
    if st.sidebar.button("Retry connection", disabled=running_horiz or running_intel):
        with st.spinner("🔄 Reconnecting to Qdrant…"):
            try_connect_qdrant()

if engine_choice == "Bio-Innovation Horizon Scanning":
    run_horiz = st.sidebar.button("▶️ Run Scan", disabled= not st.session_state.qdrant_connected or running_horiz)

    st.sidebar.header("Horizon Scanning Engine Configuration")

    n_topics = st.sidebar.slider("Max topics",1,5,3, disabled=running_horiz)
    lb = st.sidebar.selectbox("Lookback unit", list(LOOKBACK_UI), disabled=running_horiz)
    lb_min, lb_max, lb_step, lb_key = LOOKBACK_UI[lb]
    lb_len = st.sidebar.number_input("Lookback length", lb_min, lb_max, lb_min, lb_step, disabled=running_horiz)

    hz = st.sidebar.selectbox("Horizon unit", list(HORIZON_UI), disabled=running_horiz)
    hz_min, hz_max, hz_step, hz_key = HORIZON_UI[hz]
    hz_len = st.sidebar.number_input("Horizon length", hz_min, hz_max, hz_min, hz_step, disabled=running_horiz)

st.title("Bio-Innovation Horizon Scanning Prototype" 
         if engine_choice=="Bio-Innovation Horizon Scanning"
         else "Intelligent Insight Analysis Prototype")

# HORIZON SCANNING ENGINE FLOW
if engine_choice == "Bio-Innovation Horizon Scanning":
    horizon_info_ph = st.empty()

    if st.session_state.stage is None:
        horizon_info_ph.info(
            "Click **Run Scan** to analyse latest trends in Bio-tech space.  "
            "Scanning can take couple of minutes depending on the Lookback window. "
            "For faster scans, choose smaller windows."
        )
    if run_horiz and not running_horiz:
        for k in ("topics","docs","report","citations"):
            st.session_state[k] = None
        st.session_state.horiz_logger._buf = ""
        st.session_state.stage = "detect"
        st.rerun()

    engine = get_horizon_engine(
        client=st.session_state.client,
        google_api_key=os.getenv("GOOGLE_API_KEY"),
        topic_cfg=TopicDetectionConfig(n_topics=n_topics, lookback_days=convert_to_days(lb_len, lb_key)),
        synthesis_cfg=SynthesisConfig(horizon_unit=hz_len, horizon_value=hz)
    )

    if st.session_state.stage == "detect":
        horizon_info_ph.info(f"1. Agents are Identifying Trending Topics in the last {lb_len} {lb}...")
        spinner, logph = st.container(), st.empty()
        st.session_state.horiz_logger.placeholder = logph
        logph.text(st.session_state.horiz_logger._buf)
        with spinner:
            with st.spinner("🔎 Scanning topics (takes a while depending on Lookup window)…"):
                old = sys.stdout; sys.stdout = st.session_state.horiz_logger
                try: topics = engine._detect_topics()
                finally: sys.stdout = old
        st.success("✅ Topics identified")
        st.session_state.topics = topics
        st.session_state.stage = "retrieve"; st.rerun()

    elif st.session_state.stage == "retrieve":
        horizon_info_ph.info("2. Retrieving relevant contents...")
        spinner, logph = st.container(), st.empty()
        st.session_state.horiz_logger.placeholder = logph
        logph.text(st.session_state.horiz_logger._buf)
        with spinner:
            with st.spinner("📥 Querying database (might take time depending on topics identified)…"):
                old = sys.stdout; sys.stdout = st.session_state.horiz_logger
                try: docs = engine._retrieve_documents(st.session_state.topics)
                finally: sys.stdout = old
        st.success("✅ Content retrieved")
        st.session_state.docs = docs
        st.session_state.stage = "synth"; st.rerun()

    elif st.session_state.stage == "synth":
        horizon_info_ph.info("3. Agents are Analyzing and Synthesizing report...")
        spinner, logph = st.container(), st.empty()
        st.session_state.horiz_logger.placeholder = logph
        logph.text(st.session_state.horiz_logger._buf)
        with spinner:
            with st.spinner("🖋 Crafting report for you…"):
                old = sys.stdout; sys.stdout = st.session_state.horiz_logger
                try: report,cits = engine._synthesise(st.session_state.docs)
                finally: sys.stdout = old
        st.success("✅ Report ready")
        st.session_state.report    = report
        st.session_state.citations = cits
        st.session_state.stage     = "done"
        st.rerun()

    elif st.session_state.stage == "done":
        topic_names = list(st.session_state.report.keys())

        if len(topic_names) > 2:
            selected = st.selectbox("Select a topic to view", topic_names)
            display = [selected]
            use_outer_tabs = False
        else:
            display = topic_names
            use_outer_tabs = True

        if use_outer_tabs:
            outer_tabs = st.tabs([f"{i+1}. {name}" for i, name in enumerate(display)])
            containers = outer_tabs
        else:
            containers = [st.container() for _ in display]

        for container, topic_name in zip(containers, display):
            citation_map = st.session_state.citations.get(topic_name, {})
            content      = st.session_state.report[topic_name]

            with container:
                st.header(topic_name)

                brief_tab, assess_tab, tl_tab = st.tabs([
                    "📝 Intelligence Brief",
                    "⚖️ Risk & Opportunity",
                    "⏱️ Predicted Timeline",
                ])

                with brief_tab:
                    md = replace_citations(content["intelligence_brief"], citation_map)
                    st.markdown(md, unsafe_allow_html=True)

                with assess_tab:
                    md = replace_citations(content["assessments"], citation_map)
                    st.markdown(md, unsafe_allow_html=True)

                with tl_tab:
                    md = replace_citations(content["predicted_timelines"], citation_map)
                    st.markdown(md, unsafe_allow_html=True)

                log_buf = st.session_state.horiz_logger._buf
                if log_buf:
                    with st.expander("📜 Logs", expanded=False):
                        st.code(log_buf)

                if citation_map:
                    with st.expander("🔗 Source citation URLs", expanded=False):
                        for idx, url in citation_map.items():
                            st.write(f"- **<<{idx}>>** - {url}")

# INTELLIGENT INSIGHT ENGINE FLOW
else:
    with st.form("intel_form", clear_on_submit=False):
        iq = st.text_area("Enter your question", value=st.session_state.intel_query, height=100, max_chars=300, help="Max 300 characters")
        run_intel = st.form_submit_button( "▶️ Submit Query", disabled=running_intel)
    st.session_state.intel_query = iq

    intel_info_ph = st.empty()

    if not st.session_state.intel_stage:
        intel_info_ph.info("Enter a question and ▶️ Submit Query to begin analysis")

    if run_intel and not running_intel:
        if len(iq.strip()) < 1:
            st.error("Query must be at least 1 character")
        else:
            for k in ("intel_parsed","intel_retrieval","intel_viz","intel_results"):
                st.session_state[k] = None
            st.session_state.intel_logger._buf = ""
            st.session_state.intel_stage = "parse"
            st.rerun()

    engine_intel = IntelligentInsightEngine(
        qdrant_client=st.session_state.client,
        google_api_key=os.getenv("GOOGLE_API_KEY")
    )

    if st.session_state.intel_stage == "parse":
        intel_info_ph.info("1. NLU Agent is analyzing your Query…")
        spinner, logph = st.container(), st.empty()
        st.session_state.intel_logger.placeholder = logph
        logph.text(st.session_state.intel_logger._buf)
        with spinner:
            with st.spinner("🔄 Enhancing user query…"):
                old = sys.stdout; sys.stdout = st.session_state.intel_logger
                try:
                    parsed = engine_intel.nlu_module.parse(iq)
                finally: sys.stdout = old
        st.success("✅ Parsed")
        st.session_state.intel_parsed = parsed
        st.session_state.intel_stage  = "retrieve"; st.rerun()

    elif st.session_state.intel_stage == "retrieve":
        intel_info_ph.info("2. Retrieving Relevant Contents…")
        spinner, logph = st.container(), st.empty()
        st.session_state.intel_logger.placeholder = logph
        logph.text(st.session_state.intel_logger._buf)
        with spinner:
            with st.spinner("📥 Querying database…"):
                old = sys.stdout; sys.stdout = st.session_state.intel_logger
                try:
                    retrieval = engine_intel.retrieval_module.retrieve(st.session_state.intel_parsed)
                finally: sys.stdout = old
        st.success("✅ Retrieved")
        st.session_state.intel_retrieval = retrieval
        st.session_state.intel_stage     = "visualize"; st.rerun()

    elif st.session_state.intel_stage == "visualize":
        intel_info_ph.info("3. Visualizer Agent is generating Visual Insights…")
        spinner, logph = st.container(), st.empty()
        st.session_state.intel_logger.placeholder = logph
        logph.text(st.session_state.intel_logger._buf)
        with spinner:
            with st.spinner("📊 Visualizing…"):
                old = sys.stdout; sys.stdout = st.session_state.intel_logger
                try:
                    viz = engine_intel.visualizer_module.visualize(st.session_state.intel_retrieval)
                finally: sys.stdout = old
        st.success("✅ Visualized")
        st.session_state.intel_viz    = viz
        st.session_state.intel_stage  = "generate"; st.rerun()

    elif st.session_state.intel_stage == "generate":
        intel_info_ph.info("4. Narrative Agent is generating Narrative Insights…")
        spinner, logph = st.container(), st.empty()
        st.session_state.intel_logger.placeholder = logph
        logph.text(st.session_state.intel_logger._buf)
        with spinner:
            with st.spinner("🖋 Generating insights (takes time)…"):
                old = sys.stdout; sys.stdout = st.session_state.intel_logger
                try:
                    result = engine_intel.insight_gen_module.generate(st.session_state.intel_viz)
                finally: sys.stdout = old
        st.success("✅ Insights ready")
        st.session_state.intel_results  = result
        st.session_state.intel_stage    = "done"
        st.rerun()

    elif st.session_state.intel_stage == "done":
        res = st.session_state.intel_results
        plots = res.get("plots", [])
        cols = st.columns(len(plots)) if plots else []
        for c, p in zip(cols, plots):
            with c:
                st.pyplot(p["figure"])

        txt = replace_citations(res["text_insights"], res["citations_map"])
        st.markdown(txt, unsafe_allow_html=True)

        log_buf = st.session_state.intel_logger._buf
        if log_buf:
            with st.expander("📜 Logs", expanded=False):
                st.code(log_buf)

        citations = res.get("citations_map", {})
        if citations:
            with st.expander("🔗 Source citation URLs", expanded=False):
                for idx, url in citations.items():
                    st.write(f"- **[{idx+1}]** - {url}")