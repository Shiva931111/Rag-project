import streamlit as st

from app.ingestion import IngestionService
from app.retrieval import RetrievalService


st.set_page_config(page_title="Production RAG Assistant", page_icon="🧠")
st.title("🧠 Production RAG Assistant")

if "ingestion" not in st.session_state:
    st.session_state.ingestion = IngestionService()
if "retrieval" not in st.session_state:
    st.session_state.retrieval = RetrievalService()

source_type = st.selectbox("Source type", ["pdf", "website", "github"])
topic = st.text_input("Topic (optional metadata)", value="")

if source_type == "pdf":
    file = st.file_uploader("Upload PDF", type=["pdf"])
    if st.button("Ingest PDF", disabled=file is None):
        with st.spinner("Indexing PDF..."):
            import tempfile

            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                tmp.write(file.read())
                tmp_path = tmp.name
            chunks = st.session_state.ingestion.ingest(
                source_type="pdf",
                source_value=tmp_path,
                extra_metadata={"source": file.name, "topic": topic},
            )
        st.success(f"Ingested {chunks} chunks.")
else:
    url = st.text_input("URL")
    if st.button("Ingest URL", disabled=not url):
        with st.spinner("Indexing URL content..."):
            chunks = st.session_state.ingestion.ingest(
                source_type=source_type,
                source_value=url,
                extra_metadata={"source": url, "topic": topic},
            )
        st.success(f"Ingested {chunks} chunks.")

st.divider()
query = st.text_input("Ask a question")
if st.button("Ask", disabled=not query):
    with st.spinner("Running advanced RAG pipeline..."):
        result = st.session_state.retrieval.answer(query)
    st.markdown("### Answer")
    st.write(result["answer"])
    st.markdown("### Rewritten Query")
    st.code(result["rewritten_query"])
    st.markdown("### Retrieved Sources")
    st.json(result["sources"])