import streamlit as st

from config import DB_PATH
from src.rag_pipeline import TouristRAG
from src.langchain_pipeline import LangChainTouristPipeline


st.set_page_config(
    page_title="RAG Tourist Assistant",
    page_icon="✈️",
    layout="wide",
)


@st.cache_resource
def load_pipeline():
    rag = TouristRAG(DB_PATH)
    return LangChainTouristPipeline(rag)


st.title("✈️ RAG Tourist Assistant")
st.caption(
    "SQLite SQL retrieval + T5 encoder semantic ranking + T5 generation + LangChain"
)

with st.sidebar:
    st.header("Pipeline")
    st.write("1. NLP query")
    st.write("2. Intent extraction")
    st.write("3. SQL retrieval")
    st.write("4. T5 encoder embeddings")
    st.write("5. Semantic reranking")
    st.write("6. Header + retrieved context")
    st.write("7. T5 encoder-decoder generation")

query = st.text_area(
    "What are you looking for?",
    height=110,
    placeholder=(
        "Example: I have a budget of 800 SAR per night in Riyadh. "
        "Recommend a hotel, a restaurant and attractions."
    ),
)

if st.button("Generate recommendations", type="primary"):
    if not query.strip():
        st.warning("Please enter a request.")
        st.stop()

    with st.spinner("Retrieving context and generating response..."):
        try:
            pipeline = load_pipeline()
            result = pipeline.invoke(query)
        except Exception as exc:
            st.error(
                "Application error. Make sure the database has been built "
                "with `python scripts/build_database.py`."
            )
            st.exception(exc)
            st.stop()

    st.subheader("Recommendation")
    st.write(result["answer"])

    st.divider()
    st.subheader("Parsed request")
    intent = result["intent"]

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("City", intent.city or "Any")
    c2.metric("Category", intent.category)
    c3.metric(
        "Budget",
        f"{intent.budget:g} SAR" if intent.budget is not None else "Not specified",
    )
    c4.metric(
        "Trip length",
        f"{intent.days} days" if intent.days else "Not specified",
    )

    st.subheader("Retrieved database records")
    rows = []
    for item in result["ranked"]:
        rows.append(
            {
                "Category": item["category"],
                "Name": item["name"],
                "Location": item["location"],
                "Rating": item["rating"],
                "Semantic score": round(item["semantic_score"], 4),
                "Price range": item.get("price_range", ""),
                "Cuisine": item.get("cuisine", ""),
                "Type": item.get("type", ""),
            }
        )

    if rows:
        st.dataframe(rows, use_container_width=True)
    else:
        st.info("No matching database rows were retrieved.")

    with st.expander("RAG prompt sent to T5"):
        st.code(result["prompt"], language="text")
