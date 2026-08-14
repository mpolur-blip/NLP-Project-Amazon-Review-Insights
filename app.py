import streamlit as st
import pandas as pd
import faiss
from sentence_transformers import SentenceTransformer

st.set_page_config(page_title="Semantic Review Search", page_icon="🔍")

@st.cache_resource
def load_resources():
    model = SentenceTransformer('all-MiniLM-L6-v2')
    index = faiss.read_index("data/reviews_faiss.index")
    df = pd.read_csv("data/reviews_with_topics.csv", low_memory=False)
    return model, index, df

embed_model, index, df_topics = load_resources()

st.title("🔍 Semantic Review Search")
st.caption("Search Amazon device reviews by meaning, not just keywords.")

query = st.text_input("Search reviews", placeholder="e.g. screen freezes and stops responding")
k = st.slider("Number of results", min_value=3, max_value=15, value=5)

if query:
    query_vec = embed_model.encode([query], convert_to_numpy=True)
    faiss.normalize_L2(query_vec)
    scores, indices = index.search(query_vec, k)

    results = df_topics.iloc[indices[0]].copy()
    results['similarity_score'] = scores[0]

    st.subheader(f"Top {k} results")
    for _, row in results.iterrows():
        with st.container(border=True):
            st.write(row['reviews.text'])
            col1, col2 = st.columns(2)
            col1.metric("Sentiment", row['sentiment'])
            col2.metric("Similarity", f"{row['similarity_score']:.2f}")
            st.caption(f"Topic: {row['bertopic_topic_name']}")