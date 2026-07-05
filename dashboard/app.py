"""
app.py
======
CineMatch — Main entry point.
Run with: streamlit run app.py

This sets global page config + a cinema-inspired theme and serves as the
landing page. The User Page and Item Page live in dashboard/pages/ —
Streamlit auto-discovers files in a `pages/` folder and adds them to the
sidebar navigation automatically.
"""

import streamlit as st

st.set_page_config(
    page_title="CineMatch",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Light custom styling — keeps Streamlit's reliable widget behavior
# ---------------------------------------------------------------------------
st.markdown(
    """
    <style>
    .cinematch-title {
        font-size: 3rem;
        font-weight: 800;
        background: linear-gradient(90deg, #E50914, #FFB400);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0;
    }
    .cinematch-subtitle {
        font-size: 1.1rem;
        color: #9CA3AF;
        margin-top: 0;
    }
    .feature-card {
        background-color: rgba(255,255,255,0.03);
        border: 1px solid rgba(255,255,255,0.1);
        border-radius: 12px;
        padding: 24px;
        height: 100%;
    }
    .feature-card h3 {
        color: #FFB400;
        margin-top: 0;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown('<p class="cinematch-title">🎬 CineMatch</p>', unsafe_allow_html=True)
st.markdown(
    '<p class="cinematch-subtitle">A Neural Movie Recommendation System — built on MovieLens</p>',
    unsafe_allow_html=True,
)

st.divider()

col1, col2 = st.columns(2)

with col1:
    st.markdown(
        """
        <div class="feature-card">
        <h3>👤 User Page</h3>
        <p>Select a user, explore their rating history, and get personalized
        Top-N movie recommendations powered by our trained Neural Collaborative
        Filtering (NCF) model.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

with col2:
    st.markdown(
        """
        <div class="feature-card">
        <h3>🎬 Item Page</h3>
        <p>Pick any movie and discover similar titles using content-based
        Cosine Similarity over genres and user-generated tags.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

st.divider()
st.info("👈 Use the sidebar to navigate to **User Page** or **Item Page**.")

with st.expander("ℹ️ About this project"):
    st.write(
        """
        **CineMatch** is built for the AI-Pro Intake 46 Recommender Systems course project.

        - **Dataset:** MovieLens (small)
        - **User Recommender:** Neural Collaborative Filtering (NCF / NeuMF)
        - **Item Similarity:** TF-IDF + Cosine Similarity on genres & tags
        """
    )
