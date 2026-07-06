"""
pages/2_Item_Page.py
=====================
The Item Page of the CineMatch dashboard.

Required UI components (per project spec):
  - Item selection mechanism
  - Item profile (metadata)
  - Item top-N similar items
  - Similarity section navigation (next/previous/specific page)
"""

import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import streamlit as st
from utils import (
    load_movies_full,
    load_ratings_encoded,
    load_similarity_artifacts,
    paginate,
)

st.set_page_config(page_title="CineMatch — Item Page", page_icon="🎬", layout="wide")

st.title("🎬 Item Page")
st.caption("Select a movie to view its profile and discover similar titles.")

# ---------------------------------------------------------------------------
# Load all artifacts (cached — only loads from disk once per session)
# ---------------------------------------------------------------------------
try:
    movies_full = load_movies_full()
    ratings_encoded = load_ratings_encoded()
    sim_matrix, sim_mappings = load_similarity_artifacts()
except FileNotFoundError as e:
    st.error(
        f"⚠️ Missing required file: `{e.filename}`\n\n"
        "Make sure you've run Phase 1 and Phase 3 notebooks first, and that "
        "the `processed/` and `models/` folders are present at the project root."
    )
    st.stop()

movieId_to_row = sim_mappings["movieId_to_row"]
row_to_movieId = sim_mappings["row_to_movieId"]


# ---------------------------------------------------------------------------
# Core similarity lookup, built on top of the precomputed matrix
# ---------------------------------------------------------------------------
def get_similar_movies(movie_id, n=50):
    if movie_id not in movieId_to_row:
        return pd.DataFrame()

    row_idx = movieId_to_row[movie_id]
    sim_scores = sim_matrix[row_idx]

    import numpy as np

    similar_indices = np.argsort(sim_scores)[::-1]
    similar_indices = similar_indices[similar_indices != row_idx][:n]

    result_movie_ids = [row_to_movieId[i] for i in similar_indices]
    result_scores = [sim_scores[i] for i in similar_indices]

    result_df = movies_full[movies_full["movieId"].isin(result_movie_ids)].copy()
    score_map = dict(zip(result_movie_ids, result_scores))
    result_df["similarity_score"] = result_df["movieId"].map(score_map)
    result_df = result_df.sort_values("similarity_score", ascending=False).reset_index(drop=True)

    return result_df


# =============================================================================
# 1. ITEM SELECTION MECHANISM
# =============================================================================
st.subheader("1. Select a Movie")

# Build a friendly display label: "Title (Year)"
movies_full["display_label"] = movies_full.apply(
    lambda r: f"{r['clean_title']} ({int(r['year'])})" if pd.notna(r["year"]) else r["clean_title"],
    axis=1,
)

movie_options = movies_full[["movieId", "display_label"]].sort_values("display_label")

selected_label = st.selectbox(
    "Search or choose a movie:",
    options=movie_options["display_label"].tolist(),
)

selected_movie_id = int(
    movie_options.loc[movie_options["display_label"] == selected_label, "movieId"].values[0]
)
selected_movie_row = movies_full[movies_full["movieId"] == selected_movie_id].iloc[0]

st.divider()

# =============================================================================
# 2. ITEM PROFILE (metadata)
# =============================================================================
st.subheader("2. Movie Profile")

movie_ratings = ratings_encoded[ratings_encoded["movieId"] == selected_movie_id]
n_ratings = len(movie_ratings)
avg_rating = movie_ratings["rating"].mean() if n_ratings > 0 else None

profile_col1, profile_col2 = st.columns([1, 3])

with profile_col1:
    st.markdown(
        """
        <div style="background-color: rgba(255,255,255,0.05); border-radius: 12px;
                    height: 220px; display:flex; align-items:center; justify-content:center;
                    border: 1px solid rgba(255,255,255,0.1);">
            <span style="font-size: 3rem;">🎬</span>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.caption("Poster preview (connect a TMDB API key to enable real posters)")

with profile_col2:
    st.markdown(f"### {selected_movie_row['clean_title']}")
    if pd.notna(selected_movie_row.get("year")):
        st.write(f"**Release Year:** {int(selected_movie_row['year'])}")
    st.write(f"**Genres:** {selected_movie_row['genres']}")

    if isinstance(selected_movie_row.get("all_tags"), str) and selected_movie_row["all_tags"].strip():
        st.write(f"**User Tags:** {selected_movie_row['all_tags']}")

    m1, m2 = st.columns(2)
    m1.metric("Number of Ratings", n_ratings)
    m2.metric("Average Rating", f"{avg_rating:.2f} ⭐" if avg_rating is not None else "N/A")

    with st.expander("🔗 External Links"):
        if pd.notna(selected_movie_row.get("imdbId")):
            imdb_id = str(int(selected_movie_row["imdbId"])).zfill(7)
            st.markdown(f"[View on IMDb](https://www.imdb.com/title/tt{imdb_id}/)")
        if pd.notna(selected_movie_row.get("tmdbId")):
            tmdb_id = int(selected_movie_row["tmdbId"])
            st.markdown(f"[View on TMDB](https://www.themoviedb.org/movie/{tmdb_id})")

st.divider()

# =============================================================================
# 3. TOP-N SIMILAR ITEMS + 4. NAVIGATION
# =============================================================================
st.subheader("3. Similar Movies")

col_n, col_btn = st.columns([1, 2])
with col_n:
    n_per_page = st.number_input(
        "Items per page (N):",
        min_value=1,
        max_value=50,
        value=10,
        step=1,
        key="item_page_n",
    )
with col_btn:
    st.write("")
    st.write("")
    refresh_clicked = st.button("🔍 Find Similar Movies", type="primary")

SIMILARITY_POOL_SIZE = 50
sim_state_key = f"sim_page_{selected_movie_id}"
sim_pool_key = f"sim_pool_{selected_movie_id}"

if sim_state_key not in st.session_state:
    st.session_state[sim_state_key] = 1

if refresh_clicked or sim_pool_key not in st.session_state:
    with st.spinner("Calculating similarity..."):
        similar_df = get_similar_movies(selected_movie_id, n=SIMILARITY_POOL_SIZE)
    st.session_state[sim_pool_key] = similar_df
    st.session_state[sim_state_key] = 1

st.divider()

if sim_pool_key in st.session_state and len(st.session_state[sim_pool_key]) > 0:
    sim_pool = st.session_state[sim_pool_key]

    page_df, total_pages, current_page = paginate(sim_pool, st.session_state[sim_state_key], n_per_page)
    st.session_state[sim_state_key] = current_page

    for _, row in page_df.iterrows():
        with st.container(border=True):
            rc1, rc2 = st.columns([5, 1])
            with rc1:
                year_str = f" ({int(row['year'])})" if pd.notna(row.get("year")) else ""
                st.markdown(f"**{row['clean_title']}**{year_str}")
                st.caption(f"🎭 {row['genres']}")
            with rc2:
                st.metric(
                    "Similarity",
                    f"{row['similarity_score']:.2f}",
                    label_visibility="collapsed",
                )

    st.divider()

    st.caption("📖 Navigate Similar Movies")
    nav_col1, nav_col2, nav_col3, nav_col4 = st.columns([1, 1, 2, 1])

    with nav_col1:
        if st.button("⬅️ Previous", disabled=(current_page <= 1), use_container_width=True, key="item_prev"):
            st.session_state[sim_state_key] = max(1, current_page - 1)
            st.rerun()

    with nav_col2:
        if st.button("Next ➡️", disabled=(current_page >= total_pages), use_container_width=True, key="item_next"):
            st.session_state[sim_state_key] = min(total_pages, current_page + 1)
            st.rerun()

    with nav_col3:
        jump_to_page = st.number_input(
            "Go to page:",
            min_value=1,
            max_value=total_pages,
            value=current_page,
            step=1,
            label_visibility="collapsed",
            key="item_jump",
        )
        if jump_to_page != current_page:
            st.session_state[sim_state_key] = jump_to_page
            st.rerun()

    with nav_col4:
        st.markdown(f"**Page {current_page} / {total_pages}**")

else:
    st.info("👆 Click **Find Similar Movies** to see Top-N similar titles for this movie.")
