"""
pages/1_User_Page.py
=====================
The User Page of the CineMatch dashboard.

Required UI components (per project spec):
  - User selection mechanism
  - User history view (with item details)
  - Input for N (recommendations per page)
  - Top-N recommendations list
  - Recommendation section navigation (next/previous/specific page)
"""

import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import streamlit as st
import pandas as pd
from utils import (
    load_mappings,
    load_movies_full,
    load_ratings_encoded,
    load_ncf_model,
    get_top_n_recommendations,
    movie_idx_to_details,
    paginate,
)

st.set_page_config(page_title="CineMatch — User Page", page_icon="👤", layout="wide")

st.title("👤 User Page")
st.caption("Select a user to explore their history and get personalized recommendations.")

# ---------------------------------------------------------------------------
# Load all artifacts (cached — only loads from disk once per session)
# ---------------------------------------------------------------------------
try:
    mappings = load_mappings()
    movies_full = load_movies_full()
    ratings_encoded = load_ratings_encoded()
    model, model_config = load_ncf_model()
except FileNotFoundError as e:
    st.error(
        f"⚠️ Missing required file: `{e.filename}`\n\n"
        "Make sure you've run Phase 1, 2 and 3 notebooks first, and that "
        "the `processed/` and `models/` folders are present next to this dashboard."
    )
    st.stop()

idx_to_user = mappings["idx_to_user"]
user_to_idx = mappings["user_to_idx"]
idx_to_movie = mappings["idx_to_movie"]
n_movies = mappings["n_movies"]

# Build a quick lookup: movieId -> title (used everywhere below)
movieId_to_title = dict(zip(movies_full["movieId"], movies_full["clean_title"]))


# =============================================================================
# 1. USER SELECTION MECHANISM
# =============================================================================
st.subheader("1. Select a User")

all_user_ids = sorted(user_to_idx.keys())

selected_user_id = st.selectbox(
    "Choose a user from the dataset:",
    options=all_user_ids,
    format_func=lambda uid: f"User #{uid}",
)

selected_user_idx = user_to_idx[selected_user_id]

# Quick stats about this user
user_ratings = ratings_encoded[ratings_encoded["userId"] == selected_user_id].copy()
n_user_ratings = len(user_ratings)
avg_user_rating = user_ratings["rating"].mean() if n_user_ratings > 0 else 0.0

c1, c2, c3 = st.columns(3)
c1.metric("Movies Rated", n_user_ratings)
c2.metric("Average Rating", f"{avg_user_rating:.2f} ⭐")
c3.metric("User Index (model space)", selected_user_idx)

st.divider()

# =============================================================================
# 2. USER HISTORY VIEW
# =============================================================================
st.subheader("2. Rating History")

if n_user_ratings == 0:
    st.warning("This user has no ratings in the dataset.")
else:
    history_df = user_ratings.merge(
        movies_full[["movieId", "clean_title", "genres", "year"]],
        on="movieId",
        how="left",
    )
    history_df = history_df.sort_values("rating", ascending=False)[
        ["clean_title", "genres", "year", "rating"]
    ].rename(
        columns={
            "clean_title": "Title",
            "genres": "Genres",
            "year": "Year",
            "rating": "Rating",
        }
    )

    st.dataframe(
        history_df,
        use_container_width=True,
        hide_index=True,
        height=300,
    )

st.divider()

# =============================================================================
# 3. INPUT FOR N (recommendations per page) + Generate
# =============================================================================
st.subheader("3. Get Recommendations")

col_n, col_btn = st.columns([1, 2])

with col_n:
    n_per_page = st.number_input(
        "Items per page (N):",
        min_value=1,
        max_value=50,
        value=10,
        step=1,
        help="How many recommended movies to show per page.",
    )

with col_btn:
    st.write("")  # spacing
    st.write("")
    generate_clicked = st.button("🎯 Generate Recommendations", type="primary")

# We compute a larger candidate pool once (e.g. top 100), then paginate
# through it client-side — fast navigation without recomputing predictions
# every time the user clicks "next".
RECOMMENDATION_POOL_SIZE = 100

# Reset pagination state if the user changes selection
state_key = f"reco_page_{selected_user_id}"
if state_key not in st.session_state:
    st.session_state[state_key] = 1

if generate_clicked or f"reco_pool_{selected_user_id}" not in st.session_state:
    watched_movie_indices = user_ratings["movie_idx"].values
    with st.spinner("Predicting ratings for unwatched movies..."):
        recommendations = get_top_n_recommendations(
            model=model,
            user_idx=selected_user_idx,
            n_movies=n_movies,
            watched_movie_indices=watched_movie_indices,
            n=RECOMMENDATION_POOL_SIZE,
        )
    if recommendations:
        movie_idx_list = [r[0] for r in recommendations]
        scores = [r[1] for r in recommendations]
        reco_df = movie_idx_to_details(movie_idx_list, idx_to_movie, movies_full, scores=scores)
    else:
        reco_df = pd.DataFrame()

    st.session_state[f"reco_pool_{selected_user_id}"] = reco_df
    st.session_state[state_key] = 1  # reset to page 1 on new generation

st.divider()

# =============================================================================
# 4. TOP-N RECOMMENDATIONS LIST + 5. NAVIGATION
# =============================================================================
st.subheader("4. Top-N Recommendations")

pool_key = f"reco_pool_{selected_user_id}"

if pool_key in st.session_state and len(st.session_state[pool_key]) > 0:
    reco_pool = st.session_state[pool_key]

    page_df, total_pages, current_page = paginate(
        reco_pool, st.session_state[state_key], n_per_page
    )
    st.session_state[state_key] = current_page  # clamp if out of range

    # --- Display the recommendation cards ---
    for _, row in page_df.iterrows():
        with st.container(border=True):
            rc1, rc2 = st.columns([5, 1])
            with rc1:
                st.markdown(f"**{row['clean_title']}** ({int(row['year']) if pd.notna(row['year']) else 'N/A'})")
                st.caption(f"🎭 {row['genres']}")
            with rc2:
                st.metric("Predicted", f"{row['predicted_rating']:.2f} ⭐", label_visibility="collapsed")

    st.divider()

    # --- Navigation: previous / page selector / next ---
    st.caption("📖 Navigate Recommendations")
    nav_col1, nav_col2, nav_col3, nav_col4 = st.columns([1, 1, 2, 1])

    with nav_col1:
        if st.button("⬅️ Previous", disabled=(current_page <= 1), use_container_width=True):
            st.session_state[state_key] = max(1, current_page - 1)
            st.rerun()

    with nav_col2:
        if st.button("Next ➡️", disabled=(current_page >= total_pages), use_container_width=True):
            st.session_state[state_key] = min(total_pages, current_page + 1)
            st.rerun()

    with nav_col3:
        jump_to_page = st.number_input(
            "Go to page:",
            min_value=1,
            max_value=total_pages,
            value=current_page,
            step=1,
            label_visibility="collapsed",
        )
        if jump_to_page != current_page:
            st.session_state[state_key] = jump_to_page
            st.rerun()

    with nav_col4:
        st.markdown(f"**Page {current_page} / {total_pages}**")

else:
    st.info("👆 Click **Generate Recommendations** to see personalized Top-N movies for this user.")
