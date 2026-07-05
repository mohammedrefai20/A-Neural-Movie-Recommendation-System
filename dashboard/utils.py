"""
utils.py
========
Shared utilities for the CineMatch dashboard.
Handles loading of all artifacts produced in Phases 1-3:
- Phase 1: processed data + mappings
- Phase 2: trained NCF model
- Phase 3: item similarity matrix

This module is imported by both the User Page and Item Page so the
loading logic (and Streamlit caching) lives in exactly one place.
"""

import os
import pickle
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import streamlit as st

PROCESSED_DIR = "processed"
MODEL_DIR = "models"


# ---------------------------------------------------------------------------
# NCF Model Architecture (must match Phase 2 exactly so weights load correctly)
# ---------------------------------------------------------------------------
class NCF(nn.Module):
    def __init__(self, n_users, n_movies, embedding_dim=32, mlp_layers=[64, 32, 16]):
        super(NCF, self).__init__()

        self.user_embedding_mlp = nn.Embedding(n_users, embedding_dim)
        self.movie_embedding_mlp = nn.Embedding(n_movies, embedding_dim)

        self.user_embedding_mf = nn.Embedding(n_users, embedding_dim)
        self.movie_embedding_mf = nn.Embedding(n_movies, embedding_dim)

        mlp_modules = []
        input_size = embedding_dim * 2
        for layer_size in mlp_layers:
            mlp_modules.append(nn.Linear(input_size, layer_size))
            mlp_modules.append(nn.ReLU())
            mlp_modules.append(nn.Dropout(0.2))
            input_size = layer_size
        self.mlp = nn.Sequential(*mlp_modules)

        self.output_layer = nn.Linear(mlp_layers[-1] + embedding_dim, 1)

    def forward(self, user_idx, movie_idx):
        user_mf = self.user_embedding_mf(user_idx)
        movie_mf = self.movie_embedding_mf(movie_idx)
        mf_vector = user_mf * movie_mf

        user_mlp = self.user_embedding_mlp(user_idx)
        movie_mlp = self.movie_embedding_mlp(movie_idx)
        mlp_input = torch.cat([user_mlp, movie_mlp], dim=-1)
        mlp_vector = self.mlp(mlp_input)

        combined = torch.cat([mlp_vector, mf_vector], dim=-1)
        output = self.output_layer(combined)
        return output.squeeze()


# ---------------------------------------------------------------------------
# Cached loaders — Streamlit caches these so files are only read from disk
# once per session, not on every user interaction (important for the
# similarity matrix, which can be large).
# ---------------------------------------------------------------------------

@st.cache_data
def load_mappings():
    with open(os.path.join(PROCESSED_DIR, "mappings.pkl"), "rb") as f:
        return pickle.load(f)


@st.cache_data
def load_movies_full():
    df = pd.read_csv(os.path.join(PROCESSED_DIR, "movies_full.csv"))
    return df


@st.cache_data
def load_ratings_encoded():
    return pd.read_csv(os.path.join(PROCESSED_DIR, "ratings_encoded.csv"))


@st.cache_resource
def load_ncf_model():
    with open(os.path.join(MODEL_DIR, "ncf_config.pkl"), "rb") as f:
        config = pickle.load(f)

    model = NCF(
        n_users=config["n_users"],
        n_movies=config["n_movies"],
        embedding_dim=config["embedding_dim"],
        mlp_layers=config["mlp_layers"],
    )
    state_dict = torch.load(
        os.path.join(MODEL_DIR, "ncf_model.pt"), map_location="cpu"
    )
    model.load_state_dict(state_dict)
    model.eval()
    return model, config


@st.cache_resource
def load_similarity_artifacts():
    loaded = np.load(os.path.join(MODEL_DIR, "cosine_sim_matrix.npz"))
    sim_matrix = loaded["sim_matrix"]

    with open(os.path.join(MODEL_DIR, "item_similarity_mappings.pkl"), "rb") as f:
        sim_mappings = pickle.load(f)

    return sim_matrix, sim_mappings


# ---------------------------------------------------------------------------
# Inference helpers
# ---------------------------------------------------------------------------

def get_top_n_recommendations(model, user_idx, n_movies, watched_movie_indices, n=10):
    """Predict ratings for all unwatched movies and return the Top-N (movie_idx, score) pairs."""
    model.eval()
    all_movie_indices = np.arange(n_movies)
    unwatched = np.setdiff1d(all_movie_indices, watched_movie_indices)

    if len(unwatched) == 0:
        return []

    with torch.no_grad():
        user_tensor = torch.tensor([user_idx] * len(unwatched), dtype=torch.long)
        movie_tensor = torch.tensor(unwatched, dtype=torch.long)
        preds = model(user_tensor, movie_tensor)
        preds = torch.clamp(preds, 0.5, 5.0).numpy()

    order = np.argsort(preds)[::-1][:n]
    top_movie_indices = unwatched[order]
    top_scores = preds[order]

    return list(zip(top_movie_indices, top_scores))


def movie_idx_to_details(movie_idx_list, idx_to_movie, movies_full, scores=None):
    """Convert a list of movie_idx (model space) into a display-ready DataFrame, preserving order."""
    movie_ids = [idx_to_movie[idx] for idx in movie_idx_list]
    order_map = {mid: i for i, mid in enumerate(movie_ids)}

    details = movies_full[movies_full["movieId"].isin(movie_ids)].copy()
    details["_order"] = details["movieId"].map(order_map)
    details = details.sort_values("_order").drop(columns="_order").reset_index(drop=True)

    if scores is not None:
        score_map = dict(zip(movie_ids, scores))
        details["predicted_rating"] = details["movieId"].map(score_map)

    return details


def paginate(df, page, page_size):
    """Generic pagination helper for any DataFrame. Returns (page_df, total_pages, page_clamped)."""
    total_items = len(df)
    total_pages = max(1, (total_items + page_size - 1) // page_size)
    page = max(1, min(page, total_pages))
    start = (page - 1) * page_size
    end = start + page_size
    return df.iloc[start:end].reset_index(drop=True), total_pages, page
