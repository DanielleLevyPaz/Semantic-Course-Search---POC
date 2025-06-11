# -----------------------------------------
# IMPORTS
# -----------------------------------------
import numpy as np  # For numerical array operations
import pandas as pd  # For loading and processing tabular data
from sentence_transformers import SentenceTransformer  # To load pretrained models and create embeddings
from sklearn.metrics.pairwise import cosine_similarity  # For computing similarity between vectors
from rapidfuzz import process, fuzz  # For fuzzy string matching (used for correcting user typos)
import re  # For tokenizing text using regular expressions
from itertools import product  # To compute all combinations of corrected token phrases
import pickle  # For saving and loading Python objects like dictionaries

# -----------------------------------------
# LOAD COURSE DATA
# -----------------------------------------
course_details = pd.read_csv("course_details.csv")  # Load course metadata with names and combined text

# Normalize course names for consistent lookup (lowercase + strip)
course_details["course_name_norm"] = course_details["course_name"].str.strip().str.lower()

# -----------------------------------------
# LOAD EMBEDDING MODELS
# -----------------------------------------
# Load MiniLM model that was saved locally for title-based embeddings
model_st = SentenceTransformer("all-MiniLM-L6-v2")

# Load MPNet model that was saved locally for richer semantic embeddings using title + description
semantic_model = SentenceTransformer("all-mpnet-base-v2")

# -----------------------------------------
# GENERATE EMBEDDINGS
# -----------------------------------------
# Compute sentence embeddings for course titles using MiniLM
mini_title_embeddings = model_st.encode(course_details["course_name"].tolist(), show_progress_bar=True)

# Build a dictionary: normalized course name → MiniLM vector
mini_title_to_embedding = dict(zip(course_details["course_name_norm"], mini_title_embeddings))

# Compute semantic embeddings for full course descriptions using MPNet
mpnet_course_embeddings = semantic_model.encode(course_details['combined'].tolist(), show_progress_bar=True)

# Build a dictionary: normalized course name → MPNet vector
mpnet_course_to_embedding = dict(zip(course_details["course_name_norm"], mpnet_course_embeddings))

# -----------------------------------------
# SAVE EMBEDDINGS TO DISK
# -----------------------------------------
# Save MiniLM title-based embeddings
np.save("mini_title_embeddings.npy", mini_title_embeddings)
pd.Series(course_details["course_name_norm"]).to_csv("mini_title_names.csv", index=False)  # Save names

# Save MiniLM dictionary as pickle
with open("mini_title_to_embedding.pkl", "wb") as f:
    pickle.dump(mini_title_to_embedding, f)

# Save MPNet full-text embeddings
np.save("mpnet_course_embeddings.npy", mpnet_course_embeddings)
pd.Series(course_details["course_name_norm"]).to_csv("mpnet_course_names.csv", index=False)

# Save MPNet dictionary as pickle
with open("mpnet_course_to_embedding.pkl", "wb") as f:
    pickle.dump(mpnet_course_to_embedding, f)


# -----------------------------
# SAVE MODELS TO DISK
# ----------------------------- 
semantic_model.save("models/mpnet_model")
model_st.save("models/mini_model")