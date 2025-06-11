# semantic_search_poc.py

"""
POC: Semantic Course Search Engine using SentenceTransformer Embeddings + Manual Cosine Similarity

This script creates a dual-embedding semantic search engine using MiniLM and MPNet sentence transformers.
It:
- Loads course data
- Generates semantic embeddings
- Saves them to disk
- Performs dual-model search using cosine similarity and token-corrected reranking

Used for prototyping. Not optimized for persistent or scalable deployment.
"""

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
# LOAD MODELS FROM DISK
# -----------------------------------------
model_st=SentenceTransformer("models/mini_model") # load the model
semantic_model = SentenceTransformer("models/mpnet_model")  # load the model
# -----------------------------------------
# LOAD EMBEDDINGS FROM DISK (Instead of recomputing)
# -----------------------------------------

# Load MiniLM title-based embeddings
mini_title_embeddings = np.load("mini_title_embeddings.npy")
mini_title_names = pd.read_csv("mini_title_names.csv")["course_name_norm"].tolist()

with open("mini_title_to_embedding.pkl", "rb") as f:
    mini_title_to_embedding = pickle.load(f)

# Load MPNet full-text embeddings
mpnet_course_embeddings = np.load("mpnet_course_embeddings.npy")
mpnet_course_names = pd.read_csv("mpnet_course_names.csv")["course_name_norm"].tolist()

with open("mpnet_course_to_embedding.pkl", "rb") as f:
    mpnet_course_to_embedding = pickle.load(f)

# Sanity check (optional)
assert len(mini_title_names) == len(mini_title_embeddings)
assert len(mpnet_course_names) == len(mpnet_course_embeddings)

# -----------------------------------------
# SEARCH FUNCTION: TOKEN CORRECTION + COSINE SIMILARITY
# -----------------------------------------
def dual_model_search_with_title_embeddings(query,
                                            mini_model,
                                            mpnet_model,
                                            mini_title_to_embedding,
                                            mpnet_course_to_embedding,
                                            top_k=5,
                                            max_token_corrections=2,
                                            token_match_threshold=75,
                                            rerank_pool_size=30):
    """
    Perform semantic search with:
    - Typo correction (using RapidFuzz)
    - Dual-model scoring (MiniLM + MPNet)
    - Semantic + lexical reranking

    Parameters:
        query: input search string
        mini_model: MiniLM model for title embedding
        mpnet_model: MPNet model for full description embedding
        mini_title_to_embedding: dict of course → MiniLM vector
        mpnet_course_to_embedding: dict of course → MPNet vector
        top_k: how many results to return
        max_token_corrections: max fuzzy matches per word
        token_match_threshold: fuzzy matching threshold
        rerank_pool_size: how many to consider before final reranking
    """

    # Normalize input query and tokenize into lowercase words
    def normalize(text): return text.strip().lower()
    def tokenize(text): return set(re.findall(r'\w+', normalize(text)))

    # Blending factor for score fusion: short queries rely more on title
    alpha = 0.6 if len(query.strip().split()) <= 3 else 0.4

    # Get all course names shared by both embedding sources
    all_course_names = list(set(mini_title_to_embedding) & set(mpnet_course_to_embedding))

    # Build token set from all course names for fuzzy matching
    all_tokens = set(word for name in all_course_names for word in tokenize(name))

    # -----------------------------------------
    # Step 1: Correct individual tokens using fuzzy matching
    # -----------------------------------------
    original_tokens = query.strip().split()
    token_options = []  # list of correction options for each word
    for token in original_tokens:
        corrections = process.extract(token, all_tokens, scorer=fuzz.ratio, limit=max_token_corrections)
        corrected = [word for word, score, _ in corrections if score >= token_match_threshold]
        token_options.append(corrected or [token])

    # Generate all possible corrected phrases from token combinations
    corrected_combinations = list(product(*token_options))
    corrected_phrases = [" ".join(words) for words in corrected_combinations]
    print(f"Suggested corrected phrases: {corrected_phrases[:2]}")

    # Choose the most common phrase found in course titles
    phrase_freq = [sum(phrase in normalize(title) for title in all_course_names)
                   for phrase in corrected_phrases]
    best_phrase = corrected_phrases[np.argmax(phrase_freq)]
    print(f"Using most common corrected phrase: \"{best_phrase}\"")

    # -----------------------------------------
    # Step 2: Find matching courses by token overlap
    # -----------------------------------------
    matching_courses = []
    for course in all_course_names:
        course_tokens = tokenize(course)
        for phrase in corrected_phrases:
            phrase_tokens = tokenize(phrase)
            if phrase_tokens & course_tokens:
                matching_courses.append(course)
                break

    if not matching_courses:
        print("❌ No token matches found, fallback to full list.")
        matching_courses = all_course_names[:top_k * 10]

    # -----------------------------------------
    # Step 3: Semantic scoring using cosine similarity
    # -----------------------------------------
    query_emb_mini = mini_model.encode([best_phrase])[0].reshape(1, -1)
    query_emb_mpnet = mpnet_model.encode([best_phrase])[0].reshape(1, -1)

    mini_embeddings = np.array([mini_title_to_embedding[c] for c in matching_courses])
    mpnet_embeddings = np.array([mpnet_course_to_embedding[c] for c in matching_courses])

    sim_mini = cosine_similarity(query_emb_mini, mini_embeddings).flatten()
    sim_mpnet = cosine_similarity(query_emb_mpnet, mpnet_embeddings).flatten()

    # Combine MiniLM and MPNet scores
    combined_scores = alpha * sim_mini + (1 - alpha) * sim_mpnet

    # -----------------------------------------
    # Step 4: Rerank by overlap with corrected tokens
    # -----------------------------------------
    corrected_tokens = tokenize(best_phrase)

    # Sort by semantic similarity
    top_n_ranked = sorted(zip(matching_courses, combined_scores), key=lambda x: x[1], reverse=True)[:rerank_pool_size]

    # Then re-rank by number of word overlaps with the corrected phrase
    final_ranked = sorted(top_n_ranked,
                          key=lambda x: (len(tokenize(x[0]) & corrected_tokens), x[1]),
                          reverse=True)[:top_k]

    print(f"\n🔍 Top {top_k} results for: \"{query}\"")
    for i, (name, score) in enumerate(final_ranked, 1):
        print(f"{i}. {name} — Score: {score:.4f}")

    return [name for name, _ in final_ranked]

# -----------------------------------------
# RUN EXAMPLE SEARCH
# -----------------------------------------
results = dual_model_search_with_title_embeddings(
    query="data visualization",
    mini_model=model_st,
    mpnet_model=semantic_model,
    mini_title_to_embedding=mini_title_to_embedding,
    mpnet_course_to_embedding=mpnet_course_to_embedding,
    top_k=5
)

print("\nResults:", results)
