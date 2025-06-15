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
import faiss  # For efficient similarity search and clustering of dense vectors


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
# SEARCH FUNCTION: TOKEN CORRECTION + FAISS
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
    # Step 2 & 3: Semantic scoring using FAISS over all courses
    # -----------------------------------------
    # Prepare course lists and embeddings
    all_course_names = list(set(mini_title_to_embedding) & set(mpnet_course_to_embedding))
    mini_embeddings = np.array([mini_title_to_embedding[c] for c in all_course_names]).astype('float32')
    mpnet_embeddings = np.array([mpnet_course_to_embedding[c] for c in all_course_names]).astype('float32')

    # Normalize for cosine similarity
    mini_embeddings /= np.linalg.norm(mini_embeddings, axis=1, keepdims=True)
    mpnet_embeddings /= np.linalg.norm(mpnet_embeddings, axis=1, keepdims=True)

    # Build FAISS indices
    mini_index = faiss.IndexFlatIP(mini_embeddings.shape[1])
    mpnet_index = faiss.IndexFlatIP(mpnet_embeddings.shape[1])
    mini_index.add(mini_embeddings)
    mpnet_index.add(mpnet_embeddings)

    # Encode and normalize query
    query_emb_mini = mini_model.encode([best_phrase])[0].astype('float32').reshape(1, -1)
    query_emb_mpnet = mpnet_model.encode([best_phrase])[0].astype('float32').reshape(1, -1)
    query_emb_mini /= np.linalg.norm(query_emb_mini, axis=1, keepdims=True)
    query_emb_mpnet /= np.linalg.norm(query_emb_mpnet, axis=1, keepdims=True)

    # Search top rerank_pool_size candidates from both models
    _, mini_idx = mini_index.search(query_emb_mini, rerank_pool_size)
    _, mpnet_idx = mpnet_index.search(query_emb_mpnet, rerank_pool_size)

    # Union of indices from both models
    candidate_indices = set(mini_idx[0]).union(set(mpnet_idx[0]))
    candidate_names = [all_course_names[i] for i in candidate_indices]

    # Get embeddings for candidates
    mini_cand_emb = np.array([mini_title_to_embedding[c] for c in candidate_names])
    mpnet_cand_emb = np.array([mpnet_course_to_embedding[c] for c in candidate_names])

    # Normalize again for safety
    mini_cand_emb = mini_cand_emb / np.linalg.norm(mini_cand_emb, axis=1, keepdims=True)
    mpnet_cand_emb = mpnet_cand_emb / np.linalg.norm(mpnet_cand_emb, axis=1, keepdims=True)

    # Compute cosine similarity
    sim_mini = (query_emb_mini @ mini_cand_emb.T).flatten()
    sim_mpnet = (query_emb_mpnet @ mpnet_cand_emb.T).flatten()

    # Combine scores
    alpha = 0.6 if len(query.strip().split()) <= 3 else 0.4
    combined_scores = alpha * sim_mini + (1 - alpha) * sim_mpnet


    # -----------------------------------------
    # Step 4: Rerank by maximum overlap with best_phrase tokens
    # -----------------------------------------
    corrected_tokens = set(re.findall(r'\w+', best_phrase.lower()))
    # Sort by (number of overlapping tokens, combined score)
    final_ranked = sorted(
        zip(candidate_names, combined_scores),
        key=lambda x: (len(set(re.findall(r'\w+', x[0].lower())) & corrected_tokens), x[1]),
        reverse=True
    )[:top_k]

    print(f"\n🔍 Top {top_k} results for: \"{query}\"")
    for i, (name, score) in enumerate(final_ranked, 1):
        print(f"{i}. {name} — Score: {score:.4f}")

    return [name for name, _ in final_ranked]




# # -----------------------------------------
# # RUN EXAMPLE SEARCH
# # -----------------------------------------
# results = dual_model_search_with_title_embeddings(
#     query="centrifugal pumps",
#     mini_model=model_st,
#     mpnet_model=semantic_model,
#     mini_title_to_embedding=mini_title_to_embedding,
#     mpnet_course_to_embedding=mpnet_course_to_embedding,
#     top_k=10
# )

# print("\nResults:", results)
