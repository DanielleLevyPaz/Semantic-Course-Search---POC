# Semantic-Course-Search---POC
This project is a proof of concept (POC) for a semantic course search engine that uses SentenceTransformer-based embeddings and manual vector similarity matching to retrieve relevant courses based on natural language queries.
##### 🧠 Key Features

Uses MiniLM and MPNet SentenceTransformers for embedding course titles and descriptions.

Performs semantic search using cosine similarity.

Adds fuzzy matching and token correction with RapidFuzz to improve recall for user queries.

Combines both title and full text embeddings for robust ranking.

Fully in-memory and file-based; suitable for prototyping.

## MOO:
### Step 1: Normalize & Prepare Vocabulary
* Collects all course titles present in both MiniLM‐ and MPNet‐based embedding maps.
* Lowercases and tokenizes them (\w+ regex) to build a set of known tokens.
* This token set drives typo corrections grounded in your actual course catalog.

### Step 2: Correct Typos & Generate Phrases
* Splits the query into words (tokens).
* For each token, uses RapidFuzz to retrieve up to N similar tokens from the vocabulary (threshold ≥ 75%).
* Constructs every combination of corrected tokens into candidate phrases (e.g., "data scienc" → ["data science", "data scientific"]).

### Step 3: Select the Best Corrected Phrase
* Counts how often each candidate phrase (as a substring) appears in the titles.
* Chooses the phrase with the highest frequency in your catalog—most likely what the user meant.
Example: picks "data science" over "data scientific"

### Step 4: Find Matching Courses
* Scans all course titles and retains those that share any token with any corrected phrase.
      For every course title, it:
          -Tokenizes the title into words.
          -Tokenizes every corrected phrase into words.
          -Keeps the course only if at least one token from a corrected phrase appears in the title.
(If no titles match, falls back to the top (top_k * 10) courses (semantic‐only mode))

### Step 5: Compute Semantic Similarity

* Encodes the original query using both MiniLM and MPNet models.
* Retrieves embeddings for the filtered candidates (or, if no candidates matched tokens, for a fallback pool of the top top_k * 10 courses).
* Computes cosine similarities: sim_mini and sim_mpnet.
* Blends them with an alpha weight:
    * alpha = 0.6 when query ≤ 3 words (favoring MiniLM precision)
    * alpha = 0.4 when query > 3 words (favoring MPNet context)
      Combined score = alpha * sim_mini + (1 - alpha) * sim_mpnet.

Fallback strategic:
If no titles share any corrected tokens, the engine skips token filtering and instead semantically scores the top top_k * 10 courses from the full catalog—ensuring results even for queries with completely novel or rare words.

### Step 6: Final Reranking by Token Match
* Takes the top rerank_pool_size candidates by combined score.
* Reranks those by the number of tokens they share with the chosen corrected phrase, then by combined score.

Ensures that semantically strong results containing the user’s key terms bubble to the top.

### Step 7: Print Final Results
* Prints the top-K course titles with their final scores.
* Returns the list of recommended courses.

### <u>Advantegest of this search engine:</u>
* Corrects typos dynamically (e.g. "scienc" → "science")
* Deep semantic understanding via dual LLM embeddings (MiniLM + MPNet)
* Prefers familiar, real-world terms from the data
* Balanced ranking that values both meaning and exact word overlap—just like a human would



## 📁 Files

embedding_model.py #precomputng embeddings , save the model and the embeddings
semantic_search_poc.py #Main POC script for generating embeddings and running the dual-model search

course_details.csv #Input dataset containing course names and descriptions

models/mini_model #MiniLM model directory (locally saved SentenceTransformer)

models/mpnet_model #MPNet model directory (locally saved SentenceTransformer)

*.pkl, *.npy, *.csv #Saved embeddings and dictionaries for reuse

## ⚙️ How It Works

Embeddings are generated for each course using:

MiniLM on course_name
MPNet on title + description (in a combined column)
All embeddings are saved as .npy arrays and .pkl dictionaries.

During search:

A fuzzy-corrected query is generated using token-level matching.
Candidate course names are filtered based on overlap with corrected phrases.
The best match is found using cosine similarity over both embedding sets.
A final reranking step prioritizes lexical matches.

## 🏃‍♀️ Running the Code

Prerequisites:

pip install sentence-transformers pandas numpy rapidfuzz scikit-learn

Execution:

python semantic_search_poc.py

Expected output: Top ranked course names printed with similarity scores.

## 📦 Limitations

* This version uses in-memory dictionaries for embedding lookup.
* Not optimized for scale or production deployment.
* No persistent index — search is recomputed per run.
To upgrade this to a production-ready version, consider using ChromaDB or Milvus for persistent, indexed vector storage.

##### 🧱 Next Step

Next: Convert this logic into a production-ready setup using ChromaDB.
