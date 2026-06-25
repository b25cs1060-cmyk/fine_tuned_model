"""
Dataset Generator — builds instruction/response (Q&A) pairs from GitHub repos
using Groq (llama-3.3-70b-versatile) as the generator.

Fixes applied:
  1. Excludes virtualenv/dependency folders (my_venv, site-packages, node_modules,
     .git, __pycache__, etc.) so only the repo's own code gets used.
  2. Allocates a per-repo pair budget up front, so every repo gets covered
     instead of the first repo eating the entire TARGET_TOTAL_NEW_PAIRS.
  3. Lightweight question-similarity dedup per repo, so repetitive boilerplate
     (e.g. many near-identical product objects) doesn't generate 30 near-
     duplicate Q&A pairs.

Output format matches your existing dataset.json exactly:
{
    "query_id": int,
    "repo_name": str,
    "file_path": str,
    "question": str,
    "context": str,
    "answer": str
}

Run this locally (paths below are LOCAL paths, not Colab paths).
"""

import os
import json
import shutil
import time
import re
from dotenv import load_dotenv
from langchain_community.document_loaders import GitLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_groq import ChatGroq
from langchain_core.messages import SystemMessage, HumanMessage

# ---------------------------------------------------------------------------
# CONFIG
# ---------------------------------------------------------------------------

load_dotenv("/home/rudri/Documents/fine_tuning_llms/chunker/.env")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")


REPOS = [

    {
        "name": "fast_api",
        "url": "https://github.com/b25cs1060-cmyk/fastAPI-app",
        "branch": "main",
    },
    {
        "name": "traffic_bot",
        "url": "https://github.com/b25cs1060-cmyk/traffic_bot",
        "branch": "main",
    },
    {
        "name": "ICS-Project",
        "url": "https://github.com/MihirGharote/ICS-Project.git",
        "branch": "main",
    },
    {
        "name": "Insomniac-Hackathon",
        "url": "https://github.com/AishwaryaIITJ/Insomniac-Hackathon.git",
        "branch": "main",
    },
    {
        "name": "langgraph",
        "url": "https://github.com/b25cs1060-cmyk/langgraph.git",
        "branch": "main",
    },
]

FILE_EXTENSIONS = (".py", ".js", ".ts", ".md", ".txt", ".ipynb", ".html", ".css")

# Folder name fragments to exclude entirely — dependency/environment/build
# artifacts that are NOT part of the repo's actual source, even if they
# happen to live inside the cloned directory on disk.
EXCLUDED_PATH_FRAGMENTS = (
    "venv/", "/venv", "my_venv", ".venv",
    "site-packages",
    "node_modules",
    "__pycache__",
    ".git/",
    "dist/", "build/",
    ".egg-info",
)

CLONE_BASE_DIR = "/home/rudri/Documents/fine_tuning_llms/dataset_repos"
CHUNK_SIZE = 1800
CHUNK_OVERLAP = 100
PAIRS_PER_CHUNK = 3
TARGET_TOTAL_NEW_PAIRS = 100
STARTING_QUERY_ID = 25
OUTPUT_PATH = "/home/rudri/Documents/fine_tuning_llms/model/new_dataset_pairs.json"
SLEEP_BETWEEN_CALLS = 1.5

# Max number of pairs allowed per (repo, "question shape") bucket, to avoid
# the same boilerplate question being asked 30 times across near-identical
# chunks (e.g. repeated product objects in a data file).
MAX_PAIRS_PER_QUESTION_SIGNATURE = 3

llm = ChatGroq(model="llama-3.1-8b-instant", api_key=GROQ_API_KEY, temperature=0.4)


# ---------------------------------------------------------------------------
# STEP 1 — Clone + load + chunk each repo, excluding dependency folders
# ---------------------------------------------------------------------------


def load_and_chunk_repo(repo):
    repo_path = os.path.join(CLONE_BASE_DIR, repo["name"])

    if os.path.exists(repo_path):
        shutil.rmtree(repo_path)

    print(f"\nCloning {repo['name']} ...")
    ...

def is_excluded_path(file_path: str) -> bool:
    normalized = file_path.replace("\\", "/")
    return any(fragment in normalized for fragment in EXCLUDED_PATH_FRAGMENTS)


def load_and_chunk_repo(repo):
    repo_path = os.path.join(CLONE_BASE_DIR, repo["name"])
    print(f"\nCloning {repo['name']} ...")

    loader = GitLoader(
        repo_path=repo_path,
        clone_url=repo["url"],
        branch=repo["branch"],
        file_filter=lambda file_path: (
            file_path.endswith(FILE_EXTENSIONS) and not is_excluded_path(file_path)
        ),
    )
    docs = loader.load()
    print(f"  Loaded {len(docs)} files (after excluding venv/deps/etc).")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP
    )
    chunks = splitter.split_documents(docs)
    print(f"  Split into {len(chunks)} chunks.")

    return chunks


# ---------------------------------------------------------------------------
# STEP 2 — Prompt Groq to generate Q&A pairs for a single chunk
# ---------------------------------------------------------------------------

GENERATION_SYSTEM_PROMPT = """You are an expert software engineer creating a training dataset \
for a code-explanation chatbot. Given a snippet of code or documentation from a GitHub \
repository, generate clear, realistic question-answer pairs a developer might ask about it.

Rules:
- Questions should be specific to what the code actually does, not generic.
- Answers must be accurate, based only on the given snippet, and concise (1-3 sentences).
- Vary question style: "what does X do", "why is Y used", "what does Z return", "how does \
this work", "what is the purpose of...".
- Do NOT invent functionality that isn't shown in the snippet.
- Output ONLY valid JSON: a list of objects with keys "question" and "answer". No prose, \
no markdown fences, nothing else.
"""


def generate_pairs_for_chunk(chunk_text: str, n_pairs: int):
    user_prompt = f"""Code/content snippet:
---
{chunk_text}
---

Generate exactly {n_pairs} question-answer pairs about this snippet, as a JSON list."""

    try:
        response = llm.invoke(
            [
                SystemMessage(content=GENERATION_SYSTEM_PROMPT),
                HumanMessage(content=user_prompt),
            ]
        )
        raw = response.content.strip()
        raw = re.sub(r"^```json\s*|\s*```$", "", raw.strip(), flags=re.MULTILINE)

        pairs = json.loads(raw)
        if not isinstance(pairs, list):
            return []
        return [p for p in pairs if "question" in p and "answer" in p]

    except json.JSONDecodeError:
        print("  [skip] Could not parse JSON from model response.")
        return []
    except Exception as e:
        print(f"  [skip] Generation error: {e}")
        return []


# ---------------------------------------------------------------------------
# STEP 3 — Dedup helper: collapse a question to a coarse "signature" so we
# can cap how many near-identical questions we keep per repo.
# ---------------------------------------------------------------------------

def question_signature(question: str) -> str:
    q = question.lower().strip()
    q = re.sub(r"['\"`]", "", q)
    # Strip specific identifier-like tokens so "the 'rating' object" and
    # "the 'keywords' array" still collapse together if the surrounding
    # phrasing is otherwise identical.
    q = re.sub(r"\b\w+\b", lambda m: m.group(0) if len(m.group(0)) > 3 else "_", q)
    q = re.sub(r"\s+", " ", q)
    return q


# ---------------------------------------------------------------------------
# STEP 4 — Run generation with a per-repo budget, so every repo is covered
# ---------------------------------------------------------------------------

def build_dataset():
    all_new_examples = []
    query_id = STARTING_QUERY_ID

    per_repo_budget = TARGET_TOTAL_NEW_PAIRS // len(REPOS)
    print(f"Per-repo budget: ~{per_repo_budget} pairs each "
          f"({len(REPOS)} repos, target {TARGET_TOTAL_NEW_PAIRS} total)")

    for repo in REPOS:
        repo_examples = []
        signature_counts = {}

        chunks = load_and_chunk_repo(repo)

        for chunk in chunks:
            if len(repo_examples) >= per_repo_budget:
                break

            file_path = chunk.metadata.get("file_path", "unknown")
            pairs = generate_pairs_for_chunk(chunk.page_content, PAIRS_PER_CHUNK)

            kept_this_chunk = 0
            for pair in pairs:
                sig = question_signature(pair["question"])
                count = signature_counts.get(sig, 0)
                if count >= MAX_PAIRS_PER_QUESTION_SIGNATURE:
                    continue  # too many near-duplicates of this question shape already

                signature_counts[sig] = count + 1
                repo_examples.append(
                    {
                        "query_id": query_id,
                        "repo_name": repo["name"],
                        "file_path": file_path,
                        "question": pair["question"],
                        "context": chunk.page_content[:500],
                        "answer": pair["answer"],
                    }
                )
                query_id += 1
                kept_this_chunk += 1

                if len(repo_examples) >= per_repo_budget:
                    break

            print(f"  [{repo['name']}] {file_path} -> +{kept_this_chunk} pairs kept "
                  f"(repo total: {len(repo_examples)}/{per_repo_budget})")

            time.sleep(SLEEP_BETWEEN_CALLS)

        all_new_examples.extend(repo_examples)
        print(f"Finished {repo['name']}: {len(repo_examples)} pairs "
              f"(running total: {len(all_new_examples)})")

    return all_new_examples


# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    new_examples = build_dataset()

    print(f"\nGenerated {len(new_examples)} new Q&A pairs across {len(REPOS)} repos.")

    with open(OUTPUT_PATH, "w") as f:
        json.dump(new_examples, f, indent=4)

    print(f"Saved to {OUTPUT_PATH}")
    print("\nNext step: merge this with your existing 24-example dataset.json "
          "into one file before training.")