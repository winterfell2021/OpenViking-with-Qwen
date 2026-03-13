"""Full flow test: Embedding → VikingDB upsert → vector search.

Usage:
    pip install requests
    python tests/test_vikingdb_embedding.py

Requires:
    - Embedding service running on localhost:8000
    - OpenViking VikingDB server running on localhost:1933
"""

import sys
import time

import requests

EMBED_URL = "http://localhost:8000"
VIKING_URL = "http://localhost:1933"

# Bypass any local HTTP proxy (e.g. clash/v2ray) for localhost traffic
NO_PROXY = {"http": None, "https": None}

COLLECTION_NAME = "embed_test"
INDEX_NAME = "embed_test_idx"
PROJECT_NAME = "default"

SAMPLE_TEXTS = [
    "OpenViking is an agent-native context database for AI applications.",
    "Python is a popular programming language for machine learning.",
    "Docker containers provide lightweight virtualization for deployment.",
    "Vector databases enable efficient similarity search over embeddings.",
    "The quick brown fox jumps over the lazy dog.",
]

QUERY_TEXT = "vector similarity search engine"


def step(n: int, title: str):
    print(f"\n{'='*60}")
    print(f"  Step {n}: {title}")
    print(f"{'='*60}")


def embed(texts: list[str]) -> tuple[list[list[float]], int]:
    """Call embedding service, return (embeddings, dim)."""
    resp = requests.post(f"{EMBED_URL}/embed", json={"input": texts}, timeout=60, proxies=NO_PROXY)
    resp.raise_for_status()
    data = resp.json()
    return data["embeddings"], data["dim"]


def main():
    # ------------------------------------------------------------------
    # Step 1: Embed sample texts
    # ------------------------------------------------------------------
    step(1, "Embed sample texts")
    embeddings, dim = embed(SAMPLE_TEXTS)
    print(f"  Embedded {len(embeddings)} texts, dim={dim}")
    assert len(embeddings) == len(SAMPLE_TEXTS), "embedding count mismatch"

    # ------------------------------------------------------------------
    # Step 2: Create collection
    # ------------------------------------------------------------------
    step(2, "Create collection")

    # Clean up if exists from a previous run
    requests.post(
        f"{VIKING_URL}/DeleteVikingdbCollection",
        json={"CollectionName": COLLECTION_NAME, "ProjectName": PROJECT_NAME},
        timeout=10,
        proxies=NO_PROXY,
    )
    time.sleep(0.5)

    resp = requests.post(
        f"{VIKING_URL}/CreateVikingdbCollection",
        json={
            "CollectionName": COLLECTION_NAME,
            "ProjectName": PROJECT_NAME,
            "Description": "Embedding full flow test",
            "Fields": [
                {"FieldName": "id", "FieldType": "string", "IsPrimaryKey": True},
                {"FieldName": "text", "FieldType": "string"},
                {"FieldName": "vec", "FieldType": "vector", "Dim": dim},
            ],
        },
        timeout=10,
        proxies=NO_PROXY,
    )
    resp.raise_for_status()
    body = resp.json()
    print(f"  Response: code={body['code']}, message={body['message']}")
    assert body["code"] == 0, f"CreateCollection failed: {body}"

    # ------------------------------------------------------------------
    # Step 3: Create HNSW cosine index
    # ------------------------------------------------------------------
    step(3, "Create index")
    resp = requests.post(
        f"{VIKING_URL}/CreateVikingdbIndex",
        json={
            "CollectionName": COLLECTION_NAME,
            "ProjectName": PROJECT_NAME,
            "IndexName": INDEX_NAME,
            "VectorIndex": {
                "IndexType": "flat",
                "Distance": "cosine",
                "Quant": "int8",
            },
            "ScalarIndex": [],
        },
        timeout=10,
        proxies=NO_PROXY,
    )
    resp.raise_for_status()
    body = resp.json()
    print(f"  Response: code={body['code']}, message={body['message']}")
    assert body["code"] == 0, f"CreateIndex failed: {body}"

    # ------------------------------------------------------------------
    # Step 4: Upsert data
    # ------------------------------------------------------------------
    step(4, "Upsert data")
    records = [
        {"id": f"doc_{i}", "text": text, "vec": emb}
        for i, (text, emb) in enumerate(zip(SAMPLE_TEXTS, embeddings))
    ]
    resp = requests.post(
        f"{VIKING_URL}/api/vikingdb/data/upsert",
        json={
            "collection_name": COLLECTION_NAME,
            "project": PROJECT_NAME,
            "fields": records,
        },
        timeout=10,
        proxies=NO_PROXY,
    )
    resp.raise_for_status()
    body = resp.json()
    print(f"  Response: code={body['code']}, message={body['message']}")
    assert body["code"] == 0, f"Upsert failed: {body}"

    # ------------------------------------------------------------------
    # Step 5: Search by vector
    # ------------------------------------------------------------------
    step(5, "Search by vector")
    query_emb, _ = embed([QUERY_TEXT])
    resp = requests.post(
        f"{VIKING_URL}/api/vikingdb/data/search/vector",
        json={
            "collection_name": COLLECTION_NAME,
            "project": PROJECT_NAME,
            "index_name": INDEX_NAME,
            "dense_vector": query_emb[0],
            "output_fields": ["id", "text"],
            "limit": 3,
        },
        timeout=10,
        proxies=NO_PROXY,
    )
    resp.raise_for_status()
    body = resp.json()
    print(f"  Response: code={body['code']}, message={body['message']}")
    assert body["code"] == 0, f"Search failed: {body}"

    # ------------------------------------------------------------------
    # Step 6: Verify results
    # ------------------------------------------------------------------
    step(6, "Verify results")
    results = body.get("data", {})
    result_list = results.get("data", [])
    print(f"  Query: \"{QUERY_TEXT}\"")
    print(f"  Returned {len(result_list)} results:")
    for i, item in enumerate(result_list):
        fields = item.get("fields", item)
        score = item.get("score", "N/A")
        print(f"    [{i+1}] score={score}  text={fields.get('text', 'N/A')}")

    assert len(result_list) > 0, "Search returned no results"
    print("\n  All steps passed!")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"\n  FAILED: {e}", file=sys.stderr)
        sys.exit(1)
