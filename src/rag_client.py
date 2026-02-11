import requests
from typing import List, Optional

from config import RAG_SERVER_URL, BOOK_GLOBS


def book_glob(book_name: str) -> Optional[str]:
    return BOOK_GLOBS.get(book_name, None)


def retrieve_chunks(
    query: str,
    book_name: str,
    k: int = 12,
) -> List[str]:
    payload = {
        "query": query,
        "k": k,
        "metadata_filter": None,
        "filepath_globpattern": book_glob(book_name),
    }
    r = requests.post(f"{RAG_SERVER_URL}/v1/retrieve", json=payload, timeout=120)
    r.raise_for_status()
    data = r.json()

    # Pathway returns JSON with "result" in many builds; be defensive:
    results = data.get("result", data)

    chunks: List[str] = []
    if isinstance(results, list):
        for it in results:
            if isinstance(it, dict) and "text" in it:
                chunks.append(it["text"])
            else:
                chunks.append(str(it))
    return [c for c in chunks if c and c.strip()]
