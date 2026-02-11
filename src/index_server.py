import os
from dotenv import load_dotenv

import pathway as pw
from pathway.xpacks.llm.document_store import DocumentStore
from pathway.xpacks.llm.splitters import TokenCountSplitter
from pathway.xpacks.llm.servers import DocumentStoreServer

# ✅ BM25 (tantivy) factory
from pathway.stdlib.indexing.bm25 import TantivyBM25Factory

load_dotenv()

PATHWAY_HOST = os.getenv("PATHWAY_HOST", "127.0.0.1")
PATHWAY_PORT = int(os.getenv("PATHWAY_PORT", "8765"))
BOOKS_DIR = os.getenv("BOOKS_DIR", "./data/books")


def txt_parser(data: bytes):
    text = data.decode("utf-8", errors="ignore")
    return [(text, {})]


def main():
    # snapshot only (prevents re-index loops)
    docs = pw.io.fs.read(
        BOOKS_DIR,
        format="binary",
        with_metadata=True,
        mode="static",
    )

    # fewer chunks => less overhead
    splitter = TokenCountSplitter(min_tokens=900, max_tokens=1400)

    # ✅ disk-backed BM25 to reduce RAM
    retriever_factory = TantivyBM25Factory(
        ram_budget=50_000_000,      # ~50MB memory budget
        in_memory_index=False,      # store index on disk
    )

    store = DocumentStore(
        docs=docs,
        retriever_factory=retriever_factory,
        parser=txt_parser,
        splitter=splitter,
    )

    server = DocumentStoreServer(
        host=PATHWAY_HOST,
        port=PATHWAY_PORT,
        document_store=store,
    )
    server.run(threaded=True, with_cache=False)


if __name__ == "__main__":
    main()
