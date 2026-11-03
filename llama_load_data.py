import chromadb
import re
from llama_index.core import Document

from llama_index.core import (
    SimpleDirectoryReader,
    VectorStoreIndex,
    StorageContext,
    Settings,
)
from llama_index.core.node_parser import SentenceSplitter
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.vector_stores.chroma import ChromaVectorStore

DB_PATH = "./chroma_db"


def setup_embedding():
    """הגדרת מודל ה-Embeddings."""
    Settings.embed_model = HuggingFaceEmbedding(
        model_name="intfloat/multilingual-e5-small",
        query_instruction="query: ",
        text_instruction="passage: ",
    )


def load_document():
    """חלוקה לפי פסקאות, ולכל פסקה מוסיפים את שם החלק שלה."""
    raw_docs = SimpleDirectoryReader("./docs").load_data()
    documents = []

    for raw in raw_docs:
        file_name = raw.metadata.get("file_name", "Unknown source")
        section = ""

        # חיתוך לפי שורה ריקה = לפי פסקאות
        blocks = re.split(r"\n\s*\n", raw.text)

        for block in blocks:
            lines = []
            is_header = False

            for line in block.strip().splitlines():
                if line.startswith("==="):
                    is_header = True
                else:
                    lines.append(line)

            if not lines:
                continue

            if is_header:
                # פסקה עם ===== היא תחילת חלק חדש
                section = lines[0]
                text = "\n".join(lines)
            else:
                # פסקה רגילה - מוסיפים לה את שם החלק
                text = section + "\n" + "\n".join(lines)

            documents.append(Document(
                text=text.strip(),
                metadata={"source": file_name, "section": section},
            ))

    print("נוצרו", len(documents), "פסקאות")
    return documents


def setup_vector_store():
    client = chromadb.PersistentClient(path=DB_PATH)
    collection = client.get_or_create_collection(
        "documents", metadata={"hnsw:space": "cosine"}
    )

    store = ChromaVectorStore(chroma_collection=collection)
    return StorageContext.from_defaults(vector_store=store)


def save_document(documents, storage_context):
    """פיצול המסמכים, יצירת אינדקס ושמירת המידע."""
    splitter = SentenceSplitter(
    chunk_size=1024,
    chunk_overlap=40,
    )

    index = VectorStoreIndex.from_documents(
        documents,
        storage_context=storage_context,
        transformations=[splitter],
        show_progress=True,
    )

    return index


def main():
    """הפונקציה הראשית שמפעילה את התהליך."""
    setup_embedding()

    documents = load_document()

    storage_context = setup_vector_store()

    index = save_document(documents, storage_context)


if __name__ == "__main__":
    main()