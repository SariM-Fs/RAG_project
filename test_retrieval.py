import chromadb
from llama_index.core import VectorStoreIndex
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.vector_stores.chroma import ChromaVectorStore

embed_model = HuggingFaceEmbedding(
    model_name="intfloat/multilingual-e5-small",
    query_instruction="query: ",
    text_instruction="passage: ",
)
collection = chromadb.PersistentClient(path="./chroma_db").get_or_create_collection("documents")
index = VectorStoreIndex.from_vector_store(
    ChromaVectorStore(chroma_collection=collection),
    embed_model=embed_model,
)

question = input("שאלה: ")

for r in index.as_retriever(similarity_top_k=5).retrieve(question):
    print("ציון:", round(r.score, 3))
    print(r.node.get_content()[:300])
    print("-----")