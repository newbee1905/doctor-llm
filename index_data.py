from utils.retriever import create_embedding_function
from retriever import create_or_load_vectorstore

embedding_function = create_embedding_function(device="cuda")
vectorstore = create_or_load_vectorstore(embedding_function)
