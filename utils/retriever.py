import os

from langchain import chains
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings

def create_embedding_function(
	model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
	device: str = "cpu",
) -> HuggingFaceEmbeddings:
	"""
	Create the embedding function used with the FAISS index.

	Parameters:
	- model_name (str): Model name for embedding 
	- device (str): device type for model

	Returns:
	- HuggingFaceEmbeddings: The embedding function instance.
	"""
	return HuggingFaceEmbeddings(
		model_name=model_name,
		model_kwargs={'device': device},
		encode_kwargs={'normalize_embeddings': False},
	)
