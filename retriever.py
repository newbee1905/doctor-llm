import os

from langchain import chains
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_community.vectorstores import FAISS
from langchain_core.retrievers import BaseRetriever
from langchain_huggingface import HuggingFaceEmbeddings

from utils.data_processing import load_documents, split_documents

def create_history_aware_retriever(
	llm: ChatOpenAI, 
	retriever: BaseRetriever
) -> BaseRetriever:
	"""
	Create a history-aware retriever that reformulates questions based on chat history.

	Parameters:
	- llm (ChatOpenAI): The language model used for question reformulation.
	- retriever (BaseRetriever): The retriever instance used to fetch relevant documents.

	Returns:
	- BaseRetriever: A history-aware retriever instance that reformulates questions based on the given chat history.
	"""
	contextualise_q_system_prompt = (
		"Given a chat history and the latest user question "
		"which might reference context in the chat history, "
		"formulate a standalone question which can be understood "
		"without the chat history. Do NOT answer the question, "
		"just reformulate it if needed and otherwise return it as is."
	)
	contextualise_q_prompt = ChatPromptTemplate.from_messages(
		[
			("system", contextualise_q_system_prompt),
			MessagesPlaceholder("chat_history"),
			("human", "{input}"),
		]
	)
	return chains.create_history_aware_retriever(llm, retriever, contextualise_q_prompt)

def create_embedding_function() -> HuggingFaceEmbeddings:
	"""
	Create the embedding function used with the FAISS index.

	Returns:
	- HuggingFaceEmbeddings: The embedding function instance.
	"""
	return HuggingFaceEmbeddings(
		model_name="sentence-transformers/all-MiniLM-L6-v2",
		model_kwargs={'device': 'cpu'},
		encode_kwargs={'normalize_embeddings': False},
	)

def create_or_load_vectorstore(
	embedding_function: HuggingFaceEmbeddings,
	index_path: str = 'faiss_index'
) -> FAISS:
	"""
	Create a new FAISS vector store or load an existing one from the specified path.

	Parameters:
	- embedding_function (HuggingFaceEmbeddings): The embedding function to use with the FAISS index.
	- index_path (str): FAISS index path.

	Returns:
	- FAISS: The FAISS vector store instance.
	"""

	if os.path.exists(index_path):
		vectorstore = FAISS.load_local(index_path, embedding_function, allow_dangerous_deserialization=True)
	else:
		docs = load_documents()
		splits = split_documents(docs)
		vectorstore = FAISS.from_documents(splits, embedding_function)
		vectorstore.save_local(index_path)
	return vectorstore
