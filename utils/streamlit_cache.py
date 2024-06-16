from streamlit import cache_resource, cache_data

from utils.retriever import create_embedding_function
from utils.debug import log_time

from openrouter import ChatOpenRouter
from runpod import ChatRunpod
from retriever import create_or_load_vectorstore, create_qa_chain, create_qa_tool, create_prompt_react_agent

from langchain_openai import ChatOpenAI
from langchain_core.retrievers import BaseRetriever

import os

@cache_resource
def get_retriever() -> BaseRetriever:
	"""
	Create and return a retriever object for information retrieval.

	This function uses a cached resource and logs the time taken to execute.
	It creates an embedding function, loads or creates a vector store using
	that embedding function, and then converts the vector store into a retriever object.

	Returns:
		retriever: An object that can be used to retrieve information from the vector store.
	"""
	embedding_function = create_embedding_function()
	vectorstore = create_or_load_vectorstore(embedding_function)

	retriever = vectorstore.as_retriever()
	return retriever

@cache_resource
def get_llm(llm_type: str, model_name: str, agent: bool) -> ChatOpenAI:
	"""
	Create and return a language model (LLM) based on the specified type and model name.

	This function uses a cached resource and logs the time taken to execute. 
	It matches the provided LLM type to instantiate the appropriate model (OpenRouter or Runpod).
	If the 'agent' parameter is True, it binds an agent to the LLM.

	Args:
		llm_type (str): The type of language model to create ('openrouter' or 'runpod').
		model_name (str): The name of the model to use.
		agent (bool): A flag indicating whether to bind an agent to the LLM.

	Returns:
		llm: An instantiated language model object.
	
	Raises:
		NotImplementedError: If the specified LLM type is not supported.
	"""

	match llm_type:
		case "openrouter":
			llm = ChatOpenRouter(
				model_name=model_name,
				temperature=0,
			)
		case "runpod":
			runpod_endpoint_id = os.getenv("RUNPOD_ENDPOINT_ID")
			llm = ChatRunpod(
				model_name=model_name,
				openai_api_base=f"https://api.runpod.ai/v2/{runpod_endpoint_id}/openai/v1",
			)
		case _:
			raise NotImplementedError

	if agent:
		llm = llm.bind(stop=["\nFinal Answer"])
	return llm
