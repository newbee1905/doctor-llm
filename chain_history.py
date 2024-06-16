from langchain_openai import ChatOpenAI
from langchain import chains
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.retrievers import BaseRetriever

from openrouter import ChatOpenRouter
from utils.debug import log_time

@log_time
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

