from langchain_openai import ChatOpenAI
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain 
from langchain.chains.combine_documents.base import BaseCombineDocumentsChain
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.retrievers import BaseRetriever

from openrouter import ChatOpenRouter

def create_qa_chain(
	llm: ChatOpenAI,
	retriever: BaseRetriever
) -> BaseCombineDocumentsChain:
	"""
	Create a question-answering (QA) chain using the given language model and retriever.

	Parameters:
	- llm (ChatOpenRouter): The language model used for generating answers.
	- retriever (BaseRetriever): The retriever instance used to fetch relevant context.

	Returns:
	- CombineDocumentsChain: A chain that uses the language model and retrieved context to answer questions concisely.
	"""
	system_prompt = (
		"You are an assistant for question-answering tasks. "
		"Use the following pieces of retrieved context to answer "
		"the question. If you don't know the answer, say that you "
		"don't know. Use three sentences maximum and keep the "
		"answer concise."
		"\n\n"
		"{context}"
	)

	qa_prompt = ChatPromptTemplate.from_messages(
		[
			("system", system_prompt),
			MessagesPlaceholder("chat_history"),
			("human", "{input}"),
		]
	)

	return create_stuff_documents_chain(llm, qa_prompt)
