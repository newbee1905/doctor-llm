import os
import pickle

from langchain_core.retrievers import BaseRetriever
from langchain_core.tools import BaseTool
from langchain.chains.combine_documents.base import BaseCombineDocumentsChain
from langchain_core.prompts.base import BasePromptTemplate
from langchain_huggingface import HuggingFaceEmbeddings

from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.tools.retriever import create_retriever_tool
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_community.vectorstores import FAISS

from utils.data_processing import load_documents, split_documents
from utils.debug import log_time

@log_time
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

@log_time
def create_qa_tool(retriever: BaseRetriever) -> BaseTool:
	"""
	Create a question-answering (QA) tool using to get context 
	close to the given input either by the user or the agent.

	Parameters:
	- retriever (BaseRetriever): The retriever instance used to fetch relevant context.

	Returns:
	- BaseTool: A tool to be used by langchain agent to get relevant context.
	"""

	tool_rag_desc = (
		"Searches and returns data from RAG. "
		"The output of this tool will be in a format Q and A."
		"Q stands for standard questions being ask and A stands"
		"for correct answer for the question. Alaways have "
		"an observation after getting the output of this tool."
	)

	tool_rag = create_retriever_tool(
		retriever=retriever,
		name="search_rag",
		description=tool_rag_desc,
	)

	return tool_rag

@log_time
def create_prompt_react_agent() -> BasePromptTemplate:
	"""
	Create a prompt for ReAct agent based of hwchase17/react
	"""

	system_prompt = (
		"You are an assistant for question-answering tasks. "
		"Use the following pieces of retrieved context to answer "
		"the question. If you don't know the answer, say that you "
		"don't know. Use three sentences maximum and keep the "
		"answer concise. You have access to the following tools:"
		""
		"{tools}"
		""
		"Given a chat history and the latest user question "
		"which might reference context in the chat history, "
		"formulate a standalone question which can be understood "
		"without the chat history. Do NOT answer the question, "
		"just reformulate it if needed and otherwise return it as is."
		""
		"{chat_history}"
		""
		"Use the following format:"
		""
		"Question: the input question you must answer"
		"Thought: you should always think about what to do"
		"Action: the action to take, should be one of [{tool_names}]"
		"Action Input: the input to the action"
		"Observation: the result of the action"
		"... (this Thought/Action/Action Input/Observation can repeat "
		"N times with N smaller or equal to 5)"
		"Thought: I now know the final answer"
		"Final Answer: the final answer to the original input question"
		""
		"Begin!"
		""
		"Question: {input}"
		"Thought:{agent_scratchpad}"
	)

	return ChatPromptTemplate.from_template(system_prompt)
	

@log_time
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
		print("Loading Docs")
		docs = load_documents()
		print("Spliting Docs")
		splits = split_documents(docs)
		print("Building vectorstore")
		vectorstore = FAISS.from_documents(splits, embedding_function)
		vectorstore.save_local(index_path)
	return vectorstore
