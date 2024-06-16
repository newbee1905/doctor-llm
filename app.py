from langchain import hub
from langchain.chains import create_retrieval_chain
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain.agents import AgentExecutor, create_react_agent
from langchain.memory import ConversationBufferMemory

from utils.data_processing import load_documents, split_documents
from utils.retriever import create_embedding_function
from utils.history import get_session_history
from utils.debug import log_time

from openrouter import ChatOpenRouter
from retriever import create_or_load_vectorstore, create_qa_chain, create_qa_tool, create_prompt_react_agent
from chain_history import create_history_aware_retriever

import streamlit as st
from streamlit.runtime.scriptrunner.script_run_context import get_script_run_ctx
from pprint import pp
import os
import time
import argparse

from dotenv import load_dotenv
load_dotenv()

DEBUG = os.getenv("DEBUG") != None

parser = argparse.ArgumentParser(
	prog="DoctorLLM",
	description="Basic Q&A LLM acting as an doctor using openrouter, langchaian and streamlit",
)

parser.add_argument(
	"-m", "--model",
	type=str,
	default="microsoft/phi-3-mini-128k-instruct:free",
	help="Chose what model from operrouter the program will use",
)

parser.add_argument(
	"-a", "--agent",
	action=argparse.BooleanOptionalAction
)

args = parser.parse_args()

llm = ChatOpenRouter(
	model_name=args.model,
	temperature=0,
)

embedding_function = create_embedding_function()
vectorstore = create_or_load_vectorstore(embedding_function)

retriever = vectorstore.as_retriever()

ctx = get_script_run_ctx()
user_session = ctx.session_id

st.title(f"Doctor LLM")

if args.agent:
	# Bind words for react
	llm = llm.bind(stop=["\nFinal Answer"])

	tool_rag = create_qa_tool(retriever)
	tools = [tool_rag]

	# prompt = hub.pull("hwchase17/react")
	prompt = create_prompt_react_agent()

	# Construct the ReAct agent
	agent = create_react_agent(llm, tools, prompt)

	# Create an agent executor by passing in the agent and tools
	memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

	agent_executor = AgentExecutor(
		agent=agent,
		tools=tools,
		memory=memory,
		max_iterations=100,
		verbose=DEBUG,
		handle_parsing_errors=True,
	)

	runner = agent_executor

	for message in memory.buffer_as_messages:
		with st.chat_message(message.type):
			st.markdown(message.content)
else:
	### Contextualize question ###
	history_aware_retriever = create_history_aware_retriever(llm, retriever)

	### Answer question ###
	question_answer_chain = create_qa_chain(llm, retriever)

	rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)

	### Statefully manage chat history ###
	if "messages" not in st.session_state:
		st.session_state.messages = {}

	conversational_rag_chain = RunnableWithMessageHistory(
		rag_chain,
		get_session_history,
		input_messages_key="input",
		history_messages_key="chat_history",
		output_messages_key="answer",
	)

	runner = conversational_rag_chain

	for message in get_session_history(user_session).messages:
		with st.chat_message(message.type):
			st.markdown(message.content)

def res_generator(stream):
	for chunk in stream:
		if answer_chunk := chunk.get("answer"):
			yield answer_chunk

if user_inp := st.chat_input("Message..."):
	st.chat_message("human").markdown(user_inp)

	prompt_obj = {"input": user_inp}

	config = {
		"configurable": {"session_id": user_session}
	}

	with st.chat_message("ai"):
		res = log_time(runner.invoke)(prompt_obj, config=config)
		if args.agent:
			st.markdown(res["output"])
		else:
			st.markdown(res["answer"])
		if DEBUG:
			pp(res)
