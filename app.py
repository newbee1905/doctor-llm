from langchain import hub
from langchain.chains import create_retrieval_chain
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain.agents import AgentExecutor, create_react_agent
from langchain.memory import ConversationBufferMemory

from utils.data_processing import load_documents, split_documents
from utils.history import get_session_history
from utils.debug import log_time
from utils.streamlit_cache import get_retriever, get_llm

from openrouter import ChatOpenRouter
from runpod import ChatRunpod
from retriever import create_qa_chain, create_qa_tool, create_prompt_react_agent
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
	"-l", "--llm",
	type=str,
	default="openrouter",
	choices=["runpod", "openrouter"],
	help="Chose what provider of the LLM the program will use",
)

parser.add_argument(
	"-m", "--model",
	type=str,
	default="openchat/openchat-7b:free",
	help="Chose what model from operrouter the program will use",
)

parser.add_argument(
	"-a", "--agent",
	action=argparse.BooleanOptionalAction
)

args = parser.parse_args()

ctx = get_script_run_ctx()
user_session = ctx.session_id

runpod_model_name = os.getenv("RUNPOD_MODEL_NAME")

if args.llm == "runpod":
	# I am hosting runpod with serverless, so it will
	# only have one model option, which will be set in
	# the dotenv alongside with other info of Runpod
	model_name = runpod_model_name
else:
	model_name = args.model

retriever = get_retriever()

llm = get_llm(args.llm, model_name, args.agent)

st.title(f"Doctor LLM")

if args.agent:
	tool_rag = create_qa_tool(retriever)
	tools = [tool_rag]
	prompt = create_prompt_react_agent()

	agent = create_react_agent(llm, tools, prompt)

	runner = AgentExecutor(
		agent=agent,
		tools=tools,
		max_iterations=100,
		verbose=DEBUG,
		handle_parsing_errors=True,
	)
else:
	history_aware_retriever = create_history_aware_retriever(llm, retriever)
	question_answer_chain = create_qa_chain(llm, retriever)

	runner = create_retrieval_chain(history_aware_retriever, question_answer_chain)

### Statefully manage chat history ###
if "messages" not in st.session_state:
	st.session_state.messages = {}

runner_with_history = RunnableWithMessageHistory(
	runner,
	get_session_history,
	input_messages_key="input",
	history_messages_key="chat_history",
	output_messages_key="answer",
)

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
		# res = log_time(runner.invoke)(prompt_obj, config=config)
		res = runner_with_history.invoke(prompt_obj, config=config)
		if args.agent:
			st.markdown(res["output"])
		else:
			st.markdown(res["answer"])
		if DEBUG:
			pp(res)
