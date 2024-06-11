import bs4

from langchain.chains import create_retrieval_chain
from langchain_core.runnables.history import RunnableWithMessageHistory

from utils.data_processing import load_documents, split_documents
from utils.retriever import create_embedding_function
from utils.history import get_session_history

from openrouter import ChatOpenRouter
from retriever import create_history_aware_retriever, create_or_load_vectorstore
from chain_history import create_qa_chain

import streamlit as st
from streamlit.runtime.scriptrunner.script_run_context import get_script_run_ctx
from pprint import pp

from dotenv import load_dotenv
load_dotenv()

llm = ChatOpenRouter(
	model_name="nousresearch/nous-capybara-7b:free"
)

embedding_function = create_embedding_function()
vectorstore = create_or_load_vectorstore(embedding_function)

retriever = vectorstore.as_retriever()

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

ctx = get_script_run_ctx()
user_session = ctx.session_id

st.title(f"Doctor LLM")

def res_generator(stream):
	for chunk in stream:
		if answer_chunk := chunk.get("answer"):
			yield answer_chunk

for message in get_session_history(user_session).messages:
	with st.chat_message(message.type):
		st.markdown(message.content)

if user_inp := st.chat_input("Message..."):
	st.chat_message("human").markdown(user_inp)

	prompt_obj = {"input": user_inp}

	config = {
		"configurable": {"session_id": user_session}
	}

	with st.chat_message("ai"):
		stream = conversational_rag_chain.stream(prompt_obj, config=config)
		res = st.write_stream(res_generator(stream))
