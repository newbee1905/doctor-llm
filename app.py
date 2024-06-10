import bs4
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import WebBaseLoader
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_huggingface import HuggingFaceEmbeddings

from openrouter import ChatOpenRouter

import streamlit as st
from streamlit.runtime.scriptrunner.script_run_context import get_script_run_ctx
from pprint import pp

from langchain_community.chat_message_histories import ChatMessageHistory

from dotenv import load_dotenv
load_dotenv()

llm = ChatOpenRouter(
	model_name="nousresearch/nous-capybara-7b:free"
)

embedding_function = HuggingFaceEmbeddings(
	model_name="sentence-transformers/all-MiniLM-L6-v2",
	model_kwargs={'device': 'cpu'},
	encode_kwargs={'normalize_embeddings': False},
)

### Construct retriever ###
# Load, chunk and index the contents of the blog.
loader = WebBaseLoader(
	web_paths=("https://lilianweng.github.io/posts/2023-06-23-agent/",),
	bs_kwargs=dict(
		parse_only=bs4.SoupStrainer(
			class_=("post-content", "post-title", "post-header")
		)
	),
)
docs = loader.load()

text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
splits = text_splitter.split_documents(docs)
vectorstore = FAISS.from_documents(documents=splits, embedding=embedding_function)

retriever = vectorstore.as_retriever()

### Contextualize question ###
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
history_aware_retriever = create_history_aware_retriever(
	llm, retriever, contextualise_q_prompt
)

### Answer question ###
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
question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)

rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)

### Statefully manage chat history ###
if "messages" not in st.session_state:
	st.session_state.messages = {}

def get_session_history(session_id: str) -> BaseChatMessageHistory:
	if session_id not in st.session_state.messages:
		st.session_state.messages[session_id] = ChatMessageHistory()

	return st.session_state.messages[session_id]

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
