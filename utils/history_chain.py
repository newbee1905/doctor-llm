from langchain.chains import create_history_aware_retriever
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

contextualise_q_system_prompt = """Given a chat history and the latest user question \
which might reference context in the chat history, formulate a standalone question \
which can be understood without the chat history. Do NOT answer the question, \
just reformulate it if needed and otherwise return it as is."""

contextualise_q_prompt = ChatPromptTemplate.from_messages(
	[
		("system", contextualise_q_system_prompt),
		MessagesPlaceholder("chat_history"),
		("human", "{input}"),
	]
)

def history_aware_retriever_init(llm, retriever):
	return create_history_aware_retriever(
		llm, retriever, contextualise_q_prompt
	)
