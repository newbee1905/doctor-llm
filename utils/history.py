import streamlit as st
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_community.chat_message_histories import ChatMessageHistory

def get_session_history(session_id: str) -> BaseChatMessageHistory:
	"""
	Get the chat history for the given session ID from the Streamlit session state.

	Parameters:
	- session_id (str): The session ID to retrieve the chat history for.

	Returns:
	- BaseChatMessageHistory: The chat message history instance.
	"""
	if session_id not in st.session_state.messages:
		st.session_state.messages[session_id] = ChatMessageHistory()

	return st.session_state.messages[session_id]

