import bs4

from langchain import hub
from langchain.agents import AgentExecutor, create_react_agent
from langchain.chains import create_retrieval_chain
from langchain.memory import ConversationBufferMemory
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain.tools.retriever import create_retriever_tool
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

from utils.data_processing import load_documents, split_documents
from utils.retriever import create_embedding_function
from utils.history import get_session_history
from utils.debug import log_time

from openrouter import ChatOpenRouter
from retriever import create_qa_chain, create_or_load_vectorstore, create_prompt_react_agent
from chain_history import create_history_aware_retriever

from pprint import pp
import os
import time

from dotenv import load_dotenv
load_dotenv()

llm = ChatOpenRouter(
	# model_name="nousresearch/nous-capybara-7b:free"
	model_name="microsoft/phi-3-mini-128k-instruct:free"
)
# Bind words for react
llm = llm.bind(stop=["\nFinal Answer"])

embedding_function = create_embedding_function()
vectorstore = create_or_load_vectorstore(embedding_function)

retriever = vectorstore.as_retriever()

tool_rag = create_retriever_tool(
	retriever=retriever,
	name="search_rag",
	description=(
		"Searches and returns data from RAG. "
		"The output of this tool will be in a format Q and A "
		"Q stands for standard questions being ask and the "
		"A stands for correct answer for them. Alaways have "
		"an observation after getting the output of this tool."
	),
)

# Get the prompt to use - you can modify this!
# prompt = hub.pull("hwchase17/react")
prompt = create_prompt_react_agent()

tools = [tool_rag]

# Construct the ReAct agent
agent = create_react_agent(llm, tools, prompt)
# Create an agent executor by passing in the agent and tools
memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

agent_executor = AgentExecutor(
	agent=agent,
	tools=tools,
	memory=memory,
	max_iterations=20,
	verbose=True,
	handle_parsing_errors=True,
)

q1 = "Hi, I am a bit dizzy and feel tired, I can't sleep well at night."
q2 = "Do I need any pills Doctor"

chat_history = memory.buffer_as_messages
a1 = agent_executor.invoke({
	"input": q1,
	"chat_history": chat_history,
})

chat_history = memory.buffer_as_messages
a2 = agent_executor.invoke({
	"input": q2,
	"chat_history": chat_history,
})

pp(a1)
pp(a2)

print(f"User: {q1}")
print(f"AI: {a1['output']}")
print(f"User: {q2}")
print(f"AI: {a2['output']}")
