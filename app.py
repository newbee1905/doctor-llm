from langchain_community.chat_models import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv
from typing import Optional
import os
import requests
import json

class ChatOpenRouter(ChatOpenAI):
	openai_api_base: str
	openai_api_key: str
	model_name: str

	def __init__(
		self,
		model_name: str,
		openai_api_key: Optional[str] = None,
		openai_api_base: str = "https://openrouter.ai/api/v1",
		**kwargs
	):

		openai_api_key = openai_api_key or os.getenv('OPENROUTER_API_KEY')
		super().__init__(
			openai_api_base=openai_api_base,
			openai_api_key=openai_api_key,
			model_name=model_name,
			**kwargs,
		)

from langchain_core.prompts import ChatPromptTemplate

if __name__ == "__main__":
	load_dotenv()

	llm = ChatOpenRouter(
		model_name="nousresearch/nous-capybara-7b:free"
	)
	prompt = ChatPromptTemplate.from_template("tell me a short joke about {topic}")
	openrouter_chain = prompt | llm
	# print(openrouter_chain.invoke({"topic": "banana"}))

	try:
		for chunk in openrouter_chain.stream({"topic": "banana"}):
			print(chunk.content, end="", flush=True)
	except Exception as e:
		print(e)
		pass

	# openrouter_api_key = os.getenv('OPENROUTER_API_KEY')

	# response = requests.post(
	# 	url="https://openrouter.ai/api/v1/chat/completions",
	# 	headers={
	# 		"Authorization": f"Bearer {openrouter_api_key}",
	# 	},
	# 	data=json.dumps({
	# 		"model": "nousresearch/nous-capybara-7b:free", # Optional
	# 		"messages": [
	# 			{ "role": "user", "content": "What is the meaning of life?" }
	# 		]
	# 	})
	# )

	# print(response.json())
