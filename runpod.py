from langchain_openai import ChatOpenAI
from typing import Optional
import os

class ChatRunpod(ChatOpenAI):
	openai_api_base: str
	openai_api_key: str
	model_name: str

	def __init__(
		self,
		model_name: str = os.getenv("RUNPOD_MODEL_NAME"),
		openai_api_key: Optional[str] = None,
		openai_api_base: str = f"https://api.runpod.ai/v2/{os.getenv('RUNPOD_ENDPOINT_ID')}/openai/v1",
		**kwargs
	):

		openai_api_key = openai_api_key or os.getenv('RUNPOD_API_KEY')
		super().__init__(
			openai_api_base=openai_api_base,
			openai_api_key=openai_api_key,
			model_name=model_name,
			**kwargs,
		)
