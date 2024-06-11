import bs4
from langchain_community.document_loaders import WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from typing import List

def load_documents() -> List[str]:
	"""
	Load documents for processing.

	Returns:
	- List[str]: A list of loaded document texts.
	"""
	loader = WebBaseLoader(
		web_paths=("https://lilianweng.github.io/posts/2023-06-23-agent/",),
		bs_kwargs=dict(
			parse_only=bs4.SoupStrainer(
				class_=("post-content", "post-title", "post-header")
			)
		),
	)
	return loader.load()

def split_documents(docs: List[str]) -> List[str]:
	"""
	Split the given documents into smaller chunks.

	Parameters:
	- docs (List[str]): A list of document texts to split.

	Returns:
	- List[str]: A list of document text chunks.
	"""
	text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
	return text_splitter.split_documents(docs)
