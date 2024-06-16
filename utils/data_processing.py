import bs4
# from langchain_community.document_loaders import WebBaseLoader
from langchain_community.document_loaders.csv_loader import CSVLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from typing import List

from utils.debug import log_time

@log_time
def load_documents() -> List[str]:
	"""
	Load documents for processing.

	Returns:
	- List[str]: A list of loaded document texts.
	"""
	loader = CSVLoader(
		file_path="./mashqa_merged_output_all.csv",
		csv_args={
			"delimiter": ",",
			"quotechar": '"',
			"fieldnames": ["Q", "A"],
		},
	)
	return loader.load()

@log_time
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
