# Doctor LLM

## Requirements

- langchain
- langchain-community
- langchainhub
- langchain-openai
- langchain-huggingface
- bs4
- streamlit
- faiss-cpu

## Issues

- Model still sometime doesn't follow the rules set by ReAct prompt
- Sometime RAG only return question from the vectordb instead of both question and answer
- Likely need to switch to local model since Openrouter API seem to does not provide enough free api call for ReAct
