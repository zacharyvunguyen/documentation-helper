import os
import logging
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from pinecone import Pinecone   # Correctly import pinecone

# Load API keys and environment variables
PINECONE_ENVIRONMENT = os.getenv("PINECONE_ENVIRONMENT")

def initialize_pinecone(PINECONE_API_KEY):
    try:
        logging.info("Initializing Pinecone...")
        pc = Pinecone(PINECONE_API_KEY=PINECONE_API_KEY)
        logging.info("Pinecone initialized successfully.")
        return pc
    except Exception as e:
        logging.error(f"Failed to initialize Pinecone: {e}")
        raise

def create_openai_embeddings(model, api_key):
    try:
        logging.info(f"Creating OpenAI embeddings with model '{model}'...")
        if not api_key or not api_key.startswith("sk-"):
            raise ValueError("Invalid or missing OpenAI API key.")
        embeddings = OpenAIEmbeddings(model=model, openai_api_key=api_key)
        logging.info(f"OpenAI Embeddings initialized successfully with model '{model}'.")
        return embeddings
    except Exception as e:
        logging.error(f"Failed to initialize OpenAI Embeddings: {e}")
        raise

def create_openai_chat(model, api_key):
    try:
        logging.info(f"Creating ChatOpenAI with model '{model}'...")
        if not api_key or not api_key.startswith("sk-"):
            raise ValueError("Invalid or missing OpenAI API key.")
        chat = ChatOpenAI(model=model, openai_api_key=api_key)
        logging.info(f"ChatOpenAI initialized successfully with model '{model}'.")
        return chat
    except Exception as e:
        logging.error(f"Failed to initialize ChatOpenAI: {e}")
        raise
