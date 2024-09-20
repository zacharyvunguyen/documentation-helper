import os
import logging
from dotenv import load_dotenv
from langchain.chains.retrieval import create_retrieval_chain
from langchain import hub
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_pinecone import PineconeVectorStore
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from pinecone import Pinecone

# Load environment variables from .env file
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Load API keys and environment variables
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME")
EMBED_MODEL = os.getenv("EMBED_MODEL", "text-embedding-3-small")

logging.info(f"Environment variables loaded: OpenAI Model={EMBED_MODEL}, Pinecone Index={PINECONE_INDEX_NAME}")


# Initialize Pinecone client using the provided method
def initialize_pinecone():
    try:
        logging.info("Step 1: Initializing Pinecone...")
        pc = Pinecone(api_key=PINECONE_API_KEY)
        logging.info("Step 1: Pinecone initialized successfully.")
        return pc
    except Exception as e:
        logging.error(f"Step 1: Failed to initialize Pinecone: {e}")
        raise


def create_openai_embeddings(model, api_key):
    try:
        logging.info(f"Step 2: Creating OpenAI embeddings with model '{model}'...")
        if not api_key or not api_key.startswith("sk-"):
            raise ValueError("Invalid or missing OpenAI API key.")
        embeddings = OpenAIEmbeddings(model=model, openai_api_key=api_key)
        logging.info(f"Step 2: OpenAI Embeddings initialized successfully with model '{model}'.")
        return embeddings
    except Exception as e:
        logging.error(f"Step 2: Failed to initialize OpenAI Embeddings: {e}")
        raise


def create_openai_chat(model, api_key):
    try:
        logging.info(f"Step 3: Creating ChatOpenAI with model '{model}'...")
        if not api_key or not api_key.startswith("sk-"):
            raise ValueError("Invalid or missing OpenAI API key.")
        chat = ChatOpenAI(model=model, openai_api_key=api_key)
        logging.info(f"Step 3: ChatOpenAI initialized successfully with model '{model}'.")
        return chat
    except Exception as e:
        logging.error(f"Step 3: Failed to initialize ChatOpenAI: {e}")
        raise


def run_llm(query: str):
    try:
        logging.info(f"Step 4: Starting LLM execution with query: '{query}'")

        # Step 1: Initialize Pinecone client
        pc = initialize_pinecone()

        # Step 2: Create embeddings using OpenAI with the API key
        embeddings = create_openai_embeddings(EMBED_MODEL, OPENAI_API_KEY)

        # Step 3: Connect to the specified Pinecone index using embeddings
        logging.info(f"Step 5: Connecting to Pinecone index '{PINECONE_INDEX_NAME}'...")
        docsearch = PineconeVectorStore(index_name=PINECONE_INDEX_NAME, embedding=embeddings)
        logging.info(f"Step 5: Connected to Pinecone index '{PINECONE_INDEX_NAME}'.")

        # Step 4: Initialize the chat model
        CHAT_MODEL = "gpt-4o-mini"
        chat = create_openai_chat(model=CHAT_MODEL, api_key=OPENAI_API_KEY)

        # Step 5: Pull the retrieval-qa-chat prompt from LangChain Hub
        logging.info("Step 6: Pulling retrieval-qa-chat prompt from LangChain Hub...")
        retrieval_qa_chat_prompt = hub.pull("langchain-ai/retrieval-qa-chat")

        # Step 6: Create a chain that combines documents
        logging.info("Step 7: Creating document combination chain...")
        stuff_documents_chain = create_stuff_documents_chain(chat, retrieval_qa_chat_prompt)

        # Step 7: Create the retrieval chain with Pinecone retriever and the document combination chain
        logging.info("Step 8: Creating retrieval chain with Pinecone retriever and document combination chain...")
        qa = create_retrieval_chain(
            retriever=docsearch.as_retriever(), combine_docs_chain=stuff_documents_chain
        )

        # Step 8: Invoke the chain with the query
        logging.info(f"Step 9: Invoking the retrieval chain with query: '{query}'...")
        result = qa.invoke(input={"input": query})

        logging.info(f"Step 9: LLM execution completed successfully with result: {result}")
        return result

    except Exception as e:
        logging.error(f"Error during LLM execution: {e}")
        raise


if __name__ == "__main__":
    try:
        logging.info("Starting the main function...")

        # Step 9: Run the LLM with a sample query
        res = run_llm(query="How to work with langchain and pinecone?")

        logging.info(f"LLM Result: {res['answer']}")
        print(res["answer"])

    except Exception as e:
        logging.error(f"Error in main execution: {e}")
