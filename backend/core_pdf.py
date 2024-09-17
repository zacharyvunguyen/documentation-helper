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
#PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME")
PINECONE_INDEX_NAME = "langchain-pdf-index"
EMBED_MODEL = os.getenv("EMBED_MODEL", "text-embedding-3-small")

logging.info(f"Environment variables loaded: OpenAI Model={EMBED_MODEL}, Pinecone Index={PINECONE_INDEX_NAME}")

# Initialize Pinecone client using the provided method
def initialize_pinecone():
    try:
        logging.info("Initializing Pinecone...")
        pc = Pinecone(api_key=PINECONE_API_KEY)
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

def run_llm(query: str):
    try:
        logging.info(f"Starting LLM execution with query: {query}")

        # Initialize Pinecone client
        pc = initialize_pinecone()

        # Create embeddings using OpenAI with the API key
        embeddings = create_openai_embeddings(EMBED_MODEL, OPENAI_API_KEY)

        # Connect to the specified Pinecone index using embeddings
        logging.info(f"Connecting to Pinecone index '{PINECONE_INDEX_NAME}'...")
        docsearch = PineconeVectorStore(index_name=PINECONE_INDEX_NAME, embedding=embeddings)
        logging.info(f"Connected to Pinecone index '{PINECONE_INDEX_NAME}'.")

        # Initialize the chat model
        CHAT_MODEL = "gpt-4o-mini"
        chat = create_openai_chat(model=CHAT_MODEL, api_key=OPENAI_API_KEY)

        # Pull the retrieval-qa-chat prompt from LangChain Hub
        logging.info("Pulling retrieval-qa-chat prompt from LangChain Hub...")
        retrieval_qa_chat_prompt = hub.pull("langchain-ai/retrieval-qa-chat")

        # Create a chain that combines documents
        logging.info("Creating document chain...")
        stuff_documents_chain = create_stuff_documents_chain(chat, retrieval_qa_chat_prompt)

        # Create the retrieval chain with Pinecone retriever and the document combination chain
        logging.info("Creating retrieval chain...")
        qa = create_retrieval_chain(
            retriever=docsearch.as_retriever(), combine_docs_chain=stuff_documents_chain
        )

        # Invoke the chain with the query
        logging.info(f"Invoking the chain with query: {query}")
        result = qa.invoke(input={"input": query})

        logging.info(f"LLM execution completed successfully with result: {result}")
        return result

    except Exception as e:
        logging.error(f"Error during LLM execution: {e}")
        raise


if __name__ == "__main__":
    try:
        # Run the LLM with a sample query
        logging.info("Running the main function...")
        res = run_llm(query="What is the total expenses and top categories that spend most amount of money?")
        logging.info(f"Result: {res['answer']}")
        print(res["answer"])
    except Exception as e:
        logging.error(f"Error in main execution: {e}")
