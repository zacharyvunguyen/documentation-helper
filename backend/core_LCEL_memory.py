import os
import logging
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableParallel, RunnablePassthrough
from langchain_pinecone import PineconeVectorStore
from langchain_core.pydantic_v1 import BaseModel
from pinecone import Pinecone

# from core_functions import initialize_pinecone, create_openai_embeddings,create_openai_chat

# Load environment variables from .env file
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Load API keys and environment variables
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_ENVIRONMENT = os.getenv("PINECONE_ENVIRONMENT")
PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME")
EMBED_MODEL = os.getenv("EMBED_MODEL", "text-embedding-3-small")

logging.info(f"Environment variables loaded: OpenAI Model={EMBED_MODEL}, Pinecone Index={PINECONE_INDEX_NAME}")


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

# Function to format retrieved documents into a single string
def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

# Add typing for input using BaseModel
class Question(BaseModel):
    __root__: str


# Run LLM function
def run_llm(query: str, conversation_history: list):
    try:
        logging.info(f"Starting LLM execution with query: {query}")

        # Initialize Pinecone client
        pc = initialize_pinecone(PINECONE_API_KEY)

        # Create embeddings using OpenAI with the API key
        embeddings = create_openai_embeddings(EMBED_MODEL, OPENAI_API_KEY)

        # Connect to the specified Pinecone index using embeddings
        logging.info(f"Connecting to Pinecone index '{PINECONE_INDEX_NAME}'...")
        docsearch = PineconeVectorStore(index_name=PINECONE_INDEX_NAME, embedding=embeddings)
        logging.info(f"Connected to Pinecone index '{PINECONE_INDEX_NAME}'.")

        # Retrieve documents from Pinecone, set 'k' to retrieve more documents
        retriever = docsearch.as_retriever(search_kwargs={"k": 10})  # Adjust 'k' as needed
        documents = retriever.get_relevant_documents(query)

        # Initialize a list to store metadata
        metadata_list = []

        # Log the content and metadata of the retrieved documents
        for i, doc in enumerate(documents):
            logging.info(f"Document {i+1}: Content: {doc.page_content[:200]}... Metadata: {doc.metadata}")
            metadata_list.append(doc.metadata)

        # Initialize the chat model with a larger context window
        CHAT_MODEL = "gpt-3.5-turbo-16k"  # Use "gpt-4" if you have access
        chat = create_openai_chat(model=CHAT_MODEL, api_key=OPENAI_API_KEY)

        # Build the conversation history into the prompt
        conversation_str = ""
        for turn in conversation_history:
            conversation_str += f"User: {turn['user']}\nAssistant: {turn['bot']}\n"

        # Optionally, truncate or summarize conversation history
        # conversation_str = truncate_conversation(conversation_str, max_tokens=...)

        # Format documents into context string
        context_str = format_docs(documents)

        # RAG prompt with conversation history
        template = """You are a helpful assistant.
Here is the conversation so far:
{conversation}
Answer the question based only on the following context:
{context}
Question: {question}
"""
        prompt = ChatPromptTemplate.from_template(template)

        # Prepare the inputs for the chain
        inputs = {
            "conversation": conversation_str,
            "context": context_str,
            "question": query
        }

        # Create the chain
        chain = prompt | chat | StrOutputParser()

        # Invoke the chain with the inputs
        logging.info(f"Invoking the chain with query: {query}")
        result = chain.invoke(inputs)

        # Return both the result and the metadata list
        return result, metadata_list

    except Exception as e:
        logging.error(f"Error during LLM execution: {e}")
        raise


if __name__ == "__main__":
    try:
        logging.info("Running the main function...")
        # Capture both the result and metadata_list
        res, metadata_list = run_llm(query="what is langchain?")
        print("Answer:", res)
        print("Metadata List:", metadata_list)
    except Exception as e:
        logging.error(f"Error in main execution: {e}")
