import os
import logging
import json
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_pinecone import PineconeVectorStore
from langchain_core.pydantic_v1 import BaseModel
from pinecone import Pinecone  # Removed ApiException

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
    except Exception as e:  # Removed ApiException and replaced with generic Exception
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

# Function to perform reranking using Pinecone Inference API
def rerank_documents(pc, query, documents):
    try:
        logging.info("Performing reranking with Pinecone Inference API...")
        # Perform reranking using the provided Pinecone Inference API
        rerank_results = pc.inference.rerank(
            model="bge-reranker-v2-m3",  # Use the desired model for reranking
            query=query,
            documents=[doc.page_content for doc in documents],
            top_n=3,  # Number of top documents to keep after reranking
            return_documents=True
        )
        reranked_docs = [documents[i] for i in rerank_results["ranking"]]
        logging.info("Reranking completed successfully.")
        return reranked_docs
    except Exception as e:
        logging.error(f"Failed to rerank documents: {e}")
        return documents  # Return original order if reranking fails

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

        # Include both metadata and page content in the metadata list
        metadata_list = [
            {"page_content": doc.page_content, **doc.metadata} for doc in documents
        ]

        # Perform reranking on the retrieved documents
        documents = rerank_documents(pc, query, documents)

        # Initialize the chat model with a larger context window
        CHAT_MODEL = "gpt-3.5-turbo-16k"  # Use "gpt-4" if you have access
        chat = create_openai_chat(model=CHAT_MODEL, api_key=OPENAI_API_KEY)

        # Build the conversation history into the prompt
        conversation_str = "\n".join(
            f"User: {turn['user']}\nAssistant: {turn['bot']}" for turn in conversation_history
        )

        # Format documents into context string
        context_str = format_docs(documents)

        # RAG prompt with conversation history
        template = """You are a helpful assistant.
Here is the conversation so far:
{conversation}
Here is some context that may help answer the question:
{context}

Please answer the following question based only on the context provided. 
If the context does not contain enough information to answer the question, please indicate that you cannot answer based on the provided information.

Return your response in the following JSON format:

{{
    "answer": "Your answer here",
    "show_metadata": true
}}

Ensure that the JSON is valid and properly formatted.
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

        logging.debug(f"Raw LLM response: {result}")

        # Parse the JSON response
        try:
            parsed_result = json.loads(result)
            answer = parsed_result.get("answer", "I'm sorry, I couldn't find an answer to that.")
            show_metadata = parsed_result.get("show_metadata", False)

            return answer, show_metadata, metadata_list

        except json.JSONDecodeError as e:
            logging.error(f"Failed to parse LLM response as JSON: {e}")
            # Fallback: return the raw response and metadata
            return result, False, metadata_list

    except Exception as e:
        logging.error(f"Error during LLM execution: {e}")
        raise


if __name__ == "__main__":
    try:
        logging.info("Running the main function...")
        # Example conversation history; adjust as needed
        conversation_history = []
        # Capture both the result and metadata_list
        res, metadata_list = run_llm(query="what is langchain?", conversation_history=conversation_history)
        print("Answer:", res)
        print("Metadata List:", metadata_list)
    except Exception as e:
        logging.error(f"Error in main execution: {e}")
