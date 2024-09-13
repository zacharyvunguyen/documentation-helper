import os
import logging
from dotenv import load_dotenv
from pinecone import Pinecone
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import ReadTheDocsLoader
from langchain_openai import OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from pinecone import ServerlessSpec
import time

# Load environment variables from .env file
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Load API keys and environment variables
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME")
INDEX_DIMENSION = int(os.getenv("INDEX_DIMENSION", 1536))
INDEX_METRIC = os.getenv("INDEX_METRIC", "cosine")
PINECONE_CLOUD = os.getenv("PINECONE_CLOUD", "aws")
PINECONE_REGION = os.getenv("PINECONE_REGION", "us-east-1")
EMBED_MODEL = os.getenv("EMBED_MODEL", "text-embedding-ada-002")


# Initialize OpenAI and Pinecone APIs
def initialize_pinecone():
    try:
        pc = Pinecone(api_key=PINECONE_API_KEY)
        logging.info("Pinecone initialized successfully.")
        return pc
    except Exception as e:
        logging.error(f"Failed to initialize Pinecone: {e}")
        raise


def check_or_create_index(pc):
    try:
        # Define ServerlessSpec
        spec = ServerlessSpec(cloud=PINECONE_CLOUD, region=PINECONE_REGION)

        # Check if index exists
        if PINECONE_INDEX_NAME in pc.list_indexes().names():
            logging.info(f"Index '{PINECONE_INDEX_NAME}' exists, deleting the existing index...")
            pc.delete_index(PINECONE_INDEX_NAME)

        # Create new index
        logging.info(
            f"Creating new index '{PINECONE_INDEX_NAME}' with dimension {INDEX_DIMENSION} and metric {INDEX_METRIC}.")
        pc.create_index(
            PINECONE_INDEX_NAME,
            dimension=int(INDEX_DIMENSION),  # Dimensionality of the embeddings
            metric=INDEX_METRIC,
            spec=spec
        )

        # Wait for the index to be ready
        logging.info(f"Waiting for index '{PINECONE_INDEX_NAME}' to be initialized...")
        while not pc.describe_index(PINECONE_INDEX_NAME).status['ready']:
            time.sleep(1)

        logging.info(f"Index '{PINECONE_INDEX_NAME}' is ready.")
        # Connect to the index
        index = pc.Index(PINECONE_INDEX_NAME)

        # Wait a moment for connection to establish
        time.sleep(1)

        # Return index and print index stats
        logging.info("Index stats:")
        logging.info(index.describe_index_stats())
        return index

    except Exception as e:
        logging.error(f"Failed to check or create index: {e}")
        raise


# Function to create embeddings
def create_embeddings():
    try:
        embeddings = OpenAIEmbeddings(model=EMBED_MODEL, openai_api_key=OPENAI_API_KEY)
        logging.info(f"OpenAI Embeddings initialized with model '{EMBED_MODEL}' successfully.")
        return embeddings
    except Exception as e:
        logging.error(f"Failed to initialize OpenAI Embeddings: {e}")
        raise


# Function to load and preprocess documents
def load_and_split_documents(loader):
    try:
        raw_documents = loader.load()
        logging.info(f"Loaded {len(raw_documents)} documents from the source.")

        text_splitter = RecursiveCharacterTextSplitter(chunk_size=600, chunk_overlap=50)
        documents = text_splitter.split_documents(raw_documents)

        # Update metadata with correct URL format
        for doc in documents:
            new_url = doc.metadata["source"].replace("langchain-docs", "https:/")
            doc.metadata.update({"source": new_url})

        logging.info(f"Preprocessed {len(documents)} document chunks.")
        return documents
    except Exception as e:
        logging.error(f"Error loading or splitting documents: {e}")
        raise


# Function to ingest documents into Pinecone
def ingest_documents_into_pinecone(documents, embeddings):
    try:
        logging.info(f"Adding {len(documents)} documents to Pinecone index '{PINECONE_INDEX_NAME}'")
        PineconeVectorStore.from_documents(documents, embeddings, index_name=PINECONE_INDEX_NAME)
        logging.info("Documents successfully loaded into Pinecone.")
    except Exception as e:
        logging.error(f"Failed to ingest documents into Pinecone: {e}")
        raise


# Main ingestion function
def ingest_docs():
    try:
        # Initialize Pinecone
        pc = initialize_pinecone()

        # Check or create Pinecone index
        index = check_or_create_index(pc)

        # Initialize Embeddings
        embeddings = create_embeddings()

        # Load and preprocess documents
        loader = ReadTheDocsLoader("langchain-docs/api.python.langchain.com/en/latest")
        documents = load_and_split_documents(loader)

        # Ingest documents into Pinecone
        ingest_documents_into_pinecone(documents, embeddings)
    except Exception as e:
        logging.error(f"Document ingestion process failed: {e}")
        raise


if __name__ == "__main__":
    ingest_docs()
