import os
import logging
from dotenv import load_dotenv
from pinecone import Pinecone, ServerlessSpec
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import ReadTheDocsLoader
from langchain_openai import OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain.document_loaders import PyPDFLoader
import time

# Load environment variables from .env file
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Load API keys and environment variables
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
#PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME")
PINECONE_INDEX_NAME = "langchain-pdf-index"
INDEX_DIMENSION = int(os.getenv("INDEX_DIMENSION", 1536))
INDEX_METRIC = os.getenv("INDEX_METRIC", "cosine")
PINECONE_CLOUD = os.getenv("PINECONE_CLOUD", "aws")
PINECONE_REGION = os.getenv("PINECONE_REGION", "us-east-1")
EMBED_MODEL = os.getenv("EMBED_MODEL", "text-embedding-3-small")


# Initialize Pinecone client and check/create index
def initialize_pinecone():
    pc = Pinecone(api_key=PINECONE_API_KEY)
    spec = ServerlessSpec(cloud=PINECONE_CLOUD, region=PINECONE_REGION)
    if PINECONE_INDEX_NAME in pc.list_indexes().names():
        logging.info(f"Deleting existing index '{PINECONE_INDEX_NAME}'...")
        pc.delete_index(PINECONE_INDEX_NAME)
    logging.info(f"Creating index '{PINECONE_INDEX_NAME}' with dimension {INDEX_DIMENSION}.")
    pc.create_index(PINECONE_INDEX_NAME, dimension=INDEX_DIMENSION, metric=INDEX_METRIC, spec=spec)
    while not pc.describe_index(PINECONE_INDEX_NAME).status['ready']:
        time.sleep(1)
    return pc.Index(PINECONE_INDEX_NAME)


# Create OpenAI embeddings
def create_embeddings():
    return OpenAIEmbeddings(model=EMBED_MODEL, openai_api_key=OPENAI_API_KEY)


# Load and preprocess PDF documents from 'pdf_files' folder
def load_and_split_documents_from_folder(folder_path='pdf_files'):
    all_documents = []
    logging.info(f"Loading and processing PDF files from folder: {folder_path}")

    # Iterate through each file in the folder
    for filename in os.listdir(folder_path):
        if filename.endswith(".pdf"):
            file_path = os.path.join(folder_path, filename)
            logging.info(f"Loading PDF file: {file_path}")

            # Load the PDF document
            loader = PyPDFLoader(file_path)
            raw_documents = loader.load()

            # Split the document content
            splitter = RecursiveCharacterTextSplitter(chunk_size=600, chunk_overlap=50)
            split_documents = splitter.split_documents(raw_documents)

            # Add documents to the list
            all_documents.extend(split_documents)

    logging.info(f"Loaded and split {len(all_documents)} document chunks from PDFs.")
    return all_documents


# Ingest documents into Pinecone
def ingest_documents(documents, embeddings):
    logging.info(f"Ingesting {len(documents)} documents into Pinecone...")
    PineconeVectorStore.from_documents(documents, embeddings, index_name=PINECONE_INDEX_NAME)


# Main function
def ingest_docs():
    try:
        # Initialize Pinecone
        index = initialize_pinecone()

        # Initialize Embeddings
        embeddings = create_embeddings()

        # Load and preprocess documents from 'pdf_files' folder
        documents = load_and_split_documents_from_folder('pdf_files')

        # Ingest documents into Pinecone
        ingest_documents(documents, embeddings)

        logging.info("Document ingestion completed successfully.")
    except Exception as e:
        logging.error(f"Ingestion failed: {e}")
        raise


if __name__ == "__main__":
    ingest_docs()
