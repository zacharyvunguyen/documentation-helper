import os
import logging
from dotenv import load_dotenv
from langchain_community.chat_models import ChatOpenAI
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableParallel, RunnablePassthrough
from langchain_pinecone import PineconeVectorStore
from langchain_core.pydantic_v1 import BaseModel
from pinecone import Pinecone
from core_functions import initialize_pinecone, create_openai_embeddings,create_openai_chat

# Load environment variables from .env file
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Load API keys and environment variables
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_ENVIRONMENT = os.getenv("PINECONE_ENVIRONMENT")
#PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME")
PINECONE_INDEX_NAME = "langchain-pdf-index"
EMBED_MODEL = os.getenv("EMBED_MODEL", "text-embedding-3-small")

logging.info(f"Environment variables loaded: OpenAI Model={EMBED_MODEL}, Pinecone Index={PINECONE_INDEX_NAME}")


# Function to format retrieved documents into a single string
def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

# Add typing for input using BaseModel
class Question(BaseModel):
    __root__: str

# Run LLM function
def run_llm(query: str):
    try:
        logging.info(f"Starting LLM execution with query: {query}")

        # Initialize Pinecone client
        initialize_pinecone(PINECONE_API_KEY)

        # Create embeddings using OpenAI with the API key
        embeddings = create_openai_embeddings(EMBED_MODEL, OPENAI_API_KEY)

        # Connect to Pinecone index using embeddings
        logging.info(f"Connecting to Pinecone index '{PINECONE_INDEX_NAME}'...")
        vectorstore = PineconeVectorStore.from_existing_index(
            PINECONE_INDEX_NAME, embeddings
        )
        retriever = vectorstore.as_retriever()
        logging.info(f"Connected to Pinecone index '{PINECONE_INDEX_NAME}'.")

        # Retrieve documents based on the query
        docs = retriever.get_relevant_documents(query)
        if not docs:
            logging.info("No documents retrieved for the query.")
            return {"output": "No relevant documents found."}

        # Format the retrieved documents into a single string
        formatted_docs = format_docs(docs)

        # RAG prompt template
        template = """Answer the question based only on the following context:
        {context}
        Question: {question}
        """
        prompt = ChatPromptTemplate.from_template(template)

        # Initialize the Chat model
        chat_model = create_openai_chat(model="gpt-4o-mini", api_key=OPENAI_API_KEY)

        # Create the RAG chain with RunnablePassthrough for both context and question
        chain = (
            RunnableParallel({"context": RunnablePassthrough(), "question": RunnablePassthrough()})
            | prompt
            | chat_model
            | StrOutputParser()
        )

        # Add typing for input with BaseModel `Question`
        chain = chain.with_types(input_type=Question)

        # Invoke the chain with the query
        logging.info(f"Invoking the chain with query: {query}")
        result = chain.invoke({"context": formatted_docs, "question": query})

        # Log and check the result structure
        logging.info(f"Raw result from chain: {result}")

        # Check if result is a string or dictionary
        if isinstance(result, dict):
            output = result.get("output", "No output found.")
        else:
            output = result

        logging.info(f"LLM execution completed successfully with result: {output}")
        return {"output": output}

    except Exception as e:
        logging.error(f"Error during LLM execution: {e}")
        raise

# Main execution
if __name__ == "__main__":
    try:
        # Run the LLM with a sample query
        logging.info("Running the main function...")
        res = run_llm(query="Give me some information about unemployment rate in middle tennessee?")
        logging.info(f"Final result: {res['output']}")
        print(res["output"])
    except Exception as e:
        logging.error(f"Error in main execution: {e}")
