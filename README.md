
# Advanced Conversational AI Chatbot with Memory and Reranking

![Chatbot Demo](path_to_demo_image_or_gif) <!-- Optional: Add a path to an image or GIF showcasing your chatbot UI -->

## Overview

This project demonstrates an advanced conversational AI chatbot that leverages LangChain, OpenAI GPT models, and Pinecone's vector search with reranking capabilities. The chatbot is designed to provide contextually relevant responses by integrating memory features and reranking retrieved documents based on their relevance to the query.

## Features

- **Retrieval-Augmented Generation (RAG)**: Combines OpenAI GPT models with Pinecone's vector search to retrieve and utilize contextually relevant documents.
- **Memory Integration**: Maintains conversation history to generate more coherent and context-aware responses.
- **Pinecone Reranking**: Enhances the relevance of the retrieved documents by reranking them based on the input query using Pinecone's reranking model.
- **Interactive UI**: User-friendly interface built with Streamlit for seamless interaction with the chatbot.

## Tech Stack

- **Language Model**: OpenAI GPT-3.5-turbo or GPT-4
- **Vector Search & Reranking**: Pinecone
- **Frameworks**: LangChain, Streamlit
- **Environment Management**: Python, Dotenv
- **Logging**: Python Logging Module

## Setup and Installation

1. **Clone the repository:**

   ```bash
   git clone https://github.com/your-username/advanced-conversational-ai-chatbot.git
   cd advanced-conversational-ai-chatbot
   ```

2. **Create a virtual environment:**

   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
   ```

3. **Install the required dependencies:**

   ```bash
   pip install -r requirements.txt
   ```

4. **Set up environment variables:**

   Create a `.env` file in the root directory and add your API keys:

   ```bash
   OPENAI_API_KEY=your_openai_api_key
   PINECONE_API_KEY=your_pinecone_api_key
   PINECONE_ENVIRONMENT=your_pinecone_environment
   PINECONE_INDEX_NAME=your_pinecone_index_name
   EMBED_MODEL=text-embedding-3-small  # or your preferred embedding model
   ```

5. **Run the Streamlit app:**

   ```bash
   streamlit run app.py
   ```

6. **Access the app:**

   Open your browser and go to `http://localhost:8501`.

## Usage

1. **Interact with the Chatbot:**

   Enter your query in the input box and receive contextually relevant responses, enhanced by memory and reranking.

2. **Reset Conversation:**

   Use the "Reset Conversation" button to clear the conversation history and start a new session.

3. **View Metadata:**

   Expand the "Show Retrieved Documents Metadata" section within the assistant's message to view details about the documents used to generate the response.

## Project Structure

```bash
├── backend
│   ├── core_LCEL_memory.py     # Core logic for document retrieval, reranking, and response generation
├── app.py                      # Streamlit app script
├── .env                        # Environment variables
├── requirements.txt            # Python dependencies
├── README.md                   # Project documentation
```

## Contributing

Contributions are welcome! If you have suggestions for improvements or new features, please feel free to create a pull request or open an issue.

## Acknowledgements

- [OpenAI](https://openai.com) for providing the GPT language models.
- [Pinecone](https://www.pinecone.io) for the vector search and reranking API.
- [LangChain](https://langchain.readthedocs.io/en/latest/) for prompt engineering tools.

## Contact

For any questions or inquiries, please contact [zacharynguyen.ds@gmail.com](mailto:zacharynguyen.ds@gmail.com).
