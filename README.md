
# 🤖 Advanced Conversational AI Chatbot

Welcome to the **Advanced Conversational AI Chatbot** project! This chatbot leverages cutting-edge technologies such as Pinecone for vector indexing, LangChain for language model chaining, and OpenAI's GPT models for natural language understanding and generation. The application features a user-friendly interface built with Streamlit, enabling interactive conversations, dynamic settings adjustments, and seamless integration with various data sources.

---

## Table of Contents

- [🚀 Features](#-features)
- [🛠️ Technologies Used](#️-technologies-used)
- [🔧 Installation](#-installation)
- [⚙️ Configuration](#️-configuration)
- [📂 Project Structure](#-project-structure)
- [📑 Usage Guide](#-usage-guide)
- [🤝 Contributing](#-contributing)
- [📫 Contact](#-contact)
- [📸 Screenshots](#-screenshots)
- [📝 Additional Notes](#-additional-notes)

---

## 🚀 Features

- **Advanced AI Integration:** Utilizes Pinecone for efficient vector indexing, LangChain for language model operations, and OpenAI's GPT models for robust conversational capabilities.
- **Comprehensive Ingestion Pipeline:** Seamlessly loads, preprocesses, and indexes documents for quick retrieval and response generation.
- **User-Friendly Interface:** Built with Streamlit, offering an intuitive UI with customizable settings, conversation history, and metadata display.
- **Interactive Settings:** Adjust LLM configurations, toggle dark mode, reset conversations, and export chat history with ease.
- **Robust Error Handling:** Implements comprehensive logging and error management to ensure reliability and ease of maintenance.
- **Scalable Architecture:** Designed to handle large volumes of data and multiple user interactions efficiently.

---

## 🛠️ Technologies Used

- **Programming Language:** Python 3.x
- **AI & Machine Learning:**
  - [OpenAI GPT-4](https://openai.com/product/gpt-4)
  - [LangChain](https://langchain.com/)
  - [Pinecone](https://www.pinecone.io/)
- **Data Processing:**
  - [Pinecone Vector Store](https://www.pinecone.io/docs/)
  - [LangChain Text Splitter](https://langchain.com/docs/)
  - [ReadTheDocsLoader](https://langchain.com/docs/)
- **Web Framework:**
  - [Streamlit](https://streamlit.io/)
- **Environment Management:**
  - [dotenv](https://pypi.org/project/python-dotenv/)
- **Other Libraries:**
  - `logging`, `os`, `json`, `time`, etc.

---

## 🔧 Installation

Follow these steps to set up and run the Advanced Conversational AI Chatbot on your local machine.

### 1. **Clone the Repository**

```bash
git clone https://github.com/yourusername/advanced-conversational-ai-chatbot.git
cd advanced-conversational-ai-chatbot
```

### 2. **Create a Virtual Environment**

It's recommended to use a virtual environment to manage dependencies.

```bash
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 3. **Install Dependencies**

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

*If a `requirements.txt` file is not present, create one with the following content:*

```plaintext
python-dotenv
pinecone-client
langchain
langchain-community
langchain-openai
langchain-pinecone
streamlit
```

### 4. **Set Up Environment Variables**

Create a `.env` file in the root directory of the project and populate it with your API keys and configuration settings.

```bash
touch .env
```

**`.env` File Structure:**

```env
# OpenAI Configuration
OPENAI_API_KEY=your_openai_api_key
EMBED_MODEL=text-embedding-ada-002  # Or your preferred embedding model

# Pinecone Configuration
PINECONE_API_KEY=your_pinecone_api_key
PINECONE_INDEX_NAME=your_pinecone_index_name
INDEX_DIMENSION=1536  # Adjust based on your embedding model
INDEX_METRIC=cosine
PINECONE_CLOUD=aws  # Or your preferred cloud provider
PINECONE_REGION=us-east-1  # Adjust based on your cloud region
PINECONE_ENVIRONMENT=your_pinecone_environment  # e.g., "us-east1-gcp"
```

*Ensure that the `.env` file is added to `.gitignore` to prevent sensitive information from being exposed.*

---

## ⚙️ Configuration

### 1. **Pinecone Index Setup**

The ingestion script (`ingestion.py`) is responsible for setting up the Pinecone index. By default, it checks if the specified index exists and creates it if it doesn't. Ensure that your Pinecone account has the necessary permissions and that the index configuration matches your embedding dimensions and metrics.

### 2. **OpenAI API Configuration**

The chatbot utilizes OpenAI's GPT models for generating responses. Ensure that your OpenAI API key has the necessary access and that the specified embedding and chat models are available in your OpenAI account.

### 3. **Streamlit Settings**

The Streamlit frontend (`main.py`) offers various settings in the sidebar, including:

- **Reset Conversation:** Clears the current conversation history.
- **Display Options:** Toggle metadata display and dark mode.
- **LLM Configuration:** Select the LLM model, adjust temperature, and set maximum tokens.
- **Datasource Information:** View configuration details and API key statuses.

---

## 📂 Project Structure

```
advanced-conversational-ai-chatbot/
├── backend/
│   ├── core_LCEL_memory.py
│   └── __init__.py
├── ingestion.py
├── main.py
├── .env
├── .gitignore
├── README.md
└── requirements.txt
```

- **ingestion.py:** Handles document loading, preprocessing, embedding creation, and ingestion into Pinecone.
- **backend/core_LCEL_memory.py:** Contains the core logic for interacting with the LLM, including reranking and response generation.
- **main.py:** Streamlit application serving as the frontend interface for user interactions.
- **.env:** Stores environment variables and API keys.
- **requirements.txt:** Lists all Python dependencies required for the project.

---

## 📑 Usage Guide

### 1. **Data Ingestion**

Before running the chatbot, you need to ingest documents into the Pinecone index.

```bash
python ingestion.py
```

**What This Does:**

- Initializes Pinecone and checks or creates the specified index.
- Loads and preprocesses documents using `ReadTheDocsLoader`.
- Creates embeddings using OpenAI's embedding model.
- Ingests the document embeddings into Pinecone for efficient retrieval.

*Ensure that the ingestion process completes successfully by checking the logs.*

### 2. **Running the Streamlit Application**

Launch the Streamlit app to start interacting with the chatbot.

```bash
streamlit run main.py
```

**Accessing the App:**

Once the server starts, you'll see an output similar to:

```plaintext
  You can now view your Streamlit app in your browser.

  Local URL: http://localhost:8501
  Network URL: http://192.168.1.2:8501
```

Open the **Local URL** in your web browser to access the chatbot interface.

### 3. **Interacting with the Chatbot**

- **Send Messages:** Type your questions or prompts in the input box and press Enter to receive responses.
- **Customize Settings:** Use the sidebar to adjust LLM configurations, toggle display options, and reset conversations.
- **View Metadata:** Optionally display metadata from retrieved documents to understand the context of responses.
- **Export Conversation:** Download your conversation history as a JSON file for future reference.

---

## 🤝 Contributing

Contributions are welcome! Whether you're reporting a bug, suggesting a feature, or improving documentation, your input is valuable.

### 1. **Fork the Repository**

Click the **Fork** button at the top-right corner of the repository page to create a personal copy.

### 2. **Clone Your Fork**

```bash
git clone https://github.com/yourusername/advanced-conversational-ai-chatbot.git
cd advanced-conversational-ai-chatbot
```

### 3. **Create a New Branch**

```bash
git checkout -b feature/your-feature-name
```

### 4. **Make Your Changes**

Implement your feature or fix in the respective files.

### 5. **Commit Your Changes**

```bash
git add .
git commit -m "Add feature: your feature description"
```

### 6. **Push to Your Fork**

```bash
git push origin feature/your-feature-name
```

### 7. **Create a Pull Request**

Navigate to the original repository and create a pull request from your fork's branch. Provide a clear description of your changes for review.

*Please adhere to the [Code of Conduct](CODE_OF_CONDUCT.md) when contributing.*

---

## 📫 Contact

**Zachary Nguyen**

- **Email:** [zachary.nguyen@example.com](mailto:zachary.nguyen@example.com)
- **LinkedIn:** [linkedin.com/in/zacharynguyen](https://linkedin.com/in/zacharynguyen)
- **GitHub:** [github.com/zacharynguyen](https://github.com/zacharynguyen)

Feel free to reach out for any queries, suggestions, or collaborations!

---

## 📸 Screenshots

*Include screenshots of your application to give users a visual understanding.*

### **Chat Interface**

![Chat Interface](screenshots/chat_interface.png)

### **Settings Sidebar**

![Settings Sidebar](screenshots/settings_sidebar.png)

### **Metadata Display**

![Metadata Display](screenshots/metadata_display.png)

*Ensure that the `screenshots` directory contains the respective images.*

---

## 📝 Additional Notes

- **API Usage:** Be mindful of the API usage limits and costs associated with OpenAI and Pinecone services.
- **Security:** Ensure that all sensitive information, especially API keys, are securely managed and not exposed publicly.
- **Future Enhancements:** Consider adding features like multi-language support, voice interactions, or integrating additional data sources to enrich the chatbot's capabilities.

---

Thank you for checking out the Advanced Conversational AI Chatbot! We hope it serves as a robust foundation for building intelligent and interactive conversational agents.
```