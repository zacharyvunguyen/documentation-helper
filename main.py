import streamlit as st
import logging
from backend.core_LCEL_memory import run_llm
import os
import json

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Set page configuration
st.set_page_config(
    page_title="Advanced Conversational AI Chatbot",
    page_icon="🤖",
    layout="wide",
)

# Sidebar Configuration
with st.sidebar:
    st.header("Settings")

    # Reset Conversation Button
    if st.button("Reset Conversation"):
        st.session_state.conversation = []
        st.experimental_rerun()

    # Display Options
    st.subheader("Display Options")
    show_metadata_option = st.checkbox("Show Retrieved Documents Metadata", value=True)
    dark_mode = st.checkbox("Dark Mode", value=False)

    # Theme Toggle
    if dark_mode:
        st.markdown(
            """
            <style>
            body {
                background-color: #2e2e2e;
                color: #ffffff;
            }
            </style>
            """,
            unsafe_allow_html=True
        )
    else:
        st.markdown(
            """
            <style>
            body {
                background-color: #ffffff;
                color: #000000;
            }
            </style>
            """,
            unsafe_allow_html=True
        )

    # LLM Settings
    st.subheader("LLM Configuration")
    llm_model = st.selectbox("Select LLM Model", ["gpt-3.5-turbo", "gpt-4", "gpt-3.5-turbo-16k"], index=1)
    temperature = st.slider("Temperature", min_value=0.0, max_value=1.0, value=0.7, step=0.1)
    max_tokens = st.number_input("Max Tokens", min_value=50, max_value=2048, value=500, step=50)

    # Datasource Information
    with st.expander("Datasource Information"):
        st.markdown("""
        **Vector Database:** Pinecone  
        **Embedding Model:** OpenAI Embeddings  
        **Reranker Model:** bge-reranker-v2-m3  
        **LLM Model:** {model}
        """.format(model=llm_model))
        st.markdown("""
        **API Keys:**  
        - OpenAI: Configured securely on the backend.  
        - Pinecone: Configured securely on the backend.
        """)
        st.markdown("""
        **Environment:**  
        - **Pinecone Environment:** `{env}`  
        - **Index Name:** `{index}`
        """.format(env=os.getenv("PINECONE_ENVIRONMENT"), index=os.getenv("PINECONE_INDEX_NAME")))

    # Help Section
    st.markdown("""
    ---
    ### Help & Documentation
    - **Ask Questions:** Type any question into the input box and press Enter.
    - **Reset Conversation:** Clears the current conversation history.
    - **Adjust Settings:** Use the settings above to customize the chatbot's behavior.
    - **Export History:** Download your conversation for future reference.

    For more information, visit our [documentation](https://yourdocumentationlink.com).
    """)

# Title and Intro
st.title("🤖 Advanced Conversational AI Chatbot")
st.markdown("""
Welcome to the **Advanced Conversational AI Chatbot**! This chatbot leverages state-of-the-art language models and memory to provide insightful responses.

Feel free to ask any questions or have a conversation. Your conversation history will be remembered throughout the session.
""")

# Export Conversation Button
if st.session_state.get('conversation'):
    st.download_button(
        label="📥 Download Conversation",
        data=json.dumps(st.session_state.conversation, indent=2),
        file_name='conversation_history.json',
        mime='application/json'
    )

# Initialize session state variables for conversation history
if 'conversation' not in st.session_state:
    st.session_state.conversation = []

# Display the conversation history using Streamlit's chat elements
if st.session_state.conversation:
    for chat in st.session_state.conversation:
        # Display user message
        with st.chat_message("user"):
            st.write(chat['user'])
        # Display bot response
        with st.chat_message("assistant"):
            st.write(chat['bot'])
            # Display metadata in an expandable section within the assistant's message
            if show_metadata_option and chat.get('metadata'):
                with st.expander("Show Retrieved Documents Metadata"):
                    for idx, metadata in enumerate(chat['metadata'], 1):
                        st.write(f"**Document {idx} Metadata:**")
                        st.json(metadata)

# Get user input using st.chat_input
user_input = st.chat_input("Type your message here...")

# Process user input
if user_input:
    user_input = user_input.strip()
    if user_input:
        # Display a spinner while processing
        with st.spinner("Thinking..."):
            try:
                # Call run_llm with the user input and conversation history
                answer, show_metadata, metadata_list = run_llm(
                    query=user_input,
                    conversation_history=st.session_state.conversation,
                    model=llm_model,  # Pass selected LLM model
                    temperature=temperature,  # Pass temperature
                    max_tokens=max_tokens  # Pass max tokens
                )

                # Append the new interaction to the conversation history
                st.session_state.conversation.append({
                    "user": user_input,
                    "bot": answer,
                    "metadata": metadata_list  # Always include metadata list
                })

                # Display the new user message
                with st.chat_message("user"):
                    st.write(user_input)

                # Display the bot's response
                with st.chat_message("assistant"):
                    st.write(answer)
                    # Display metadata if available and if the user opted to show it
                    if show_metadata_option and metadata_list:
                        with st.expander("Show Retrieved Documents Metadata"):
                            for idx, metadata in enumerate(metadata_list, 1):
                                st.write(f"**Document {idx} Metadata:**")
                                st.json(metadata)
            except Exception as e:
                st.error(f"An error occurred: {e}")
                logging.exception("Error during chatbot processing")

# Footer or credits
st.markdown("""
---
*Developed by Zachary Nguyen.*
""")
