"""
Streamlit app for a LANGCHAIN Chatbot with Memory and Metadata.

This app allows users to interact with a chatbot that uses LangChain and maintains conversation memory,
along with displaying metadata from retrieved documents.
"""

import streamlit as st
import logging
from backend.core_LCEL_memory import run_llm

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Set the title of the Streamlit app
st.title("🧠 LANGCHAIN Chatbot with Memory and Metadata")

# Add a button to reset the conversation
if st.button("Reset Conversation"):
    st.session_state.conversation = []

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
            if chat['metadata']:
                with st.expander("Show Retrieved Documents Metadata"):
                    for idx, metadata in enumerate(chat['metadata'], 1):
                        st.write(f"**Document {idx} Metadata:**")
                        st.json(metadata)

# Get user input using st.chat_input
user_input = st.chat_input("You:")

# Check if user_input is not None before processing
if user_input is not None:
    user_input = user_input.strip()
    if user_input:
        # Display a spinner while processing
        with st.spinner("Thinking..."):
            try:
                # Call run_llm with the user input and conversation history
                answer, metadata_list = run_llm(user_input, st.session_state.conversation)

                # Append the new interaction to the conversation history, including metadata
                st.session_state.conversation.append({
                    "user": user_input,
                    "bot": answer,
                    "metadata": metadata_list or []
                })

                # Display the new user message
                with st.chat_message("user"):
                    st.write(user_input)

                # Display the bot's response
                with st.chat_message("assistant"):
                    st.write(answer)
                    if metadata_list:
                        with st.expander("Show Retrieved Documents Metadata"):
                            for idx, metadata in enumerate(metadata_list, 1):
                                st.write(f"**Document {idx} Metadata:**")
                                st.json(metadata)
            except Exception as e:
                st.error(f"An error occurred: {e}")
                logging.exception("Error during chatbot processing")
