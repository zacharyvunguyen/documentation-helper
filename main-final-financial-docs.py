import streamlit as st
import logging
from backend.core_LCEL_memory import run_llm

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Set page configuration
st.set_page_config(
    page_title="Advanced Conversational AI Chatbot",
    page_icon="🤖",
    layout="wide",
)

# Set the title of the Streamlit app
st.title("🤖 Advanced Conversational AI Chatbot")

# Add an introductory message
st.markdown("""
Welcome to the **Advanced Conversational AI Chatbot**! This chatbot leverages state-of-the-art language models and memory to provide insightful responses.

Feel free to ask any questions or have a conversation. Your conversation history will be remembered throughout the session.
""")

# Add a sidebar with reset button and additional info
with st.sidebar:
    st.header("Settings")
    if st.button("Reset Conversation"):
        st.session_state.conversation = []
        st.experimental_rerun()
    st.markdown("### Display Options")
    show_metadata_option = st.checkbox("Show Retrieved Documents Metadata", value=True)
    st.markdown("""
    ### Instructions
    - Type your message in the input box at the bottom.
    - The chatbot will respond accordingly.
    - You can reset the conversation using the button above.

    ### About
    This chatbot is powered by OpenAI's GPT models and Pinecone for document retrieval and reranking.
    """)

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

# Check if user_input is not None before processing
if user_input is not None:
    user_input = user_input.strip()
    if user_input:
        # Display a spinner while processing
        with st.spinner("Thinking..."):
            try:
                # Call run_llm with the user input and conversation history
                answer, show_metadata, metadata_list = run_llm(user_input, st.session_state.conversation)

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

# Optionally, we can add a footer or credits
st.markdown("""
---
*Developed by Zachary Nguyen*
""")
