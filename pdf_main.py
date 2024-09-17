import streamlit as st
from backend.core_LCEL_memory_pdf import run_llm
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Set the title of the Streamlit app
st.title("🧠 Chatbot with Memory and Metadata")

# Initialize session state variables for conversation history
if 'conversation' not in st.session_state:
    st.session_state.conversation = []

# Display the conversation history using Streamlit's chat elements
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
                "metadata": metadata_list
            })

            # Rerun the app to display the new message
            st.experimental_rerun()
        except Exception as e:
            st.error(f"An error occurred: {e}")
            logging.error(f"Error during chatbot processing: {e}")
