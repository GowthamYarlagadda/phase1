import streamlit as st
from langchain.chat_models import ChatOpenAI
from langchain.memory import ChatMessageHistory

# Initialize ChatMessageHistory
history = ChatMessageHistory()

# Streamlit app
def main():
    st.title("OpenAI Chatbot with Streamlit")

    # Get OpenAI API key from user input
    openai_api_key = st.text_input("Enter your OpenAI API key:", type="password")

    # Check if API key is provided
    if openai_api_key:
        # Initialize ChatOpenAI model with the provided API key
        llm = ChatOpenAI(api_key=openai_api_key, temperature=0, model="gpt-3.5-turbo-0613")

        # Get user input
        user_input = st.text_input("User Input:", "")

        if st.button("Submit"):
            # Run the chatbot model
            model_output = llm.run(user_input)

            # Display model output
            st.text("Chatbot Output:")
            st.text(model_output)

            # Add to history
            history.add_user_message(user_input)
            history.add_ai_message(model_output)

    # Display conversation history
    st.text("Conversation History:")
    for user_msg, ai_msg in zip(history.user_messages, history.ai_messages):
        st.text(f"User: {user_msg}")
        st.text(f"AI: {ai_msg}")
        st.text("-" * 30)

if __name__ == "__main__":
    main()
