import streamlit as st
import openai
from langchain.chat_models import ChatOpenAI
from langchain.memory import ChatMessageHistory
from langchain import PromptTemplate

# Set your OpenAI API key
openai.api_key = "YOUR_OPENAI_API_KEY"

# Initialize ChatOpenAI model
llm = ChatOpenAI(temperature=0, model="gpt-3.5-turbo-0613")

# Initialize ChatMessageHistory
history = ChatMessageHistory()

# Define basic_prompt and PromptTemplate
basic_prompt = """
Role: Compassionate therapist maintaining warmth and empathy. Respond with shorter responses.
Instructions:
- Start warmly: Greet and acknowledge emotions to encourage openness.
- Listen actively: Respond thoughtfully, keeping responses short.
...
- Avoid clinical jargon: Use accessible language.
- No quick solutions: Prioritize understanding before suggesting.
Goal:
- At the end, After gaining insight into their concerns, categorize their challenges into Depression, Social Issues, Anxiety, or Bipolar disorder.
  and use this statement exactly as it is: "It seems that your concerns might be connected to ____."
Maintain empathy and connection for a supportive environment.
Question: {question}
Answer:
"""

prompt = PromptTemplate(template=basic_prompt, input_variables=["question"])

# Streamlit app
def main():
    st.title("OpenAI Chatbot with Streamlit")

    # Get OpenAI API key from user input
    openai_api_key = st.text_input("Enter your OpenAI API key:", type="password")

    # Check if API key is provided
    if openai_api_key:
        openai.api_key = openai_api_key

        # Get user input
        user_input = st.text_input("User Input:", "")

        if st.button("Submit"):
            # Run the model
            model_output = llm.run(user_input)

            # Display model output
            st.text("Model Output:")
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
