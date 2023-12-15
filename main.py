import os
import streamlit as st
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.chat_models import ChatOpenAI
from langchain.memory import ChatMessageHistory, ConversationBufferMemory

# Function to initialize the ChatOpenAI model with the provided API key
def initialize_chat_model(api_key):
    return ChatOpenAI(temperature=0, model="gpt-3.5-turbo-0613", api_key=api_key)

# Function to initialize the LLMChain with the provided model and memory
basic_prompt = """
Role: Compassionate therapist maintaining warmth and empathy. Respond with shorter responses.
Instructions:
- Start warmly: Greet and acknowledge emotions to encourage openness.
- Listen actively: Respond thoughtfully, keeping responses short.
- Build connection: Avoid conclusions; understand the client's situation.
- Be present: Engage for 10-15 exchanges, categorize challenges sensitively.
- Communicate understanding: Use the specific statement provided.
- Don't rush or interrupt: Allow the client to share at their pace.
- Avoid persistent questioning: Talk naturally, don't demand straight answers.
- Delay specific disorders: Understand before labeling; focus on experiences.
- Refrain from stereotyping: Treat each client individually.
- Use emojis thoughtfully: Express support without excess.
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
    memory = ConversationBufferMemory(llm=model, chat_memory=history, memory_key="history")
    return LLMChain(prompt=prompt, llm=model, memory=memory)

# Function to trim input string
def trim(input_str):
    input_str = re.sub(r'\([^]+)\*', '', input_str)
    question_index = input_str.find('Question:')
    
    if question_index == -1:
        return input_str.strip()

    trimmed_str = input_str[:question_index].strip()
    return trimmed_str

# Function to add user input and model output to history
def add_to_history(user_input, model_output, history):
    history.add_user_message(user_input)
    history.add_ai_message(model_output)

# Function to run the model
def run_model(message, llm_chain, history):
    if message.lower() == 'bye':
        history.clear()
        return None
    else:
        model_output = llm_chain.run(message)
        add_to_history(message, trim(model_output), history)
        return trim(model_output)

# Streamlit App
def main():
    st.title("Chatbot with OpenAI")

    # Get OpenAI API key from user input
    api_key = st.text_input("Enter your OpenAI API key:", type="password")

    if st.button("Initialize Chatbot"):
        if api_key:
            st.success("Chatbot Initialized!")
            model = initialize_chat_model(api_key)
            history = ChatMessageHistory()
            llm_chain = initialize_llm_chain(model, history)

            # Chat interface
            user_input = st.text_input("You:", "")
            if st.button("Send"):
                if user_input:
                    response = run_model(user_input, llm_chain, history)
                    st.text_area("Chatbot:", value=response, height=100)

    st.warning("Remember to keep your API key secure and do not share it publicly.")

if __name__ == "__main__":
    main()
