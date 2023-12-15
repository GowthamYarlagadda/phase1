import streamlit as st
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.chat_models import ChatOpenAI
from langchain.memory import ChatMessageHistory, ConversationBufferMemory
import re

llm = ChatOpenAI(temperature=0, model="gpt-3.5-turbo-0613")

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
history = ChatMessageHistory()
memory = ConversationBufferMemory(llm=llm, chat_memory=history, memory_key="history")
llm_chain = LLMChain(prompt=prompt, llm=llm, memory=memory)


def trim(input_str):
    input_str = re.sub(r'\([^]+)\*', '', input_str)
    question_index = input_str.find('Question:')
    if question_index == -1:
        return input_str.strip()
    trimmed_str = input_str[:question_index].strip()
    return trimmed_str


def add_to_history(user_input, model_output):
    history.add_user_message(user_input)
    history.add_ai_message(model_output)


def run_model(message, history):
    if message.lower() == 'bye':
        history.clear()
        return None
    else:
        model_output = llm_chain.run(message)
        return trim(model_output)


def main():
    st.title("Chatbot with Streamlit")

    api_key = st.text_input("Enter your OpenAI API key:", type="password")

    if api_key:
        st.write("You've entered the OpenAI API key.")
        user_input = st.text_input("You:", "")

        if st.button("Send"):
            st.write("Bot:", run_model(user_input, history))

        if st.button("Clear History"):
            history.clear()
    else:
        st.warning("Please enter your OpenAI API key.")


if __name__ == "__main__":
    main()
