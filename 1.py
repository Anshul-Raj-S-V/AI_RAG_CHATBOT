import streamlit as st
import os
from langchain_groq import ChatGroq
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv



load_dotenv() 
st.title("RAG CHATBOT !")
 # ensures GROQ_API_KEY is loaded


# SET UP A SESSION STATE VARIABLE TO HOLD ALL THE OLD MESSAGES

if 'messages' not in st.session_state:
    st.session_state.messages=[]

# DISPLAY ALL THE HISTORICAL MESSAGES
for message in st.session_state.messages:
    st.chat_message(message['role']).markdown(message['content'])

prompt = st.chat_input("GIVE YOUR PROMPT")

if prompt:
    st.chat_message("user").markdown(prompt)

    st.session_state.messages.append({'role':'user', 'content':prompt})

    groq_sys_prompt = ChatPromptTemplate.from_template(
    """You are very smart at everything, you always give the best, most accurate answer.
    Answer the following question: {user_prompt}.
    Start the answer directly. No small talk please."""
)

    model = "llama-3.1-8b-instant"
    groq_chat = ChatGroq(
        groq_api_key = os.environ.get("GROQ_API_KEY"),
        model_name= model
    )


    parser = StrOutputParser()
    chain = groq_sys_prompt | groq_chat | parser

    response= chain.invoke({"user_prompt":prompt})


    st.chat_message("assistant").markdown(response)
    st.session_state.messages.append({'role':'assistant', 'content':response})