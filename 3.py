import streamlit as st
import os
from langchain_groq import ChatGroq
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain.indexes import VectorstoreIndexCreator
from langchain.chains import RetrievalQA

load_dotenv() 
st.title("RAG CHATBOT !")
 # ensures GROQ_API_KEY is loaded


# SET UP A SESSION STATE VARIABLE TO HOLD ALL THE OLD MESSAGES

if 'messages' not in st.session_state:
    st.session_state.messages=[]

# DISPLAY ALL THE HISTORICAL MESSAGES
for message in st.session_state.messages:
    st.chat_message(message['role']).markdown(message['content'])

@st.cache_resource
def get_vectorstore():
    pdf_name = "./1.pdf"
    loaders = [PyPDFLoader(pdf_name)]

    #CREATE CHUNKS/VECTORS

    index = VectorstoreIndexCreator(
        embedding = HuggingFaceEmbeddings(model_name="all-MiniLM-L12-v2"),
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    ).from_loaders(loaders)
    return index.vectorstore







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
    try:
        vector_store =get_vectorstore()
        if vector_store is None:
            st.error("failed to load the document")
        
        chain = RetrievalQA.from_chain_type(
            llm = groq_chat,
            chain_type="stuff",
            retriever = vector_store.as_retriever(search_kwargs={"k": 3}),
            return_source_documents = True
        )

        result = chain({"query":prompt})
        response = result["result"]

        st.chat_message("assistant").markdown(response)
        st.session_state.messages.append({'role':'assistant', 'content':response})

    except Exception as e:
        st.error(f"Error:[{str(e)}]")