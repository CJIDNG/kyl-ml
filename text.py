import streamlit as st
from langchain.chat_models import ChatOpenAI
from langchain.prompts.chat import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate
from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv()

# Get the API key for OpenAI from environment variables
openai_api_key = os.getenv("OPENAI_API_KEY")

# Initialize ChatOpenAI with the API key
llm = ChatOpenAI(temperature=0, openai_api_key=openai_api_key, model="gpt-4o")

def chat_with_document(document_text, prompt):
    try:
        # Create a prompt template for the chat model
        system_message = SystemMessagePromptTemplate.from_template("You are an assistant that provides information based on the given {document}. Limit your responses to 250 words or less")
        human_message = HumanMessagePromptTemplate.from_template("{prompt}")

        chat_prompt = ChatPromptTemplate.from_messages([system_message, human_message])

        # Generate the response using the chat model
        response = llm(chat_prompt.format_prompt(document=document_text, prompt=prompt).to_messages())
        return response.content
    except Exception as e:
        return f"Error: {e}"

st.set_page_config(layout='wide')

st.title("ChatDocument powered by Langchain and OpenAI")

input_doc = st.file_uploader("Upload your text document", type=['txt'])

if input_doc is not None:
    document_text = input_doc.read().decode('utf-8')

    col1, col2 = st.columns([1, 1])

    with col1:
        st.info("Document Uploaded Successfully")
        st.text_area("Document Content", document_text, height=300)

    with col2:
        st.info("Chat Below")

        input_text = st.text_area("Enter your query")

        if input_text:
            if st.button("Chat with Document"):
                st.info("Your Query: " + input_text)
                result = chat_with_document(document_text, input_text)
                if "Error" in result:
                    st.error(result)
                else:
                    st.success(result)

