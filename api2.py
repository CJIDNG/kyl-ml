from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import JSONResponse
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

app = FastAPI()

def chat_with_document(document_text, prompt):
    try:
        # Create a prompt template for the chat model
        system_message = SystemMessagePromptTemplate.from_template("You are an assistant that provides information based on the given {document}")
        human_message = HumanMessagePromptTemplate.from_template("{prompt}")

        chat_prompt = ChatPromptTemplate.from_messages([system_message, human_message])

        # Generate the response using the chat model
        response = llm(chat_prompt.format_prompt(document=document_text, prompt=prompt).to_messages())
        return response.content
    except Exception as e:
        return f"Error: {e}"

@app.post("/chat-with-document/")
async def chat_with_document_endpoint(file: UploadFile = File(...), query: str = Form(...)):
    try:
        document_text = (await file.read()).decode('utf-8')
        result = chat_with_document(document_text, query)
        return JSONResponse(content={"response": result})
    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)
