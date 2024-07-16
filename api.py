import tempfile
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from langchain.document_loaders.csv_loader import CSVLoader
from langchain.vectorstores import FAISS
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.prompts import PromptTemplate
from langchain.chat_models import ChatOpenAI
from langchain.chains import LLMChain
from dotenv import load_dotenv

load_dotenv()

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Adjust this to allow only specific origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Declare the vector_store as a global variable
vector_store = None

class QueryRequest(BaseModel):
    question: str

class QueryResponse(BaseModel):
    response: str

# Process uploaded csv file
def process_document(file: UploadFile): 
    with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
        tmp_file.write(file.file.read())
        tmp_file_path = tmp_file.name

    loader = CSVLoader(file_path=tmp_file_path, encoding="utf-8", csv_args={'delimiter': ','})
    document = loader.load()
    return document

# Initialize and return a chat bot
def init_chat_bot():
    llm = ChatOpenAI(temperature=0, model="gpt-4o")

    template = """
    The data provided in the csv files contains scraped tweets from the twitter accounts of political figures. You will be using this data to answer user queries on political figures that are in the CSV files. While answering user questions,
    follow the guidelines below;

    1/ Make your response as concise and clear as possible
    2/ Do not pick sides or have any political allegiance with anyone or any party
    3/ Ensure your response resonates with the user's query.
    4/ Keep it concise and easy to grasp.
    5/ NOTE: MAKE YOUR RESPONSES VERY WELL-EXPLAINED
    6/ Pay close attention to the full_text column in the csv and also use it to answer questions the user throws at you

    User Question:
    {question}

    Now, provide us with your answer
    """

    prompt = PromptTemplate(input_variables=["question"], template=template)
    chat_bot = LLMChain(llm=llm, prompt=prompt)
    return chat_bot

# Query Vector Store similar embeddings
def query_store(query: str, store: FAISS):
    response = store.similarity_search(query, k=3)
    contents = [doc.page_content for doc in response]
    return contents

# Queries the chat bot for a response
def query_chat_bot(question: str, vector_store: FAISS):
    ideas = query_store(question, vector_store)
    chat_bot = init_chat_bot()
    response = chat_bot.run(question=question, ideas=ideas)
    return response

@app.post("/upload-csv/")
async def upload_csv(file: UploadFile = File(...)):
    global vector_store
    try:
        document = process_document(file)
        embeddings = OpenAIEmbeddings()
        vector_store = FAISS.from_documents(document, embeddings)
        return {"message": "File processed and vector store created successfully"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/query/", response_model=QueryResponse)
async def query(request: QueryRequest):
    global vector_store
    try:
        if vector_store is None:
            raise HTTPException(status_code=400, detail="Vector store not initialized. Upload a CSV file first.")
        response = query_chat_bot(request.question, vector_store)
        return QueryResponse(response=response)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
