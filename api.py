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
    The data provided in the csv files contains scraped tweets from twitter accounts of political leaders across the world. We want to use this data to build a tool that tells the public about the stance of these political figures on different issues. I will be asking you certain questions based on their positions on different issues and i expect you to use the data provided only to give me responses. Please find the project scope below:
Know your leader Project charter and scope statement Last updated:May 23, 2024 Introduction 2024 is set to be the biggest election in history with national elections being held in more than 60 countries totaling half of the world’s population. One of the most important aspects of any election is understanding where a candidate stands on a variety of issues. This means not only looking at what they’re promising but rather what their track record says about the positions on a variety of topics such as on the economy, social issues, foreign policy and more. Objective Know Your Leader will be an AI powered chatbot that will allow the public and journalists to check political speech. We will allow the public to ask questions such as “What has X person said about Y topic in the past?” The data will come from different sources to most accurately represent a candidate's position. By scouring video interviews, press releases, social media posts, news articles and PDF transcripts of their statements in Congress the platform will then extract all instances of what a person has said, how consistently they’ve said it, or perhaps most interestingly what they’ve not said about a particular topic. This will allow us to present a dynamic analysis of their changing viewpoints over time. Beyond elections, this platform could also help journalists, researchers, or the greater public gauge which public figures have been consistent with their messaging and values, who has spoken up, and who has remained silent. Scope The core of this project is to offer the public and journalists an easy-to-use platform to check political speech. Our ultimate goal is to build a generative artificial intelligence model with open-source technology that allows us to hold political leaders or public figures accountable for what they say in the public media. Administrators will have the ability to easily adapt the model for any kind of leader, by providing the information needed. We are aiming to use and refine the platform by applying it to some of the major upcoming presidential elections starting with South Africa, the UK, the US elections and beyond. While the goal is to be able to profile potentially dozens of candidates including local leaders, if we just focus on the presidential candidates we could still provide a lot of value to hundreds of millions of readers in several countries around the world. The minimum output we want to reach by the end of the year is an application that checks political speech for the US presidential candidates - likely between Joe Biden and Donald Trump. Our goal is to democratize access to political accountability, making it easier for citizens and journalists alike to fact-check and understand the politician's view about certain topics over the years. This project stands at the intersection of technology, journalism, and civic empowerment, harnessing the potential of generative AI to encourage our audience to be more informed and engaged with the news. ● End users: AI will be used to enhance the process for these two primary users Journalist - Anyone who wants to ‘generate’ a package for any upcoming election. User - Anyone who interacts with the page to learn about elections. Human resources ● AI mentor - 8 hours of consultation Timeline May - South Africa elections POC June - Data gathering July - UK elections POC August - Development September - Development October - Development November - US elections - main deliverable Budget ● TBD PART II - Editorial requirements Functional requirements Seven general categories divided by user journey: Category 1, 2 and 3 are all general knowledge to set the scene as a starter pack Category 4, 5 and 6 are the backbone of the experience as know your leader Category 7 are the user engagement features as get me involved Category Description [starter pack] South Africa elections at a glance Basic stats about the election so that people can familiarise themselves with the overall process. Journalist - Generate a prompt that spits out information for the following: Election day Country population Registered voters Voting age Time is polling Provinces/states Municipalities/constituencies Polling stations Presidential term Limit of terms Type of system E.g. Journalist types in: Generate Ghana elections at a glance Output 1: data of all these values Output 2: infographic of all these values User - Guide the user through these basic stats. Could be: What do I need to know about the South African elections? ARROWS - tell me more | tell me less [starter pack] What are the main election issues? Journalist - Generate a prompt that spits out information for the following: Top 20 election issues unique to that country E.g. Journalist types in: Generate Ghana elections main issues Output 1: data of all these values Output 2: infographic of all these values User - Guide the user through the basic election issues. Could be: What are the main election issues in South Africa? ARROWS - tell me more | tell me less [starter pack] Who are the candidates? Journalist - Generate a prompt that spits out information for the following: List out all the main candidates in the elections E.g. Journalist types in: Generate Ghana elections main candidates User - Guide the user through the list of candidates. Could be: Who are the main candidates? ARROWS - tell me more | tell me less Generate South Africa elections starter pack [know your leader] What is each leader’s position on X? Journalist - Generate a prompt that spits out information for the following: List of each candidate and their positions on all the main election issues. User - E.g. User types in: What is Jacob Zuma’s position on the economy? ARROWS - tell me more | tell me less [know your leader] How consistent has my candidate been on X? Journalist - Generate a prompt that spits out information for the following: How consistent has X candidate been on Y issue? User - E.g. User types in: Generate Jacob Zuma’s consistency scorecard on immigration ARROWS - tell me more | tell me less [know your leader] Who is most similar to my leader? Journalist - Generate a prompt that spits out information for the following: How similar is this leader to this other leader? User - E.g. User given the option to: Generate Jacob Zuma’s and Julius Malema similarity scorecard ARROWS - tell me more | tell me less Generate South Africa elections know your leader [get me involved] Who should I vote for? Might be a really valuable feature to then give readers some suggestion based on some principles they have Generate South Africa elections get me involved Story principles REUSABLE - The experience should be easy to replicate for any election SIMPLICITY - Users should never feel overwhelmed in the decisions they have to take JOURNEY - Users need to know what questions to ask. The experience should be a journey to get them to better understand their country’s election and their leaders. ENGAGEMENT - The experience should be very engaging by being very visual, interactive and never leave the reader with the question “What am I supposed to do next?” SHARABLE - The experience should allow people to share infographics or charts about their candidate on social media Editorial risk assessment and mitigation steps Risk Mitigation Incomplete data Utilise existing content from multiple LLMs Hallucination or misrepresenting a candidate Fields should not be totally open ended, rather they should focus solely on major election issues. Users should be told that upfront. Prompt injection Industry best practices Source of the data should be visible.
I want you to use the data provided in the file ONLY to generate responses and I want you to provide references of the tweets that you are using to generate  each of your responses


Here is the user question;
{question}

Here are the CSV;

{csv}


    """

    prompt = PromptTemplate(input_variables=["question", "csv"], template=template)
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
