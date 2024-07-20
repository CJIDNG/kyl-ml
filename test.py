import streamlit as st 
from pandasai import SmartDataframe
from pandasai.llm import OpenAI
from dotenv import load_dotenv
import os
import pandas as pd

load_dotenv()

openai_api_key = os.getenv("OPENAI_API_KEY")

def chat_with_csv(df, prompt):
    # Instantiate the LLM
    # Convert to SmartDataframe
    smart_df = SmartDataframe(df, config = {
    "llm": OpenAI(api_token=openai_api_key, model="gpt-4o", temperature=0, seed=26),
    "custom_whitelisted_dependencies": ["ast"]  # Add 'ast' here
})

    # Create a prompt template
    prompt_template = """
    Introduction:
    You are an AI-powered chatbot for the "Know Your Leader" project, designed to help users understand the positions of political leaders based on their public statements. You will use the provided data from CSV files, which contain scraped tweets from political leaders' Twitter accounts. Your responses should only use the data from these tweets, and you must provide references for each tweet you use.

    Objective:
    Help users learn about political leaders' positions on various issues by answering their questions using the data provided. Always include references to the tweets you used for your responses.

    

    Response Format:
    - Answer the query based on the data provided in the CSV files.
    - Provide references for each tweet you used in the format: "Reference: [Tweet text] - [Date of Tweet] by [Leader's Name]".

    Example Queries and Responses:
    1. What has X person said about Y topic in the past?
    - Answer: [Answer based on data]
    - References:
        - [Tweet text] - [Date of Tweet] by [Leader's Name]
        - [Tweet text] - [Date of Tweet] by [Leader's Name]

    2. How consistent has X person been on Y topic?
    - Answer: [Answer based on data]
    - References:
        - [Tweet text] - [Date of Tweet] by [Leader's Name]
        - [Tweet text] - [Date of Tweet] by [Leader's Name]

    3. Cite your references with bullet points

    NOTE: Limit your references to a maximum of 3. Your responses should also a follow a life-like conversation model. 

    Please use this template for generating responses based on the provided data only.

    Prompt Template:
    Please answer the following query based on the provided DataFrame. Be concise and provide only the necessary information. If the query involves a list, present the results as a bullet list. Always include references to the tweets you used.

    NOTE: MAKE SURE YOUR CODE IS CORRECT AND COMPLETE BEFORE RUNNING IT

    Query: {prompt}

    """
    
    # Format the prompt
    formatted_prompt = prompt_template.format(prompt=prompt)

    # Query the SmartDataframe
    result = smart_df.chat(formatted_prompt)
    print(result)
    return result

st.set_page_config(layout='wide')

st.title("ChatCSV powered by LLM")

input_csv = st.file_uploader("Upload your CSV file", type=['csv'])

if input_csv is not None:
    col1, col2 = st.columns([1, 1])

    with col1:
        st.info("CSV Uploaded Successfully")
        data = pd.read_csv(input_csv)
        st.dataframe(data, use_container_width=True)

    with col2:
        st.info("Chat Below")
        
        input_text = st.text_area("Enter your query")

        if input_text:
            if st.button("Chat with CSV"):
                st.info("Your Query: " + input_text)
                try:
                    result = chat_with_csv(data, input_text)
                    st.success(result)
                except ValueError as e:
                    st.error(f"Error: {e}")
