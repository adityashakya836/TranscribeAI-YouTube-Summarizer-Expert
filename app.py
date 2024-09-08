from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
import google.generativeai as genai
from langchain.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from langchain_community.document_loaders.youtube import YoutubeLoader
import json
import streamlit as st 
import re

# Set Google API Key
config_file_path = 'config.json'

# Open and read the JSON file
with open(config_file_path, 'r') as file:
    config_data = json.load(file)

GOOGLE_API_KEY = config_data['GOOGLE_API_Key']

# Transcribe YouTube Video
def transcribe_video(url):
    language_codes = ['en', 'hi', 'es', 'fr', 'de', 'ru', 'zh-Hans', 'ar', 'pt', 'ta', 'te', 'sw', 'ur', 'bn', 'mr', 'pa', 'vi']
    loader = YoutubeLoader.from_youtube_url(url, add_video_info=False, language = language_codes, translation='en')
    data = loader.load()
    return data[0].page_content

# Splitting the text into chunks
def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=10000,
        chunk_overlap=1000
    )

    chunks = text_splitter.split_text(text)
    return chunks

# Create a vector store
def get_vector_store(text_chunks):
    embeddings = GoogleGenerativeAIEmbeddings(model = 'models/embedding-001', google_api_key = GOOGLE_API_KEY)
    vector_store = FAISS.from_texts(text_chunks, embedding = embeddings)
    vector_store.save_local('faiss_index')

# Create a conversational_chain
def get_conversational_chain():
    prompt_template = """
        You are an insightful assistant with a talent for crafting clear, engaging, and well-structured answers. 
        Context (Video Transcript):
        {context}

        Question:
        {question}

        Answer : 
    """
    model = ChatGoogleGenerativeAI(model = 'gemini-pro', temperature = 0.7, google_api_key = GOOGLE_API_KEY)
    prompt = PromptTemplate(template = prompt_template, input_variables = ['context','question'])
    chain = load_qa_chain(model, chain_type="stuff", prompt = prompt)
    return chain

# Get Response
def get_response(user_query):
    embeddings = GoogleGenerativeAIEmbeddings(model = 'models/embedding-001', google_api_key = GOOGLE_API_KEY)
    new_db = FAISS.load_local('faiss_index', embeddings, allow_dangerous_deserialization=True)
    docs = new_db.similarity_search(user_query)
    chain = get_conversational_chain()

    response = chain(
        {
            "input_documents":docs,
            'question':user_query
        },
        return_only_outputs = True
    )

    st.write(response['output_text'])

# Streamlit app
def main():
    st.title("TranscribeAI: Unlock Insights from YouTube Video")
    user_query = st.text_input("Enter your query")
    if user_query:
        get_response(user_query)

    with st.sidebar:
        st.title("Transcribe Video")
        youtube_url = st.sidebar.text_input("Enter YouTube URL")
        # print(youtube_url)
        # Button to strart transcription
        if st.sidebar.button("Transcribe"):
            if youtube_url:
                with st.spinner('Transcribing video...'):
                    transcript = transcribe_video(youtube_url)
                    text_chunks = get_text_chunks(transcript)
                    get_vector_store(text_chunks)
                    st.success("Transcription successful!")

if __name__ == "__main__":
    main()