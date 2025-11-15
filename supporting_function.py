import re 
import os 
import streamlit as st 
from dotenv import load_dotenv
from youtube_transcript_api import YouTubeTranscriptApi
import time


from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_chroma import Chroma
from langchain.embeddings import HuggingFaceEmbeddings 
from sentence_transformers import SentenceTransformer
load_dotenv()

api_key = os.getenv("GEMINI_API_KEY")


# Function to extract video ID from a YouTube URL (Helper Funtion)
def extract_video_id(url):

    """
    Extract the YouTube video ID from any valid YouTube URL.
    """

    match = re.search(r"(?:v=|\/)([0-9A-Za-z_-]{11}).*",url) 
    if match:
        return match.group(1)

    st.error("Invalid YouTube URL. Please enter a valid video link.")
    return None  


# Function to get transcript from the video.
def get_transcript(video_id, language):
    ytt_api = YouTubeTranscriptApi()
    try:
        transcript = ytt_api.fetch(video_id, languages=[language])
        full_transcript = " ".join([i.text for i in transcript])
        time.sleep(10)
        return full_transcript
    
    except Exception as e:
        st.error(f"Error fetching video {e}")


# Function to translate the transcript into english. 
    # initialize the gemini model 

llm  = ChatGoogleGenerativeAI(
    model = "gemini-2.5-flash-lite",
    temperature=0.2,
    api_key=api_key
)


def translate_transcript(transcript):
    try:
        prompt = ChatPromptTemplate.from_template("""
        you are an expert translator with deep cultural and linguistic knowledge.
        I will provide you with a transcript. Your task is to translate it into English with absolute accuracy, preserving
        - Full meaning and context (one omissions, no additions).
        - Tone and style (formal/informal, emotional/neutral as in original).
        - Nuances, idioms, and cultural expressions (adapt appropriately while keeping intent).
        - Speaker's voice (same perspective, no rewriting into third-person).
        Do not summarize or simplify, The translation should read naturally in the target language but stay as close as possible to the original intent
                                                  
        Transcript:
        {transcript}
""")
        
        ## Runable chain 
        chain = prompt | llm

        # Run chain 
        response = chain.invoke({"transcript": transcript})

        return response.content

    except Exception as e : 
        st.error(f"Error fetching video {e}")


# id = extract_video_id("https://youtu.be/-xSJA8-o6Eg?si=114JoypJVK54q7DE")
# transcript = get_transcript(id, language='hi')
# print(transcript)
# final_output = translate_transcript(transcript)
# print(final_output)

# Function to get important topics 

def get_important_topics(transcript):

    try:
        prompt = ChatPromptTemplate.from_template("""
        You are an assistant that extracts the 5 most important topics discussed in a video transcript or summary.

        Rules:
        - Summarize into exactly 5 major points.
        - Each point should represent a key topic or concept, not small details.
        - Keep wording concise and focused on the technical content.
        - Do not phrase them as questions or opinions.
        - Output should be a numbered list.
        - show only points that are discussed in the transcript.
        Here is the transcript:
        {transcript}
        """)

        ## Runable chain 
        chain = prompt | llm

        # Run chain 
        response = chain.invoke({"transcript": transcript})

        return response.content

    except Exception as e : 
        st.error(f"Error fetching video {e}")

## FUNCTION TO GET NOTES FROM THE VIDEO

def generate_notes(transcript):

    try: 
        prompt = ChatPromptTemplate.from_template("""
                You are an AI note-taker. Your task is to read the following Youtube video transcript
                and produce well-structure, concise notes.
                                                  
                Requirements:
                - Present the output as **bulleted points**, grouped into clear sections.
                - Highlight key takeaways, important facts, and examples.
                - Use **short, clear sentences** (no long paragraphs).
                - If the transcript includes multiple themes, organize them under **subheadings**.
                - Do not add information that is not present in the transcript.
                                                  
                Here the transcript:
                {transcript}
             """)
        
        ## Runable chain 
        chain = prompt | llm

        # Run chain 
        response = chain.invoke({"transcript": transcript})

        return response.content

    except Exception as e : 
        st.error(f"Error fetching video {e}")



## FUNCTION TO CREATE CHUNKS 

def create_chunks(transcript):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size = 10000, chunk_overlap=1000)
    docs = text_splitter.create_documents([transcript])

    return docs

## function to create embeddings and store in vectore store 
def create_vector_store(docs):
    embeddings = GoogleGenerativeAIEmbeddings(model = "models/text-embedding-004",transport="rest",google_api_key=api_key)
    vector_store = Chroma.from_documents(docs,embeddings)

    return vector_store
    # embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    # vector_store = Chroma.from_documents(docs, embeddings)
    # return vector_store


# Rag Function

def rag_answer(question, vectorstore):
    results= vectorstore.similarity_search(question,k=4)
    context_text = "\n".join([i.page_content for i in results])

    prompt = ChatPromptTemplate.from_template("""
                You are a kind, polite, and precise assistant.
                - Begin with a warm and respectful greeting (avoid repeating greetings every turn).
                - Understand the user’s intent even with typos or grammatical mistakes.
                - Answer ONLY using the retrieved context.
                - If answer not in context, say:
                  "I couldn’t find that information in the database. Could you please rephrase or ask something else?"
                - Keep answers clear, concise, and friendly.

                Context:
                {context}

                User Question:
                {question}

                Answer:
                """)

    #chain
    chain = prompt|llm
    response= chain.invoke({"context":context_text,"question":question})

    return response.content
