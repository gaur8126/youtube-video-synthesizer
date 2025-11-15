__import__("pysqlite3")
import sys
sys.modules["sqlite3"] = sys.modules.pop("pysqlite3")



import streamlit as st
from supporting_function import (
    extract_video_id,
    get_important_topics,
    get_transcript,
    translate_transcript,
    generate_notes,
    create_chunks,
    create_vector_store,
    rag_answer
)

# sidebar (Inputs)

with st.sidebar:
    st.title("🎬 VidSynth AI")
    st.markdown("---")
    st.markdown("Transform any Youtube video into key topics, a podcast, or a chatbot.")
    st.markdown("### Input Details")

    youtube_url = st.text_input("YouTube URL", placeholder="https://www.youtube.com/watch?v=...")
    language = st.text_input("Video Langauge Code", placeholder="e.g. en, hi, es, fr", value="en")

    task_option = st.radio(
        "Choose what you want to generate:",
        ["Chat with Video", "Notes For You"]
    )

    submit_button = st.button("✨ Start Processing")
    st.markdown("---")

    # The "New Chat" has been removed from here 


# Main Page 
st.title("Youtube Content Synthesizer")
st.markdown("Paste a video link and select a task from the sidebar.")



# --- Process Flow --- 
if submit_button:
    if youtube_url and language:
        video_id = extract_video_id(youtube_url)
        if video_id:
            with st.spinner("Step 1/3: Fetching Transcript......"):
                full_transcript = get_transcript(video_id,language)

                if language != "en":
                    with st.spinner("Step 1.5/3: Translating Transcript into English, This may take few moments"):
                        full_transcript = translate_transcript(video_id)

            if task_option == "Notes For You":
                with st.spinner("Step 2/3: Extracting important Topics..."):
                    important_topic = get_important_topics(full_transcript)
                    st.subheader("Important Topics")
                    st.write(important_topic)
                    st.markdown("---")

                with st.spinner("Step 3/3: Generating Notes for You"):

                    notes = generate_notes(full_transcript)
                    st.subheader("Notes for you")
                    st.write(notes)

                st.success("Summary and Notes Generated")

            if task_option == "Chat with Video":
                with st.spinner("2/3: Creating Chunks and vector store....."):
                    chunks = create_chunks(full_transcript)
                    vectorestore = create_vector_store(chunks)
                    st.session_state.vector_store = vectorestore

                st.session_state.messages = []

            


###  Chatbot session 
if task_option == "Chat with Video" and "vector_store" in st.session_state:
    st.divider()
    st.subheader("Chatwith Video")

    # Display th eentire history
    for message in st.session_state.get("messages",[]):
        with st.chat_message(message['role']):
            st.write(message['content'])

    # user_input 
    prompt = st.chat_input("Ask me anything about video.")
    if prompt:
        st.session_state.messages.append({"role":"user","content":prompt})
        with st.chat_message("user"):
            st.write(prompt)

        with st.chat_message("assistant"):
            response = rag_answer(prompt,st.session_state.vector_store)
            st.write(response)

        st.session_state.messages.append({"role":"assistant","content":response})
        
