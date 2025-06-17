import os
import streamlit as st
from youtube_transcript_api import YouTubeTranscriptApi, TranscriptsDisabled
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.chains.question_answering import load_qa_chain
from langchain_core.prompts import PromptTemplate

# Set API Key
os.environ["OPENAI_API_KEY"] = st.secrets["OPENAI_API_KEY"]

# Page Config
st.set_page_config(page_title="📺 YouTube RAG Chatbot", page_icon="🤖", layout="wide")

# Header
st.markdown(
    """
    <style>
    /* Title styling */
    .title {
        font-size: 48px;
        font-weight: 800;
        color: white;
        background: linear-gradient(to right, #1e3c72, #2a5298);
        padding: 20px;
        border-radius: 12px;
        text-align: center;
        margin-bottom: 20px;
    }

    /* Subtitle styling */
    .subtitle {
        font-size: 20px;
        font-weight: 400;
        color: #dddddd;
        text-align: center;
        margin-bottom: 30px;
    }

    /* Input box styling */
    .stTextInput > div > div > input {
        background-color: #1e1e1e !important;
        color: #ffffff !important;
        border: 1px solid #444 !important;
        border-radius: 8px;
        padding: 10px;
    }
    </style>

    <div class="title">🎬 NOVA – Your Personalized YouTube Assistant</div>
    <p class="subtitle">Ask query based on a Youtube video transcript</p>
    """,
    unsafe_allow_html=True
)


# Sidebar Controls
with st.sidebar:
    st.image("https://upload.wikimedia.org/wikipedia/commons/e/ef/Youtube_logo.png", width=180)
    st.markdown("### 🧾 Video Input")
    video_id = st.text_input("Enter YouTube Video ID", value="X0btK9X0Xnk", help="Paste the ID (the part after 'v=')")
    lang = st.selectbox("Select Transcript Language", ["en", "hi","ja","de","fr","es","ar"])
    k_chunks = st.slider("Top K Chunks to Search", 1, 10, 3)
    if st.button("⚡ Load & Process"):
        st.session_state.load_transcript = True

# Session State Initialization
if "load_transcript" not in st.session_state:
    st.session_state.load_transcript = False

# Transcript Loader
def get_transcript(video_id, lang):
    try:
        transcript_text = YouTubeTranscriptApi.get_transcript(video_id, languages=[lang])
        return " ".join(chunk['text'] for chunk in transcript_text)
    except TranscriptsDisabled:
        st.error("Transcript not available for this video.")
        return None

# Text Processing
def process_transcript(transcript):
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    chunks = splitter.split_text(transcript)
    vector_db = FAISS.from_texts(chunks, embedding=OpenAIEmbeddings())
    return vector_db

# Answer Generator
def generate_response(vectordb, query, k_chunks):
    retriever = vectordb.as_retriever(search_kwargs={"k": k_chunks})
    docs = retriever.invoke(query)
    llm = ChatOpenAI(model_name="gpt-3.5-turbo")
    prompt_template = PromptTemplate(
        input_variables=["context", "question"],
        template="Answer the question based on the context:\n\nContext: {context}\n\nQuestion: {question}"
    )
    chain = load_qa_chain(llm, chain_type="stuff", prompt=prompt_template)
    return chain.run(input_documents=docs, question=query)

# Main Workflow
if st.session_state.load_transcript:
    with st.spinner("⏳ Fetching and embedding transcript..."):
        transcript = get_transcript(video_id, lang)
        if transcript:
            vectordb = process_transcript(transcript)
            st.success("✅ Transcript processed successfully!")

            # Input field with chat-style
            question = st.text_input("🧠 Ask a query about the video:")
            if question:
                with st.spinner("🔍 Generating answer..."):
                    response = generate_response(vectordb, question, k_chunks)
                    st.markdown("### 💬 Answer:")
                    st.info(response)
