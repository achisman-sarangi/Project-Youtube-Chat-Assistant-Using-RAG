# 🎬 NOVA – Your Personalized YouTube Chatbot

This Streamlit app allows you to ask questions based on any YouTube video's transcript using Retrieval-Augmented Generation (RAG) with LangChain and OpenAI.

## 🚀 Live App

🔗 [Try the app on Streamlit Cloud](https://project-youtube-chat-assistant-using-rag-fvvvqsacphiknxh9otaey.streamlit.app/)

[![Open in Streamlit](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://project-youtube-chat-assistant-using-rag-fvvvqsacphiknxh9otaey.streamlit.app/)

## 📦 Features

- Extracts YouTube video transcript
- Splits and embeds the transcript using LangChain and FAISS
- Retrieves relevant content and answers questions using GPT-3.5 or GPT-4

## 🛠 Tech Stack

- Streamlit
- LangChain
- OpenAI (via `langchain-openai`)
- FAISS
- YouTube Transcript API

## 📁 Setup

1. Clone the repo
2. Install dependencies:
    ```bash
    pip install -r requirements.txt
    ```
3. Create a `.streamlit/secrets.toml` file with your OpenAI API key:
    ```toml
    OPENAI_API_KEY = "your-openai-api-key"
    ```

4. Run the app:
    ```bash
    streamlit run Youtube_chatbot_app.py
    ```

---