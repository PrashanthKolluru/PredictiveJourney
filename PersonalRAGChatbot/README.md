# Personal_RAG_chatbot

# 🤖 Multilingual Personal RAG Chatbot

A fast, multilingual chatbot that:
- Accepts questions in **English, Hindi, Telugu, and French** (via text or voice)
- Answers in **English**, with bracketed translations in the original input language
- Uses **Retrieval-Augmented Generation (RAG)** to fetch personalized answers from Prashanth's information storage database
- Falls back to **Google Gemini 2.0 Flash** for general knowledge queries
- Fully deployable on **Azure** using Python, Qdrant, and Azure Cognitive Services

---

## 🚀 Features
- 🗣️ **Multilingual I/O**: Language detection, text-to-speech & voice-to-text for English, Hindi, Telugu, and French  
- 📄 **Document-Grounded Q&A**: Semantic search over `profile.pdf` via Qdrant vector database  
- 🧠 **Hybrid RAG Pipeline**: Combines semantic retrieval with LLM reasoning (Gemini 2.0 Flash API)  
- ☁️ **Cloud-First**: End-to-end hosting on Azure (Embeddings, Vector DB, Cognitive Services)  
- 🔗 **Embeddings**: Uses `paraphrase-multilingual-mpnet-base-v2` hosted on Azure

---

## 🛠️ Tech Stack
- **Language**: Python  
- **PDF Parsing**: PyMuPDF  
- **Embeddings & Semantic Models**: Azure-hosted `sentence-transformers/paraphrase-multilingual-mpnet-base-v2`  
- **Vector Database**: Qdrant on Azure  
- **LLM API**: Google Gemini 2.0 Flash  
- **Voice & Translation**: Azure Cognitive Services (Speech & Translator APIs)  
- **Deployment**: Azure App Services, Functions, Container Instances

---

## 🏗️ Architecture Overview
1. **Document Ingestion**  
   Extract text from `profile.pdf`  
2. **Chunking & Embedding**  
   Split into chunks → generate embeddings  
3. **Indexing**  
   Store vectors in Qdrant for fast semantic search  
4. **User Query**  
   - Detect language & transcribe voice input  
   - Retrieve top-k context chunks from Qdrant  
   - Formulate RAG prompt & send to Gemini 2.0 Flash  
5. **Response Assembly**  
   - Receive LLM answer  
   - Translate to original language if needed  
   - TTS for spoken replies

---
