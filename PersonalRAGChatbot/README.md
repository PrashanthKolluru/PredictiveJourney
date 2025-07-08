# Friday: Personal RAG Chatbot

This repo powers “Friday,” a multilingual RAG chatbot using:
- **Qdrant Cloud** for vector storage
- **Sentence-Transformers** (MPNet) for local 768-d embeddings
- **Google Gemini 2.0 Flash** for LLM generation and translation
- **Flask + Gunicorn** on Azure App Service

## Files

- `personal_chatbot_backend.py` – Flask API (endpoints `/chat`, `/reload`, etc.)
- `ingestpdf.py`                – PDF-→-Qdrant ingestion script
- `profile.pdf`                 – your bio PDF
- `requirements.txt`            – Python dependencies
- `Procfile`                    – Azure startup command
- `.env.example`                – sample environment variables
- `.gitignore`                  – files/folders to exclude

## Deployment

1. **Configure Azure App Service**  
   - Connect this GitHub repo under Deployment Center  
   - In “Configuration,” set environment variables:  
     ```ini
     QDRANT_URL=
     QDRANT_API_KEY=
     GOOGLE_API_KEY=
     PROFILE_DOC_PATH=profile.pdf
     QDRANT_COLLECTION=profile
     ```
2. **CORS**  
   - Under API → CORS, add your frontend origin.
3. **Test**  
   ```bash
   curl -X POST https://<your-app>.azurewebsites.net/chat \
     -H "Content-Type: application/json" \
     -d '{"question":"Who is Prashanth?"}'