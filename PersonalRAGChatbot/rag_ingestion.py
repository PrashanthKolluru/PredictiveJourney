import os
import re
import uuid

from dotenv import load_dotenv
import pdfplumber
from qdrant_client import QdrantClient
from qdrant_client.http.models import VectorParams, Distance, PointStruct
from sentence_transformers import SentenceTransformer
from qdrant_client.http.models import MatchText
from typing import Iterator, Tuple
from langchain.text_splitter import RecursiveCharacterTextSplitter


def split_into_sections(full_text: str) -> Iterator[Tuple[str, str]]:
    lines = full_text.splitlines()
    indices: list[int] = []
    titles: list[str] = []

    # 1) Detect all-caps headings ending with ':'
    for i, ln in enumerate(lines):
        stripped = ln.strip()
        if stripped.endswith(':') and stripped.isupper():
            raw = stripped[:-1].strip()                   
            raw = re.sub(r'\s*\(.*?\)$', '', raw)          
            key = raw.lower()

            # 2) Substring-based canonical mapping
            if "personal projects" in key:
                title = "Personal Projects"
            elif "education" in key:
                title = "Education"
            elif "research" in key:
                title = "Research"
            elif "work experience" in key:
                title = "Work Experience"
            elif "contact information" in key:
                title = "Contact Information"
            elif "languages" in key:
                title = "Languages"
            elif "tools" in key or "technologies" in key:
                title = "Tools and Technologies"
            elif "technological inquisitiveness" in key:
                title = "Technological Inquisitiveness"
            else:
                # Fallback: title-case whatever remains
                title = raw.title()

            indices.append(i)
            titles.append(title)

    # 3) Emit an intro section if there's text before the first heading
    if indices and indices[0] > 0:
        intro = "\n".join(lines[:indices[0]]).strip()
        if intro:
            yield "About Prashanth and Education", intro

    # 4) Yield each section’s title + content
    for idx, title in zip(indices, titles):
        start = idx + 1
        # next heading index or EOF
        next_idxs = [j for j in indices if j > idx]
        end = next_idxs[0] if next_idxs else len(lines)
        content = "\n".join(lines[start:end]).strip()
        yield title, content

def main():
    load_dotenv()
    qdrant_url = os.getenv("QDRANT_URL")
    qdrant_api_key = os.getenv("QDRANT_API_KEY")
    collection_name = os.getenv("QDRANT_COLLECTION", "profile")
    pdf_path = os.getenv("PROFILE_DOC_PATH", "profile.pdf")

    # 1) Connect & (re)create the Qdrant collection
    client = QdrantClient(url=qdrant_url, api_key=qdrant_api_key)
    if client.collection_exists(collection_name):
        print(f"🔗 Deleting existing collection '{collection_name}'…")
        client.delete_collection(collection_name)

    print(f"🔗 Creating collection '{collection_name}'…")
    client.create_collection(
        collection_name=collection_name,
        vectors_config=VectorParams(size=768, distance=Distance.COSINE)
    )

     # 2) Read & clean the full PDF text *but keep line breaks*
    print(f"📄 Reading '{pdf_path}'…")
    pages = []
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            txt = page.extract_text() or ""
            # 2a) replace bullets/dashes
            txt = txt.replace("•", "-").replace("–", "-")
            # 2b) clean each line but keep the \n
            lines = txt.splitlines()
            clean_lines = [re.sub(r" +", " ", ln).strip() for ln in lines]
            pages.extend(clean_lines)
    # Now full_text has real line breaks for headings
    full_text = "\n".join(pages)

    # 3) Prepare a recursive splitter for deep chunks
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=400,
        chunk_overlap=200,
        length_function=len,
        separators=["\n\n", "\n", ". ", " "],
        keep_separator=True
    )

    # 4) Load the embedding model
    print("📦 Loading embedding model…")
    model = SentenceTransformer("sentence-transformers/paraphrase-multilingual-mpnet-base-v2")

    # 5) Build PointStructs for summaries + sub-chunks
    points = []
    section_count = 0
    for section_title, section_text in split_into_sections(full_text):
        section_count += 1

        # a) Summary chunk (first 2 sentences)
        sentences = re.split(r'(?<=[.!?]) +', section_text)
        summary = " ".join(sentences[:2]).strip()
        emb_sum = model.encode(summary, normalize_embeddings=True).tolist()
        points.append(PointStruct(
            id=str(uuid.uuid4()),
            vector=emb_sum,
            payload={
                "section_title": section_title,
                "text": summary,
                "is_summary": True
            }
        ))

        # b) Deeper overlapping sub-chunks
        sub_chunks = splitter.split_text(section_text)
        for chunk in sub_chunks:
            emb = model.encode(chunk, normalize_embeddings=True).tolist()
            points.append(PointStruct(
                id=str(uuid.uuid4()),
                vector=emb,
                payload={
                    "section_title": section_title,
                    "text": chunk,
                    "is_summary": False
                }
            ))

    print(f"🔢 Prepared {section_count} sections → total {len(points)} chunks.")

    # 6) Upload all points to Qdrant
    print("📡 Uploading to Qdrant…")
    client.upsert(collection_name=collection_name, wait=True, points=points)
    print("✅ Ingestion complete. Your PDF is now searchable in Qdrant.")


if __name__ == "__main__":
    main()
