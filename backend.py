# backend.py

import os
import shutil
import tempfile
import uuid
from dotenv import load_dotenv 
import ssl
import torch
import requests
import json

# SSL context fix for macOS
ssl._create_default_https_context = ssl._create_unverified_context

import chromadb
import fitz  # PyMuPDF
import uvicorn
import whisper
from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from sentence_transformers import SentenceTransformer
from transformers import BlipProcessor, BlipForConditionalGeneration
from PIL import Image

# Load environment variables from .env file
load_dotenv()

# --- 1. Initialization ---

# Create necessary directories
os.makedirs("uploads", exist_ok=True)
os.makedirs("images_data", exist_ok=True)
os.makedirs("chroma_db", exist_ok=True)  # Directory for persistent ChromaDB

# Initialize FastAPI app
app = FastAPI(title="Multimodal RAG Backend")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize models and database
print("Initializing models and database...")

# Check if MPS (M1 GPU) is available
device = "mps" if torch.backends.mps.is_available() else "cpu"
print(f"Using device: {device}")

# Persistent ChromaDB (THIS IS THE KEY CHANGE)
print("Initializing persistent ChromaDB...")
client = chromadb.PersistentClient(path="./chroma_db")
collection = client.get_or_create_collection(name="multimodal_rag")
print(f"ChromaDB initialized with {collection.count()} existing documents")

# Sentence Transformer for text embeddings
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

# Whisper for audio transcription
try:
    whisper_model = whisper.load_model("base")
    print("Whisper model loaded successfully.")
except Exception as e:
    print(f"Warning: Could not load Whisper model: {e}")
    whisper_model = None

# BLIP model for image captioning
print("Loading BLIP model for image captioning...")
try:
    blip_processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
    blip_model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base").to(device)
    print("BLIP model loaded successfully.")
except Exception as e:
    print(f"Warning: Could not load BLIP model: {e}")
    blip_processor = None
    blip_model = None

# Mistral API configuration
MISTRAL_API_KEY = os.getenv("MISTRAL_API_KEY")
MISTRAL_API_URL = "https://api.mistral.ai/v1/chat/completions"

if not MISTRAL_API_KEY:
    print("WARNING: Mistral API key is not set. Advanced text generation will be disabled.")
    print("You can get a free API key from https://mistral.ai/")
else:
    print("Mistral API key loaded successfully.")

print("Initialization complete.")

# --- 2. Helper Functions ---

def chunk_text(text, chunk_size=512, chunk_overlap=50):
    """Splits text into overlapping chunks."""
    words = text.split()
    chunks = []
    for i in range(0, len(words), chunk_size - chunk_overlap):
        chunks.append(" ".join(words[i:i + chunk_size]))
    return chunks

def get_image_reasoning(image_path: str, question: str) -> str:
    """
    Generate image captions using LOCAL BLIP model for basic description.
    """
    if not blip_processor or not blip_model:
        return "Image processing is disabled due to model loading failure."
    
    try:
        # Load and process image
        image = Image.open(image_path).convert('RGB')
        inputs = blip_processor(image, return_tensors="pt").to(device)
        
        # Generate caption using BLIP
        with torch.no_grad():
            out = blip_model.generate(**inputs, max_length=100, num_beams=5)
        
        caption = blip_processor.decode(out[0], skip_special_tokens=True)
        return f"Figure/Diagram: {caption}"
        
    except Exception as e:
        print(f"Error processing image with BLIP: {e}")
        return f"Error processing image: {e}"

def generate_mistral_response(user_question: str, text_context: str, image_context: str) -> str:
    """
    Generate intelligent response using Mistral's free API.
    """
    if not MISTRAL_API_KEY:
        return None
    
    try:
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {MISTRAL_API_KEY}"
        }
        
        # Construct the prompt for Mistral
        prompt = f"""You are an AI assistant helping users understand documents. Based on the following context from a document, provide a clear, detailed, and helpful answer to the user's question.
        If the question cannot be answered using the document context, you must respond with "I cannot answer this question based on the uploaded document."

Document Text Context:
{text_context}

Image/Figure Context (if any):
{image_context}

User Question: {user_question}
Answer ONLY based on the document context above. Do not use external knowledge.
Please provide a comprehensive and well-structured answer based on the document content. If the question is about technical concepts, architecture, or diagrams, explain them clearly using the provided context. Be specific and informative."""

        payload = {
            "model": "mistral-tiny",  # Free tier model
            "messages": [
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            "max_tokens": 800,
            "temperature": 0.7
        }
        
        print("Sending request to Mistral API...")
        response = requests.post(MISTRAL_API_URL, headers=headers, json=payload, timeout=30)
        
        if response.status_code == 200:
            result = response.json()
            answer = result['choices'][0]['message']['content']
            print("Successfully received response from Mistral API")
            return answer
        else:
            print(f"Mistral API error {response.status_code}: {response.text}")
            return None
            
    except requests.exceptions.Timeout:
        print("Mistral API request timed out")
        return None
    except Exception as e:
        print(f"Mistral API error: {e}")
        return None

def document_exists(filename: str) -> bool:
    """Check if a document has already been processed."""
    try:
        results = collection.get(where={"source": filename})
        return len(results['ids']) > 0
    except:
        return False

def process_pdf(file_path: str, filename: str):
    """
    Extracts text and images from a PDF, processes them, and stores them in ChromaDB.
    """
    # Check if document already exists
    if document_exists(filename):
        print(f"Document {filename} already exists in database. Skipping processing.")
        return
    
    print(f"Processing PDF: {filename}")
    doc = fitz.open(file_path)
    full_text = ""
    image_counter = 0

    # 1. Extract text and images
    for page_num, page in enumerate(doc):
        # Extract text
        full_text += page.get_text() + "\n"

        # Extract images
        image_list = page.get_images(full=True)
        for img_index, img in enumerate(image_list):
            xref = img[0]
            base_image = doc.extract_image(xref)
            image_bytes = base_image["image"]
            
            # Save image to a file
            image_ext = base_image["ext"]
            image_filename = f"images_data/{filename}_p{page_num}_{img_index}.{image_ext}"
            
            # Only save if image doesn't already exist
            if not os.path.exists(image_filename):
                with open(image_filename, "wb") as img_file:
                    img_file.write(image_bytes)
            
            # Get caption using LOCAL BLIP model
            print(f"Getting caption for image: {image_filename}")
            image_caption = get_image_reasoning(image_filename, "What is this image about?")
            
            # Store image caption in ChromaDB
            collection.add(
                ids=[f"{filename}_img_{image_counter}"],
                documents=[image_caption],
                metadatas=[{
                    "type": "image",
                    "source": filename,
                    "page_num": page_num,
                    "image_path": image_filename
                }]
            )
            image_counter += 1

    # 2. Chunk and store text
    text_chunks = chunk_text(full_text)
    chunk_ids = [f"{filename}_txt_{i}" for i in range(len(text_chunks))]
    
    collection.add(
        ids=chunk_ids,
        documents=text_chunks,
        metadatas=[{"type": "text", "source": filename} for _ in text_chunks]
    )
    print(f"Finished processing PDF: {filename}. Stored {len(text_chunks)} text chunks and {image_counter} images.")

def process_audio(file_path: str, filename: str):
    """
    Transcribes audio, chunks the text, and stores it in ChromaDB.
    """
    if not whisper_model:
        raise HTTPException(status_code=500, detail="Audio processing is disabled.")
    
    # Check if document already exists
    if document_exists(filename):
        print(f"Document {filename} already exists in database. Skipping processing.")
        return
    
    print(f"Processing audio: {filename}")
    result = whisper_model.transcribe(file_path)
    transcription = result["text"]

    text_chunks = chunk_text(transcription)
    chunk_ids = [f"{filename}_aud_{i}" for i in range(len(text_chunks))]
    
    collection.add(
        ids=chunk_ids,
        documents=text_chunks,
        metadatas=[{"type": "audio", "source": filename} for _ in text_chunks]
    )
    print(f"Finished processing audio: {filename}. Stored {len(text_chunks)} chunks.")

def clear_all_data():
    """Clear both ChromaDB and image files for fresh start."""
    global collection
    try:
        client.delete_collection("multimodal_rag")
        print("Deleted existing ChromaDB collection")
    except:
        pass
    
    collection = client.get_or_create_collection(name="multimodal_rag")
    
    # Clear images directory
    if os.path.exists("images_data"):
        shutil.rmtree("images_data")
        print("Cleared images directory")
    os.makedirs("images_data", exist_ok=True)
    
    print("All data cleared successfully")

# --- 3. API Endpoints ---

@app.post("/upload_pdf/")
async def upload_pdf(file: UploadFile = File(...)):
    """Endpoint to upload and process a PDF file."""
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf", dir="uploads") as tmp:
            shutil.copyfileobj(file.file, tmp)
            tmp_path = tmp.name
        
        process_pdf(tmp_path, file.filename)
        os.remove(tmp_path)
        
        return {"status": "success", "filename": file.filename, "message": "PDF processed and stored."}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"An error occurred: {e}")
    finally:
        await file.close()

@app.post("/upload_audio/")
async def upload_audio(file: UploadFile = File(...)):
    """Endpoint to upload and process an audio file."""
    try:
        allowed_formats = ['.mp3', '.wav', '.m4a']
        file_ext = os.path.splitext(file.filename)[1]
        if file_ext not in allowed_formats:
            raise HTTPException(status_code=400, detail=f"Unsupported audio format.")

        with tempfile.NamedTemporaryFile(delete=False, suffix=file_ext, dir="uploads") as tmp:
            shutil.copyfileobj(file.file, tmp)
            tmp_path = tmp.name

        process_audio(tmp_path, file.filename)
        os.remove(tmp_path)
        
        return {"status": "success", "filename": file.filename, "message": "Audio processed and stored."}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"An error occurred: {e}")
    finally:
        await file.close()

@app.post("/query/")
async def query_index(user_question: str = Form(...)):
    """Endpoint to query the vector database and get an intelligent response using Mistral API."""
    try:
        print(f"Received query: {user_question}")
        
        # 1. Query ChromaDB for relevant context
        results = collection.query(
            query_texts=[user_question],
            n_results=5
        )
        
        # 2. Separate context into text and image reasoning
        retrieved_docs = results['documents'][0]
        retrieved_metadatas = results['metadatas'][0]
        
        text_chunks = "\n".join([doc for doc, meta in zip(retrieved_docs, retrieved_metadatas) if meta['type'] != 'image'])
        image_reasoning = "\n".join([doc for doc, meta in zip(retrieved_docs, retrieved_metadatas) if meta['type'] == 'image'])

        # 3. Try to use Mistral API for intelligent response
        mistral_response = generate_mistral_response(user_question, text_chunks, image_reasoning)
        
        if mistral_response:
            return {"answer": mistral_response}
        
        # 4. Fallback: Construct a local response if Mistral API fails
        if not text_chunks and not image_reasoning:
            final_answer = "I couldn't find relevant information in the uploaded documents to answer your question. Please make sure you've uploaded a document and that your question relates to its content."
        else:
            # Create a structured local response
            response_parts = []
            
            if text_chunks:
                response_parts.append(f"**Document Content:**\n{text_chunks}")
            
            if image_reasoning:
                response_parts.append(f"\n**Visual Elements:**\n{image_reasoning}")
            
            # Add a brief summary based on question type
            if any(keyword in user_question.lower() for keyword in ["transformer", "architecture", "attention"]):
                response_parts.append(f"\n**Summary:** Based on the document content above, this relates to the Transformer architecture and its attention mechanisms. The Transformer uses self-attention instead of recurrence or convolution for processing sequences.")
            
            final_answer = "\n".join(response_parts)
        
        return {"answer": final_answer}
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"An error occurred during query: {e}")

@app.get("/")
def read_root():
    return {"message": "Welcome to the Multimodal RAG API"}

@app.get("/health")
def health_check():
    return {
        "status": "healthy", 
        "models": {
            "whisper": whisper_model is not None,
            "blip": blip_model is not None,
            "embedding": True,
            "mistral_api": MISTRAL_API_KEY is not None
        },
        "device": device,
        "documents_count": collection.count()
    }

@app.get("/list_documents/")
async def list_documents():
    """List all stored documents."""
    try:
        all_docs = collection.get()
        sources = set()
        for metadata in all_docs['metadatas']:
            if 'source' in metadata:
                sources.add(metadata['source'])
        return {"documents": list(sources), "total_chunks": len(all_docs['ids'])}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error listing documents: {e}")

@app.delete("/clear_all/")
async def clear_all_documents():
    """Clear all stored documents and images."""
    try:
        clear_all_data()
        return {"status": "success", "message": "All documents and images cleared successfully."}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error clearing data: {e}")

@app.delete("/delete_document/{filename}")
async def delete_document(filename: str):
    """Delete a specific document and its associated images."""
    try:
        # Get all IDs for this document
        results = collection.get(where={"source": filename})
        if len(results['ids']) == 0:
            raise HTTPException(status_code=404, detail=f"Document '{filename}' not found.")
        
        # Delete from ChromaDB
        collection.delete(ids=results['ids'])
        
        # Delete associated image files
        deleted_images = 0
        for metadata in results['metadatas']:
            if metadata.get('type') == 'image' and 'image_path' in metadata:
                image_path = metadata['image_path']
                if os.path.exists(image_path):
                    os.remove(image_path)
                    deleted_images += 1
        
        return {
            "status": "success", 
            "message": f"Document '{filename}' deleted successfully.",
            "deleted_chunks": len(results['ids']),
            "deleted_images": deleted_images
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error deleting document: {e}")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
