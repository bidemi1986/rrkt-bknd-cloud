import os
import logging
from flask import Flask, jsonify, request
from google.cloud import storage, firestore
from firebase_admin import credentials, initialize_app
from langchain_experimental.text_splitter import SemanticChunker
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from google.cloud.firestore_v1.vector import Vector
from google.cloud.firestore_v1.base_vector_query import DistanceMeasure
import fitz  # PyMuPDF
import hashlib
from dotenv import load_dotenv

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# Firebase setup
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CRED_PATH = os.path.join(BASE_DIR, './rrkt-firebase-adminsdk.json')
PROJECT_ID = os.getenv("FIREBASE_PROJECT_ID")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
FIREBASE_STORAGE_BUCKET = os.getenv("FIREBASE_STORAGE_BUCKET")

# Initialize Firebase
cred = credentials.Certificate(CRED_PATH)
initialize_app(cred, {
    'storageBucket': FIREBASE_STORAGE_BUCKET
})
storage_client = storage.Client()
firestore_client = firestore.Client()

# Initialize embedding model
embedding_model = GoogleGenerativeAIEmbeddings(model="models/text-embedding-004", google_api_key=GOOGLE_API_KEY)

# Flask app
app = Flask(__name__)

def load_and_prepare_pdfs(user_id, max_chunk_size=1000):
    """Load and prepare PDFs from Firebase Storage for vectorization."""
    folder_path = f'room_files/{user_id}/'
    bucket = storage_client.bucket(FIREBASE_STORAGE_BUCKET)
    blobs = bucket.list_blobs(prefix=folder_path)

    for blob in blobs:
        if blob.name.endswith('.pdf'):
            pdf_path = '/tmp/temp.pdf'
            try:
                # Download the file to /tmp
                blob.download_to_filename(pdf_path)
                original_path = blob.name
                original_name = os.path.basename(original_path)
                pdf_document = fitz.open(pdf_path)
                num_pages = pdf_document.page_count
                chunk_size = min(num_pages, max_chunk_size)
                logger.info(f"Processing {blob.name} with {num_pages} pages, using chunk size {chunk_size}")
                yield pdf_path, original_path, original_name, chunk_size
            finally:
                # Clean up the temporary file
                if os.path.exists(pdf_path):
                    os.remove(pdf_path)

def get_file_hash(pdf_path):
    """Generate a hash for the PDF file."""
    hash_sha256 = hashlib.sha256()
    with open(pdf_path, "rb") as f:
        while chunk := f.read(8192):
            hash_sha256.update(chunk)
    return hash_sha256.hexdigest()

def is_file_processed(pdf_hash):
    """Check if the file with the given hash has been processed."""
    processed_files_collection = firestore_client.collection('processed_files')
    processed_doc_ref = processed_files_collection.document(pdf_hash)
    doc = processed_doc_ref.get()
    return doc.exists

def vectorize_and_store_documents(pdf_path, pdf_hash, original_path, original_name, chunk_size):
    """Vectorize and store documents in Firestore."""
    collection = firestore_client.collection('files-knowledge-base')
    metadata = {'pdf_name': original_name, 'pdf_hash': pdf_hash, 'file_path': original_path}
    text_splitter = SemanticChunker(embedding_model)
    doc = fitz.open(pdf_path)

    for page_num in range(0, doc.page_count, chunk_size):
        page = doc[page_num]
        text = page.get_text()
        docs = text_splitter.create_documents([text])
        chunked_content = [doc.page_content for doc in docs]
        chunked_embeddings = embedding_model.embed_documents(chunked_content)

        for i, (content, embedding) in enumerate(zip(chunked_content, chunked_embeddings)):
            chunk_metadata = metadata.copy()
            chunk_metadata['page_number'] = page_num
            doc_id = f"{pdf_hash[:8]}_page_{page_num}_chunk_{i}"
            doc_ref = collection.document(doc_id)
            doc_ref.set({
                "content": content,
                "embedding": Vector(embedding),
                "metadata": chunk_metadata
            })

    processed_files_collection = firestore_client.collection('processed_files')
    processed_doc_ref = processed_files_collection.document(pdf_hash)
    processed_doc_ref.set({"pdf_hash": pdf_hash})


@app.route('/')
def home():
    return jsonify({"message": "Hello, Flask Cloud Function!"})



@app.route('/greet', methods=['GET'])
def greet():
    name = request.args.get('name', 'Guest')
    return jsonify({"message": f"Hello, {name}!"})



@app.route('/vectorize/<userId>', methods=['POST'])
def vectorize_documents(userId):
    """Endpoint to vectorize documents for a specific user."""
    try:
        for pdf_path, original_path, original_name, chunk_size in load_and_prepare_pdfs(userId):
            pdf_hash = get_file_hash(pdf_path)
            if not is_file_processed(pdf_hash):
                vectorize_and_store_documents(pdf_path, pdf_hash, original_path, original_name, chunk_size)
            else:
                logger.info(f"File {original_name} has already been processed.")
        return jsonify({"message": "Documents processed successfully."}), 200
    except Exception as e:
        logger.error(f"Error processing documents: {e}")
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=int(os.getenv("PORT", 8080)))
