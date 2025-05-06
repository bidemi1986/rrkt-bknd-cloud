# from langchain_openai import OpenAIEmbeddings  
from langchain_google_genai import GoogleGenerativeAIEmbeddings 
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_experimental.text_splitter import SemanticChunker
from google.cloud import firestore
from google.cloud.firestore_v1.vector import Vector
from google.cloud.firestore_v1.base_vector_query import DistanceMeasure
from dotenv import load_dotenv
import os
# import fitz  # PyMuPDF
import pymupdf  # PyMuPDF
from firebase_admin import credentials, initialize_app, storage, firestore
import logging
import os
import getpass
from pathlib import Path 
from PIL import Image
import hashlib
import math

load_dotenv()
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CRED_PATH = os.path.join(BASE_DIR, './rrkt-firebase-adminsdk.json')
PROJECT_ID = os.getenv("FIREBASE_PROJECT_ID")
LOCATION = os.getenv("FIREBASE_SERVER_LOCATION")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
FILES = 'room_files/KnowledgeBaseDocs/' # Folder path in the storage bucket
# Cu3DdQMksLP7UEnZXmasiNlcEko1
print(PROJECT_ID)

if "GOOGLE_API_KEY" not in os.environ:
    os.environ["GOOGLE_API_KEY"] = getpass.getpass("Provide your Google API key here")


def initialize_firebase():
    """Initialize Firebase with credentials if not already initialized"""
    try:
        # Try to get an existing app
        storage.bucket()
        return storage, firestore.client()
    except ValueError:
        # Initialize new app if none exists
        cred = credentials.Certificate(CRED_PATH)
        initialize_app(cred, {
            'storageBucket': 'sample-firebase-ai-app-3e813.firebasestorage.app'
        })
        return storage, firestore.client()
    


storage, db = initialize_firebase() 
embedding_model = GoogleGenerativeAIEmbeddings(model="models/text-embedding-004",google_api_key=GOOGLE_API_KEY)  
# Use a variable called collection to create a reference to a collection named food-safety.
processed_files_collection = db.collection('processed_files')  # A collection to track processed files
collection = db.collection('wine-knowledge-base')



def load_and_prepare_pdfs(folder_path, max_chunk_size=1000):
    bucket = storage.bucket()
    blobs = bucket.list_blobs(prefix=folder_path)

    for blob in blobs:
        if blob.name.endswith('.pdf'):
            pdf_path = '/tmp/temp.pdf'
            blob.download_to_filename(pdf_path)
            # Store original file path and name
            original_path = blob.name
            original_name = os.path.basename(original_path)
            # Open the PDF file with fitz to check the number of pages
            pdf_document = pymupdf.open(pdf_path)
            num_pages = pdf_document.page_count
            
            # Dynamically determine the chunk size based on the number of pages
            chunk_size = min(num_pages, max_chunk_size)  # Don't exceed max_chunk_size
            print(f"Processing {blob.name} with {num_pages} pages, using chunk size {chunk_size}")
            
            loader = PyMuPDFLoader(pdf_path)
            data = loader.load()
            
            # Yield data in chunks based on the chunk_size
            for i in range(0, num_pages, chunk_size):
                yield data[i:i + chunk_size], original_path, original_name



def extract_screenshot_and_text(pdf_path, page_num):
    """Extracts screenshot (image) and text from the PDF page"""
    # Extract text using PyMuPDFLoader
    loader = PyMuPDFLoader(pdf_path)
    page = loader.load()[page_num]  # Extract the text of the current page

    # Open the PDF with fitz for image extraction
    doc = pymupdf.open(pdf_path)
    pdf_page = doc.load_page(page_num)  # Load the actual page using fitz

    # Extract screenshot as in-memory image
    pix = pdf_page.get_pixmap()
    img_filename = f"/tmp/screenshot_{page_num}.png"
    pix.save(img_filename)

    return page.page_content, img_filename


def create_markdown_content(text, img_filename):
    """Create markdown content for each page"""
    # markdown = f"## Page content\n\n{text}\n\n![Screenshot]({img_filename})"
    markdown = f"## Page content\n\n{text}\n\n"
    return markdown


def get_file_hash(pdf_path):
    """Generate a hash for the PDF file"""
    hash_sha256 = hashlib.sha256()
    with open(pdf_path, "rb") as f:
        # Read the file in chunks to avoid memory issues with large files
        while chunk := f.read(8192):
            hash_sha256.update(chunk)
    return hash_sha256.hexdigest()


def is_file_processed(pdf_hash):
    """Check if the file with the given hash has been processed"""
    processed_doc_ref = processed_files_collection.document(pdf_hash)
    doc = processed_doc_ref.get()
    return doc.exists  # If the document exists, it means the file has been processed


def clean_page(markdown_content):
    # return page.page_content.replace("-\n", "")\ 
    """Clean the markdown content before vectorization"""
    return markdown_content.replace("-\n", "")\
                            .replace("\n", " ")\
                            .replace("\x02", "")\
                            .replace("\x03", "")\
                            .replace("## Page content", "")\
                            .replace("fo d P R O T E C T I O N  T R A I N I N G  M A N U A L", "")

 
def vectorize_and_store_documents(data_chunk, pdf_path, pdf_hash, original_path, original_name):
    # Extract metadata for this PDF (e.g., file name, hash, etc.)
    metadata = {
        'pdf_name': original_name,
        'pdf_hash': pdf_hash,
        'file_path': original_path  # Use the original path from Storage
    }
    
    # Iterate through each page and process
    for page_num, page in enumerate(data_chunk):
        # Extract text and screenshot (saved to temporary file)
        text, img_filename = extract_screenshot_and_text(pdf_path, page_num)
        
        # Create markdown content
        markdown_content = create_markdown_content(text, img_filename)
        
        # Clean the markdown content before vectorization
        cleaned_markdown_content = clean_page(markdown_content)
        
        # Now, vectorize the cleaned markdown content
        text_splitter = SemanticChunker(embedding_model)
        docs = text_splitter.create_documents([cleaned_markdown_content])  # Create documents from cleaned markdown
        chunked_content = [doc.page_content for doc in docs]
        chunked_embeddings = embedding_model.embed_documents(chunked_content)
        
        # Create a file-unique identifier using part of the hash
        file_id = pdf_hash[:8]  # Use first 8 characters of hash as unique file identifier
    
        # Store in Firestore with metadata
        collection = db.collection('wine-knowledge-base')
        for i, (content, embedding) in enumerate(zip(chunked_content, chunked_embeddings)):
             # Update metadata to include page_number for this specific chunk
            chunk_metadata = metadata.copy()
            chunk_metadata['page_number'] = page_num
            # Create unique document ID that includes file identifier
            doc_id = f"{file_id}_page_{page_num}_chunk_{i}"
            doc_ref = collection.document(doc_id)
            doc_ref.set({
                "content": content,
                "embedding": Vector(embedding),
                "metadata": chunk_metadata,  # Storing metadata including page number
                "page_number": page_num,  # Keep this for backward compatibility
                "file_name": metadata['pdf_name']  # Include file name in the stored data
            })
        
        # Delete the temporary screenshot file after processing
        if os.path.exists(img_filename):
            os.remove(img_filename)
 

    # After processing the PDF, store the hash in Firestore to mark it as processed
    processed_doc_ref = processed_files_collection.document(pdf_hash)
    processed_doc_ref.set({"pdf_hash": pdf_hash})



for data_chunk,original_path, original_name in load_and_prepare_pdfs(FILES):
    pdf_path = '/tmp/temp.pdf'  # Path to the downloaded PDF
    pdf_hash = get_file_hash(pdf_path)  # Generate the file hash

    # Check if the file has already been processed
    if not is_file_processed(pdf_hash):
        # vectorize_and_store_documents(data_chunk, pdf_path, pdf_hash)
        vectorize_and_store_documents(data_chunk, pdf_path, pdf_hash, original_path, original_name)

    else:
        print(f"File {pdf_path} has already been processed.")



def search_vector_database(query: str):
    context = []
    query_embedding = embedding_model.embed_query(query)

    # Specify a distance result field to retrieve similarity scores
    distance_result_field = "similarity_score"  # You can name this field as per your setup
    
    # Perform the vector search query with the distance_result_field
    vector_query = collection.find_nearest(
        vector_field="embedding",
        query_vector=Vector(query_embedding),
        distance_measure=DistanceMeasure.EUCLIDEAN,
        limit=5,
        distance_result_field=distance_result_field  # Specify the field for similarity score
    )
    
    docs = vector_query.stream()

    for result in docs:  # Iterate over each result in the docs stream
        doc_data = result.to_dict()  # Get the document data as a dictionary
        
        # Retrieve content, metadata, and similarity score (from the specified field)
        content = doc_data.get('content')
        metadata = doc_data.get('metadata', {})  # Retrieve metadata from the document
        
        # Retrieve the similarity score from the specified distance_result_field
        similarity_score = doc_data.get(distance_result_field, None)

        if similarity_score is None:
            print("Warning: Similarity score not found for this result.")

        # Append to context as a dictionary with content, similarity score, and metadata
        context.append({
            'content': content,
            'metadata': metadata,
            'similarity_score': similarity_score
        })
    
    return context


final_result = search_vector_database("who is DSS Coordinator?")
print(f"Search results: {final_result}")
# Note:
# when prompted for the Google API key, provide the GEMINI_API_KEY in .env.development or .production and press Enter. 
# The key will be stored in the environment variable GOOGLE_API_KEY for future use.
# ref: https://medium.com/google-cloud/building-a-rag-application-with-vector-search-in-firestore-71da2e6e7e77