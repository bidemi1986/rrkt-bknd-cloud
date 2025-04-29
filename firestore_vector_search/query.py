import os
import logging
from typing import List, Optional, Dict, Any
from dotenv import load_dotenv
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.chains.qa_with_sources.retrieval import RetrievalQAWithSourcesChain
from langchain_community.utilities import GoogleSearchAPIWrapper
from langchain_community.retrievers import WebResearchRetriever
from langchain.schema import Document
from langchain.vectorstores.base import VectorStore
from google.cloud.firestore_v1.vector import Vector
from google.cloud.firestore_v1.base_vector_query import DistanceMeasure
import firebase_admin
from firebase_admin import credentials, firestore
# from langchain.vectorstores import Chroma

# Initialize logging
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()
PROJECT_ID = os.getenv("FIREBASE_PROJECT_ID")
LOCATION = os.getenv("FIREBASE_SERVER_LOCATION")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Initialize Firebase
def initialize_firebase():
    """Initialize Firebase Admin SDK and return Firestore client."""
    if not firebase_admin._apps:
        # If using service account key file
        if os.path.exists("serviceAccountKey.json"):
            cred = credentials.Certificate("serviceAccountKey.json")
        # If running in Google Cloud environment (using default credentials)
        else:
            cred = credentials.ApplicationDefault()
        
        firebase_admin.initialize_app(cred, {
            'projectId': PROJECT_ID,
        })
    
    return firestore.client()

# Initialize Firebase clients
db = initialize_firebase()

# Collection for vector embeddings
COLLECTION_NAME = os.getenv("FIRESTORE_COLLECTION", "web_research_embeddings")
collection = db.collection(COLLECTION_NAME)

# Initialize embedding model
embedding_model = GoogleGenerativeAIEmbeddings(
    model="models/text-embedding-004", 
    google_api_key=os.getenv("GEMINI_API_KEY")
)



class FirestoreVectorStore(VectorStore):
    """
    Custom vector store implementation that uses Firestore's vector search.
    """
    def __init__(self, embedding_function):
        self.embedding_function = embedding_function

    def add_texts(
        self,
        texts: List[str],
        metadatas: Optional[List[Dict[str, Any]]] = None,
        **kwargs
    ) -> List[str]:
        """
        Add texts to the vector store with their embeddings.
        """
        if metadatas is None:
            metadatas = [{} for _ in texts]
        
        ids = []
        for i, (text, metadata) in enumerate(zip(texts, metadatas)):
            # Generate embedding for the text
            embedding = self.embedding_function.embed_query(text)
            
            # Create document data
            doc_data = {
                'content': text,
                'metadata': metadata,
                'embedding': Vector(embedding)
            }
            
            # Add to Firestore
            doc_ref = collection.document()
            doc_ref.set(doc_data)
            ids.append(doc_ref.id)
            
            logger.info(f"Added document {doc_ref.id} to Firestore")
        
        return ids

    def similarity_search(
        self,
        query: str,
        k: int = 4,
        **kwargs
    ) -> List[Document]:
        """
        Perform a similarity search in Firestore.
        """
        query_embedding = self.embedding_function.embed_query(query)
        
        # Perform the vector search query
        vector_query = collection.find_nearest(
            vector_field="embedding",
            query_vector=Vector(query_embedding),
            distance_measure=DistanceMeasure.COSINE,
            limit=k,
            distance_result_field="similarity_score"
        )
        
        docs = vector_query.stream()
        
        documents = []
        for result in docs:
            doc_data = result.to_dict()
            content = doc_data.get('content')
            metadata = doc_data.get('metadata', {})
            documents.append(Document(page_content=content, metadata=metadata))
        
        return documents

    def as_retriever(self, **kwargs):
        """
        Return a retriever that uses this vector store.
        """
        return FirestoreRetriever(self.embedding_function)



class FirestoreRetriever:
    """
    Custom retriever that uses Firestore's vector search functionality.
    """
    def __init__(self, embedding_function):
        self.embedding_function = embedding_function
        self.vectorstore = FirestoreVectorStore(embedding_function)

    def get_relevant_documents(self, query: str) -> List[Document]:
        """
        Get documents relevant to the query.
        """
        return self.vectorstore.similarity_search(query)




def query_the_web(query_text):
    """
    Query the web using the QA chain via the web research retriever.
    This function uses the provided query text to perform web research and returns a dictionary containing the answer and its sources.
    """
    logger.info(f"Processing query: {query_text}")
    
    # Initialize the language model and search wrapper
    llm = ChatOpenAI(api_key=OPENAI_API_KEY, temperature=0)
    search = GoogleSearchAPIWrapper(k=3)
    
    # Create vector store with the embedding model
    # vectorstore = FirestoreVectorStore(embedding_function=embedding_model)
    vectorstore = Chroma(embedding_function=OpenAIEmbeddings(), persist_directory="./chroma_db_oai")

    # Build the WebResearchRetriever from the LLM, vectorstore, and search API
    web_research_retriever = WebResearchRetriever.from_llm(
        vectorstore=vectorstore,
        llm=llm,
        search=search,
        allow_dangerous_requests=True  # Allow dangerous requests
    )
    
    # Create the QA chain using the web research retriever
    qa_chain = RetrievalQAWithSourcesChain.from_chain_type(llm, retriever=web_research_retriever)
    
    # Execute the QA chain with the query
    result = qa_chain({"question": query_text})
    
    # Extract the answer and sources from the result dictionary
    answer = result.get("answer", "No answer found.")
    sources = result.get("sources", [])
    if not isinstance(sources, list):
        sources = [sources]
    
    logger.info(f"Web research result: {result}")
    
    return {"answer": answer, "sources": sources}