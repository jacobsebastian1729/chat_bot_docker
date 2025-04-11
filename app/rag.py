from flask import session
#from chromadb.utils import embedding_functions
from langchain_groq import ChatGroq
from dotenv import load_dotenv
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.document_loaders import Docx2txtLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings.sentence_transformer import SentenceTransformerEmbeddings
#from langchain_chroma import Chroma
from langchain.schema.runnable import RunnablePassthrough
#from gtts import gTTS
#import speech_recognition as sr
#import pygame
import os
import io
#import tempfile
import time
import random
import string
import shutil
#import psutil
from langchain_community.vectorstores import FAISS
#from langchain.vectorstores import FAISS
from pathlib import Path



# Function to generate a random string for temporary filenames
def random_string(length=8):
    return ''.join(random.choice(string.ascii_letters) for _ in range(length))

# Initialize the speech recognizer
# Adjust for ambient noise and recognition sensitivity


# Text Splitter Configuration
class RAGPipeline:
    def __init__(self, file_path = Path(__file__).parent / "docs" / "rag_doc.docx", model_name="all-MiniLM-L6-v2", chunk_size=500, chunk_overlap=200, collection_name='doc_collection_1', top_k=2):
        """Initialize the RAG pipeline with defaults and immediately run setup steps."""
        self.file_path = file_path
        self.model_name = model_name
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.collection_name = collection_name
        self.top_k = top_k

        # Run the setup steps directly in __init__
        try:
            print("🚀 Initializing RAG Pipeline...")
            self.load_document()
            self.split_document()
            self.init_embedding()
            self.create_vector_store()
            self.initialize_retriever()
            print("✅ RAG Pipeline initialized successfully!")

            # Clear the chat history after new file upload
            
        except Exception as e:
            print(f"❌ Error during initialization: {e}")
            self.documents = self.splits = self.embedding_function = self.vector_store = self.retriever = None

    def load_document(self):
        """Load the document file."""
        print("📄 Loading document...")
        loader = Docx2txtLoader(self.file_path)
        self.documents = loader.load()

    def split_document(self):
        """Split the document into chunks for better retrieval."""
        print("🔪 Splitting document into chunks...")
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            length_function=len
        )
        self.splits = text_splitter.split_documents(self.documents)

    def init_embedding(self):
        """Initialize the sentence transformer embedding model."""
        print("🧠 Initializing embeddings...")
        self.embedding_function = SentenceTransformerEmbeddings(model_name=self.model_name)

    def create_vector_store(self):
        try:
            self.vector_store = FAISS.from_documents(
                documents=self.splits,
                embedding=self.embedding_function,
            )
            print("FAISS vector store created successfully.")
        except Exception as create_error:
            print(f"Error creating FAISS vector store: {create_error}")
            self.vector_store = None
    
    #def create_vector_store(self):
        #collection_name = 'doc_collection_1'
    #    try:
    #        self.vector_store = Chroma.from_documents(
    #        documents=self.splits,
    #        embedding=self.embedding_function,
    #        collection_name=self.collection_name,
    #        persist_directory="./chroma_db",
    #    )
    #        print("Chroma vector store created successfully.")
    #    except Exception as create_error:
    #        print(f"Error creating Chroma vector store: {create_error}")
    #        self.vector_store = None
        
    def initialize_retriever(self):
        """Set up a retriever for fetching relevant documents."""
        print("🔍 Setting up retriever...")
        if self.vector_store:
            self.retriever = self.vector_store.as_retriever(search_kwargs={"k": self.top_k})
            print("🚀 Retriever initialized and ready!")
        else:
            print("⚠️ Vector store not available — retriever can't be initialized.")
            self.retriever = None
    

    def run_query(self, query):
        """Run a query on the retriever and return the top results."""
        if self.retriever:
            print(f"🔎 Running query: '{query}'")
            results = self.retriever.get_relevant_documents(query)
            return results
        else:
            print("⚠️ Retriever not initialized. Can't run the query.")
            return []

    
# ==============================
# 🚀 Main execution starts here
# ==============================


env_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '.env'))
load_dotenv(dotenv_path=env_path)

# Now this will work:
api_key = os.getenv("GROQ_API_KEY")
# Initialize language model
model_name = "llama3-70b-8192"
llm = ChatGroq(temperature=0.7, model = model_name)


# Prompt Template
template = """
You are a helpful assistant. Maintain awareness of user details from chat history (e.g., their name or role). 
Answer the question based **only** on the provided context and chat history. 

If the context doesn’t cover the question, respond with "The context doesn’t provide that information."

Chat History (keep this in mind):  
{chat_history}

Context (only use this for facts):  
{context}

Question: {question}

Answer:
"""


prompt = ChatPromptTemplate.from_template(template)

def doc2str(docs):
    return "\n\n".join(doc.page_content for doc in docs)

# RAG Chain
def create_rag_chain(pipeline_instance):
    rag_chain = (
        {
            "context": lambda x: doc2str(pipeline_instance.retriever.invoke(x["question"])),
            "question": RunnablePassthrough(),
            "chat_history": RunnablePassthrough(),
        }
        | prompt
        | llm
        | StrOutputParser()
    )
    return rag_chain


#from your_rag_module import rag_chain  # Adjust this to your RAG setup

def get_rag_response(question, pipeline_instance):
    """
    Takes a text input (question), gets response from the RAG chain.
    Handles chat history per user session.
    Exits gracefully if user says 'bye', 'thank you', or 'quit'.
    """
    try:
        # Ensure the session persists beyond the current browser window
        session.permanent = True  

        # Common exit phrases
        exit_phrases = ["bye", "thank you", "quit", "exit"]

        # Initialize chat history for this user if not present
        if 'chat_history' not in session:
            session['chat_history'] = []

        # Check if the user wants to end the conversation
        if any(phrase in question.lower() for phrase in exit_phrases):
            print("\n=== Chat Session Ended ===")
            for chat in session['chat_history']:
                print(chat)

            session.pop('chat_history', None)  # Clear the session history
            return "It’s been a pleasure chatting with you. Take care! 👋"

        # Pass question, context, and chat history to RAG chain
        if pipeline_instance: #check if pipeline_instance exists.
            rag_chain = create_rag_chain(pipeline_instance) #create rag chain with the instance.
            response = rag_chain.invoke({
                "question": question,
                "chat_history": session['chat_history']
            })
            session['chat_history'].append(f"User: {question}")
            session['chat_history'].append(f"Nova: {response}")
            return response
        else:
            return "Please upload a document first."


    except Exception as e:
        import traceback
        traceback.print_exc()
        print(f"Error processing question: {e}")
        return "Sorry, I encountered an error processing your question. Please try again."



if __name__ == "__main__":
    # Define the file path to your document
    file_path = r"C:\Users\JACOB\Desktop\New folder (2)\chat_bot\docs\rag_doc.docx"
    
    # Instantiate the pipeline (this runs everything up to the retriever)
    pipeline = RAGPipeline(file_path)

    # Check if everything was created successfully
    if pipeline.vector_store and pipeline.retriever:
        print("✅ Vector store and retriever are ready to use!")

        # Example query to test the pipeline
        query = "What event caused the people of Eldoria to disappear?"
        print(pipeline.retriever.invoke(query))
        #results = pipeline.run_query(query)

        # Display results nicely
        '''print("\n🔍 Query Results:")
        for i, result in enumerate(results):
            print(f"📌 Result {i+1}:\n{result.page_content}\n")
        '''
    else:
        print("❌ Failed to initialize pipeline components. Check document or model setup.")

