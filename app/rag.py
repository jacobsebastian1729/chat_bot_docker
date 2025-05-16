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

# Website page navigation mapping
# Add all pages that your website has - update with your actual URLs
WEBSITE_PAGES = {
    "home": "https://www.4labsinc.com/",
    "services": "https://www.4labsinc.com/services",
    "service": "https://www.4labsinc.com/services",  # Handle singular version
    "about": "https://www.4labsinc.com/about-us",
    "about us": "https://www.4labsinc.com/about-us",  # Common variation
    "contact": "https://www.4labsinc.com/lets-connect",
    "contact us": "https://www.4labsinc.com/lets-connect",  # Common variation  # Handle singular version
    "connect": "https://www.4labsinc.com/lets-connect",
    "connect you" : "https://www.4labsinc.com/lets-connect",
    "connect with you": "https://www.4labsinc.com/lets-connect",
    "blog": "https://www.4labsinc.com/blogs",
    "blogs": "https://www.4labsinc.com/blogs",
    "casestudies": "https://www.4labsinc.com/case-studies",
    "case studies": "https://www.4labsinc.com/case-studies",
    "case study": "https://www.4labsinc.com/case-studies",
    "support": "https://www.4labsinc.com/support",
    "careers": "https://www.4labsinc.com/careers",
    "career": "https://www.4labsinc.com/careers",  # Handle singular version
    "industry": "https://www.4labsinc.com/case-studies",
    "industries": "https://www.4labsinc.com/case-studies",
    "referrals": "https://www.4labsinc.com/business-referral",
    "referral": "https://www.4labsinc.com/business-referral",
    "models": "https://www.4labsinc.com/engagement-model",
    "model": "https://www.4labsinc.com/engagement-model",
    "business model": "https://www.4labsinc.com/engagement-model",
    "business models": "https://www.4labsinc.com/engagement-model",
    "programs": "https://www.4labsinc.com/partnership-programs",
    "ourprograms": "https://www.4labsinc.com/partnership-programs",
    "partnership programs": "https://www.4labsinc.com/partnership-programs"
  # Common variation
    # Add more pages as needed
}

# Function to generate a random string for temporary filenames
def random_string(length=8):
    return ''.join(random.choice(string.ascii_letters) for _ in range(length))

# Navigation intent detection function
def detect_navigation_intent(user_message):
    """
    Detect if the user wants to navigate to a specific page.
    Returns the page name if detected, None otherwise.
    """
    user_msg = user_message.lower()
    
    # Navigation phrases to detect
    nav_phrases = [
        "take me to", "go to", "navigate to", "visit", "show me", 
        "open the", "open", "direct me to", "bring me to",
        "can you take me to", "i want to see", "i need to go to",
        "show the", "access the", "send me to", "get me to",
        "lead me to", "can i see", "would like to see"
    ]
    
    # Check if any navigation phrase is present
    for phrase in nav_phrases:
        if phrase in user_msg:
            # Find which page they want to navigate to
            for page in WEBSITE_PAGES.keys():
                if page in user_msg or f"{page} page" in user_msg:
                    # If page is found, canonicalize it to handle variations
                    # For example, map "service" to "services" by URL
                    canonical_url = WEBSITE_PAGES[page]
                    # Find the primary key that uses this URL (usually the plural/standard form)
                    for primary_key, url in WEBSITE_PAGES.items():
                        if url == canonical_url:
                            # Return the first match as the canonical page name
                            return primary_key
                    # Fallback to the matched page name
                    return page
    
    # Check for direct mentions of pages without navigation phrases
    for page in WEBSITE_PAGES.keys():
        if user_msg == page or user_msg == f"{page} page":
            return page
    
    return None


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
You are Nova, the AI assistant for 4Labs Technologies. Your job is to help visitors by answering questions about the company, its leadership, services, or navigation—just like a knowledgeable and professional team member.

Rules:

    Use only the information provided in the context.

    Do not refer to the "context" or "provided information" in your replies. Just respond naturally and confidently.

    If specific information is missing, say:

        "I appreciate your interest. While I don’t have that specific information at the moment."

    Use a helpful, polished, and human tone. Avoid sounding robotic or overly generic.
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
            return "It's been a pleasure chatting with you. Take care! 👋"
            
        # Check for navigation intent
        navigation_page = detect_navigation_intent(question)
        if navigation_page:
            # Get the canonical URL for this page
            page_url = WEBSITE_PAGES.get(navigation_page)
            if page_url:
                # Make the display name more user-friendly
                display_name = navigation_page.replace("_", " ").title()
        
                # Handle special cases for cleaner display

                #if navigation_page == "home":
                #    display_name = "Home"
                #elif navigation_page in ["services", "service"]:
                #    display_name = "Services"
                #elif navigation_page in ["about", "about us"]:
                #    display_name = "About Us"
                #elif navigation_page in ["contact", "contact us"]:
                #    display_name = "Contact Us"
                if navigation_page.lower() == "home":
                    display_name = "Home"
                elif navigation_page.lower() in ["services", "service"]:
                    display_name = "Services"
                elif navigation_page.lower() in ["about", "about us"]:
                    display_name = "About Us"
                elif navigation_page.lower() in ["contact", "contact us", "connect", "connect with you", "connect you"]:
                    display_name = "Contact Us"
                elif navigation_page.lower() in ["blog", "blogs"]:
                    display_name = "Blog"
                elif navigation_page.lower() in ["case studies", "casestudies", "case study"]:
                    display_name = "Case Studies"
                elif navigation_page.lower() in ["referral", "referrals"]:
                    display_name = "Business Referral"
                elif navigation_page.lower() in ["models", "model", "business model", "business models"]:
                    display_name = "Engagement Model"
                elif navigation_page.lower() in ["programs", "ourprograms", "partnership programs"]:
                    display_name = "Partnership Programs"
        
                # Return a response with a clickable link (HTML format)
                response = f"Sure! You can <a href='{page_url}' class='chatbot-link'>click here</a> to visit the {display_name} page."
        
                # Add to chat history
                session['chat_history'].append(f"User: {question}")
                session['chat_history'].append(f"Nova: {response}")
        
                return response
            else:
                # Page not found in mapping – handle professionally
                polite_response = (
                    "I couldn't find a page that matches your request. "
                    "Please double-check the page name or let me know what you're looking for, and I’ll do my best to help!"
                )
                session['chat_history'].append(f"User: {question}")
                session['chat_history'].append(f"Nova: {polite_response}")
                return polite_response
#        navigation_page = detect_navigation_intent(question)
#        if navigation_page:
            # Get the canonical URL for this page
#            page_url = WEBSITE_PAGES.get(navigation_page)
#            if page_url:
                # Make the display name more user-friendly
#                display_name = navigation_page.replace("_", " ").title()
                
                # Handle special cases for cleaner display
#                if navigation_page == "home":
#                    display_name = "Home"
#                elif navigation_page in ["services", "service"]:
#                    display_name = "Services"
#                elif navigation_page in ["about", "about us"]:
#                    display_name = "About Us"
#                elif navigation_page in ["contact", "contact us"]:
#                    display_name = "Contact Us"
                
                # Return a response with a clickable link (HTML format)
#                response = f"Sure! You can <a href='{page_url}' class='chatbot-link'>click here</a> to visit the {display_name} page."
                
                # Add to chat history
#                session['chat_history'].append(f"User: {question}")
#                session['chat_history'].append(f"Nova: {response}")
#                
#                return response

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