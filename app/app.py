from flask import Flask, render_template, request, jsonify, session
from rag import get_rag_response, RAGPipeline
from flask_cors import CORS
import os
from dotenv import load_dotenv
from datetime import timedelta
#import shutil
#import time
from pathlib import Path

app = Flask(__name__)
#load_dotenv()
#app.secret_key = os.getenv('SECRET_KEY')

env_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '.env'))
load_dotenv(dotenv_path=env_path)

app.secret_key = os.getenv("SECRET_KEY")
CORS(app)

app.permanent_session_lifetime = timedelta(days=1)

UPLOAD_FOLDER = 'docs'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

file_path = Path(__file__).parent / "docs" / "rag_doc.docx"
pipeline_instance = RAGPipeline(file_path)
app.config['pipeline_instance'] = pipeline_instance

@app.get("/")
def index_get():
    return render_template("base.html")

# Add routes for all website pages that the chatbot might direct users to
@app.route('/services')
def services():
    # You would normally render your services page template here
    return render_template("services.html")

@app.route('/about')
def about():
    return render_template("about.html")

@app.route('/contact')
def contact():
    return render_template("contact.html")

@app.route('/products')
def products():
    return render_template("products.html")

# Add more routes as needed for all pages in the WEBSITE_PAGES dictionary in rag.py

@app.route('/predict', methods=['POST'])
def predict():
    """Handles incoming user messages and returns chatbot responses."""
    data = request.get_json()
    user_message = data.get("message")
    pipeline_instance = app.config.get('pipeline_instance')

    # Ensure the pipeline exists (fallback check)
    if not pipeline_instance:
        return jsonify({"answer": "Pipeline is not initialized. Please restart the server."})

    bot_response = get_rag_response(user_message, pipeline_instance)
    return jsonify({"answer": bot_response}) 

#app.config['pipeline_instance'] = None

if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0')