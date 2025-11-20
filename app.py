from flask import Flask, jsonify, render_template, request
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_pinecone.vectorstores import PineconeVectorStore
from langchain_groq import ChatGroq
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_history_aware_retriever
from src.prompt import *
from src.helper import get_session_history, RunnableWithMessageHistory

import os
from dotenv import load_dotenv

load_dotenv()

os.environ['HF_TOKEN'] = os.getenv('HF_TOKEN')
os.environ['GROQ_API_KEY'] = os.getenv('GROQ_API_KEY')
os.environ['PINECONE_API_KEY'] = os.getenv('PINECONE_API_KEY')

index_name = 'medical-chatbot'

# Initialize components
embeddings = HuggingFaceEmbeddings(model='all-MiniLM-L6-v2')
llm = ChatGroq(model='llama-3.1-8b-instant')
docsearch = PineconeVectorStore.from_existing_index(
    index_name=index_name, 
    embedding=embeddings
)
retriever = docsearch.as_retriever(
    search_type='similarity', 
    search_kwargs={'k': 3}
)

history_retriever = create_history_aware_retriever(llm, retriever, history_prompt)
qa_chain = create_stuff_documents_chain(llm, prompt)
rag_chain = create_retrieval_chain(history_retriever, qa_chain)

get_history = RunnableWithMessageHistory(
    rag_chain,
    get_session_history,
    input_messages_key='input',
    history_messages_key='chat_history',
    output_messages_key='answer',
)

app = Flask(__name__)

@app.route("/")
def index():
    return render_template('chat.html')

@app.route("/get", methods=["POST"])
def chat():
    try:
        msg = request.form.get("msg", "").strip()
        sess_id = request.form.get("sess_id", "").strip()
        
        # Validate inputs
        if not msg:
            return "Please provide a message.", 400
        if not sess_id:
            return "Session ID is required.", 400
        
        print(f"Session ID: {sess_id}")
        print(f"User Input: {msg}")
        
        # Get response from chatbot
        response = get_history.invoke(
            {"input": msg},
            config={"configurable": {"session_id": sess_id}}
        )
        
        print(f"Bot Response: {response['answer']}")
        return str(response["answer"])
        
    except Exception as e:
        print(f"Error: {str(e)}")
        return "Sorry, I encountered an error processing your request.", 500

if __name__ == '__main__':
    app.run(debug=True, port=8080)
