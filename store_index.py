from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from pinecone import Pinecone,ServerlessSpec
from pinecone import Pinecone
from langchain_pinecone.vectorstores import PineconeVectorStore
from src.helper import *

from dotenv import load_dotenv
import os

load_dotenv()

os.environ['HF_TOKEN']=os.getenv('HF_TOKEN')
os.environ['GROQ_API_KEY']=os.getenv('GROQ_API_KEY')
os.environ['PINECONE_API_KEY']=os.getenv('PINECONE_API_KEY')

embeddings=HuggingFaceEmbeddings(model='all-MiniLM-L6-v2')
llm=ChatGroq(model='llama-3.1-8b-instant')

path=r'C:\Users\Chetan\OneDrive\Desktop\LANGCHAIN MEDICAL CHATBOT\Medical-ChatBot\data'

pc=Pinecone()

index_name='medical-chatbot'

if not pc.has_index(index_name):
    pc.create_index(
        name=index_name,
        dimension=384,
        metric='cosine',
        spec=ServerlessSpec(cloud='aws',region='us-east-1')
    )

index=pc.Index(index_name)

data=extract_data(path)
data=minimal_extract(data)

docsearch=PineconeVectorStore.from_documents(
    documents=data,
    embedding=embeddings,
    index_name=index_name
)