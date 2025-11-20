from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.runnables import RunnableWithMessageHistory

store={}

def get_session_history(session_id:str)->BaseChatMessageHistory:
    if session_id not in store:
        store[session_id]=ChatMessageHistory()
    return store[session_id]


def extract_data(path:str)->Document:
    document=PyPDFDirectoryLoader(path).load()
    document=RecursiveCharacterTextSplitter(chunk_size=1000,chunk_overlap=200).split_documents(document)
    return document

def minimal_extract(doc):
    for i in range(0,len(doc)-1):
        doc[i]=Document(
        page_content=doc[i].page_content,
        metadata={"source": doc[i].metadata['source']}
    )
    return doc