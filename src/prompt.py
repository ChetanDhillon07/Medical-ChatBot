from langchain_core.prompts import ChatPromptTemplate
from langchain_core.prompts import MessagesPlaceholder

system_prompt='''
    you are a medical assistant for question-answering task. use the following pieces of retrieved context to answer the question. if you dont know the answer say that you dont know. use three sentences maximum and keep the answer conscise. 
    \n\n
    context:{context}
'''
prompt=ChatPromptTemplate.from_messages(
    [
        ('system',system_prompt),
        ('human','{input}')
    ]
)
history_prompt=ChatPromptTemplate.from_messages(
    [
        ('system', 'you are a medical assistant for question-answering task. use the following pieces of retrieved context to answer the question. if you dont know the answer say that you dont know. use three sentences maximum and keep the answer conscise. If the question is out of the medical domain simply say I DONT KNOW. keep it small and conscise'),
        MessagesPlaceholder(variable_name='chat_history'),
        ('human','{input}')
    ]
)