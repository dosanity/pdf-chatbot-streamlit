# Imports
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter, RecursiveCharacterTextSplitter
from langchain.vectorstores import DocArrayInMemorySearch
from langchain.chains.summarize import load_summarize_chain
from langchain.chains import RetrievalQA,  ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain.chat_models import ChatOpenAI
from langchain.document_loaders import TextLoader
from langchain.document_loaders import PyPDFLoader
from langchain.prompts import PromptTemplate
from langchain.llms import OpenAI
import tiktoken 
import os
import sys
sys.path.append('../..')

import datetime
current_date = datetime.datetime.now().date()
if current_date < datetime.date(2023, 9, 2):
    llm_name = "gpt-3.5-turbo-0301"
else:
    llm_name = "gpt-3.5-turbo"

def load_db(file, api_key):
    os.environ['OPENAI_API_KEY'] = api_key
    # load documents
    loader = PyPDFLoader(file)
    # loader = file
    documents = loader.load()
    # documents = loader.read()
    # split documents
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
    docs = text_splitter.split_documents(documents)
    # define embedding
    embeddings = OpenAIEmbeddings()
    # create vector database from data
    db = DocArrayInMemorySearch.from_documents(docs, embeddings)
    
    # add in the prompt
    prompt_template_doc = """

    Use the following pieces of context to answer the question at the end. {context}
    You can also look into chat history. {chat_history}
    If you still can't find the answer, please respond: "Please ask a question related to the document."

    Question: {question}
    Answer:
    """
    prompt_doc = PromptTemplate(
        template=prompt_template_doc,
        input_variables=["context", "question", "chat_history"],
    )
    # define retriever
    retriever = db.as_retriever(search_type="similarity", search_kwargs={"k": 4})
    # keeps a buffer of history and process it
    memory = ConversationBufferMemory(
        memory_key="chat_history",
        output_key="answer",
        return_messages=True
    )
    # create a chatbot chain
    qa = ConversationalRetrievalChain.from_llm(
        llm=ChatOpenAI(model_name=llm_name, temperature=0), 
        chain_type="stuff", 
        retriever=retriever,
        combine_docs_chain_kwargs={"prompt": prompt_doc},
        memory=memory
    )
    return qa 

def load_db_sum(file, api_key):
    os.environ['OPENAI_API_KEY'] = api_key
    # load documents
    loader = PyPDFLoader(file)
    # loader = file
    documents = loader.load()
    # documents = loader.read()
    # split documents
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=5000, chunk_overlap=150)
    docs = text_splitter.split_documents(documents)
    # create string of documents
    str_docs = str(documents)

    # define number of tokens from text
    def num_tokens_from_string(string: str, encoding_name: str) -> int:    
        encoding = tiktoken.encoding_for_model(encoding_name)
        num_tokens = len(encoding.encode(string))
        return num_tokens
    
    # get tokens
    num_tokens = num_tokens_from_string(str_docs, llm_name)
    model_max_tokens = 4097
    # define embedding
    embeddings = OpenAIEmbeddings()
    # create vector database from data
    db = DocArrayInMemorySearch.from_documents(docs, embeddings)
    # define retriever
    retriever = db.as_retriever(search_type="similarity", search_kwargs={"k": 4})
    #Keeps a buffer of history and process it
    memory = ConversationBufferMemory(
        memory_key="chat_history",
        output_key="answer",
        return_messages=True
    )

    # create a chatbot chain based on tokens
    if num_tokens < model_max_tokens:
        chain = load_summarize_chain(llm=OpenAI(temperature=0, model="text-davinci-003", openai_api_key=api_key), chain_type="stuff")
        qa = chain.run(documents)
    else:
        chain = load_summarize_chain(llm=OpenAI(temperature=0, model="text-davinci-003", openai_api_key=api_key), chain_type="map_reduce")
        qa = chain.run(documents)

    return qa 

def save_pdf(pdf_file):
    with open("uploaded.pdf", "wb") as file:
        file.write(pdf_file.getvalue())
    file = "uploaded.pdf"
    return file