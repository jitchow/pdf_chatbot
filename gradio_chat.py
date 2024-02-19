import gradio as gr

from langchain.vectorstores import Chroma
from langchain.llms import OpenAI
from langchain.chains import RetrievalQA
from langchain.embeddings.openai import OpenAIEmbeddings

# from InstructorEmbedding import INSTRUCTOR
from langchain.embeddings import HuggingFaceInstructEmbeddings

from langchain.chat_models import ChatOpenAI
from langchain.agents import ConversationalChatAgent, ZeroShotAgent, Tool, AgentExecutor
from langchain.memory import ConversationBufferMemory
from langchain import OpenAI, LLMChain, PromptTemplate
from langchain.document_loaders import DirectoryLoader, TextLoader, PyPDFLoader, PyPDFDirectoryLoader, Docx2txtLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter, CharacterTextSplitter

from langchain.chat_models import ChatOpenAI
from langchain.chains.combine_documents.stuff import StuffDocumentsChain
from langchain.prompts import PromptTemplate
from langchain.chains import create_qa_with_sources_chain

import json
import os
os.environ["OPENAI_API_KEY"] = "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"

retrieval_qa = None

def generate_response(files, query):
  try:
      
    documents = []
    embeddings = OpenAIEmbeddings()

    for file in files:
      loader = PyPDFLoader(file)
      documents.extend(loader.load_and_split())

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    texts = text_splitter.split_documents(documents=documents)

    db = Chroma.from_documents(persist_directory='db', documents=texts, embedding=OpenAIEmbeddings())
    retriever = db.as_retriever()

    llm_src = ChatOpenAI(temperature=0, model="gpt-3.5-turbo-0613")

    qa_chain = create_qa_with_sources_chain(llm_src)

    doc_prompt = PromptTemplate(
        template="""
        You are a study helper chatbot. Use the following pieces of context to answer the question at the end. If you don't know the answer, just say that you don't know, don't try to make up an answer.
        As an AI assistant you provide answers based on the given context, ensuring accuracy and brifness. 
        You always follow these guidelines:

        -If the answer isn't available within the context, state that fact
        -Otherwise, answer to your best capability, refering to source of documents provided
        -Only use examples if explicitly requested
        -Do not introduce examples outside of the context
        -Do not answer if context is absent
        -Limit responses to three or four sentences for clarity and conciseness
        
        Content: {page_content}
        Source: {source}
        Page:{page}
        """, # look at the prompt does have page#
        input_variables=["page_content", "source","page"],
    )

    final_qa_chain = StuffDocumentsChain(
        llm_chain=qa_chain, 
        document_variable_name='context',
        document_prompt=doc_prompt,
    )
    retrieval_qa = RetrievalQA(
        retriever=retriever,
        combine_documents_chain=final_qa_chain
    )
    ans = retrieval_qa.run(query)
    return ans

  except Exception as e:
    # Return an error code or message when an exception occurs
    error_message = str(e)  # You can customize the error message as needed
    return {'error': error_message}

  

# Create an interface for chatbot interaction
chatbot_interface = gr.Interface(
    fn=generate_response,
    inputs=["files","text"],
    outputs="json",
    title="DocuSearch Assistance and Retrieval Tool (DART)",
    description="Upload a PDF document(s) and then enter a prompt to chat with the bot.",
)

# Combine the two interfaces to create a sequential workflow
# workflow = gr.Interface([load_docs, chatbot_interface],inputs=None, outputs=None, live=False)

# Launch the combined workflow
chatbot_interface.launch(share=True, server_name="0.0.0.0", server_port=7880)
