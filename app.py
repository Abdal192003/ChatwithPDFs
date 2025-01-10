# MADE BY ==
            # ABDAL AHMAD KHAN (2230139)

# # " "------ Importing Libraries ------" "

# Used to make web pages 
import streamlit as st

# Importing this library for PDF reader as this will help in pdf page merging,cropping,adding custom data,
# also for PDF password lock
from PyPDF2 import PdfReader

# Importing this library from langchain for text splitting in PDF
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os
# From langchain importing google made embedding techniques or an integration package for
# connecting langchain and google gemini pro features
from langchain_google_genai import GoogleGenerativeAIEmbeddings

 # importing google packeages features to empower our GENAI apps
import google.generativeai as genai

# Importing this libraries for embedding techniques from text to vectors using FACEBOOK AI SIMILARITY SEARCH
from langchain.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI

# This software package will help us to question our pdf , langchain simply this to question from the given text or huge paragraph 
from langchain.chains.question_answering import load_qa_chain

# This will allows us to send us request or asking question as prompt 
from langchain.prompts import PromptTemplate

# To load the environment variables
from dotenv import load_dotenv
# " ---------------------------------------------------------------------------------"

# loading the environment variable to access our Google Gemini Pro API 
load_dotenv()
os.getenv("GOOGLE_API_KEY")
 # We want the google api key to configure accordingly
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))


# This function will give information when we submit our pdf 
# Read the pdf , extract information from each and every pages of pdf ,return the extracted text 
def get_pdf_text(pdf_docs):
    text=""
    for pdf in pdf_docs:
        pdf_reader= PdfReader(pdf)
        for page in pdf_reader.pages:
            text+= page.extract_text()
    return  text


# For vector embedding we divide the text into chunks
def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
    chunks = text_splitter.split_text(text)
    return chunks

#  converting these chunks of text into vectors 
def get_vector_store(text_chunks):
    embeddings = GoogleGenerativeAIEmbeddings(model = "models/embedding-001")
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
    vector_store.save_local("faiss_index")


# lOGIC HOW GEMINI PRO WILL BEHAVE AND WILL WITH HOW EFFICIENCY , ACCURACY AS WE HAVE GIVEN TEMPERATURE=0.3 TO GIVE MORE ACCURACY.
     
def get_conversational_chain():

    prompt_template = """
    Answer the question as detailed as possible from the provided context, make sure to provide all the details, if the answer is not in
    provided context just say, "answer is not available in the context", don't provide the wrong answer\n\n
    Context:\n {context}?\n
    Question: \n{question}\n

    Answer:
    """

    model = ChatGoogleGenerativeAI(model="gemini-pro",
                             temperature=0.3)

    prompt = PromptTemplate(template = prompt_template, input_variables = ["context", "question"])
    chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)

    return chain


# lOGIC FOR AI PDF HOW IT WILL TACKLE THE QUESTION OR PROMPT WHICH WE WILL GIVE  

def user_input(user_question):
    embeddings = GoogleGenerativeAIEmbeddings(model = "models/embedding-001")
    
    # new_db = FAISS.load_local("faiss_index", embeddings)
    new_db = FAISS.load_local("faiss_index", embeddings,allow_dangerous_deserialization=True)
    docs = new_db.similarity_search(user_question)

    chain = get_conversational_chain()

    
    response = chain(
        {"input_documents":docs, "question": user_question}
        , return_only_outputs=True)

    print(response)
    st.write("Reply: ", response["output_text"])

# -----------------------------------lOGIC FOR AI PDF IS OVER HERE-------------------------------------------------------------------------

# STREAMLIT CODE FOR UI OF WEB PAGE

def main():
    st.set_page_config("Chat PDF")
    st.header("Chat with PDF using Gemini API Made with Passion and interest By ABDAL💁")

    user_question = st.text_input("Ask a Question from the PDF Files")

    if user_question:
        user_input(user_question)

    with st.sidebar:
        st.title("Menu:")
        pdf_docs = st.file_uploader("Upload your PDF Files and Click on the Submit & Process Button", accept_multiple_files=True)
        if st.button("Submit & Process"):
            with st.spinner("Processing..."):
                raw_text = get_pdf_text(pdf_docs)
                text_chunks = get_text_chunks(raw_text)
                get_vector_store(text_chunks)
                st.success("Done")



if __name__ == "__main__":
    main()

# -------------------END------------------------------------------------------------------------
