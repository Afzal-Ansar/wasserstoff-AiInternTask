from PyPDF2 import PdfReader
import streamlit as st
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.prompts import PromptTemplate
from langchain_groq import ChatGroq
from langchain_core.runnables import RunnablePassthrough,RunnableParallel
import uuid
from langchain.docstore.document import Document
import pytesseract
from PIL import Image
from pdf2image import convert_from_path
template1="""
You are a helpful document assistant.
When a user asks a question, your task is to provide a detailed answer from the provided context ({context}) and present them in two parts.

Part 1 - Tabular format:
For each relevant document, extract the answer, document ID and citation (page number and paragraph number if available) and present it in a table with three columns:
'Document ID' | 'Extracted Answer' | 'Citation'

Example:
| Document ID | Extracted Answer                                                                  | Citation         |
|-------------|-----------------------------------------------------------------------------------|------------------|
| DOC001      | The order states that the fine was imposed under section 15 of the SEBI Act.     | Page 4, Para 2   |
| DOC002      | Tribunal observed delay in disclosure violated Clause 49 of LODR.                | Page 2, Para 1   |

Part 2 – Final Synthesized Response in Chat Format:
After the table, give a clear, structured synthesis of the findings by organizing them into themes.
For each theme, mention relevant Document IDs and summarize the insights.

Example:
Theme 1 - Regulatory Non-Compliance:
Documents (DOC001, DOC002) highlight regulatory non-compliance with SEBI Act and LODR.

Theme 2 - Penalty Justification:
DOC001 explicitly justifies penalties under statutory frameworks.
"""
prompt=PromptTemplate(template=template1,input_variables=['context'])
model=ChatGroq(api_key="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",model="Llama-3.3-70B-Versatile")

embed=HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
st.header(""" :blue[DocumentResearch & Theme Identification Chatbot]""")
files=st.file_uploader("upload multiple pdf files",type=["pdf"],accept_multiple_files=True)

def pdf_text(files):
    try:
        docs=[]
        for pdf in files:
            doc_id = str(uuid.uuid4())[:6].upper()
            pdf_reader=PdfReader(pdf)
            for page_num,page in enumerate(pdf_reader.pages):
                page_text=page.extract_text()
                docs.append({"doc_id": doc_id,"source": pdf.name,"page": page_num + 1,   "text": page_text})
        return docs
    except:
        docs=[]
        for pdf in files:
           doc_id=str(uuid.uuid4())[:6].upper()
           images=convert_from_path(pdf,dpi=300)
           for page_num,page in enumerate(images):
                page_text=pytesseract.image_to_string(images)
                docs.append({"doc_id":doc_id,"source":pdf.name,"page":page_num+1,"text":page_text})
        return docs
def chunk_files(docs):
    splitter = RecursiveCharacterTextSplitter(chunk_size=512, chunk_overlap=128, separators=["\n\n", "\n", ".", "?"])
    chunks = []
    for doc in docs:
        split_chunks = splitter.split_text(doc["text"])
        for idx, chunk in enumerate(split_chunks):
            chunks.append(
                Document(
                    page_content=chunk,
                    metadata={
                        "doc_id": doc["doc_id"],
                        "source": doc["source"],
                        "page": doc["page"],
                        "chunk_id": idx + 1
                    }
                )
            )
    return chunks
def vector_store(chunks):
    vector = FAISS.from_texts([doc.page_content for doc in chunks], embed)

    retriever=vector.as_retriever(search_kwargs={"k": 6})
    return retriever
def chain1(retriever):
    chain=(RunnableParallel(
        {'context':retriever,'user_query':RunnablePassthrough()})|prompt|model)
    return chain

user_query=st.text_input("enter your query")
if user_query:
    if st.button("generate response"):
        text=pdf_text(files)
        chunks=chunk_files(text)
        retriever=vector_store(chunks)
        chain=chain1(retriever)
        response=chain.invoke(user_query)
        st.write(response)
