#Install Following Libraries

#!pip install langchain
#!pip install openai
#!pip install PyPDF2
#!pip install faiss-cpu
#!pip install tiktoken

from PyPDF2 import PdfReader
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import FAISS

import os
os.environ["OPENAI_API_KEY"] = ""


# provide the path of  pdf file/files.
pdfreader = PdfReader('Saqib Resume.pdf') 

from typing_extensions import Concatenate
# read text from pdf
raw_text = ''
for i, page in enumerate(pdfreader.pages):
    content = page.extract_text()
    if content:
        raw_text += content

  raw_text

# We need to split the text using Character Text Split such that it sshould not increse token size
text_splitter = CharacterTextSplitter(
    separator = "\n",
    chunk_size = 800,
    chunk_overlap  = 200,
    length_function = len,
)
texts = text_splitter.split_text(raw_text)

len(texts)

# Download embeddings from OpenAI
embeddings = OpenAIEmbeddings()

document_search = FAISS.from_texts(texts, embeddings)

document_search


from langchain.chains.question_answering import load_qa_chain
from langchain.llms import OpenAI

chain = load_qa_chain(OpenAI(), chain_type="stuff")

query = "Vision for 2030"
docs = document_search.similarity_search(query)
chain.run(input_documents=docs, question=query)

query = "How much the agriculture target will be increased to and what the focus will be"
docs = document_search.similarity_search(query)
chain.run(input_documents=docs, question=query)

# Online PDF Analyzer

from langchain.document_loaders import OnlinePDFLoader
loader = OnlinePDFLoader("https://arxiv.org/pdf/1706.03762.pdf")

# !pip install unstructured
data = loader.load()

data
# Download embeddings from OpenAI
embeddings = OpenAIEmbeddings()


# !pip install chromadb
from langchain.indexes import VectorstoreIndexCreator
index = VectorstoreIndexCreator().from_loaders([loader])
query = "Explain me about Attention is all you need"
index.query(query)
