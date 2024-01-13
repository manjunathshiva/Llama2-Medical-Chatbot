from langchain.embeddings import HuggingFaceEmbeddings
from langchain.document_loaders import PyPDFLoader, DirectoryLoader,UnstructuredWordDocumentLoader
from langchain.document_loaders import UnstructuredExcelLoader, UnstructuredPowerPointLoader
from langchain.document_loaders import ConfluenceLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma

from constants import CHROMA_SETTINGS
import chromadb
import os

DATA_PATH = 'data/'
persist_directory =  CHROMA_SETTINGS.persist_directory


# Create vector database
def create_vector_db():
    
    loaders =  {
                 '.pdf':  DirectoryLoader(DATA_PATH,
                             glob='*.pdf',
                             loader_cls=PyPDFLoader,
                             use_multithreading=True,
                             show_progress=True,
                             silent_errors=True
                ),
    
                 '.docx':  DirectoryLoader(DATA_PATH,
                             glob='*.docx',
                             loader_cls=UnstructuredWordDocumentLoader,
                             use_multithreading=True,
                             show_progress=True,
                             silent_errors=True
                ),
    
                 '.xlsx':  DirectoryLoader(DATA_PATH,
                             glob='*.xlsx',
                             loader_cls=UnstructuredExcelLoader,
                             use_multithreading=True,
                             show_progress=True,
                             silent_errors=True
                 ),

                 '.pptx':  DirectoryLoader(DATA_PATH,
                             glob='*.pptx',
                             loader_cls=UnstructuredPowerPointLoader,
                             use_multithreading=True,
                             show_progress=True,
                             silent_errors=True
                 ),

               }


    documents = []

    config = {
          "confluence_url":"https://templates.atlassian.net/wiki/",
          "username":None,
          "api_key":None,
          }
    
    confluence_url = config.get("confluence_url",None)
    username = config.get("username",None)
    api_key = config.get("api_key",None)

    embeddings = HuggingFaceEmbeddings(model_name='BAAI/bge-large-en-v1.5',
                                       model_kwargs={'device': 'cpu'})
    documents = []
   ## 1. Extract the documents
    loader = ConfluenceLoader(
       url=confluence_url,
       username = username,
       api_key= api_key
   )
    
   # documents = loader.load(space_key="RD",limit=1000)
   # documents.extend(loader.load(space_key="SWPRJ",limit=1000))
   # documents.extend(loader.load(space_key="CW",limit=1000))
 
    for loader in loaders.values():
         documents.extend(loader.load())



    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500,
                                                   chunk_overlap=100)
    texts = text_splitter.split_documents(documents)


    # Chroma client
    chroma_client = chromadb.PersistentClient(settings=CHROMA_SETTINGS , path=persist_directory)
    

    # Create and store locally vectorstore
    print(f"Creating embeddings. May take some minutes...")
    db = Chroma.from_documents(texts, embeddings, persist_directory=persist_directory, client_settings=CHROMA_SETTINGS, client=chroma_client)
    
    db.persist()
    db = None
    print(f"Run chainlit run model.py -w")


if __name__ == "__main__":
    create_vector_db()

