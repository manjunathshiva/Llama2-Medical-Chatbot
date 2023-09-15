from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.document_loaders import PyPDFLoader, DirectoryLoader,UnstructuredWordDocumentLoader
from langchain.document_loaders import UnstructuredExcelLoader, UnstructuredPowerPointLoader
from langchain.document_loaders import ConfluenceLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter 
from langchain.vectorstores import Chroma

from constants import CHROMA_SETTINGS
import chromadb

DATA_PATH = 'data/'
#DB_FAISS_PATH = 'vectorstore/db_faiss'
persist_directory =  CHROMA_SETTINGS.persist_directory


# Create vector database
def create_vector_db():
 #   text_loader_kwargs={'autodetect_encoding': True}
    
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
    for loader in loaders.values():
         documents.extend(loader.load())



    text_splitter = RecursiveCharacterTextSplitter(chunk_size=512,
                                                   chunk_overlap=80)
    texts = text_splitter.split_documents(documents)

    embeddings = HuggingFaceEmbeddings(model_name='sentence-transformers/all-mpnet-base-v2',
                                       model_kwargs={'device': 'cpu'})
    # Chroma client
    chroma_client = chromadb.PersistentClient(settings=CHROMA_SETTINGS , path=persist_directory)
    

    # Create and store locally vectorstore
    print(f"Creating embeddings. May take some minutes...")
    db = Chroma.from_documents(texts, embeddings, persist_directory=persist_directory, client_settings=CHROMA_SETTINGS, client=chroma_client)
    db.persist()
    db = None
    print(f"Run chainlit run model.py -w")

   # db = FAISS.from_documents(texts, embeddings)
   # db.save_local(DB_FAISS_PATH)

if __name__ == "__main__":
    create_vector_db()

