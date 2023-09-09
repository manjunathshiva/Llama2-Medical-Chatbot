from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.document_loaders import PyPDFLoader, DirectoryLoader,UnstructuredWordDocumentLoader
from langchain.document_loaders import UnstructuredExcelLoader, UnstructuredPowerPointLoader
from langchain.document_loaders import ConfluenceLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter 

DATA_PATH = 'data/'
DB_FAISS_PATH = 'vectorstore/db_faiss'

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

#                  'wiki ':   ConfluenceLoader(url="https://yoursite.atlassian.com/wiki", 
#                                          token="12345")
               }


    documents = []
    for loader in loaders.values():
       if loader == 'wiki':
            documents.extend(loader.load(space_key="SPACE", include_attachments=True, limit=50, max_pages=50))
       else:
            documents.extend(loader.load())



    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500,
                                                   chunk_overlap=50)
    texts = text_splitter.split_documents(documents)

    embeddings = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2',
                                       model_kwargs={'device': 'cpu'})

    db = FAISS.from_documents(texts, embeddings)
    db.save_local(DB_FAISS_PATH)

if __name__ == "__main__":
    create_vector_db()

