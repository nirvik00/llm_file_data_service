
import os 
from dotenv import load_dotenv
load_dotenv()
openai_key = os.getenv("OPENAI_API_KEY")

from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma



embeddings = OpenAIEmbeddings()

text_splitter=CharacterTextSplitter(
    separator = "\n",
    chunk_size = 1000,
    chunk_overlap = 200
)

DATA_PATH = "pdf_data/"
loader = DirectoryLoader(DATA_PATH, glob='*.pdf', loader_cls=PyPDFLoader)
docs = loader.load_and_split(text_splitter)

# generate embeddings and wrtie to db
# if you run again and again, you duplicate data
db = Chroma.from_documents(
    docs, 
    embedding = embeddings,
    persist_directory = "embeddings"
)




 
