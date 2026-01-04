import os
import shutil
from dotenv import load_dotenv
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_chroma import Chroma

load_dotenv()

LAWS_PATH = "./laws"
DB_PATH = "./chroma_db"

def create_vector_db():
    if not os.path.exists(LAWS_PATH) or not os.listdir(LAWS_PATH):
        print(f"Xato: '{LAWS_PATH}' papkasi bo'sh")
        return
    
    print("Qonunlar yuklanmoqda")

    loader = DirectoryLoader(LAWS_PATH, glob="*.txt", loader_cls=TextLoader)
    documents = loader.load()
    print(f"{len(documents)} ta hujjat yuklandi.")

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        separators=["\n\n", "\n", " ", ""]
    )
    chunks = text_splitter.split_documents(documents)
    print(f"Hujjatlar {len(chunks)} ta qismga bo'lindi.")

    print("Vektor baza yaratilmoqda...")
    embedding_funcion = GoogleGenerativeAIEmbeddings(model="models/embedding-001")

    Chroma.from_documents(
        documents=chunks,
        embedding=embedding_funcion,
        persist_directory=DB_PATH
    )
    print("Baza muvaffaqiyatli yaratildi!")

if __name__ == "__main__":
    if not os.path.exists(DB_PATH):
        shutil.rmtree(DB_PATH)
        print("Eski baza o'chirildi")
    create_vector_db()