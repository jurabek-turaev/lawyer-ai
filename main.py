import os
import shutil
from fastapi import FastAPI, UploadFile, File, HTTPException
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
import pypdf
import docx
from time import time

load_dotenv()

app = FastAPI(title="AI Yurist - Shartnoma Tahlili")

DB_PATH = "./chroma_db"
embedding_function = GoogleGenerativeAIEmbeddings(model="models/embedding-001")

vector_store = Chroma(persist_directory=DB_PATH, embedding_function=embedding_function)
retriever = vector_store.as_retriever(search_kwargs={"k": 10})


llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.2)

template = """
Sen O'zbekiston qonunchiligi bo'yicha tajribali yuristsan. 
Sening vazifang: Foydalanuvchi yuklagan shartnoma matnini quyida keltirilgan qonun hujjatlari asosida tahlil qilish.

Tahlil qoidalari:
1. Shartnomaning qaysi bandlari O'zbekiston qonunchiligiga zid ekanligini aniqla.
2. Har bir e'tiroz uchun asos bo'lgan qonun moddasini "Qonun hujjatlaridan parchalar" qismidan olib ko'rsat.
3. Agar hammasi to'g'ri bo'lsa, shuni tasdiqla.
4. Javobni o'zbek tilida, aniq va tushunarli qilib ber.
5. Foydalanuvchiga "taqdim etilgan qonun hujjatlaridan parchalardan tahlil qilib chiqdim" deb aytma, o'zingni o'zbekistonning butun qonunlarini bilgandek izoh ber, ya'ni berilgan qonunlar emas, ularni hammasini o'zing biladigandek javob ber
6. O'zbekiston qonunchilig bo'yicha shartnomani yuridik jihatdan o'rganib chiq va sotuvchi yoki sotib oluvchilarni manfaatlari kamsitilishi yoki qonunchilikka to'g'ri kelmaydigan shartlarni o'rganib chiq
7. Javob iloji boricha qisqaroq bo'lishi kerak, foydalanuvchi tahlilni o'qishga ko'p vaqt ketqizmasligi kerak

Qonun hujjatlaridan parchalar:
{context}

Tekshirilayotgan Shartnoma Matni:
{question}
"""

prompt = ChatPromptTemplate.from_template(template)

def format_docs(docs):
    return "\n\n".join([d.page_content for d in docs])

rag_chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

def extract_text_from_file(file_path: str, content_type: str) -> str:
    text = ""
    try:
        if "pdf" in content_type:
            reader = pypdf.PdfReader(file_path)
            for page in reader.pages:
                text += page.extract_text() + "\n"
        elif "word" in content_type or "docx" in content_type:
            doc = docx.Document(file_path)
            for para in doc.paragraphs:
                text += para.text + "\n"
        else:
            with open(file_path, "r", encoding="utf-8") as f:
                text = f.read()
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Faylni o'qishda xatolik: {str(e)}")
    return text


@app.post("/analyze")
async def analyze_contract(file: UploadFile = File(...)):
    
    temp_filename = f"temp/temp_{file.filename}"
    with open(temp_filename, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    try:
        start = time()
        contract_text = extract_text_from_file(temp_filename, file.content_type)

        if len(contract_text) < 50:
            raise HTTPException(status_code=400, detail="Fayl bo'sh yoki matn o'qilmadi.")
        
        response = rag_chain.invoke(contract_text)
        end = time()

        return {
            "filename": file.filename,
            "analysis": response,
            "time": float(f"{end-start:.2f}")
        }
    
    finally:
        if os.path.exists(temp_filename):
            os.remove(temp_filename)