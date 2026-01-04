import streamlit as st
import requests

st.set_page_config(page_title="AI Yurist", page_icon="⚖️", layout="wide")

st.title("⚖️ O'zbekiston Qonunchiligi bo'yicha Shartnomalar Tahlili")
st.write("Shartnoma faylini yuklang (PDF, DOCX yoki TXT) va AI uni qonunchilikka muvofiqligini tekshiradi.")


# st.sidebar.header("Sozlamalar")
# api_url = st.sidebar.text_input("FastAPI URL", value="http://127.0.0.1:8000/analyze")

upload_file = st.file_uploader("Shartnomani tanlang", type=["pdf", "docx", "txt"])

if upload_file is not None:
    st.info(f"Fayl yuklandi: {upload_file.name}")

    if st.button("Tahlil qilishni boshlash"):
        with st.spinner("AI shartnomani o'rganmoqda..."):
            try:
                files = {"file": (upload_file.name, upload_file.getvalue(), upload_file.type)}
                response = requests.post("http://127.0.0.1:8000/analyze", files=files)

                if response.status_code == 200:
                    result = response.json()

                    st.success("Tahlil yakunlandi!")

                    st.subheader("📋 AI Xulosasi:")
                    st.markdown(result.get("analysis", "Xulosa topilmadi"))

                else:
                    st.error(f"Xatolik yuz berdi: {response.status_code}")
                    st.write(response.text)

            except Exception as e:
                st.error(f"Serverga bog'lanishda xatolik: {e}")

