import os
import sys
import streamlit as st
import chromadb
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.llms.google_genai import GoogleGenAI
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core import VectorStoreIndex, Settings
from dotenv import load_dotenv

load_dotenv()
MY_API_KEY = os.getenv("GEMINI_API_KEY")
MODEL_NAME = os.getenv("MODEL_NAME", "models/gemini-2.5-flash")
chroma_tenant = os.getenv("CHROMA_TENANT")
chroma_database = os.getenv("CHROMA_DATABASE")
chroma_api_key = os.getenv("CHROMA_API_KEY")
Settings.llm = GoogleGenAI(model=MODEL_NAME,api_key=MY_API_KEY,temperature=0.1 )
Settings.embed_model = HuggingFaceEmbedding(model_name="jhgan/ko-sroberta-multitask")

st.set_page_config(page_title="CNU AI 챗봇", layout="centered")
st.title("🎓 충남대학교 AI 챗봇")

def main():
    if "query_engine" not in st.session_state:
        try:
            
            db = chromadb.CloudClient(
                tenant=chroma_tenant,
                database=chroma_database,
                api_key=chroma_api_key,
            )
            
            collection_name = "pikmin326_cnu_ai_project_v2" 
            chroma_collection = db.get_or_create_collection(collection_name)
            vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
            index = VectorStoreIndex.from_vector_store(vector_store)
            st.session_state.query_engine = index.as_query_engine(similarity_top_k=3)
            
        except Exception as e:
            st.error(f"데이터베이스 연결 실패, 인덱싱을 먼저 완료하세요.: {e}")
            st.stop()
            
    if "messages" not in st.session_state:
        st.session_state.messages = []
        
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            
    if prompt := st.chat_input("질문을 입력하세요"):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"): st.markdown(prompt)
        
        with st.chat_message("assistant"):
            response = st.session_state.query_engine.query(prompt)
            st.markdown(response.response)
            with st.expander("📍 참조 출처"):
                for node in response.source_nodes:
                    title = node.node.metadata.get('title', '제목 없음')
                    url = node.node.metadata.get('url', '#')
                    st.write(f"- [{title}]({url})")
                    
        st.session_state.messages.append({"role": "assistant", "content": response.response})
        
if __name__ == "__main__":
    main()