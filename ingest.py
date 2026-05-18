import os
import re
import time
import json
import requests
import chromadb
import pandas as pd
import fitz  # PyMuPDF
from bs4 import BeautifulSoup
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.core import StorageContext, VectorStoreIndex, Document, Settings
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from dotenv import load_dotenv
from datetime import datetime, timedelta

load_dotenv()
DATA_DIR = "./data"
FILE_DIR = "./data/attachments"
STATE_FILE = "./crawl_state.json" # 증분 수집을 위한 상태 저장 파일

# 1. 파일 파서 정의 (PDF, Excel)[cite: 1]
def parse_pdf(file_path):
    text = ""
    try:
        with fitz.open(file_path) as doc:
            for page in doc:
                text += page.get_text() + "\n"
    except Exception as e:
        print(f"⚠️ PDF 파싱 에러: {e}")
    return text

def parse_excel(file_path):
    try:
        df = pd.read_excel(file_path)
        return df.to_markdown(index=False)
    except Exception as e:
        print(f"⚠️ Excel 파싱 에러: {e}")
        return ""

# 2. 증분 수집 상태 관리
def get_last_crawl_date():
    if os.path.exists(STATE_FILE):
        with open(STATE_FILE, "r") as f:
            return datetime.strptime(json.load(f)["last_date"], "%y.%m.%d")
    return datetime.now() - timedelta(days=365) # 초기 실행 시 1년 전

def save_last_crawl_date(date_obj):
    with open(STATE_FILE, "w") as f:
        json.dump({"last_date": date_obj.strftime("%y.%m.%d")}, f)

# 3. 통합 크롤링 함수
def crawl_cnupa_notices():
    last_limit_date = get_last_crawl_date()
    new_max_date = last_limit_date
    base_url = "https://cnupa.cnu.ac.kr/pa/notice/notice.do"
    headers = {"User-Agent": "Mozilla/5.0"}

    if not os.path.exists(DATA_DIR): os.makedirs(DATA_DIR)
    if not os.path.exists(FILE_DIR): os.makedirs(FILE_DIR)

    limit = 10
    offset = 0
    total_new_docs = 0
    stop_signal = False

    while not stop_signal:
        page_url = f"{base_url}?articleLimit={limit}&article.offset={offset}"
        print(f"🔍 Analyzing Offset: {offset}")
        
        try:
            res = requests.get(page_url, headers=headers, timeout=10)
            soup = BeautifulSoup(res.text, 'html.parser')
            rows = soup.select('table.board-table tbody tr')

            if not rows: break

            for row in rows:
                # 날짜 추출
                date_match = re.search(r'(\d{2}\.\d{2}\.\d{2})', row.get_text())
                if not date_match: continue
                
                curr_date = datetime.strptime(date_match.group(1), "%y.%m.%d")

                # 증분 수집 로직: 저장된 날짜보다 이전 글이면 크롤링 완전 중단
                if curr_date <= last_limit_date:
                    stop_signal = True
                    break

                if curr_date > new_max_date: new_max_date = curr_date

                # 상세 정보 수집 및 첨부파일 처리[cite: 3]
                link = row.find('a', href=True)
                if link:
                    title = link.get_text(strip=True)
                    detail_url = base_url + link['href'] if link['href'].startswith('?') else link['href']
                    
                    p_res = requests.get(detail_url, headers=headers)
                    p_soup = BeautifulSoup(p_res.text, 'html.parser')
                    
                    content = p_soup.select_one('div.fr-view').get_text(strip=True) if p_soup.select_one('div.fr-view') else ""
                    
                    # 첨부파일 다운로드 및 본문에 병합[cite: 1, 3]
                    attach_text = ""
                    for a_tag in p_soup.select('a[href*="download"]'):
                        f_url = base_url + a_tag['href'] if a_tag['href'].startswith('?') else a_tag['href']
                        f_name = a_tag.get_text(strip=True)
                        f_path = os.path.join(FILE_DIR, f_name)
                        
                        f_res = requests.get(f_url)
                        with open(f_path, "wb") as f: f.write(f_res.content)
                        
                        ext = os.path.splitext(f_name)[1].lower()
                        if ext == ".pdf": attach_text += f"\n[PDF내용: {f_name}]\n" + parse_pdf(f_path)
                        elif ext in [".xls", ".xlsx"]: attach_text += f"\n[Excel내용: {f_name}]\n" + parse_excel(f_path)

                    # 저장[cite: 3]
                    combined_data = {"title": title, "url": detail_url, "content": content + attach_text}
                    safe_name = "".join([c for c in title if c.isalnum() or c in (' ', '_')]).strip()
                    with open(os.path.join(DATA_DIR, f"{safe_name}.json"), "w", encoding="utf-8") as f:
                        json.dump(combined_data, f, ensure_ascii=False, indent=4)
                    
                    total_new_docs += 1

            offset += limit
            time.sleep(2.0) # 서버 매너
            
        except Exception as e:
            print(f"❌ Critical Error during crawl: {e}")
            break

    save_last_crawl_date(new_max_date)
    print(f"✅ 수집 완료: 새로운 문서 {total_new_docs}개 발견.")
        
def load_json_documents():
    print("📂 [데이터 로드] JSON 파일에서 문서와 출처를 읽어옵니다...")
    docs = []
    if not os.path.exists(DATA_DIR): return docs
    
    for filename in os.listdir(DATA_DIR):
        if filename.endswith(".json"):
            with open(os.path.join(DATA_DIR, filename), "r", encoding="utf-8") as f:
                data = json.load(f)
                # 본문(text)과 메타데이터(metadata)를 명확히 분리하여 Document 객체 생성
                doc = Document(text=data['content'],metadata={"title": data['title'],"url": data['url']})
                docs.append(doc)
    return docs

def configure_ingestion_settings():
    Settings.embed_model = HuggingFaceEmbedding(
        model_name="jhgan/ko-sroberta-multitask"
    )

def main():
    try:
        configure_ingestion_settings() # 설정 호출
        crawl_cnupa_notices()
        
        documents = load_json_documents()
        if not documents:
            print("⚠️ 수집된 문서가 없어 인덱스를 생성할 수 없습니다.")
            return

        print("🧠 [인덱스 생성] 벡터 저장소를 구축 중입니다. 잠시만 기다려주세요...")
        
        db = chromadb.CloudClient(
            tenant=chroma_tenant,
            database=chroma_database,
            api_key=chroma_api_key,
        )
        
        collection_name = "pikmin326_cnu_ai_project_v2" 
        chroma_collection = db.get_or_create_collection(collection_name)
        vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
        storage_context = StorageContext.from_defaults(vector_store=vector_store)
        VectorStoreIndex.from_documents(documents, storage_context=storage_context)
        
        print("\n" + "="*50)
        print("🚀 크로마 클라우드 원격 인덱싱 완료! 이제 app.py를 준비하세요.")
        print("="*50)
        
    except Exception as e:
        print(f"\n❌ 실행 중 치명적 오류 발생: {e}")
    
    # EXE 실행 시 창이 바로 닫히는 것을 방지
    input("\n계속하려면 엔터 키를 누르세요...")
    
if __name__ == "__main__": 
    try:
        main()
    except Exception as e:
        print(f"\n❌ 실행 중 치명적 오류 발생: {e}")
        traceback.print_exc()