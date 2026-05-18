import os
import re
import time
import json
import traceback
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
chroma_tenant = os.getenv("CHROMA_TENANT")
chroma_database = os.getenv("CHROMA_DATABASE")
chroma_api_key = os.getenv("CHROMA_API_KEY")

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
    # 현재 연도의 1월 1일 00:00:00으로 설정
    now = datetime.now()
    return datetime(now.year, 1, 1)

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

    processed_ids = set() # 게시글 고유 번호(articleNo) 저장용
    offset, limit = 0, 10
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
                num_td = row.select_one('td.b-num-box')
                is_notice = num_td and "공지" in num_td.get_text()
                
                link = row.find('a', href=True)
                if not link: continue
                
                # 1. 고유 ID 추출 및 중복 체크 (가장 먼저 수행)
                id_match = re.search(r'articleNo=(\d+)', link['href'])
                if not id_match: continue
                article_id = id_match.group(1)

                if article_id in processed_ids: continue
                processed_ids.add(article_id)

                # 2. URL 정규화 (offset 등 불필요한 파라미터 제거)
                clean_url = f"{base_url}?mode=view&articleNo={article_id}"
                title = link.get_text(strip=True)

                date_match = re.search(r'(\d{2}\.\d{2}\.\d{2})', row.get_text())
                if not date_match: continue
                curr_date = datetime.strptime(date_match.group(1), "%y.%m.%d")

                # 3. 증분 중단 조건 (일반 게시글만 해당)
                if not is_notice:
                    if curr_date <= last_limit_date:
                        print(f"  🛑 [중단] 일반 게시글 기한 만료: {title}")
                        stop_signal = True
                        break 
                
                # 4. 수집 및 파일 처리[cite: 1, 3]
                print(f"  📥 {'[중요]' if is_notice else '[일반]'} 수집: {title}")
                p_res = requests.get(clean_url, headers=headers)
                p_soup = BeautifulSoup(p_res.text, 'html.parser')
                content = p_soup.select_one('div.fr-view').get_text(strip=True) if p_soup.select_one('div.fr-view') else ""
                
                attach_text = ""
                file_tags = p_soup.select('a[href*="download"]')
                for f_tag in file_tags:
                    f_url = base_url + f_tag['href'] if f_tag['href'].startswith('?') else f_tag['href']
                    f_name = f_tag.get_text(strip=True)
                    f_path = os.path.join(FILE_DIR, f_name)
                    
                    with open(f_path, "wb") as f: 
                        f.write(requests.get(f_url).content)
                    
                    ext = os.path.splitext(f_name)[1].lower()
                    if ext == ".pdf": 
                        attach_text += f"\n[PDF: {f_name}]\n" + parse_pdf(f_path)
                    elif ext in [".xls", ".xlsx"]: 
                        attach_text += f"\n[Excel: {f_name}]\n" + parse_excel(f_path)
                    print(f"    └📎 첨부파일 완료: {f_name}")

                # JSON 저장 시 clean_url 사용[cite: 3]
                combined_data = {"title": title, "url": clean_url, "content": content + attach_text}
                safe_name = "".join([c for c in title if c.isalnum() or c in (' ', '_')]).strip()
                with open(os.path.join(DATA_DIR, f"{safe_name}.json"), "w", encoding="utf-8") as f:
                    json.dump(combined_data, f, ensure_ascii=False, indent=4)
                
                if curr_date > new_max_date: new_max_date = curr_date
                total_new_docs += 1

            if stop_signal: break
            offset += limit
            time.sleep(1.0)
            
        except Exception as e:
            print(f"❌ 오류 발생: {e}"); break

    save_last_crawl_date(new_max_date)
    print(f"✨ 완료: {total_new_docs}개 수집됨.")
        
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