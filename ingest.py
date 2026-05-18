import os
import re
import sys
import time
import json
import requests
import chromadb
import traceback
from bs4 import BeautifulSoup
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.core import StorageContext,VectorStoreIndex, Document, Settings
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from dotenv import load_dotenv
from datetime import datetime, timedelta

load_dotenv()
os.environ["HF_TOKEN"] = os.getenv("HF_TOKEN")
os.environ["TIKTOKEN_CACHE_DIR"] = "./tiktoken_cache"
PERSIST_DIR = os.getenv("PERSIST_DIR", "./default_storage")
DATA_DIR = "./data"
chroma_tenant = os.getenv("CHROMA_TENANT")
chroma_database = os.getenv("CHROMA_DATABASE")
chroma_api_key = os.getenv("CHROMA_API_KEY")

def crawl_cnupa_notices():
    print("🌐 [크롤링 시작] 최근 12개월 이내의 공지를 수집합니다...")
    
    limit_date = datetime.now() - timedelta(days=365)
    base_url = "https://cnupa.cnu.ac.kr/pa/notice/notice.do"
    headers = {"User-Agent": "Mozilla/5.0"}

    if not os.path.exists(DATA_DIR):
        os.makedirs(DATA_DIR)

    try:
        response = requests.get(base_url, headers=headers, timeout=10)
        soup = BeautifulSoup(response.text, 'html.parser')
        # 테이블의 모든 행(tr)을 가져옵니다.
        rows = soup.select('table.board-table tbody tr') 

        count = 0
        for row in rows:
            tds = row.select('td')
            if not tds: continue

            # 행 전체의 텍스트를 하나로 합칩니다.
            row_text = row.get_text(separator=" ", strip=True)

            # 정규표현식으로 '26.04.30' 같은 패턴을 찾습니다.
            # \d{2}는 숫자 2개, \.은 점을 의미합니다.
            match = re.search(r'(\d{2}\.\d{2}\.\d{2})', row_text)

            if not match:
                continue # 날짜 패턴이 없으면 스킵

            try:
                # 패턴에 맞는 첫 번째 결과(날짜)만 추출합니다.
                clean_date = match.group(1)
                post_date = datetime.strptime(clean_date, "%y.%m.%d")
                
                if post_date < limit_date:
                    continue
                
                # 4. 상세 페이지 링크 추출
                link_tag = row.find('a', href=True)
                if link_tag and "mode=view" in link_tag['href']:
                    title = link_tag.get_text(strip=True)
                    href = link_tag['href']
                    detail_url = base_url + href if href.startswith('?') else href
                    
                    # 상세 내용 수집
                    time.sleep(2.0)
                    post_res = requests.get(detail_url, headers=headers)
                    post_soup = BeautifulSoup(post_res.text, 'html.parser')
                    
                    selectors = ['div.fr-view', 'div.board-view-content', 'div.v-content']
                    content_area = next((post_soup.select_one(s) for s in selectors if post_soup.select_one(s)), None)
                    content = content_area.get_text(separator="\n", strip=True) if content_area else "본문 내용을 찾을 수 없습니다."

                    # JSON 저장
                    notice_data = {"title": title, "url": detail_url, "content": content}
                    safe_title = "".join([c for c in title if c.isalnum() or c in (' ', '_')]).strip()
                    
                    with open(os.path.join(DATA_DIR, f"{safe_title}.json"), "w", encoding="utf-8") as f:
                        json.dump(notice_data, f, ensure_ascii=False, indent=4)
                    
                    count += 1
                    print(f"  [{count}] 수집 성공: {title} ({clean_date})")
                    
            except Exception as inner_error:
                # 내부 루프 에러 시 'inner_error' 객체를 안전하게 출력
                print(f"⚠️ 개별 항목 처리 중 오류 발생: {inner_error}")
                continue

        print(f"✅ 총 {count}개의 최신 공지가 수집되었습니다.")
    except Exception as outer_error:
        print(f"❌ 크롤링 중 치명적 오류 발생: {outer_error}")
        
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