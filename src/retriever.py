"""실습 ③ 다중 벡터 검색기 + 최종 체인.

기준 노트북: practice/03_임베딩_최종체인_연습.ipynb

실행 (프로젝트 루트에서, extract.py → summarize.py 를 먼저 실행해야 합니다):
    python src/retriever.py

⚠️ 임베딩·LLM API 를 호출하므로 실행할 때마다 비용이 발생합니다.
   (질의응답 5건 + Before/After 비교 포함)

표시 규칙:
    # [.py 변환] : 노트북을 .py 로 옮기면서 바꾼 부분
    # [수정]     : 노트북 코드의 버그를 고친 부분
    (표시가 없는 코드는 노트북 코드 그대로입니다)
"""

# ── 0. 결과 복원 (import) ──────────────────────────────────
import os, re, time, uuid, pickle
from pathlib import Path
from dotenv import load_dotenv

# ── 2. 검색기 만들기 (import) ──────────────────────────────
# 1) MultiVectorRetriever import (langchain 1.x : langchain_classic)
try:
    from langchain_classic.retrievers.multi_vector import MultiVectorRetriever
except:
    print("MultiVectorRetriever 클래스가 필요합니다.")

# 2) InMemoryStore import (langchain 1.x : langchain_core.stores)
try:
    from langchain_core.stores import InMemoryStore
except:
    print("InMemoryStore 클래스가 필요합니다.")

# 3) Chroma(langchain_chroma), OpenAIEmbeddings(langchain_openai) import
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings

# ── 3·5. Document import ───────────────────────────────────
from langchain_core.documents import Document as _Doc
from langchain_core.documents import Document

# ── 6. 최종 멀티모달 체인 (import) ─────────────────────────
from langchain_core.messages import HumanMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableLambda, RunnablePassthrough
from langchain_openai import ChatOpenAI

# [.py 변환] 노트북에서 셀 중간에 하던 import 를 파일 맨 위로 모았습니다.

# 6) 연결 키(= 청구기호) : id_key = "doc_id"
# [.py 변환] add_documents 함수가 id_key 를 사용하므로 파일 맨 위(전역)에 둡니다.
id_key = "doc_id"


# ── 3. 요약과 원본을 같은 ID로 짝짓기 ─────────────────────
def add_documents(retriever, doc_summaries, doc_contents):
    """요약본은 벡터 DB에, 원본은 docstore에 '같은 id'로 저장합니다."""
    doc_ids = [str(uuid.uuid4()) for _ in doc_contents]

    summary_docs = [
        _Doc(page_content=s, metadata={id_key: doc_ids[i]})
        for i, s in enumerate(doc_summaries)
    ]

    # TODO 1: 요약본은 어디에 넣을까요? (vectorstore / docstore)
    retriever.vectorstore.add_documents(summary_docs)

    # TODO 2: 원본은 어디에 넣을까요?
    retriever.docstore.mset(list(zip(doc_ids, doc_contents)))


# ── 5. 검색 결과를 텍스트와 이미지로 분리 ─────────────────
def split_image_text_types(docs):
    """검색 결과를 Base64 이미지와 텍스트로 나눕니다."""
    b64_images = []
    texts_out = []

    # docs : retriever가 검색한 '원본 목록'
    for doc in docs:
        doc = doc.page_content if isinstance(doc, Document) else doc
        if re.match("^[A-Za-z0-9+/]{100,}={0,2}$", doc):
            b64_images.append(doc)
        else:
            texts_out.append(doc)

    return {"images" : b64_images, "texts" : texts_out}


# ── 6. 최종 멀티모달 체인 만들기 ──────────────────────────
def img_prompt_func(data_dict):
    """검색된 원본을 멀티모달 LLM이 읽을 형태로 조립합니다."""
    messages = []

    # 1. 이미지를 먼저 넣는다.
    for image in data_dict["context"]["images"]:
        messages.append({
            "type" : "image_url",
            "image_url" : {"url" : f"data:image/jpeg;base64,{image}"},
        })

    # 2. 텍스트 context와 질문을 넣는다.
    formatted_texts = "\n".join(data_dict["context"]["texts"])
    messages.append({
                "type" : "text",
                "text" : ("자료를 근거로 질문에 한국어로 답하세요. 자료의 수치를 값과 단위까지 포함해서 일의 자리 이상부터 답변해주세요.\n"
                          "자료에 없는 내용은 모른다고 답하세요.\n\n"
                          f"질문: {data_dict['question']}\n\n"
                          f"참고 텍스트:\n {formatted_texts}"
                )
    })

    return [HumanMessage(content=messages)]


# ── 8-1. 비교용 텍스트 전용 RAG ────────────────────────────
# 검색된 Document 들의 page_content 를 줄바꿈으로 합치는 함수
def format_docs(docs):
    return "\n".join(doc.page_content for doc in docs)


# [.py 변환] 노트북은 셀을 순서대로 실행했지만, .py 는 아래 블록에서 한 번에 실행합니다.
#           다른 파일에서 import 할 때는 이 블록이 실행되지 않습니다.
if __name__ == "__main__":
    # ── 0. 결과 복원 ──────────────────────────────────────
    # 3) .env 가 있는 폴더를 프로젝트 루트로 자동 탐색
    PROJECT_ROOT =  next(
        ( p for p in [Path.cwd(), *Path.cwd().parents] if(p / ".env").exists()), Path.cwd(),
    )
    print("프로젝트 루트 : ", PROJECT_ROOT)

    # 4) PROJECT_ROOT 의 .env 파일 로드 (load_dotenv)
    # [수정] 노트북은 load_dotenv(".env") 라서 실행 위치에 .env 가 없으면 읽지 못했습니다
    load_dotenv(PROJECT_ROOT / ".env")

    # 5) OPENAI_API_KEY 가 없으면 ValueError 발생시키기
    try:
        openai_api_key = os.environ["OPENAI_API_KEY"]
        if not openai_api_key.strip():
            raise ValueError("OPENAI_API_KEY가 비어 있습니다. .env 파일을 확인해주세요!")
    except KeyError:
        raise ValueError("OPENAI_API_KEY가 존재하지 않습니다. .env 파일을 확인해 주세요.")

    # 6) processed 경로 = PROJECT_ROOT / "data" / "processed"
    processed = PROJECT_ROOT / "data" / "processed"

    # 7) 추출 결과 복원: processed/"extracted.pkl" 을 pickle 로 읽기 ("rb")
    with open(processed/"extracted.pkl", "rb") as f:
        doc1 = pickle.load(f)
        texts, tables, texts_2k_token = doc1["texts"], doc1["tables"], doc1["texts_2k_token"]

    # 8) 요약 결과 복원: processed/"summaries.pkl" 을 pickle 로 읽기
    with open(processed/"summaries.pkl", "rb") as img_f:
        doc2 = pickle.load(img_f)
        text_summaries, table_summaries, image_summaries, img_base64_list = doc2["text_summaries"], doc2["table_summaries"], doc2["image_summaries"], doc2["img_base64_list"]

    # 9) "복원 완료" 와 함께 텍스트·표·이미지의 원본 수 / 요약 수 출력
    print("\n\n")
    print("복원 완료")
    print(f"텍스트 원본 :{len(texts_2k_token):3d}건 / 텍스트 요약 : {len(text_summaries)}건")
    print(f"표 원본 :{len(tables):3d}건 / 표 요약 : {len(table_summaries)}건")
    print(f"이미지 원본 :{len(img_base64_list):3d}건 / 이미지 요약 : {len(image_summaries)}건")

    # ── 2. 검색기 만들기 ──────────────────────────────────
    # 4) 요약문이 들어갈 벡터 DB (= 색인 카드 서랍)
    vectorstore = Chroma(
            collection_name="multimodal_rag",
            embedding_function = OpenAIEmbeddings(model="text-embedding-3-small")
    )

    # 같은 커널에서 이 셀을 다시 실행해도 Chroma 의 이전 요약은 메모리에 남아 있습니다.
    # docstore(InMemoryStore)는 새로 비워지므로, 짝이 맞도록 벡터 DB 도 함께 비웁니다.
    vectorstore.reset_collection()

    # 5) 원본이 들어갈 저장소 (= 서가) : InMemoryStore()
    store = InMemoryStore()

    # 7) MultiVectorRetriever(vectorstore=..., docstore=..., id_key=...) 생성
    retriever = MultiVectorRetriever(vectorstore=vectorstore, docstore=store, id_key=id_key)

    # 8) "검색기 생성 완료" 출력
    print("검색기 생성 완료")

    # ── 3. 요약과 원본을 같은 ID로 짝짓기 ─────────────────
    # 재실행해도 중복되지 않도록 기존 색인을 먼저 비웁니다 (벡터 DB + docstore 모두)
    retriever.vectorstore.reset_collection()
    retriever.docstore.mdelete(list(retriever.docstore.yield_keys()))

    if text_summaries:
        add_documents(retriever, text_summaries, texts_2k_token)
    if table_summaries:
        add_documents(retriever, table_summaries, tables)
    if image_summaries:
        add_documents(retriever, image_summaries, img_base64_list)

    print("색인 완료")
    print("  벡터 DB에 저장된 요약 수:", len(vectorstore.get()["ids"]))

    # ── 4. 검색 테스트 ────────────────────────────────────
    # TODO 3: 그래프나 표를 봐야만 답할 수 있는 질문을 query 에 적으세요
    query = "24년말 대비 26년 1분기의 누적 평가이익을 알려줘"

    # 1) retriever.invoke(query) 결과를 found 에 저장
    # 2) 검색된 원본 수 출력
    found = retriever.invoke(query)

    print("검색된 원본 수:", len(found))
    print()

    # 3) found 를 순회하며 종류, 길이, 앞 120자(줄바꿈 제거) 출력
    for i, doc in enumerate(found):
        content = doc.page_content if hasattr(doc, "page_content") else doc
        is_image = bool(re.match("^[A-Za-z0-9+/]{100,}={0,2}$", content))
        # Base64에 쓰는 문자만 100개 이상 이어지고 AND 끝에 = 가 0~2개 붙은 문자열은, 이미지로 분류한다.
        kind = "이미지(Base64)" if is_image else "텍스트/표"
        print(f"[{i}] {kind} — 길이 {len(content):,}자")
        print("   ", content[:120].replace("\n", " "), "...")
        print()

    # ── 5. 검색 결과를 텍스트와 이미지로 분리 ─────────────
    # 3) 앞서 검색한 결과(found)로 동작 확인 → 이미지 / 텍스트 건수 출력
    split_data = split_image_text_types(found)
    print("이미지:", len(split_data["images"]), "건")
    print("표:", len(split_data["texts"]), "건")

    # ── 6. 최종 멀티모달 체인 만들기 ──────────────────────
    # 3) 모델: ChatOpenAI(temperature=0, model="gpt-4o-mini", max_tokens=1024)
    model = ChatOpenAI(temperature=0, model="gpt-4o-mini", max_tokens=1024)

    # 4) 체인 조립 (LCEL)
    multimodal_chain = (
            {
                "context" : retriever | RunnableLambda(split_image_text_types),
                "question" : RunnablePassthrough(),
            }
            | RunnableLambda(img_prompt_func)
            | model                                # TODO 5: 멀티모달 LLM
            | StrOutputParser()
    )
    # 5) "최종 체인 조립 완료" 출력
    print("최종 체인 조립 완료")

    # ── 7. 질의 실행 ──────────────────────────────────────
    # 1) 시작 시간 기록 (time.time())
    start = time.time()

    # 2) chain.invoke(query) 결과를 answer 에 저장
    answer = multimodal_chain.invoke(query)

    # 3) 소요 시간, 질문(Q), 구분선, 답변 출력
    print(f"답변생성: {time.time() - start:.1f}초 \n")
    print("질문: ", query)
    print("---"*60)
    print(answer)

    # ── 8-1. 비교용 텍스트 전용 RAG ───────────────────────
    # 이미지와 표를 뺀 것 말고는 모든 조건이 같아야 합니다.

    # 1) 텍스트만 들어간 벡터 DB 생성
    text_vectorstore = Chroma(
        collection_name="text_only_rag",
        embedding_function=OpenAIEmbeddings(model="text-embedding-3-small"),
    )
    text_vectorstore.reset_collection()          # 재실행해도 중복 저장되지 않도록 비우기
    text_vectorstore.add_texts(texts_2k_token)   # add_texts 는 '메서드 호출'이어야 저장됨
    text_retriever = text_vectorstore.as_retriever(search_kwargs={"k": 4})

    # 2) 프롬프트: 멀티모달 체인과 같은 지시문 + {question} + {context}
    #    ⚠️ ChatPromptTemplate 은 f-string 이 아니라 {변수} 자리표시자를 그대로 둬야 합니다
    text_prompt = ChatPromptTemplate.from_template(
        "자료를 근거로 질문에 한국어로 답하세요\n"
        "그래프로 나타난 수치는 이미지를 분석하여 대략적인 값과 단위를 요약에 포함하세요\n"
        "자료에 없는 내용은 모른다고 답하세요.\n\n"
        "질문: {question}\n\n"
        "참고 텍스트 :\n {context}"
    )

    # 3) 체인 조립
    text_chain = (
        {
            "context": text_retriever | RunnableLambda(format_docs),
            "question": RunnablePassthrough(),
        }
        | text_prompt
        | model
        | StrOutputParser()
    )

    # 4) 완료 출력
    print("비교용 텍스트 전용 RAG 준비 완료")

    # ── 8-2. Before / After 출력 ──────────────────────────
    # 1) 구분선과 질문 출력
    print("=" * 80)
    print("질문:", query)
    print("=" * 80)

    # 2) [BEFORE] 텍스트 전용 RAG
    before = text_chain.invoke(query)
    print("\n[BEFORE] 텍스트 전용 RAG")
    print("-" * 80)
    print(before)

    # 3) [AFTER] 멀티모달 RAG
    after = multimodal_chain.invoke(query)
    print("\n[AFTER] 멀티모달 RAG")
    print("-" * 80)
    print(after)

    # 4) 안내 출력
    print("\n" + "=" * 80)
    print("이 출력 전체를 캡처해 README.md에 넣으세요.")

    # [.py 변환] 9번 번역 체인(Ollama) 셀은 노트북에서도 주석만 있고 코드가 없어 옮기지 않았습니다.

    # ── 10. 질의응답 5건 기록 (과제 제출물) ───────────────
    questions = [
        query,
        "이 문서의 핵심 주제를 한 문장으로 요약해줘",
        "문서에 나온 표의 내용을 설명해줘",
        "가장 중요한 수치 3가지를 알려줘",
        "이 문서가 다루지 않는 내용은 무엇인지 알려줘",
    ]

    rows = []
    for q in questions:
        found_q = retriever.invoke(q)
        kinds = split_image_text_types(found_q)
        answer = multimodal_chain.invoke(q)
        rows.append({
            "질문": q,
            "근거": f"이미지 {len(kinds['images'])}건 / 텍스트 {len(kinds['texts'])}건",
            "답변": answer,
        })
        print("=" * 70)
        print("Q:", q)
        print("근거:", rows[-1]["근거"])
        print("A:", answer[:400])
        print()

    # README에 붙일 마크다운 표로 출력
    print("\n" + "=" * 70)
    print("아래를 복사해 README.md에 붙여넣으세요")
    print("=" * 70 + "\n")
    print("| 질문 | 근거가 된 원본 | 답변 |")
    print("| --- | --- | --- |")
    for r in rows:
        short = r["답변"].replace("\n", " ")[:120]
        print(f"| {r['질문']} | {r['근거']} | {short}... |")
