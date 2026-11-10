"""실습 ② 요약 생성 — 텍스트·표·이미지를 검색 가능한 요약문으로 바꿉니다.

기준 노트북: practice/02_요약생성_학습_(주석만).ipynb

실행 (프로젝트 루트에서, src/extract.py 를 먼저 실행해야 합니다):
    python src/summarize.py

⚠️ OpenAI API 를 호출하므로 실행할 때마다 비용이 발생하고,
   data/processed/summaries.pkl 을 덮어씁니다.

표시 규칙:
    # [.py 변환] : 노트북을 .py 로 옮기면서 바꾼 부분
    # [수정]     : 노트북 코드의 버그를 고친 부분
    (표시가 없는 코드는 노트북 코드 그대로입니다)
"""

# ── 0. API 키 불러오기 ─────────────────────────────────────
# ⌨️ 0-1. os, time, pickle, Path(pathlib), load_dotenv(dotenv) 를 임포트합니다.
import os
import time
import pickle
from pathlib import Path
from dotenv import load_dotenv

# ⌨️ 2-1. StrOutputParser, ChatPromptTemplate, ChatOpenAI 를 임포트합니다.
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

# ⌨️ 6-1. base64 모듈과 HumanMessage(langchain_core.messages)를 임포트합니다.
import base64
from langchain_core.messages import HumanMessage

# [.py 변환] 노트북에서 셀 중간에 하던 import 를 파일 맨 위로 모았습니다.

# ⌨️ 0-2. PROJECT_ROOT 를 정합니다.
#        현재 폴더 이름이 "invest" 또는 "practice" 이면 상위 폴더, 아니면 현재 폴더
# [.py 변환] .py 는 src 폴더에서 실행될 수도 있으므로 "src" 를 목록에 추가했습니다.
PROJECT_ROOT = Path.cwd().parent if Path.cwd().name in ["invest", "practice", "src"] else Path.cwd()


# ── 2. 요약 체인 만들기 ────────────────────────────────────
# ⌨️ 2-2. generate_text_summaries(texts, tables, summarize_texts=False) 함수를 정의합니다.
def generate_text_summaries(texts, tables, summarize_texts=False):

    # ⌨️ (1) 한국어 요약 프롬프트 문자열을 작성합니다. 반드시 {element} 자리표시자를 포함합니다.
    prompt_template_kor = """
                    표와 텍스트를 검색용으로 요약하는 도우미 / 임베딩되어 원본 검색에 쓰임 /
                    검색에 최적화된 간결한 요약 작성 / 요약할 표 또는 텍스트: {element}"""

    prompt_template_eng = """
                    You are an assistant tasked with summarizing tables and text for retrieval.
                    These summaries will be embedded and used to retrieve the raw text or table elements.
                    Give a concise summary of the table or text that is well optimized for retrieval.
                    Table or text: {element}"""

    # ⌨️ (2) ChatPromptTemplate.from_template 으로 prompt 를 만듭니다.
    prompt = ChatPromptTemplate.from_template(prompt_template_kor)

    # ⌨️ (3) ChatOpenAI 로 model 을 만듭니다. (temperature=0, model="gpt-4o-mini")
    model = ChatOpenAI(
        temperature=0,
        model = "gpt-4o-mini"
    )

    # ⌨️ (4) 컨베이어 벨트 4칸을 | 로 연결해 summarize_chain 을 만듭니다.
    summarize_chain = prompt | model | StrOutputParser()

    # ⌨️ (5) text_summaries, table_summaries 빈 리스트를 만듭니다.
    text_summaries = []
    table_summaries = []

    # ⌨️ (6) texts 가 있고 summarize_texts 가 True 면 → batch 로 요약 (max_concurrency 5)
    #        texts 는 있지만 summarize_texts 가 False 면 → 원문을 그대로 사용
    if texts and summarize_texts:
        text_summaries = summarize_chain.batch(texts, {"max_concurrency" : 5})
    elif texts and not summarize_texts:
        text_summaries = texts

    # ⌨️ (7) tables 가 있으면 → batch 로 요약 (max_concurrency 5)
    if tables:
        table_summaries = summarize_chain.batch(tables, {"max_concurrency" : 5})
    else:
        table_summaries = tables

    # ⌨️ (8) text_summaries, table_summaries 를 반환합니다.
    return text_summaries, table_summaries


# ── 6. 이미지를 Base64로 포장하기 ──────────────────────────
# ⌨️ 6-2. encode_image(image_path) 함수를 정의합니다.
#        파일을 "rb" 로 열어 → b64encode → "utf-8" 로 decode 한 문자열을 반환
def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")


# ── 7. 이미지 요약 함수 만들기 ─────────────────────────────
# ⌨️ 7-1. image_summarize(img_base64, prompt) 함수를 정의합니다.
def image_summarize(img_base64, prompt):
    # ⌨️ (1) ChatOpenAI 로 chat 을 만듭니다. (model="gpt-4o-mini", max_tokens=1024)
    chat = ChatOpenAI(
        model="gpt-4o-mini",
        max_tokens=1024
    )

    # ⌨️ (2) 위 구조대로 HumanMessage 를 만들어 chat.invoke([...]) 로 호출합니다.
    #        이미지 url 은 "data:image/jpeg;base64," 접두사 뒤에 img_base64 를 붙입니다.
    msg = chat.invoke([
        HumanMessage(content=[
            {"type": "text",      "text": prompt},
            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{img_base64}"},
                },
            ])
        ])

    # ⌨️ (3) 응답의 .content 를 반환합니다.
    return msg.content


# ── 8. 폴더 내 전체 이미지 요약 ────────────────────────────
def generate_img_summaries(path):
    """
    이미지의 요약과 Base64 인코딩 문자열을 생성

    path: Unstructured에 의해 추출된 이미지 파일들의 경로
    """
    img_base64_list = []   # Base64로 인코딩된 이미지
    image_summaries = []   # 이미지 요약문

    # 지시문은 영어, 결과는 한국어로 (토큰 효율 + 품질)
    prompt_kor = """You are an assistant tasked with summarizing images for retrieval.
These summaries will be embedded and used to retrieve the raw image.
Provide a concise summary of the image that is well optimized for retrieval.
The summary should be written in Korean (Hangul)."""

    # 영어 요약이 필요하면 이쪽을 사용
    # prompt_eng = """You are an assistant tasked with summarizing images for retrieval.
    # These summaries will be embedded and used to retrieve the raw image.
    # Provide a concise summary of the image that is well optimized for retrieval."""

    for img_file in sorted(os.listdir(path)):
        # .jpg, .png, .jpeg 확장자만 처리합니다
        if img_file.endswith((".jpg", ".png", ".jpeg")):
            img_path = os.path.join(path, img_file)
            base64_image = encode_image(img_path)
            img_base64_list.append(base64_image)
            image_summaries.append(image_summarize(base64_image, prompt_kor))

    return img_base64_list, image_summaries


# [.py 변환] 노트북은 셀을 순서대로 실행했지만, .py 는 아래 블록에서 한 번에 실행합니다.
#           다른 파일에서 import 할 때는 이 블록이 실행되지 않습니다.
if __name__ == "__main__":
    # ⌨️ 0-3. PROJECT_ROOT / ".env" 파일을 load_dotenv 로 읽습니다.
    # [수정] 노트북은 "..env" 로 적혀 있어 .env 를 읽지 못했습니다 → ".env"
    load_dotenv(PROJECT_ROOT / ".env")

    # ⌨️ 0-4. 환경변수 "OPENAI_API_KEY" 를 openai_api_key 에 담습니다.
    openai_api_key = os.getenv("OPENAI_API_KEY")

    # ⌨️ 0-5. 키가 없으면 ValueError 를 발생시킵니다.
    if not openai_api_key:
        raise ValueError("환경변수 OPENAI_API_KEY가 없습니다. .env 파일을 확인하세요.")

    # ⌨️ 0-6. 키를 "앞 7자 + ... + 뒤 4자" 형태로만 출력합니다.
    print(f"API KEY : {openai_api_key[:7]} + ... + {openai_api_key[-4:]}")

    # ⌨️ 0-7. 키가 실제로 유효한지 확인합니다.
    from openai import AuthenticationError, OpenAI

    try:
        OpenAI(api_key=openai_api_key).models.list()
    except AuthenticationError:
        raise ValueError(f"환경변수 OPENAI_API_KEY가 유효하지 않습니다. .env 파일을 확인하세요. (API KEY : {openai_api_key[:7]} + ... + {openai_api_key[-4:]})")

    # ── 1. 지난 시간 결과 복원 ────────────────────────────
    # ⌨️ 1-1. PROJECT_ROOT / "data" / "processed" / "extracted.pkl" 경로를 pkl_path 에 담습니다.
    pkl_path = PROJECT_ROOT / "data" / "processed" / "extracted.pkl"

    # ⌨️ 1-2. 파일을 바이너리 읽기 모드("rb")로 열어 pickle.load 로 data 에 불러옵니다.
    with open(pkl_path, "rb") as f:
        data = pickle.load(f)

    # ⌨️ 1-3. data 딕셔너리에서 texts, tables, texts_2k_token 을 꺼냅니다.
    texts = data["texts"]
    tables = data["tables"]
    texts_2k_token = data["texts_2k_token"]

    # ⌨️ 1-4. 각 리스트의 개수를 출력합니다.
    print("texts 개수:", len(texts))
    print("tables 개수:", len(tables))
    print("texts_2k_token 개수:", len(texts_2k_token))

    # ⌨️ 1-5. 지난 시간에 이미지를 저장한 폴더를 figures_dir 로 잡고 존재 여부를 출력합니다.
    figures_dir = PROJECT_ROOT / "data" / "figures"
    print("존재:", figures_dir.exists())

    # ── 3. 텍스트·표 요약 실행 ────────────────────────────
    # ⌨️ 3-1. 시작 시각을 기록합니다.
    start = time.time()

    # ⌨️ 3-2. texts_2k_token 과 tables 를 넣어 generate_text_summaries 를 호출합니다.
    text_summaries, table_summaries = generate_text_summaries(texts_2k_token, tables, summarize_texts=True)

    # ⌨️ 3-3. 소요 시간, 텍스트 요약 수, 표 요약 수를 출력합니다.
    print(f"요약 완료 => {time.time() - start :.1f}초 걸림")
    print(f"텍스트 요약 수 => {len(text_summaries)}개")
    print(f"표 요약 수 => {len(table_summaries)}개")

    # ── 4. 요약 결과 확인 ─────────────────────────────────
    # ⌨️ 4-1. 구분선과 함께 texts_2k_token[0] 의 앞 300자를 출력합니다.
    print("-"*60)
    print("texts_2k_token 결과:", texts_2k_token[0][:300])
    print("\n\n")

    # ⌨️ 4-2. 구분선과 함께 text_summaries[0] 전체를 출력합니다.
    print("-"*60)
    print("text_summaries 결과:", text_summaries[0])
    print("\n\n")

    # ⌨️ 4-3. 원본 글자 수, 요약 글자 수, 압축률(%)을 출력합니다.
    original = len(texts_2k_token[0])
    short = len(text_summaries[0])

    print("원본 글자 수:", original)
    print("요약 글자 수:", short)
    print("압축률(%) :", f"{original}자 -> {short}자 (%) {short/original*100:.1f}%")

    # ── 5. table_summaries 가 비어 있다면? ────────────────
    # ⌨️ 5-1. table_summaries 가 있으면 → 개수와 첫 번째 표 요약을 출력합니다.
    if table_summaries:
        # 원본 표와 함께 비교하고 싶다.
        for i, (table, summary) in enumerate(zip(tables[:5], table_summaries[:5])):
            print("="*60)
            print(f"[표 {i+1} 원본 (앞 300자)]")
            print("="*60)
            print(table[:300])
            print()
            print("="*60)
            print(f"[표 {i+1} 요약본]")
            print("="*60)
            print(summary)
            print()

    # ⌨️ 5-2. 비어 있으면 → 위 두 가지 원인을 점검하라는 안내 문구를 출력합니다.
    else:
        print("표 요약이 비어 있습니다."
            "원인은 1. startegy = 'hi_res'"
            "2. 진짜 표인지, 그림으로 그린 표인지 확인"
            )

    # ── 6. 이미지를 Base64로 포장하기 ─────────────────────
    # ⌨️ 6-3. figures_dir 에서 .jpg / .jpeg / .png 파일만 골라 정렬해 image_files 에 담습니다.
    image_files = sorted(
        f for f in figures_dir.iterdir()
        if f.suffix.lower() in (".jpg", ".jpeg", ".png")
        )

    # ⌨️ 6-4. 첫 번째 이미지를 인코딩해 sample_b64 에 담고
    #        파일 이름, 문자열 길이, 앞 100자를 출력합니다.
    sample_b64 = encode_image(image_files[0])
    print("첫번째 이미지 파일 이름:", image_files[0].name)
    print("문자열 길이:", len(sample_b64))
    print("앞 100글자:", sample_b64[:100])

    # ── 7. 이미지 요약 테스트 ─────────────────────────────
    # ⌨️ 7-2. 테스트용 영어 프롬프트 test_prompt 를 작성합니다.
    test_prompt = """You are an assistant tasked with summarizing images for retrieval.
These summaries will be embedded and used to retrieve the raw image.
Provide a concise summary of the image that is well optimized for retrieval.
The summary should be written in Korean (Hangul)."""

    # ⌨️ 7-3. sample_b64 와 test_prompt 로 image_summarize 를 호출해 결과를 출력합니다.
    print(image_summarize(sample_b64, test_prompt))

    # ── 8. 폴더 내 전체 이미지 요약 ───────────────────────
    start = time.time()
    img_base64_list, image_summaries = generate_img_summaries(str(figures_dir))

    print(f"이미지 요약 완료 — {time.time() - start:.1f}초 소요")
    print("요약된 이미지 수:", len(image_summaries))

    # [.py 변환] 9번 요약 품질 검증 셀(IPython.display 로 이미지 표시)은 .py 에서 쓸 수 없어 뺐습니다.

    # ── 10. 토큰 사용량 측정 ──────────────────────────────
    import tiktoken

    enc = tiktoken.encoding_for_model("gpt-4o-mini")

    input_tokens = sum(len(enc.encode(t)) for t in texts_2k_token)
    output_tokens = sum(len(enc.encode(s)) for s in text_summaries)

    print("텍스트 요약 토큰 사용량 (추정)")
    print(f"입력   : {input_tokens:,} 토큰")
    print(f"출력   : {output_tokens:,} 토큰")
    print(f"합계   : {input_tokens + output_tokens:,} 토큰")
    print()
    print("요약한 이미지 수:", len(image_summaries), "장")
    print()
    print("※ 실제 비용 = 토큰 수 x 단가. 단가는 OpenAI 공식 가격 페이지에서 확인하세요.")
    print("※ 이미지 토큰은 해상도에 따라 달라지므로 위 수치에 포함되지 않았습니다.")

    # ── 11. 저장 ──────────────────────────────────────────
    # ⌨️ 11-1. PROJECT_ROOT / "data" / "processed" 를 output_dir 로 잡고 폴더를 만듭니다.
    output_dir = PROJECT_ROOT / "data" / "processed"
    output_dir.mkdir(parents=True, exist_ok=True)

    # ⌨️ 11-2. output_dir / "summaries.pkl" 에 4개 키를 가진 딕셔너리를 pickle.dump 합니다.
    # "wb" : 바이너리 파일로 저장
    with open(output_dir / "summaries.pkl", "wb") as f:
        pickle.dump(
            {
                "text_summaries": text_summaries,
                "table_summaries": table_summaries,
                "image_summaries": image_summaries,
                "img_base64_list": img_base64_list
            },
            f,
        )

    # ⌨️ 11-3. 저장 경로를 출력합니다.
    print("저장 완료:", output_dir / "summaries.pkl")
