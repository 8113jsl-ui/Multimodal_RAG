"""실습 ① 데이터 추출 및 분할 — PDF를 텍스트 / 표 / 이미지로 분리합니다.

기준 노트북: practice/01_데이터추출__연습본.ipynb

실행 (프로젝트 루트에서):
    python src/extract.py

표시 규칙:
    # [.py 변환] : 노트북을 .py 로 옮기면서 바꾼 부분
    (표시가 없는 코드는 노트북 코드 그대로입니다)
"""

# ── 1. 라이브러리 임포트 ───────────────────────────────────
import os
import time
import pickle
from pathlib import Path
from collections import Counter

from langchain_text_splitters import CharacterTextSplitter
from unstructured.partition.pdf import partition_pdf


# ── 2. 경로 설정 ──────────────────────────────────────────
# [.py 변환] 노트북은 practice 폴더에서 실행했으므로 상위 폴더를 루트로 잡았습니다.
#           .py 는 src 폴더에서 실행될 수도 있으므로 "src" 를 목록에 추가했습니다.
PROJECT_ROOT = Path.cwd().parent if Path.cwd().name in ("invest", "practice", "src") else Path.cwd()

# TODO 1: 본인의 PDF 파일 이름으로 바꾸세요 (예: "invest.pdf")
fname = "invest.pdf"
# [.py 변환] 노트북의 Path("../data") 는 practice 폴더 기준 상대경로라서
#           실행 위치에 상관없도록 PROJECT_ROOT 기준으로 바꿨습니다.
pdf_dir = PROJECT_ROOT / "data"
pdf_path = pdf_dir / fname

fpath = str(pdf_dir) + os.sep


# ── 3. 추출 함수 만들기 ────────────────────────────────────
def extract_pdf_elements(path, fname):
    """
    PDF 파일에서 이미지, 테이블, 텍스트 블록을 추출합니다.

    path : 파일 경로 (추출된 이미지 파일이 저장될 위치)
    fname: 파일 이름
    """
    return partition_pdf(
        filename=path + fname,
        strategy="hi_res",                  # TODO 2: 정밀 추출 전략
        extract_images_in_pdf=True,       # TODO 3: 이미지를 추출할까요?
        infer_table_structure=True,       # TODO 4: 표 구조를 추론할까요?
        chunking_strategy="by_title",     # 타이틀 기준으로 텍스트 분할
        max_characters=4000,              # 최대 4000자로 블록 제한
        new_after_n_chars=3800,           # 3800자 이후 새 블록 생성
        combine_text_under_n_chars=2000,  # 2000자 이하는 결합
        extract_image_block_output_dir=path+"/figures",       # 이미지 저장 경로
    )


# ── 4. 요소 분류 함수 만들기 ───────────────────────────────
def categorize_elements(raw_pdf_elements):
    """
    PDF에서 추출한 요소들을 테이블과 텍스트로 분류합니다.

    raw_pdf_elements: unstructured.documents.elements 리스트
    """
    tables = []
    texts = []

    for element in raw_pdf_elements:
        if "unstructured.documents.elements.Table" in str(type(element)):
            tables.append(str(element))     # TODO 5: 표는 어느 리스트로?
        elif "unstructured.documents.elements.CompositeElement" in str(type(element)):
            texts.append(str(element))     # TODO 6: 텍스트는 어느 리스트로?

    return texts, tables


# [.py 변환] 노트북은 셀을 순서대로 실행했지만, .py 는 아래 블록에서 한 번에 실행합니다.
#           다른 파일에서 import 할 때는 이 블록이 실행되지 않습니다.
if __name__ == "__main__":
    print("현재 위치 : " , Path.cwd())
    print("프로젝트 루트:", PROJECT_ROOT)
    print("PDF 폴더     :", pdf_dir ,pdf_path.resolve())
    print("PDF 존재 여부:", pdf_path.exists())

    # ── 5. 추출 실행 ──────────────────────────────────────
    start = time.time()

    raw_pdf_elements = extract_pdf_elements(fpath, fname)

    print(f"추출 완료 — {time.time() - start:.1f}초 소요")
    print("추출된 전체 요소 수:", len(raw_pdf_elements))

    texts, tables = categorize_elements(raw_pdf_elements)

    print("텍스트 블록 수:", len(texts))
    print("표 블록 수    :", len(tables))

    # ── 6. 어떤 요소들이 나왔는지 집계 ────────────────────
    type_counts = Counter(str(type(el)).split(".")[-1].strip("'>") for el in raw_pdf_elements)

    for element_type, count in type_counts.most_common():
        print(f"{element_type:22s}: {count}개")

    # ── 7. 추출된 이미지 확인 ─────────────────────────────
    figures_dir = Path(fpath) / "figures"

    if figures_dir.exists():
        image_files = sorted(
            f for f in figures_dir.iterdir()
            if f.suffix.lower() in (".jpg", ".jpeg", ".png")
        )
        print(f"추출된 이미지: {len(image_files)}개\n")
        for f in image_files[:10]:
            print(f"  {f.name}  ({f.stat().st_size // 1024} KB)")
    else:
        print("✗ figures 폴더가 없습니다.")
        print("  → TODO 2, TODO 3 을 다시 확인하세요.")

    # [.py 변환] 노트북의 이미지 미리보기 셀(IPython.display)은 .py 에서 쓸 수 없어 뺐습니다.

    # ── 8. 텍스트 분할 ────────────────────────────────────
    text_splitter = CharacterTextSplitter.from_tiktoken_encoder(
        chunk_size=2000,       # 각 텍스트 블록의 최대 크기
        chunk_overlap=200,    # TODO 8: 청크 간 겹칠 글자 수
    )

    joined_texts = " ".join(texts)
    texts_2k_token = text_splitter.split_text(joined_texts)

    print("원본 블록 수  :", len(texts))
    print("분할 후 블록 수:", len(texts_2k_token))
    print("전체 글자 수  :", len(joined_texts))

    # ── 9. 분할 결과 비교 ─────────────────────────────────
    print("=" * 60)
    print("[원본 texts[0]] 앞 300자")
    print("=" * 60)
    print(texts[0][:300])
    print()

    print("=" * 60)
    print("[분할 texts_2k_token[0]] 앞 300자")
    print("=" * 60)
    print(texts_2k_token[0][:300])

    # ── 10. 다음 교시를 위한 저장 ─────────────────────────
    output_dir = PROJECT_ROOT / "data" / "processed"
    output_dir.mkdir(parents=True, exist_ok=True)

    with open(output_dir / "extracted.pkl", "wb") as f:
        pickle.dump(
            {"texts": texts, "tables": tables, "texts_2k_token": texts_2k_token},
            f,
        )

    print("저장 완료:", output_dir / "extracted.pkl")
