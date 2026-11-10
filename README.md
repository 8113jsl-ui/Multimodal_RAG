# 📄 Multimodal RAG

> PDF 속 **텍스트 · 표 · 이미지**를 함께 검색해 답하는 멀티모달 RAG 수업 실습 프로젝트

![Python](https://img.shields.io/badge/Python-3.11.9-3776AB?logo=python&logoColor=white)
![LangChain](https://img.shields.io/badge/LangChain-RAG-1C3C3C?logo=langchain&logoColor=white)
![Chroma](https://img.shields.io/badge/VectorDB-Chroma-FF6F00)
![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-F37626?logo=jupyter&logoColor=white)

---

## 🧭 전체 흐름

```
data/invest.pdf
   │  ① 데이터 추출 (텍스트 · 표 · 이미지)
   ▼
요약 생성 (텍스트 요약 / 이미지 설명)
   │  ② 요약을 임베딩해 Chroma에 저장, 원본은 docstore에 보관
   ▼
다중 벡터 검색기 (요약으로 검색 → 원본 반환)
   │  ③ 원본 텍스트 + 이미지를 멀티모달 LLM에 전달
   ▼
최종 답변
```

| 단계 | 실습 노트북 | 내용 |
| --- | --- | --- |
| 01 | `practice/01_데이터추출_*.ipynb` | PDF에서 텍스트·표·이미지 분리 |
| 02 | `practice/02_요약생성_*.ipynb` | 텍스트 요약 · 이미지 설명 생성 |
| 03 | `practice/03_임베딩_최종체인_연습.ipynb` | 임베딩, 다중 벡터 검색기, 멀티모달 최종 체인 |

---


## Before & After : 일반 RAG vs 멀티모달 RAG
<img width="2346" height="569" alt="Image" src="https://github.com/user-attachments/assets/0f119b1b-5cd8-439f-bddc-df86d7baacbf" />



## 💬 질의응답 결과 (5건)

멀티모달 체인에 질문 5개를 넣고, 검색된 원본 종류와 답변을 기록했습니다.

| 질문 | 근거가 된 원본 | 답변 |
| --- | --- | --- |
| 24년말 대비 26년 1분기의 누적 평가이익을 알려줘 | 이미지 3건 / 텍스트 1건 | 자료에 따르면, 24년말 대비 26년 1분기의 누적 평가이익은 571조 원입니다.... |
| 이 문서의 핵심 주제를 한 문장으로 요약해줘 | 이미지 0건 / 텍스트 4건 | 이 문서의 핵심 주제는 최근 한국 경제의 민간소비 회복세가 반도체 경기 호조에도 불구하고 상대적으로 완만하며, 소비 증가가 일부 계층에 집중되어 있다는 점을 평가하고 향후 소비 회복세의 확산 가능성을 점검하는 것이다... |
| 문서에 나온 표의 내용을 설명해줘 | 이미지 4건 / 텍스트 0건 | 자료에 나온 표는 두 개의 그래프와 두 개의 로고로 구성되어 있습니다.  첫 번째 그래프는 특정 기간 동안의 수치를 나타내며, y축은 수치(단위: 만호)를 나타내고, x축은 날짜를 나타냅니다. 그래프에서 가장 높은 ... |
| 가장 중요한 수치 3가지를 알려줘 | 이미지 4건 / 텍스트 0건 | 1. 첫 번째 그래프에서 3MA의 값은 60입니다. 2. 두 번째 그래프에서 전체 평균 수준은 50%입니다. 3. 세 번째 그래프에서 26.2/4분기의 제조업 기대비는 -10입니다.... |
| 이 문서가 다루지 않는 내용은 무엇인지 알려줘 | 이미지 2건 / 텍스트 2건 | 이 문서가 다루지 않는 내용은 "배당소득이 없는 가구의 소비지출 변화"입니다. 문서에서는 배당소득이 있는 가구에 대한 소비지출 변화만을 다루고 있습니다.... |

> 💡 **관찰**: 수치·그래프를 묻는 질문은 이미지 원본이, 주제 요약 질문은 텍스트 원본이 주로 검색되었습니다.

---

## 📁 폴더 구조

```
multimodal-rag/
├── data/
│   ├── invest.pdf          # 실습 원본 PDF
│   └── figures/            # 추출 이미지 (git 제외)
├── notebooks/
│   └── Multi_modal_RAG.ipynb
├── practice/               # 단계별 실습 노트북 (01 ~ 03)
├── src/
│   ├── extract.py          # PDF 파싱
│   ├── summarize.py        # 요약 체인
│   └── retriever.py        # 다중 벡터 검색기
├── settings.py
├── requirements.txt        # 직접 의존성
├── requirements.lock.txt   # 버전 고정 목록
└── .env.example            # API 키 템플릿 (실제 키 X)
```

---

## ⚙️ 실행 방법

```bash
git clone https://github.com/8113jsl-ui/Multimodal_RAG.git
cd Multimodal_RAG

# 가상환경 생성 및 활성화
python -m venv .venv
source .venv/bin/activate          # 윈도우: .venv\Scripts\activate

# 의존성 설치
pip install --upgrade pip
pip install -r requirements.txt     # 또는 requirements.lock.txt

# 커널 등록
python -m ipykernel install --user --name multimodal-rag

# .env 작성
cp .env.example .env                # 윈도우: copy .env.example .env
# .env 파일을 열어 실제 API 키를 입력
```

VS Code에서는 Python 인터프리터와 노트북 커널을 `.venv\Scripts\python.exe`로 선택합니다.

> ⚠️ `.env`에는 실제 API 키가 들어가므로 **절대 커밋하지 않습니다** (`.gitignore`에 등록됨).
