# Multimodal RAG

PDF의 텍스트, 표, 이미지를 처리하는 수업 실습 프로젝트입니다.

## 현재 상태

- 기존 Python 3.11.9 가상환경과 의존성 파일을 보존했습니다.
- 폴더 및 실습 파일의 기본 틀을 준비했습니다. RAG 기능 구현은 아직 포함하지 않습니다.
- 실제 원본 PDF를 `data/invest.pdf`에 넣어 주세요.
- `src/`의 세 파일은 교안 코드를 작성할 자리입니다.

## 실행 준비 (Windows CMD)

프로젝트 폴더에서 다음 명령으로 기존 환경을 활성화합니다.

```cmd
.venv\Scripts\activate.bat
python --version
```

VS Code에서 Python 인터프리터와 노트북 커널을 이 프로젝트의
`.venv\Scripts\python.exe`로 선택합니다.
API 키는 프로젝트 최상위 `.env`에 직접 입력합니다.
`.env.example`에는 실제 키를 넣지 않습니다.

## 파일 구성

- `requirements.txt`: 기존 직접 의존성 목록
- `requirements.lock.txt`: 기존 버전 고정 목록 (이번 작업에서 변경하지 않음)
- `data/invest.pdf`: 사용자가 준비할 원본 PDF
- `data/figures/`: 추출 이미지 저장 폴더
- `notebooks/Multi_modal_RAG.ipynb`: 실습 노트북
- `src/extract.py`: PDF 파싱
- `src/summarize.py`: 요약 체인
- `src/retriever.py`: 다중 벡터 검색기

이전 Python 3.14.6 환경의 보관본은 프로젝트 밖 `C:\workspaces\multimodal-rag-backup-20261104-142950\.venv_backup`에 있습니다.
현재 `.venv`는 독립된 Python 3.11.9 설치를 사용합니다.
새 환경에서 실습이 정상 작동하는지 확인한 뒤 필요 없다면 보관본을 삭제할 수 있습니다.





git clone https://github.com/8113jsl-ui/UnstructuredParser.git
cd multimodal-rag

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