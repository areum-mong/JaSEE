# 🪑 자세히봐 (JASEE)

<p align="center">
  <img src="logo.png" alt="자세히봐 Logo" width="400">
</p>

> **RULA + VDT 고시 제2020-17호 기반 AI 실시간 자세 & 작업환경 측정 및 RAG 피드백 서비스**

사무직 근로자의 고질적인 문제인 **근골격계 질환(VDT 증후군) 예방**을 목적으로 개발된 스마트 헬스케어 AI 솔루션입니다. 웹캠을 통해 사용자의 자세와 주변 작업환경(모니터, 책상, 의자)을 실시간으로 분석하고, 수집된 9개 인체공학적 지표를 기반으로 AI 피드백 및 맞춤형 스트레칭 가이드를 제공합니다.

---

## 🖥️ 시연 영상 및 프레젠테이션
* **발표 자료**: [`document/JASSE(자세히봐)_발표자료.pptx`](document/JASSE(자세히봐)_발표자료.pptx)
* **데모 영상**: [`document/스트림릿시연영상.mp4`](document/스트림릿시연영상.mp4)

---

## ✨ 핵심 기능

### 1. 실시간 자세 측정 & 분석 (Step 1)
* **YOLOv8-pose 모델**을 활용하여 사용자의 측면 상반신 관절(17개 키포인트)을 실시간 검출합니다.
* **CVA(목굴곡각)** 및 **TIA(몸통굴곡각)** 등을 계산하고, 학습된 **Attention MLP Classifier**가 실시간으로 자세의 Good/Bad 상태를 이중 분류합니다.
* 오버레이 화면을 통해 나쁜 자세(FHP, 전굴) 감지 시 파란색 교정 방향 화살표와 음성 안내(TTS)로 즉각 피드백을 전달합니다.

### 2. 작업환경 자동 감지 & 인체공학 분석 (Step 2)
* 자세 상태가 일정 시간 동안 'GOOD'으로 유지되면, 자동으로 **작업환경 측정 모드**로 전환됩니다.
* 의자 등받이(`chair_back`), 의자 시트(`chair_seat`), 책상 상면(`desk_surface`), 모니터(`monitor`) 객체를 커스텀 YOLOv8 모델로 탐지합니다.
* 사용자의 관절 랜드마크와 탐지된 객체의 바운딩 박스를 융합 연산하여 **시선각**, **작업대 높이 편차**, **등받이 거리 비율**을 산출하고, VDT 고시 기준 적합 여부를 판정합니다.

### 3. RAG 기반 인체공학 전문 챗봇 (chatbot.py)
* 사용자의 측정 데이터와 질문을 바탕으로 맞춤형 인체공학 전문 상담을 제공합니다.
* 고용노동부 VDT 고시 제2020-17호, KOSHA GUIDE(E-G-3-2025, E-G-6-2025), 자세분석 기준서 및 의학 Q&A 문서를 기반으로 한 **RAG(검색 증강 생성) 시스템**을 탑재하였습니다.
* 임베딩 모델(`jhgan/ko-sroberta-multitask`)과 벡터 데이터베이스(`ChromaDB`)를 활용하여 신뢰할 수 있는 출처가 명시된 전문 답변을 보장합니다.
* 질병관리청 국가건강정보포털 오픈 API와 실시간 연동되어 데이터베이스 외부의 질환/통증 정보에도 유연하게 대처합니다.

### 4. 사용자 대시보드 & 게이미피케이션
* **바른자세 점수 추이**: 날짜별 측정 기록 및 종합 점수 변화를 시각적인 차트로 제공합니다.
* **부위별 통증 완화 운동 추천**: 통증 부위(목/허리/손목 등)에 맞는 맞춤형 스트레칭 영상 및 이미지를 매핑하여 제공합니다.
* **인체공학 제품 추천**: 교정이 필요한 부위에 적합한 인체공학 의자, 모니터 받침대, 손목 보호대 등을 맞춤 제안합니다.
* **바른자세 챌린지**: 팀원 또는 그룹 간 자세 유지 포인트를 통해 건강 관리에 재미를 더합니다.

---

## 📐 측정 및 판정 기준 (RULA + VDT 고시)

인체공학 자세 평가 기법인 **RULA(Rapid Upper Limb Assessment)**와 **고용노동부 VDT 작업관리지침**에 기반한 과학적인 판정 알고리즘을 사용합니다.

| 지표 | 정상 범위 (GOOD) | 산출 기준 및 기술적 배경 |
| :--- | :--- | :--- |
| **CVA (목굴곡각)** | 0° ~ 20° | 수직축 대비 귀(Ear)와 어깨(Shoulder) 라인이 이루는 각도 (RULA Neck Score 연계) |
| **TIA (몸통굴곡각)** | 0° ~ 20° | 수직축 대비 어깨 중점과 골반 중점이 이루는 각도 (RULA Trunk Score 연계) |
| **무릎 각도** | 85° ~ 100° | 골반(Hip) - 무릎(Knee) - 발목(Ankle) 사이의 내각 (VDT 고시 기준) |
| **팔꿈치 각도** | 90° ~ 120° | 어깨(Shoulder) - 팔꿈치(Elbow) - 손목(Wrist) 사이의 내각 (VDT 고시 기준) |
| **손목 각도** | ±15° 이내 | 중립 자세 기준 손목 꺾임 편차 (VDT 고시 기준) |
| **모니터 시선각** | 하방 10° ~ 15° | 눈(Eye) 중점과 모니터 중심점 사이의 하방 시선각 (VDT 고시 제6조) |
| **작업대 높이** | 팔꿈치 기준 ±10% 이내 | 팔꿈치 높이 대비 책상 상면의 높이 비율 편차 (VDT 고시 기준) |
| **의자 등받이** | 골반 너비의 20% 이내 | 골반 위치와 의자 등받이 끝부분 사이의 수평 거리 비율 (VDT 고시 기준) |

---

## 📂 프로젝트 구조

```
📁 JASEE/
├── 📁 Yolo_env/                     # 작업환경 인식 모델 학습 및 처리
│   ├── 📄 step1_split.py            # 데이터셋 분할 (train/val/test 8:1:1)
│   ├── 📄 step2_augmentation.py     # Albumentations 이미지 증강
│   └── 📄 step3_train.py            # YOLOv8n 학습 스크립트
│
├── 📁 Yolo_pose/                    # 자세 분류 머신러닝/딥러닝 모델 개발
│   ├── 📁 01_data_preprocessing/    # 프레임 별 YOLO 포즈 키포인트 추출
│   ├── 📁 02_model_comparison/      # ML(RF, XGB) vs DL(MLP, ResNet) 성능 비교
│   ├── 📁 03_model(attetion MLP)_improvement/  # 하이퍼파라미터 튜닝 및 성능 고도화
│   ├── 📁 04_final_model/           # 최적화된 Attention MLP 모델 추출 및 검증
│   │   └── 📁 output/
│   │       └── 📄 final_attention_mlp.pt  # ⭐ 자세 분류 최종 가중치 (Attention MLP)
│   └── 📄 README.md                 # 모델 개발 아카이빙 문서
│
├── 📁 자세히봐_RAG/                 # RAG 파이프라인 구축을 위한 원천 데이터
│   ├── 📁 Q&A/                      # 부위별/점수별 Q&A 매핑 데이터
│   ├── 📁 기능1~3_자료/             # VDT 고시 PDF 및 자세분석기준 DOCX 지침 문서
│   ├── 📁 기능4_부위별 통증 완화 방법 답변/  # 근골격계 통증 DB 및 질병청 API 테스트
│   └── 📁 기능5_개인화 맞춤 운동 추천(자세맵핑)/  # UCS/LCS 체형 별 맞춤 운동 추천 DB
│
├── 📁 processed_data/               # 전처리된 JSON 청크 데이터 저장 폴더
├── 📁 vector_db/                    # ChromaDB 벡터 데이터베이스 저장 폴더
├── 📁 assets/                       # UI 리소스, 스트레칭 애니메이션 및 아이콘
├── 📁 document/                     # 프로젝트 발표 PPTX 및 시연 영상 MP4
│
├── 📄 preprocess_jasee.py           # RAG 데이터 파이프라인 (PDF/DOCX/JSON 파싱 및 청크 변환)
├── 📄 build_vectordb.py             # ChromaDB 구축 및 ko-sroberta 벡터 임베딩 생성
├── 📄 chatbot.py                    # RAG 인체공학 챗봇 핵심 실행 파일 (CLI / API 연동)
├── 📄 jasee_core.py                 # 비전 인식 핵심 백엔드 엔진 (YOLO + MLP + 각도 연산)
│
├── 📄 app_mobile.py                 # 모바일 레이아웃 최적화 Streamlit 웹앱
├── 📄 app_desktop.py                # 데스크톱 레이아웃 최적화 Streamlit 웹앱
├── 📄 app_web.py                    # 모바일/데스크톱 기능 통합 올인원 Streamlit 웹앱
│
├── 📄 yolov8n-pose.pt               # ⭐ YOLOv8 공식 Pose 검출 가중치 파일
├── 📄 requirements.txt              # 전체 프로젝트 의존성 라이브러리 목록
├── 📄 logo.png                      # 프로젝트 로고 이미지
└── 📄 README.md                     # 프로젝트 설명서 (본 문서)
```

---

## 🛠️ 기술 스택

| 분류 | 기술 및 라이브러리 |
| :--- | :--- |
| **프로그래밍 언어** | Python 3.10 |
| **프론트엔드 / 웹앱 UI** | Streamlit, Streamlit-webrtc |
| **포즈 검출 (Pose)** | YOLOv8-pose (Ultralytics), OpenCV |
| **객체 탐지 (Object)** | YOLOv8 Custom Object Detection (의자, 책상, 모니터) |
| **자세 이중 분류** | PyTorch, Attention-based MLP |
| **임베딩 & 벡터 DB** | SentenceTransformers (`ko-sroberta-multitask`), ChromaDB |
| **생성형 AI (Chatbot)** | OpenAI API (`gpt-4o` / `gpt-3.5-turbo`) |
| **음성 안내 (TTS)** | pyttsx3 |
| **데이터 처리 / 시각화** | Pandas, Numpy, Altair, Pillow, python-docx, PyMuPDF (`fitz`) |
| **데이터 증강** | Albumentations |

---

## 🚀 시작 가이드 (Quick Start)

### 1. 리포지토리 클론 및 폴더 이동
```bash
git clone https://github.com/areum-mong/JaSEE.git
cd JaSEE
```

### 2. 가상환경 생성 및 활성화
```bash
conda create -n jasee python=3.10
conda activate jasee
```

### 3. 필수 라이브러리 설치
```bash
pip install -r requirements.txt
# CUDA 가속이 가능한 PyTorch 버전을 설치하는 것을 권장합니다 (RTX 그래픽카드 보유 시)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

### 4. 환경 변수 설정
프로젝트 루트 디렉토리에 `.env` 파일을 생성하고 아래 키를 입력합니다.
```env
OPENAI_API_KEY=your_openai_api_key_here
```

### 5. RAG 벡터 데이터베이스 구축 (데이터 변경 시 실행)
```bash
# 1단계: 지침 및 DB 문서 전처리 및 청크 분할
python preprocess_jasee.py

# 2단계: ko-sroberta 임베딩 생성 및 ChromaDB 저장
python build_vectordb.py
```

### 6. 서비스 실행
Streamlit 애플리케이션을 구동하여 실시간 측정 서비스를 웹 브라우저에서 이용할 수 있습니다.

* **통합 웹 페이지 버전 실행 (추천)**
  ```bash
  streamlit run app_web.py
  ```
* **모바일 뷰 전용 버전 실행**
  ```bash
  streamlit run app_mobile.py
  ```
* **데스크톱 뷰 전용 버전 실행**
  ```bash
  streamlit run app_desktop.py
  ```

---

## 👥 팀원 정보 (4팀 — 척추처척추)
* **개발 기간**: 2026년 5월 ~ 2026년 6월
* **교육 기관**: 아시아경제교육센터 인공지능 세미 프로젝트

---

## ⚠️ 면책 조항 (Disclaimer)
* 본 서비스는 의학적 진단 및 치료를 대체할 수 없습니다.
* 분석된 수치와 리포트는 바른 자세 및 올바른 작업 환경을 유지하기 위한 **가이드 및 참고용**으로만 활용하시기 바랍니다.
* 지속적인 통증이나 척추 질환이 의심되는 경우, 반드시 정형외과 전문의 등 의료 전문가와의 상담을 권장합니다.
