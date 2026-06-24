# 2026 Visionaries Kiosk
> 스마트 카페 키오스크 - 연령 맞춤형 UI 시스템

### 프로젝트 개요
- 웹캠 기반 실시간 얼굴 인식
- MobileNetV2를 통한 연령 분류 (50대 이상 / 미만)
- 50대 이상 감지 → 자동으로 큰글씨 + 쉬운 한국어 모드 전환


### 시스템 아키텍처
```text
[웹캠] -> [OpenCV 얼굴 검출] -> [이미지 전처리(Padding/Resize)]
                                     |
                                     v
[MobileNetV2 추론] <--- [FastAPI 백엔드 서버] ---> [고령자 맞춤 UI 전환]
```


## 협업 내역
- **총 개발 기간**: 2025. 05. 13 ~ 2025. 06. 25 (43일)
- **커밋 기록**: [Activity 페이지에서 확인 가능](https://github.com/mxrch5th/2026-Visionaries-Kiosk/activity)


### 팀 구성 및 역할
| 이름 | 역할 |
|------|------|
| mxrch5th (황유하) | 프로젝트 리드, UI/UX 최적화, 데이터 수집 및 학습 |
| sjhsppp (서정현) | AI 모델 개발, UI/UX 개발, 데이터 수집 및 학습 | 
| chaeun59ed (양채은) | 데이터 라벨링, 모델 검증, 데이터 수집 및 학습 |


### 설치 방법
#### 1단계: 저장소 클론
```bash
git clone https://github.com/mxrch5th/2026-Visionaries-Kiosk.git
cd 2026-Visionaries-Kiosk
```

#### 2단계: 가상 환경 생성 (권장)
```bash
python -m venv venv
source venv/bin/activate  # Mac/Linux
venv\Scripts\activate     # Windows
```

#### 3단계: 패키지 설치
```bash
pip install -r requirements.txt
```

### 실행방법
```bash
uvicorn app.main:app --reload
```

다음과 같은 메시지가 나오면 성공:
```code
INFO:     Uvicorn running on http://127.0.0.1:8000
```


### 사용방법

1. 브라우저에서 http://127.0.0.1:8000 접속
2. 웹캠 영역에 얼굴을 가까이 대기
3. 프로그램이 자동으로 나이 판별 시작 (최초 1회, 이후 새로고침 될 때마다)
4. 50대 이상 감지 → 자동으로 쉬운 모드로의 전환
5. 메뉴 선택 → 장바구니에 추가
6. "카드 결제하기" 버튼으로 주문 완료

### 트러블슈팅
| 문제 | 해결 방법 |
|------|------|
| 웹캠이 안 보임 | 브라우저 카메라 권한 허용 필요 |
| "⚠️ 모델 파일 없음" 오류 | model/best_model_v3_final.pth 파일 확인 | 
| 얼굴이 감지 안 됨 | 조명 개선, 얼굴을 웹캠에 더 가까이 |


### **주요 기능**
**1. 실시간 얼굴 감지**: OpenCV 기반 Haar Cascade

**2. 연령 분류**: PyTorch MobileNetV2 모델 (정확도 91% ~ 94%)

**3. UI 모드 전환**:
- 일반 모드: 일반 테마, 작은 글씨, 작은 아이콘
- 고령자 모드: 강조된 테마, 큰 글씨, 큰 아이콘

**4. 메뉴 관리**: 8가지 음료/디저트 (일반모드 4종 + 쉬운모드 4종)


### **기술 스택**
- FastAPI (백엔드 서버)
- Tkinter (UI)
- PyTorch (AI 모델)
- OpenCV (얼굴 감지)
- PIL (이미지 처리)


### 프로젝트 구조
```
2026-Visionaries-Kiosk/
├── app/
│   └── main.py              # FastAPI 백엔드 서버
├── static/
│   ├── index.html           # 키오스크 UI
│   ├── styles.css           # 스타일링
│   └── app.js               # 웹캠 + 메뉴 기능
├── model/
│   └── best_model_v3_final.pth  # 학습된 AI 모델
└── requirements.txt         # 패키지 목록
```
