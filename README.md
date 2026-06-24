# 2026 Visionaries Kiosk
> 스마트 카페 키오스크 - 연령 맞춤형 UI 시스템

### 프로젝트 개요
- 웹캠 기반 실시간 얼굴 인식
- MobileNetV2를 통한 연령 분류 (50대 이상 / 미만)
- 50대 이상 감지 → 자동으로 큰글씨 + 쉬운 한국어 모드 전환

## 협업 내역
- **총 개발 기간**: 2025. 05. 13 ~ 2025. 06. 25 (43일)
- **총 커밋**: [Activity 페이지에서 확인 가능](https://github.com/mxrch5th/2026-Visionaries-Kiosk/activity)


### 팀 구성 및 역할
| 이름 | 역할 |
|------|------|
| mxrch5th (황유하) | 프로젝트 리드, UI/UX 최적화, 데이터 수집 및 학습 |
| sjhsppp (서정현) | AI 모델 개발, UI/UX 개발, 데이터 수집 및 학습 | 
| chaeun59ed (양채은) | 데이터 라벨링, 모델 검증, 데이터 수집 및 학습 |


### 설치 및 실행 방법
```bash
pip install -r requirements.txt
uvicorn app.main:app --reload
```
브라우저에서 `http://localhost:8000` 으로 접속하여 키오스크 UI 사용 가능


### **주요 기능**
**1. 실시간 얼굴 감지**: OpenCV 기반 Haar Cascade

**2. 연령 분류**: PyTorch MobileNetV2 모델 (정확도 91% ~ 94%)

**3. UI 모드 전환**:
- 일반 모드: 일반 테마, 작은 글씨, 작은 아이콘
- 고령자 모드: 강조된 테마, 큰 글씨, 큰 아이콘
- 메뉴 관리: 4가지 음료/디저트


### **기술 스택**
Python, Tkinter (UI)
PyTorch (AI 모델)
OpenCV (얼굴 감지)
PIL (이미지 처리)
