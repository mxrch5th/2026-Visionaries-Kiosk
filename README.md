# Face Age Cafe Kiosk

얼굴 이미지에서 50대 이상으로 보이면 카페 키오스크가 자동으로 `쉬운모드`로 바뀌는 팀 프로젝트용 프로토타입입니다.

## 구성

- `app/main.py`: 웹 화면을 띄우고, 카메라 프레임을 받아 나이대 판별을 하는 서버
- `static/`: 실제 카페 키오스크 화면
- `training/train_age_group.py`: Google Colab GPU에서 학습하는 스크립트
- `training/dataset_guide.md`: 팀원들이 얼굴 데이터를 넣는 방법
- `model/`: 학습 완료 모델을 넣는 위치

## 실행 방법

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
uvicorn app.main:app --reload
```

브라우저에서 `http://127.0.0.1:8000`을 열면 됩니다.

모델 파일이 아직 없으면 서버가 자동으로 모의 판별 모드로 동작합니다. 그래서 UI 개발과 발표 시연은 먼저 진행할 수 있고, 학습이 끝난 뒤 `model/age_group_resnet18.pt`를 넣으면 실제 모델을 사용합니다.

## 학습 데이터 구조

팀원이 각자 모은 얼굴 이미지는 아래 구조로 정리합니다.

```text
dataset/
  train/
    under50/
    over50/
  val/
    under50/
    over50/
```

50대 이상 이미지는 `over50`, 50대 미만 이미지는 `under50`에 넣습니다.

## Colab 학습 흐름

1. Google Drive에 프로젝트 폴더를 올립니다.
2. Colab에서 GPU 런타임을 켭니다.
3. `training/train_age_group.py`를 실행합니다.
4. 생성된 `model/age_group_resnet18.pt`를 GitHub 또는 로컬 프로젝트의 `model/` 폴더에 넣습니다.

## GitHub 협업 팁

- 얼굴 원본 데이터는 개인정보라서 GitHub에 올리지 않는 것을 권장합니다.
- 모델 파일은 크기가 크면 Git LFS를 쓰거나 Google Drive 링크로 공유하세요.
- `.gitignore`에 데이터셋과 개인 이미지 폴더를 제외하도록 넣어두었습니다.
