# 강수량 예측 모델

이 프로젝트는 다양한 기상 데이터를 사용하여 특정 날짜에 비가 올지(1) 또는 오지 않을지(0)를 예측하는 이진 분류 모델을 구축합니다.

## 프로젝트 구조

- `train.csv`: 훈련 데이터 (날짜별 기상 데이터와 강수 여부 정보)
- `test.csv`: 테스트 데이터 (강수 여부를 예측해야 하는 데이터)
- `sample_submission.csv`: 제출 양식
- `rainfall_prediction.py`: 전체 모델링 과정을 담은 파이썬 스크립트
- `notebook.ipynb`: Jupyter 노트북 (동일한 코드를 대화형으로 실행)
- `requirements.txt`: 필요한 라이브러리 목록

## 데이터 설명

- `id`: 고유 식별자
- `day`: 일자 (1-365)
- `pressure`: 기압 (hPa)
- `maxtemp`: 최고 온도 (°C)
- `temparature`: 평균 온도 (°C)
- `mintemp`: 최저 온도 (°C)
- `dewpoint`: 이슬점 (°C)
- `humidity`: 습도 (%)
- `cloud`: 운량 (%)
- `sunshine`: 일조 시간 (시간)
- `winddirection`: 풍향 (각도)
- `windspeed`: 풍속 (km/h)
- `rainfall`: 강수 여부 (0=비 안 옴, 1=비 옴) - 예측해야 할 타겟 변수

## 사용한 모델

- 로지스틱 회귀
- 랜덤 포레스트
- 그래디언트 부스팅
- XGBoost

## 실행 방법

1. 필요한 라이브러리 설치:
```
pip install -r requirements.txt
```

2. 모델 학습 및 예측 실행:
```
python rainfall_prediction.py
```

3. 또는 Jupyter 노트북으로 실행:
```
jupyter notebook notebook.ipynb
```

## 결과물

- `submission.csv`: 테스트 데이터에 대한 강수 여부 예측 결과
- 다양한 시각화 파일들:
  - `correlation_heatmap.png`: 특성 간 상관관계
  - `feature_importance.png`: 특성 중요도
  - `roc_curve.png`: ROC 곡선
  - 각 모델별 혼동 행렬 