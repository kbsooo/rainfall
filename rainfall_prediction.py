#!/usr/bin/env python
# coding: utf-8

# 필요한 라이브러리 임포트
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, roc_auc_score, roc_curve
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
import xgboost as xgb

# 경고 무시
import warnings
warnings.filterwarnings('ignore')

print("# 강수량 예측 모델")
print("기상 데이터를 이용하여 비가 올지(1) 또는 오지 않을지(0)를 예측하는 이진 분류 모델을 구축합니다.")

# 데이터 로드
train_data = pd.read_csv('train.csv')
test_data = pd.read_csv('test.csv')

# 데이터 확인
print("\n# 데이터 확인")
print("훈련 데이터 형태:", train_data.shape)
print(train_data.head())

# 데이터 정보 확인
print("\n# 데이터 정보")
print(train_data.info())

# 통계량 확인
print("\n# 통계량 확인")
print(train_data.describe())

# 결측치 확인
print("\n# 결측치 확인")
print("훈련 데이터 결측치:\n", train_data.isnull().sum())
print("\n테스트 데이터 결측치:\n", test_data.isnull().sum())

# 타겟 변수 분포 확인
print("\n# 타겟 변수 분포")
print("강수 여부 분포:")
print(train_data['rainfall'].value_counts())
print(f"비율: {train_data['rainfall'].value_counts(normalize=True)}")

# 상관관계 시각화
print("\n# 상관관계 분석")
correlation = train_data.corr()
plt.figure(figsize=(12, 10))
sns.heatmap(correlation, annot=True, cmap='coolwarm', linewidths=0.5)
plt.title('특성 간 상관관계')
plt.tight_layout()
plt.savefig('correlation_heatmap.png')
plt.close()

# rainfall과 다른 특성들의 상관관계 확인
rainfall_corr = correlation['rainfall'].sort_values(ascending=False)
print("강수 여부와 각 특성의 상관관계:")
print(rainfall_corr)

# 특성 엔지니어링
print("\n# 특성 엔지니어링")
# 1. 온도차이 특성 추가
train_data['temp_diff'] = train_data['maxtemp'] - train_data['mintemp']
test_data['temp_diff'] = test_data['maxtemp'] - test_data['mintemp']

# 2. 온도와 이슬점 차이 특성 추가
train_data['temp_dewpoint_diff'] = train_data['temparature'] - train_data['dewpoint']
test_data['temp_dewpoint_diff'] = test_data['temparature'] - test_data['dewpoint']

# 3. 사이클릭 특성으로 day 변환 (계절성 표현)
train_data['day_sin'] = np.sin(2 * np.pi * train_data['day'] / 365.25)
train_data['day_cos'] = np.cos(2 * np.pi * train_data['day'] / 365.25)
test_data['day_sin'] = np.sin(2 * np.pi * test_data['day'] / 365.25)
test_data['day_cos'] = np.cos(2 * np.pi * test_data['day'] / 365.25)

# 원본 'id'와 'day' 컬럼은 제거 (모델링에 필요 없음)
train_features = train_data.drop(['id', 'day', 'rainfall'], axis=1)
test_features = test_data.drop(['id', 'day'], axis=1)

# 타겟 변수
train_target = train_data['rainfall']

# 훈련/검증 데이터 분할
X_train, X_val, y_train, y_val = train_test_split(train_features, train_target, test_size=0.2, random_state=42, stratify=train_target)

# 특성 스케일링
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)
test_scaled = scaler.transform(test_features)

print("훈련 데이터 형태:", X_train.shape)
print("검증 데이터 형태:", X_val.shape)
print("테스트 데이터 형태:", test_features.shape)

# 다양한 모델 훈련 및 비교
print("\n# 모델 훈련 및 평가")
models = {
    'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
    'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
    'Gradient Boosting': GradientBoostingClassifier(n_estimators=100, random_state=42),
    'XGBoost': xgb.XGBClassifier(n_estimators=100, random_state=42)
}

results = {}

for name, model in models.items():
    # 모델 훈련
    model.fit(X_train_scaled, y_train)
    
    # 검증 데이터로 예측
    val_pred = model.predict(X_val_scaled)
    val_prob = model.predict_proba(X_val_scaled)[:, 1] if hasattr(model, 'predict_proba') else None
    
    # 성능 평가
    accuracy = accuracy_score(y_val, val_pred)
    report = classification_report(y_val, val_pred)
    cm = confusion_matrix(y_val, val_pred)
    auc = roc_auc_score(y_val, val_prob) if val_prob is not None else None
    
    # 결과 저장
    results[name] = {
        'model': model,
        'accuracy': accuracy,
        'auc': auc,
        'report': report,
        'confusion_matrix': cm
    }
    
    # 결과 출력
    print(f"\n{name} 모델 성능:")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"AUC: {auc:.4f}" if auc is not None else "AUC: None")
    print("\nClassification Report:")
    print(report)
    
    # 혼동 행렬 시각화
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)
    plt.title(f'{name} - Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.savefig(f'{name}_confusion_matrix.png')
    plt.close()

# 최적 모델 선택
print("\n# 최적 모델 선택")
best_model_name = max(results, key=lambda x: results[x]['accuracy'])
best_model = results[best_model_name]['model']
print(f"최고 성능 모델: {best_model_name}")
print(f"정확도: {results[best_model_name]['accuracy']:.4f}")
print(f"AUC: {results[best_model_name]['auc']:.4f}" if results[best_model_name]['auc'] is not None else "AUC: None")

# 특성 중요도 확인 (트리 기반 모델만)
if hasattr(best_model, 'feature_importances_'):
    feature_importances = pd.DataFrame({
        'feature': train_features.columns,
        'importance': best_model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    plt.figure(figsize=(12, 8))
    sns.barplot(x='importance', y='feature', data=feature_importances)
    plt.title(f'{best_model_name} - 특성 중요도')
    plt.tight_layout()
    plt.savefig('feature_importance.png')
    plt.close()
    print("\n특성 중요도:")
    print(feature_importances)

# 하이퍼파라미터 튜닝
print("\n# 하이퍼파라미터 튜닝")
if best_model_name == 'Random Forest':
    param_grid = {
        'n_estimators': [100, 200, 300],
        'max_depth': [None, 10, 20, 30],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4]
    }
elif best_model_name == 'Gradient Boosting':
    param_grid = {
        'n_estimators': [100, 200, 300],
        'learning_rate': [0.01, 0.05, 0.1],
        'max_depth': [3, 5, 7],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4]
    }
elif best_model_name == 'XGBoost':
    param_grid = {
        'n_estimators': [100, 200, 300],
        'learning_rate': [0.01, 0.05, 0.1],
        'max_depth': [3, 5, 7],
        'subsample': [0.7, 0.8, 0.9],
        'colsample_bytree': [0.7, 0.8, 0.9]
    }
else:  # Logistic Regression
    param_grid = {
        'C': [0.01, 0.1, 1, 10, 100],
        'penalty': ['l1', 'l2'],
        'solver': ['liblinear', 'saga']
    }

# 그리드 서치 수행 (시간이 오래 걸릴 수 있음)
grid_search = GridSearchCV(
    estimator=best_model,
    param_grid=param_grid,
    cv=5,
    scoring='accuracy',
    verbose=1,
    n_jobs=-1
)

# 시간 단축을 위해 작은 샘플로 튜닝 (선택적)
X_train_sample, _, y_train_sample, _ = train_test_split(
    X_train_scaled, y_train, train_size=0.5, random_state=42, stratify=y_train
)

# 그리드 서치 실행
print("그리드 서치 실행 중... (시간이 오래 걸릴 수 있습니다)")
grid_search.fit(X_train_sample, y_train_sample)

print(f"최적 하이퍼파라미터: {grid_search.best_params_}")
print(f"최적 CV 점수: {grid_search.best_score_:.4f}")

# 최적화된 모델로 최종 훈련
best_model = grid_search.best_estimator_
best_model.fit(X_train_scaled, y_train)

# 검증 데이터로 성능 평가
y_pred = best_model.predict(X_val_scaled)
y_prob = best_model.predict_proba(X_val_scaled)[:, 1] if hasattr(best_model, 'predict_proba') else None

print(f"\n최종 모델 성능 (검증 데이터):")
print(f"Accuracy: {accuracy_score(y_val, y_pred):.4f}")
print(f"AUC: {roc_auc_score(y_val, y_prob):.4f}" if y_prob is not None else "AUC: None")
print("\nClassification Report:")
print(classification_report(y_val, y_pred))

# ROC 곡선 시각화 (최종 모델)
if y_prob is not None:
    fpr, tpr, _ = roc_curve(y_val, y_prob)
    auc = roc_auc_score(y_val, y_prob)
    
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, label=f'AUC = {auc:.4f}')
    plt.plot([0, 1], [0, 1], 'k--', label='Random')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.legend(loc='lower right')
    plt.grid(alpha=0.3)
    plt.savefig('roc_curve.png')
    plt.close()

# 테스트 데이터 예측
print("\n# 테스트 데이터 예측 및 제출 파일 생성")
test_pred = best_model.predict(test_scaled)
test_prob = best_model.predict_proba(test_scaled)[:, 1] if hasattr(best_model, 'predict_proba') else None

# 제출 파일 생성
submission = pd.DataFrame({
    'id': test_data['id'],
    'rainfall': test_pred
})

# 결과 확인
print("예측 결과 분포:")
print(submission['rainfall'].value_counts())

# 제출 파일 저장
submission.to_csv('submission.csv', index=False)
print("제출 파일이 생성되었습니다: 'submission.csv'") 