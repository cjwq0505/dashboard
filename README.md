# 🏫 고려대 세종캠퍼스 민원 분석 대시보드

## 실행 방법

### 1. 필요한 라이브러리 설치
```bash
pip install -r requirements.txt
```

### 2. Java 설치 확인 (KoNLPy 필요)
```bash
java -version
```
없으면: `conda install -c conda-forge openjdk`

### 3. 대시보드 실행
```bash
streamlit run app.py
```

### 4. 브라우저에서 확인
자동으로 브라우저가 열리며, 보통 `http://localhost:8501` 에서 실행됩니다.

## 파일 구성
- `app.py` — Streamlit 대시보드 메인 코드
- `에브리타임_전처리완료.csv` — 전처리 완료된 데이터 (370건)
- `model.pkl` — 분류기 모델 + 사전 데이터
- `requirements.txt` — 필요한 파이썬 라이브러리 목록

## 기능
1. 📊 **전체 현황** — 카테고리별 건수, 시기 분포, 인기 민원 Top 10
2. 🔍 **키워드 분석** — 카테고리별 워드클라우드 + TF-IDF Top 10
3. 📅 **시기별 비교** — 시험기간 vs 평소 / 일반 vs 추천수 가중 TF-IDF
4. 🔮 **민원 자동 분류** — 불만글 입력 → 카테고리 예측 + 담당 부서 안내
