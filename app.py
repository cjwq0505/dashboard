import streamlit as st
import pandas as pd
import numpy as np
import pickle
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from collections import Counter
from konlpy.tag import Okt

# =====================================================
# 페이지 설정
# =====================================================
st.set_page_config(
    page_title="고려대 세종캠퍼스 민원 분석",
    page_icon="🏫",
    layout="wide",
    initial_sidebar_state="expanded"
)

# =====================================================
# 데이터 & 모델 로드 (캐싱)
# =====================================================
@st.cache_data
def load_data():
    df = pd.read_csv('에브리타임_전처리완료.csv')
    # 추천수 복구
    def fix_upvote(val):
        if pd.isna(val): return 0
        try: return int(val)
        except (ValueError, TypeError): pass
        s = str(val)
        if s == '00:00:00': return 0
        if '1900-01-' in s:
            try: return int(s.split('1900-01-')[1].split(' ')[0])
            except: return 0
        try:
            ts = pd.Timestamp(val)
            if ts.year == 1900: return ts.day
        except: pass
        return 0
    df['추천수'] = df['추천수'].apply(fix_upvote)
    return df

@st.cache_resource
def load_model():
    with open('model.pkl', 'rb') as f:
        return pickle.load(f)

@st.cache_resource
def load_okt():
    return Okt()

df = load_data()
model_data = load_model()
okt = load_okt()

clf = model_data['classifier']
vectorizer_clf = model_data['vectorizer']
profanity_dict = model_data['profanity_dict']
custom_dict = model_data['custom_dict']
merge_dict = model_data['merge_dict']
stopwords_set = model_data['stopwords']

# 한글 폰트 설정
import platform
if platform.system() == 'Windows':
    FONT_PATH = 'C:/Windows/Fonts/malgun.ttf'
    plt.rcParams['font.family'] = 'Malgun Gothic'
elif platform.system() == 'Darwin':
    FONT_PATH = '/System/Library/Fonts/AppleSDGothicNeo.ttc'
    plt.rcParams['font.family'] = 'AppleGothic'
else:
    FONT_PATH = '/usr/share/fonts/truetype/nanum/NanumGothic.ttf'
    plt.rcParams['font.family'] = 'NanumGothic'
plt.rcParams['axes.unicode_minus'] = False

# =====================================================
# 전처리 함수
# =====================================================
def clean_text(text):
    text = str(text)
    text = re.sub(r'http\S+', '', text)
    text = re.sub(r'\\n|\n', ' ', text)
    for abbr, full in profanity_dict.items():
        text = text.replace(abbr, full)
    text = re.sub(r'[ㄱ-ㅎㅏ-ㅣ]+', ' ', text)
    text = re.sub(r'[^\w\s가-힣]', ' ', text)
    text = re.sub(r'\d+', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def apply_custom_dict(text):
    for key in sorted(custom_dict.keys(), key=len, reverse=True):
        text = text.replace(key, custom_dict[key])
    return text

def extract_morphs(text):
    pos_tags = okt.pos(text, norm=False, stem=True)
    words = []
    for word, pos in pos_tags:
        clean_word = word.replace('_','')
        if pos in ('Noun','Adjective','Verb') and len(clean_word) >= 2 and clean_word not in stopwords_set:
            words.append(clean_word)
    return ' '.join(words)

def merge_compound(text):
    for split, merged in merge_dict.items():
        text = text.replace(split, merged)
    return text

def full_preprocess(text):
    t = clean_text(text)
    t = apply_custom_dict(t)
    t = extract_morphs(t)
    t = merge_compound(t)
    return t

# =====================================================
# 부서 매핑
# =====================================================
dept_map = {
    '기숙사/주거': {'부서': '생활관 관리팀', '연락처': '044-860-1234', 'icon': '🏠', 'color': '#E53935'},
    '시설/환경': {'부서': '시설관리팀', '연락처': '044-860-1235', 'icon': '🔧', 'color': '#1E88E5'},
    '교통/이동': {'부서': '안전관리팀', '연락처': '044-860-1236', 'icon': '🛴', 'color': '#43A047'},
    '학사/행정': {'부서': '학사지원팀 (교무처)', '연락처': '044-860-1237', 'icon': '📚', 'color': '#7B1FA2'},
    '복지/음료': {'부서': '학생복지팀 (학생처)', '연락처': '044-860-1238', 'icon': '🍱', 'color': '#FB8C00'},
}

cat_colors = {
    '기숙사/주거': '#E53935', '시설/환경': '#1E88E5', '교통/이동': '#43A047',
    '학사/행정': '#7B1FA2', '복지/음료': '#FB8C00',
}

# =====================================================
# CSS 스타일
# =====================================================
st.markdown("""
<style>
    .main-header {
        font-size: 2.2rem;
        font-weight: 700;
        text-align: center;
        padding: 1rem 0 0.5rem;
        color: #8B0000;
    }
    .sub-header {
        font-size: 1rem;
        text-align: center;
        color: #666;
        margin-bottom: 2rem;
    }
    .metric-card {
        background: #f8f9fa;
        border-radius: 12px;
        padding: 1.2rem;
        text-align: center;
        border: 1px solid #e9ecef;
    }
    .metric-number {
        font-size: 2rem;
        font-weight: 700;
        color: #8B0000;
    }
    .metric-label {
        font-size: 0.85rem;
        color: #666;
        margin-top: 4px;
    }
    .dept-card {
        background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
        border-radius: 16px;
        padding: 2rem;
        text-align: center;
        margin: 1rem 0;
    }
    .result-badge {
        display: inline-block;
        padding: 6px 16px;
        border-radius: 20px;
        font-weight: 600;
        font-size: 1.1rem;
        color: white;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    .stTabs [data-baseweb="tab"] {
        border-radius: 8px;
        padding: 8px 20px;
    }
</style>
""", unsafe_allow_html=True)

# =====================================================
# 사이드바
# =====================================================
with st.sidebar:
    st.markdown("## 🏫 메뉴")
    page = st.radio(
        "페이지 선택",
        ["📊 전체 현황", "🔍 키워드 분석", "📅 시기별 비교", "🔮 민원 자동 분류"],
        label_visibility="collapsed"
    )
    st.markdown("---")
    st.markdown("**프로젝트 정보**")
    st.markdown("📌 고려대학교 세종캠퍼스")
    st.markdown(f"📄 분석 데이터: **{len(df)}건**")
    st.markdown("🔬 TF-IDF + Word2Vec")
    st.markdown("👥 최정민 · 김민제")

# =====================================================
# 페이지 1: 전체 현황
# =====================================================
if page == "📊 전체 현황":
    st.markdown('<div class="main-header">🏫 고려대 세종캠퍼스 민원 분석 대시보드</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">에브리타임 불만글 텍스트 분석 | TF-IDF & Word2Vec 기반</div>', unsafe_allow_html=True)

    # 상단 메트릭
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown(f"""<div class="metric-card">
            <div class="metric-number">{len(df)}</div>
            <div class="metric-label">전체 민원 수</div>
        </div>""", unsafe_allow_html=True)
    with col2:
        exam_count = len(df[df['시기구분']=='시험기간'])
        st.markdown(f"""<div class="metric-card">
            <div class="metric-number">{exam_count}</div>
            <div class="metric-label">시험기간 민원</div>
        </div>""", unsafe_allow_html=True)
    with col3:
        normal_count = len(df[df['시기구분']=='평소'])
        st.markdown(f"""<div class="metric-card">
            <div class="metric-number">{normal_count}</div>
            <div class="metric-label">평소 민원</div>
        </div>""", unsafe_allow_html=True)
    with col4:
        avg_upvote = df['추천수'].mean()
        st.markdown(f"""<div class="metric-card">
            <div class="metric-number">{avg_upvote:.1f}</div>
            <div class="metric-label">평균 추천수</div>
        </div>""", unsafe_allow_html=True)

    st.markdown("---")

    # 카테고리별 분포
    col_left, col_right = st.columns(2)

    with col_left:
        st.subheader("📊 카테고리별 민원 분포")
        cat_counts = df['카테고리'].value_counts()
        fig, ax = plt.subplots(figsize=(8, 5))
        colors = [cat_colors.get(c, '#999') for c in cat_counts.index]
        bars = ax.barh(cat_counts.index[::-1], cat_counts.values[::-1], color=colors[::-1], alpha=0.85)
        for bar, val in zip(bars, cat_counts.values[::-1]):
            ax.text(bar.get_width() + 1, bar.get_y() + bar.get_height()/2, f'{val}건', va='center', fontsize=12)
        ax.set_xlabel('건수', fontsize=12)
        ax.grid(axis='x', alpha=0.3)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

    with col_right:
        st.subheader("📅 시기별 분포")
        fig, ax = plt.subplots(figsize=(8, 5))
        period_counts = df['시기구분'].value_counts()
        colors_period = ['#E53935', '#1E88E5']
        wedges, texts, autotexts = ax.pie(
            period_counts.values, labels=period_counts.index,
            colors=colors_period, autopct='%1.1f%%',
            startangle=90, textprops={'fontsize': 14}
        )
        for autotext in autotexts:
            autotext.set_fontsize(13)
            autotext.set_fontweight('bold')
        ax.set_title('시험기간 vs 평소', fontsize=15, fontweight='bold', pad=15)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

    # 최근 인기 민원
    st.markdown("---")
    st.subheader("🔥 추천수 높은 민원 Top 10")
    top_posts = df.nlargest(10, '추천수')[['카테고리', '제목', '원문텍스트', '추천수', '시기구분']].reset_index(drop=True)
    top_posts.index = top_posts.index + 1
    top_posts['제목'] = top_posts['제목'].str.replace('\n', '', regex=False).str[:40]
    top_posts['원문텍스트'] = top_posts['원문텍스트'].str.replace('\n', ' ', regex=False).str[:60]
    st.dataframe(top_posts, use_container_width=True)

# =====================================================
# 페이지 2: 키워드 분석
# =====================================================
elif page == "🔍 키워드 분석":
    st.markdown('<div class="main-header">🔍 카테고리별 키워드 분석</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">TF-IDF 기반 핵심 불만 키워드 추출</div>', unsafe_allow_html=True)

    # 카테고리 탭
    categories = df['카테고리'].unique()
    tabs = st.tabs([f"{dept_map[c]['icon']} {c}" for c in categories])

    colormap_dict = {
        '기숙사/주거': 'Reds', '시설/환경': 'Blues', '교통/이동': 'Greens',
        '학사/행정': 'Purples', '복지/음료': 'Oranges',
    }

    for tab, cat in zip(tabs, categories):
        with tab:
            texts = df[df['카테고리'] == cat]['형태소'].dropna().tolist()

            # TF-IDF
            vec = TfidfVectorizer(max_features=50)
            tfidf = vec.fit_transform(texts)
            scores = tfidf.mean(axis=0).A1
            words = vec.get_feature_names_out()
            word_freq = dict(zip(words, scores))
            top_kw = sorted(word_freq.items(), key=lambda x: -x[1])[:10]

            col_wc, col_bar = st.columns([3, 2])

            with col_wc:
                st.markdown(f"**워드클라우드** ({len(texts)}건)")
                try:
                    wc = WordCloud(
                        font_path=FONT_PATH, background_color='white',
                        width=800, height=500, colormap=colormap_dict[cat],
                        max_words=50, prefer_horizontal=0.9,
                    ).generate_from_frequencies(word_freq)
                    fig, ax = plt.subplots(figsize=(10, 6))
                    ax.imshow(wc, interpolation='bilinear')
                    ax.axis('off')
                    plt.tight_layout()
                    st.pyplot(fig)
                    plt.close()
                except Exception as e:
                    st.error(f"워드클라우드 생성 실패: {e}")

            with col_bar:
                st.markdown("**TF-IDF Top 10**")
                fig, ax = plt.subplots(figsize=(6, 5))
                kw_words = [w for w, _ in top_kw][::-1]
                kw_scores = [s for _, s in top_kw][::-1]
                ax.barh(kw_words, kw_scores, color=cat_colors[cat], alpha=0.8)
                for i, v in enumerate(kw_scores):
                    ax.text(v + 0.001, i, f'{v:.3f}', va='center', fontsize=9)
                ax.grid(axis='x', alpha=0.3)
                ax.spines['top'].set_visible(False)
                ax.spines['right'].set_visible(False)
                plt.tight_layout()
                st.pyplot(fig)
                plt.close()

# =====================================================
# 페이지 3: 시기별 비교
# =====================================================
elif page == "📅 시기별 비교":
    st.markdown('<div class="main-header">📅 시기별 불만 키워드 비교</div>', unsafe_allow_html=True)

    analysis_type = st.radio(
        "분석 유형 선택",
        ["시험기간 vs 평소 (TF-IDF 차이)", "일반 vs 추천수 가중 TF-IDF"],
        horizontal=True
    )

    if analysis_type == "시험기간 vs 평소 (TF-IDF 차이)":
        st.markdown('<div class="sub-header">시험기간에 유독 많이 나오는 키워드 vs 평소에 유독 많이 나오는 키워드</div>', unsafe_allow_html=True)

        results = {}
        for period in ['시험기간', '평소']:
            texts = df[df['시기구분'] == period]['형태소'].tolist()
            vec = TfidfVectorizer(max_features=100)
            tfidf = vec.fit_transform(texts)
            scores = tfidf.mean(axis=0).A1
            words = vec.get_feature_names_out()
            top = sorted(zip(words, scores), key=lambda x: -x[1])[:20]
            results[period] = dict(top)

        all_words = set(results['시험기간'].keys()) | set(results['평소'].keys())
        diff_data = []
        for word in all_words:
            e = results['시험기간'].get(word, 0)
            n = results['평소'].get(word, 0)
            diff_data.append((word, e, n, e - n))

        exam_chars = sorted(diff_data, key=lambda x: -x[3])[:10]
        normal_chars = sorted(diff_data, key=lambda x: x[3])[:10]

        col1, col2 = st.columns(2)
        with col1:
            st.markdown("### 🎯 시험기간에 유독 많이 나오는 단어")
            fig, ax = plt.subplots(figsize=(8, 6))
            words_e = [x[0] for x in exam_chars][::-1]
            diffs_e = [x[3] for x in exam_chars][::-1]
            ax.barh(words_e, diffs_e, color='#E53935', alpha=0.85)
            for i, v in enumerate(diffs_e):
                ax.text(v + 0.001, i, f'+{v:.3f}', va='center', fontsize=10)
            ax.set_xlabel('TF-IDF 차이값', fontsize=11)
            ax.grid(axis='x', alpha=0.3)
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()

        with col2:
            st.markdown("### 🏠 평소에 유독 많이 나오는 단어")
            fig, ax = plt.subplots(figsize=(8, 6))
            words_n = [x[0] for x in normal_chars][::-1]
            diffs_n = [-x[3] for x in normal_chars][::-1]
            ax.barh(words_n, diffs_n, color='#1E88E5', alpha=0.85)
            for i, v in enumerate(diffs_n):
                ax.text(v + 0.001, i, f'+{v:.3f}', va='center', fontsize=10)
            ax.set_xlabel('TF-IDF 차이값', fontsize=11)
            ax.grid(axis='x', alpha=0.3)
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()

        st.markdown("---")
        st.info("💡 **인사이트**: 시험기간엔 **공간·환경 불만**(자리, 학술정보원, 자판기, 조용하다)이 급증하고, 평소엔 **시스템·일상 불만**(수강신청, 서버, 택배)이 주를 이룹니다.")

    else:  # 추천수 가중
        st.markdown('<div class="sub-header">많이 올라오는 불만 vs 많이 공감받는 불만</div>', unsafe_allow_html=True)

        period_filter = st.radio("시기 선택", ["전체", "시험기간", "평소"], horizontal=True)

        if period_filter == "전체":
            mask = pd.Series([True]*len(df))
        else:
            mask = df['시기구분'] == period_filter

        texts = df[mask]['형태소'].tolist()
        upvotes = df[mask]['추천수'].values
        weights = np.log(upvotes + 2)

        vec = TfidfVectorizer(max_features=100)
        tfidf = vec.fit_transform(texts)
        words = vec.get_feature_names_out()

        plain_scores = tfidf.mean(axis=0).A1
        top_plain = sorted(zip(words, plain_scores), key=lambda x: -x[1])[:12]

        tfidf_w = tfidf.multiply(weights.reshape(-1, 1))
        w_scores = np.asarray(tfidf_w.mean(axis=0)).flatten()
        top_weighted = sorted(zip(words, w_scores), key=lambda x: -x[1])[:12]

        plain_set = set([w for w, _ in top_plain])
        weighted_set = set([w for w, _ in top_weighted])
        newly = weighted_set - plain_set
        gone = plain_set - weighted_set

        col1, col2 = st.columns(2)
        with col1:
            st.markdown("### 📊 일반 TF-IDF")
            fig, ax = plt.subplots(figsize=(8, 6))
            p_words = [w for w, _ in top_plain[:10]][::-1]
            p_scores = [s for _, s in top_plain[:10]][::-1]
            colors = ['#E53935' if w in gone else '#90A4AE' for w in p_words]
            ax.barh(p_words, p_scores, color=colors, alpha=0.85)
            for i, v in enumerate(p_scores):
                ax.text(v + 0.001, i, f'{v:.3f}', va='center', fontsize=10)
            ax.grid(axis='x', alpha=0.3)
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()

        with col2:
            st.markdown("### 🔥 추천수 가중 TF-IDF")
            fig, ax = plt.subplots(figsize=(8, 6))
            w_words = [w for w, _ in top_weighted[:10]][::-1]
            w_scores_list = [s for _, s in top_weighted[:10]][::-1]
            colors = ['#43A047' if w in newly else '#90A4AE' for w in w_words]
            ax.barh(w_words, w_scores_list, color=colors, alpha=0.85)
            for i, v in enumerate(w_scores_list):
                ax.text(v + 0.001, i, f'{v:.3f}', va='center', fontsize=10)
            ax.grid(axis='x', alpha=0.3)
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()

        if newly:
            st.success(f"🔥 **추천수 가중 후 새로 등장한 키워드**: {', '.join(newly)}")
        if gone:
            st.warning(f"💤 **추천수 가중 후 밀려난 키워드**: {', '.join(gone)}")

        st.info("💡 **인사이트**: 많이 올라오는 불만(빈도)과 많이 공감받는 불만(추천수)은 다릅니다. 학교는 빈도가 아닌 공감도 기준으로 시설 개선 우선순위를 정해야 합니다.")

# =====================================================
# 페이지 4: 민원 자동 분류
# =====================================================
elif page == "🔮 민원 자동 분류":
    st.markdown('<div class="main-header">🔮 민원 자동 분류 시스템</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">불만글을 입력하면 카테고리를 자동으로 분류하고 담당 부서를 안내합니다</div>', unsafe_allow_html=True)

    st.markdown("")

    # 예시 버튼
    st.markdown("**💬 예시 불만글 (클릭해서 입력)**")
    example_cols = st.columns(4)
    examples = [
        "기숙사 와이파이 또 끊겼다 시험기간인데 진짜 답답해",
        "킥보드 역주행하는 사람들 제발 좀 단속해주세요",
        "수강신청 서버 왜 이렇게 느려요 접속이 안 돼요",
        "자판기 음료 유통기한 지난 거 팔지 마세요",
    ]
    selected_example = None
    for i, (col, ex) in enumerate(zip(example_cols, examples)):
        with col:
            if st.button(ex[:15] + "...", key=f"ex_{i}", use_container_width=True):
                selected_example = ex

    # 입력창
    user_input = st.text_area(
        "불만글을 입력하세요",
        value=selected_example if selected_example else "",
        height=120,
        placeholder="예: 기숙사 와이파이 또 끊겼다 시험기간인데 진짜 답답해"
    )

    if st.button("🔍 분석하기", type="primary", use_container_width=True):
        if user_input.strip():
            with st.spinner("분석 중..."):
                # 전처리
                morphs = full_preprocess(user_input)

                # 분류
                vec = vectorizer_clf.transform([morphs])
                pred = clf.predict(vec)[0]
                proba = clf.predict_proba(vec)[0]
                confidence = proba.max()
                proba_dict = dict(zip(clf.classes_, proba))

                # 유사 민원 검색
                all_vec = vectorizer_clf.transform(df['형태소'].tolist())
                from sklearn.metrics.pairwise import cosine_similarity
                similarities = cosine_similarity(vec, all_vec).flatten()
                top_sim_idx = similarities.argsort()[-5:][::-1]
                similar_posts = df.iloc[top_sim_idx]

                # 키워드 추출
                feature_names = vectorizer_clf.get_feature_names_out()
                vec_array = vec.toarray().flatten()
                top_keyword_idx = vec_array.argsort()[-5:][::-1]
                keywords = [feature_names[i] for i in top_keyword_idx if vec_array[i] > 0]

                dept_info = dept_map.get(pred, {'부서': '미정', '연락처': '-', 'icon': '❓', 'color': '#999'})

            # 결과 표시
            st.markdown("---")
            st.markdown("## 📋 분석 결과")

            # 상단 결과 카드
            r_col1, r_col2, r_col3 = st.columns(3)
            with r_col1:
                st.markdown(f"""<div class="metric-card">
                    <div style="font-size:2.5rem;">{dept_info['icon']}</div>
                    <div class="metric-number" style="color:{dept_info['color']}">{pred}</div>
                    <div class="metric-label">예측 카테고리</div>
                </div>""", unsafe_allow_html=True)
            with r_col2:
                st.markdown(f"""<div class="metric-card">
                    <div style="font-size:2.5rem;">📊</div>
                    <div class="metric-number">{confidence:.0%}</div>
                    <div class="metric-label">분류 신뢰도</div>
                </div>""", unsafe_allow_html=True)
            with r_col3:
                similar_count = len(df[df['카테고리'] == pred])
                st.markdown(f"""<div class="metric-card">
                    <div style="font-size:2.5rem;">📑</div>
                    <div class="metric-number">{similar_count}건</div>
                    <div class="metric-label">같은 유형 민원</div>
                </div>""", unsafe_allow_html=True)

            st.markdown("")

            # 담당 부서 안내
            st.markdown(f"""<div class="dept-card">
                <h3>📞 담당 부서: {dept_info['부서']}</h3>
                <p style="font-size:1.2rem; margin:0;">연락처: {dept_info['연락처']}</p>
            </div>""", unsafe_allow_html=True)

            # 핵심 키워드
            if keywords:
                st.markdown("### 🏷️ 추출된 핵심 키워드")
                keyword_html = " ".join([
                    f'<span style="background:{dept_info["color"]}22; color:{dept_info["color"]}; '
                    f'padding:6px 14px; border-radius:20px; margin:4px; display:inline-block; '
                    f'font-weight:600;">{kw}</span>'
                    for kw in keywords[:5]
                ])
                st.markdown(keyword_html, unsafe_allow_html=True)

            # 카테고리별 확률
            st.markdown("### 📊 카테고리별 분류 확률")
            fig, ax = plt.subplots(figsize=(10, 3))
            sorted_proba = sorted(proba_dict.items(), key=lambda x: -x[1])
            cats_sorted = [c for c, _ in sorted_proba]
            probs_sorted = [p for _, p in sorted_proba]
            bar_colors = [cat_colors.get(c, '#999') for c in cats_sorted]
            bars = ax.barh(cats_sorted[::-1], probs_sorted[::-1], color=bar_colors[::-1], alpha=0.85)
            for bar, val in zip(bars, probs_sorted[::-1]):
                ax.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height()/2,
                        f'{val:.1%}', va='center', fontsize=11, fontweight='bold')
            ax.set_xlim(0, 1)
            ax.set_xlabel('확률', fontsize=11)
            ax.grid(axis='x', alpha=0.3)
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()

            # 유사 민원
            st.markdown("### 📑 유사 민원 Top 5")
            for _, row in similar_posts.iterrows():
                sim_score = similarities[row.name] if row.name < len(similarities) else 0
                with st.expander(f"[{row['카테고리']}] {str(row['제목'])[:40].replace(chr(10),'')} (유사도 {sim_score:.1%}, 추천 {int(row['추천수'])}개)"):
                    st.write(str(row['원문텍스트'])[:300])

        else:
            st.warning("불만글을 입력해주세요!")
