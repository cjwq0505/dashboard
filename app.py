import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.feature_extraction.text import TfidfVectorizer
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import io, os, re, json, pickle, subprocess

# ============================================================
# 한글 폰트 설정 (Streamlit Cloud 호환)
# ============================================================
FONT_PATH = None

def setup_font():
    global FONT_PATH
    candidates = [
        "/usr/share/fonts/truetype/nanum/NanumGothic.ttf",
        "/usr/share/fonts/truetype/nanum/NanumGothicBold.ttf",
        "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
        "C:/Windows/Fonts/malgun.ttf",
        "/System/Library/Fonts/AppleGothic.ttf",
    ]
    for p in candidates:
        if os.path.exists(p):
            FONT_PATH = p
            break
    if FONT_PATH is None:
        import glob
        results = glob.glob("/usr/share/fonts/**/Nanum*.ttf", recursive=True)
        if results:
            FONT_PATH = results[0]
    if FONT_PATH:
        fm.fontManager.addfont(FONT_PATH)
        font_name = fm.FontProperties(fname=FONT_PATH).get_name()
        plt.rcParams['font.family'] = font_name
        plt.rcParams['axes.unicode_minus'] = False
        return font_name
    return None

font_name = setup_font()

# ============================================================
# 설정
# ============================================================
st.set_page_config(page_title="에타 불만글 분석 대시보드", page_icon="📊", layout="wide")

CAT_COLORS = {
    '기숙사/주거': '#FF6B6B', '시설/환경': '#4ECDC4', '교통/이동': '#45B7D1',
    '학사/행정': '#FFA07A', '복지/음료': '#98D8C8'
}
PLOTLY_FONT = "Nanum Gothic, Malgun Gothic, NanumGothic, Noto Sans CJK KR, AppleGothic, sans-serif"

def plotly_korean(fig, **kwargs):
    fig.update_layout(font=dict(family=PLOTLY_FONT, size=13), **kwargs)
    return fig

# ============================================================
# 담당부서 & 내선번호
# ============================================================
DEPT_INFO = {
    '기숙사/주거': {
        'main': {'부서': '학생복지처 / 학생생활지원팀', '담당': '사생, 스카이카페', '내선': '1854'},
        'sub': [
            {'부서': '학생생활지원팀-시설지원', '담당': '자유관 (남)', '내선': '40+1***'},
            {'부서': '학생생활지원팀-시설지원', '담당': '정의관 (여)', '내선': '40+2***'},
            {'부서': '학생생활지원팀-시설지원', '담당': '진리관 (남)', '내선': '40+3***'},
            {'부서': '학생생활지원팀-시설지원', '담당': '진리관 (여)', '내선': '40+5***'},
            {'부서': '학생생활지원팀-시설지원', '담당': '미래관 (여)', '내선': '40+6***'},
            {'부서': '학생생활지원팀-시설지원', '담당': '미래관 (남)', '내선': '40+7***'},
        ]
    },
    '시설/환경': {
        'main': {'부서': '학생복지처 / 학생생활지원팀', '담당': '시설, 보수', '내선': '1855'},
        'sub': [
            {'부서': '사무처 / 시설안전관리팀', '담당': '전기, 통신, 승강기, 와이파이', '내선': '1066'},
            {'부서': '사무처 / 시설안전관리팀', '담당': '캠퍼스안전, 산업안전', '내선': '1068'},
            {'부서': '사무처 / 행정지원팀', '담당': '대관, 경비, 미화, 조경, 주차, 우편', '내선': '1043'},
            {'부서': '학생복지처 / 학생생활지원팀', '담당': '복지매장 관리', '내선': '1032'},
            {'부서': '학생복지처 / 학생생활지원팀', '담당': '휘트니스 센터', '내선': '1881'},
        ]
    },
    '교통/이동': {
        'main': {'부서': '사무처 / 행정지원팀', '담당': '차량, 주차, 불법주차', '내선': '1043'},
        'sub': [{'부서': '사무처 / 시설안전관리팀', '담당': '킥보드, 캠퍼스안전', '내선': '1068'}]
    },
    '학사/행정': {
        'main': {'부서': '교학처 / 교무학사팀', '담당': '교과과정, 교무기획, 성적', '내선': '1123'},
        'sub': [
            {'부서': '교학처 / 교무학사팀', '담당': '학적, 다전공, 소속변경', '내선': '1125'},
            {'부서': '교학처 / 교무학사팀', '담당': '수업, 학점교류, DSC, 프로젝트학기', '내선': '1127'},
            {'부서': '교학처 / 교양교육원', '담당': '교양과정 (글로벌잉글리시 등)', '내선': '1901'},
            {'부서': '총학생회', '담당': '학생자치 관련', '내선': '1960'},
        ]
    },
    '복지/음료': {
        'main': {'부서': '학생복지처 / 학생생활지원팀', '담당': '복지매장 관리', '내선': '1032'},
        'sub': [
            {'부서': '학생복지처 / 학생생활지원팀', '담당': '사생, 스카이카페', '내선': '1854'},
            {'부서': '총학생회', '담당': '학생자치 관련', '내선': '1960'},
        ]
    },
}

KEYWORD_DEPT_MAP = [
    (['와이파이','인터넷','통신','데이터','엘리베이터','승강기','전기'], '사무처 / 시설안전관리팀', '1066'),
    (['성적','교과','강사','교무'], '교학처 / 교무학사팀', '1123'),
    (['이중전공','복수전공','부전공','전과','소속변경','학적','다전공'], '교학처 / 교무학사팀', '1125'),
    (['수업','학점교류','수강신청','수강','정정','시스템','서버','로그인'], '교학처 / 교무학사팀', '1127'),
    (['교양','글로벌잉글리시','아카데믹잉글리시','교양교육'], '교학처 / 교양교육원', '1901'),
    (['기숙사','소음','시끄럽다','사생','룸메','통금'], '학생복지처 / 학생생활지원팀', '1854'),
    (['자유관'], '학생생활지원팀-시설지원 (자유관 남)', '40+1***'),
    (['정의관'], '학생생활지원팀-시설지원 (정의관 여)', '40+2***'),
    (['진리관'], '학생생활지원팀-시설지원 (진리관)', '40+3***/40+5***'),
    (['미래관'], '학생생활지원팀-시설지원 (미래관)', '40+6***/40+7***'),
    (['헬스','운동','휘트니스','아이파크'], '학생복지처 / 학생생활지원팀', '1881'),
    (['담배','냄새','흡연'], '학생복지처 · 행정지원팀', '1855 / 1043 / 1032'),
    (['킥보드','전동킥보드','역주행','인도'], '사무처 / 시설안전관리팀', '1068'),
    (['주차','차량','불법주차','청소','미화','조경','경비','택배','우편'], '사무처 / 행정지원팀', '1043'),
    (['등록금','총학생회','학생회'], '총학생회', '1960'),
    (['학생식당','학식','매점','음료','카페','이디야','브래댄코','자판기'], '학생복지처 / 학생생활지원팀', '1032'),
    (['보수','시설','냉방','에어컨','난방'], '학생복지처 / 학생생활지원팀', '1855'),
]

def find_specific_dept(text):
    matched = []
    for keywords, dept, phone in KEYWORD_DEPT_MAP:
        for kw in keywords:
            if kw in text:
                matched.append({'키워드': kw, '담당부서': dept, '내선번호': phone})
                break
    return matched

# ============================================================
# 데이터 로드
# ============================================================
@st.cache_data
def load_data():
    df = pd.read_csv('에브리타임_전처리완료.csv')
    df['추천수'] = pd.to_numeric(df['추천수'], errors='coerce').fillna(0).astype(int)
    # 복합어 재결합 (형태소 분석에서 쪼개진 것 보정)
    merge = {
        '학술 정보원':'학술정보원', '공공정책 대학':'공공정책대학',
        '공공 정책':'공공정책대학', '석원 경상관':'석원경상관',
        '전동 킥보드':'전동킥보드', '총학 생회':'총학생회',
        '수강 신청':'수강신청', '학생 식당':'학생식당',
        '학생 회관':'학생회관', '글로벌 잉글리시':'글로벌잉글리시',
        '근로 장학생':'근로장학생',
    }
    for k, v in merge.items():
        df['형태소'] = df['형태소'].str.replace(k, v, regex=False)
    # '학술' 단독 → 학술정보원, '보드' 단독 → 전동킥보드
    df['형태소'] = df['형태소'].str.replace(r'(?<!\S)학술(?!정보원)(?!\S)', '학술정보원', regex=True)
    df['형태소'] = df['형태소'].str.replace(r'(?<!\S)보드(?!\S)', '전동킥보드', regex=True)
    # 추가 불용어 제거 (욕설/감탄사/의미 약한 단어)
    extra_stop = ['시발','씨발','존나','미치다','제발','지금','진짜','레알',
                  '병신','새끼','지랄','개','좆','씹','대학',
                  '좋다','싫다','크다','작다','많다','적다','이렇다','그렇다',
                  '어디','언제','얼마나','정도','대체','아예','맨날','계속']
    def remove_stopwords(text):
        if pd.isna(text): return text
        words = str(text).split()
        return ' '.join([w for w in words if w not in extra_stop])
    df['형태소'] = df['형태소'].apply(remove_stopwords)
    return df

@st.cache_data
def load_w2v_json():
    with open('word2vec_results.json', 'r', encoding='utf-8') as f:
        return json.load(f)

@st.cache_resource
def load_classifier():
    with open('classifier.pkl', 'rb') as f:
        pipe = pickle.load(f)
    return pipe

df = load_data()
w2v_data = load_w2v_json()
clf_pipe = load_classifier()

# 분류기 정확도 계산
texts_all = df['형태소'].dropna()
labels_all = df.loc[texts_all.index, '카테고리']
clf_acc = clf_pipe.score(texts_all, labels_all)

# ============================================================
# 유틸
# ============================================================
def get_tfidf_top(texts, n=10):
    vec = TfidfVectorizer(max_features=500)
    mat = vec.fit_transform(texts)
    scores = dict(zip(vec.get_feature_names_out(), mat.mean(axis=0).A1))
    return dict(sorted(scores.items(), key=lambda x: -x[1])[:n])

def get_weighted_tfidf_top(texts, weights, n=10):
    vec = TfidfVectorizer(max_features=500)
    mat = vec.fit_transform(texts)
    w = np.array(weights).reshape(-1, 1)
    weighted = mat.multiply(w)
    scores = dict(zip(vec.get_feature_names_out(), weighted.mean(axis=0).A1))
    return dict(sorted(scores.items(), key=lambda x: -x[1])[:n])

def make_wordcloud(freq_dict):
    if not FONT_PATH or not os.path.exists(FONT_PATH):
        return None, "한글 폰트를 찾을 수 없습니다. packages.txt에 fonts-nanum을 추가해 주세요."
    try:
        wc = WordCloud(font_path=FONT_PATH, width=800, height=400,
                       background_color='white', colormap='viridis',
                       max_words=50, prefer_horizontal=0.7
                       ).generate_from_frequencies(freq_dict)
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.imshow(wc, interpolation='bilinear')
        ax.axis('off')
        buf = io.BytesIO()
        fig.savefig(buf, format='png', bbox_inches='tight', dpi=150)
        plt.close(fig)
        buf.seek(0)
        return buf, None
    except Exception as e:
        return None, str(e)

def preprocess_input(text):
    custom = {
        '긱사':'기숙사','미래관':'기숙사','자유관':'기숙사',
        '정의관':'기숙사','진리관':'기숙사','학식':'학생식당',
        '킥라니':'전동킥보드','킥보드':'전동킥보드',
        '학관':'학생회관','총학':'총학생회','수신':'수강신청',
        '와파':'와이파이','wifi':'와이파이','도서관':'학술정보원',
        '에어콘':'에어컨','글잉':'글로벌잉글리시',
    }
    text = re.sub(r'http\S+', '', str(text))
    text = re.sub(r'\\n|\n', ' ', text)
    text = re.sub(r'[^\w\s가-힣]', ' ', text)
    for k, v in custom.items():
        text = text.replace(k, v)
    text = re.sub(r'\s+', ' ', text).strip()
    from konlpy.tag import Okt
    okt = Okt()
    sw = set(['것','수','때','좀','진짜','너무','정말','그냥','이거',
              '사람','학교','우리','나','내','제','말','글',
              '하다','되다','있다','없다','이다','아니다',
              '주다','받다','오다','가다','나다','들다','내다',
              '보다','알다','모르다','싶다','같다','문제','시간'])
    tagged = okt.pos(text, stem=True)
    words = [w for w, p in tagged if p in ('Noun','Adjective','Verb') and len(w)>=2 and w not in sw]
    merge = {'학술 정보원':'학술정보원','전동 킥보드':'전동킥보드','수강 신청':'수강신청',
             '총학 생회':'총학생회','학생 식당':'학생식당','학생 회관':'학생회관'}
    result = ' '.join(words)
    for k, v in merge.items():
        result = result.replace(k, v)
    return result

# ============================================================
# 사이드바
# ============================================================
st.sidebar.title("📊 에타 불만글 분석")
st.sidebar.markdown("**고려대학교 세종캠퍼스**")
st.sidebar.markdown(f"분석 대상: **{len(df)}건** 불만 게시글")
st.sidebar.markdown("---")
page = st.sidebar.radio("페이지 선택",
    ["🏠 전체 현황", "🔍 키워드 분석", "📅 시기별 비교", "🤖 민원 자동 분류기"])
st.sidebar.markdown("---")
st.sidebar.caption("2024380535 최정민 · 2024380537 김민제")
st.sidebar.caption("데이터사이언스세미나 중간고사 프로젝트")

# ============================================================
# 페이지 1: 전체 현황
# ============================================================
if page == "🏠 전체 현황":
    st.title("🏠 에브리타임 불만글 전체 현황")
    st.markdown("고려대학교 세종캠퍼스 에브리타임에서 수집한 **370건**의 불만글을 분석합니다.")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("총 게시글", f"{len(df)}건")
    c2.metric("카테고리", f"{df['카테고리'].nunique()}개")
    c3.metric("평균 추천수", f"{df['추천수'].mean():.1f}")
    c4.metric("최다 추천", f"{df['추천수'].max()}")
    st.markdown("---")

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("카테고리별 게시글 수")
        cc = df['카테고리'].value_counts().reset_index()
        cc.columns = ['카테고리','건수']
        fig = px.bar(cc, x='카테고리', y='건수', color='카테고리',
                     color_discrete_map=CAT_COLORS, text='건수')
        plotly_korean(fig, showlegend=False, height=400)
        fig.update_traces(textposition='outside')
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.subheader("시기별 분포")
        pc = df['시기구분'].value_counts().reset_index()
        pc.columns = ['시기','건수']
        fig = px.pie(pc, values='건수', names='시기',
                     color='시기', color_discrete_map={'시험기간':'#5B8DEF','평소':'#FF6B6B'})
        plotly_korean(fig, height=400, title_text="시험기간 vs 평소")
        fig.update_traces(textinfo='label+percent', textfont_size=16, textfont_family=PLOTLY_FONT)
        st.plotly_chart(fig, use_container_width=True)

    st.subheader("시기별 × 카테고리 분포")
    pc2 = df.groupby(['시기구분','카테고리']).size().reset_index(name='건수')
    fig = px.bar(pc2, x='카테고리', y='건수', color='시기구분', barmode='group', text='건수',
                 color_discrete_map={'시험기간':'#5B8DEF','평소':'#FF6B6B'})
    plotly_korean(fig, height=400)
    fig.update_traces(textposition='outside')
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("🔥 추천수 Top 10 게시글")
    top10 = df.nlargest(10, '추천수')[['카테고리','제목','추천수','시기구분']].reset_index(drop=True)
    top10.index = top10.index + 1
    st.dataframe(top10, use_container_width=True)

# ============================================================
# 페이지 2: 키워드 분석
# ============================================================
elif page == "🔍 키워드 분석":
    st.title("🔍 카테고리별 키워드 분석")
    tab1, tab2, tab3 = st.tabs(["📊 TF-IDF Top 키워드", "☁️ 워드클라우드", "🔗 Word2Vec 연관어"])

    with tab1:
        cat = st.selectbox("카테고리 선택", df['카테고리'].unique())
        texts = df[df['카테고리']==cat]['형태소'].dropna()
        top = get_tfidf_top(texts, 15)
        fig = px.bar(x=list(top.values()), y=list(top.keys()), orientation='h',
                     color=list(top.values()), color_continuous_scale='viridis')
        plotly_korean(fig, yaxis={'categoryorder':'total ascending'}, height=500,
                      title=f"{cat} — TF-IDF 상위 15개 키워드", coloraxis_showscale=False)
        st.plotly_chart(fig, use_container_width=True)

    with tab2:
        wc_cat = st.selectbox("카테고리 선택 ", df['카테고리'].unique(), key='wc')
        texts_wc = df[df['카테고리']==wc_cat]['형태소'].dropna()
        freq = get_tfidf_top(texts_wc, 50)
        buf, err = make_wordcloud(freq)
        if buf:
            st.image(buf, caption=f"{wc_cat} 워드클라우드", use_container_width=True)
        else:
            st.warning(f"워드클라우드 생성 실패: {err}")
            fig = px.bar(x=list(freq.values())[:20], y=list(freq.keys())[:20],
                         orientation='h', color=list(freq.values())[:20], color_continuous_scale='viridis')
            plotly_korean(fig, yaxis={'categoryorder':'total ascending'}, height=500, coloraxis_showscale=False)
            st.plotly_chart(fig, use_container_width=True)

    with tab3:
        st.subheader("Word2Vec 핵심 키워드 연관어")
        available = list(w2v_data.keys())
        sel = st.selectbox("키워드 선택", available)
        if sel in w2v_data:
            sims = w2v_data[sel]
            sim_df = pd.DataFrame(sims).rename(columns={'word':'연관어','score':'유사도'})
            sim_df.index = sim_df.index + 1
            c1, c2 = st.columns(2)
            with c1:
                st.dataframe(sim_df, use_container_width=True)
            with c2:
                fig = px.bar(sim_df, x='유사도', y='연관어', orientation='h',
                             color='유사도', color_continuous_scale='tealgrn')
                plotly_korean(fig, yaxis={'categoryorder':'total ascending'}, height=400,
                              coloraxis_showscale=False, title=f"'{sel}' 연관어 유사도")
                st.plotly_chart(fig, use_container_width=True)

# ============================================================
# 페이지 3: 시기별 비교
# ============================================================
elif page == "📅 시기별 비교":
    st.title("📅 시험기간 vs 평소 비교 분석")
    tab1, tab2 = st.tabs(["📊 일반 TF-IDF 비교", "⚖️ 추천수 가중 TF-IDF 비교"])

    with tab1:
        st.subheader("시험기간 vs 평소 — TF-IDF 상위 키워드")
        c1, c2 = st.columns(2)
        for col, period in zip([c1,c2], ['시험기간','평소']):
            with col:
                t = df[df['시기구분']==period]['형태소'].dropna()
                top = get_tfidf_top(t, 10)
                fig = px.bar(x=list(top.values()), y=list(top.keys()), orientation='h',
                             color=list(top.values()),
                             color_continuous_scale='blues' if period=='시험기간' else 'oranges')
                plotly_korean(fig, yaxis={'categoryorder':'total ascending'}, height=400,
                              title=f"🔹 {period}", coloraxis_showscale=False,
                              xaxis_title='TF-IDF 점수', yaxis_title='')
                st.plotly_chart(fig, use_container_width=True)

        st.subheader("시기별 TF-IDF 점수 차이 (시험기간 − 평소)")
        et = get_tfidf_top(df[df['시기구분']=='시험기간']['형태소'].dropna(), 30)
        nt = get_tfidf_top(df[df['시기구분']=='평소']['형태소'].dropna(), 30)
        aw = set(list(et.keys())[:15] + list(nt.keys())[:15])
        dd = [{'키워드':w, '차이':et.get(w,0)-nt.get(w,0)} for w in aw]
        ddf = pd.DataFrame(dd).sort_values('차이')
        fig = px.bar(ddf, x='차이', y='키워드', orientation='h', color='차이',
                     color_continuous_scale='RdBu_r', color_continuous_midpoint=0)
        plotly_korean(fig, yaxis={'categoryorder':'total ascending'}, height=600,
                      title="시험기간에 더 많이 언급(→) vs 평소에 더 많이 언급(←)")
        st.plotly_chart(fig, use_container_width=True)
        st.info("💡 **핵심 발견**: 시험기간에는 자리·학술정보원·자판기·벌레 등 **공간 관련 불만**이, 평소에는 수강신청·서버·택배 등 **시스템 관련 불만**이 두드러집니다.")

    with tab2:
        st.subheader("일반 TF-IDF vs 추천수 가중 TF-IDF")
        st.markdown("추천수 가중치: `log(추천수 + 2)`를 적용하여 **많이 공감받은 글**에 더 높은 가중치를 부여합니다.")
        ps = st.radio("시기 선택", ['전체','시험기간','평소'], horizontal=True)
        sub = df if ps=='전체' else df[df['시기구분']==ps]
        tw = sub['형태소'].dropna()
        ww = np.log(sub.loc[tw.index, '추천수'] + 2)
        normal = get_tfidf_top(tw, 10)
        weighted = get_weighted_tfidf_top(tw, ww, 10)

        c1, c2 = st.columns(2)
        with c1:
            st.markdown("**📊 일반 TF-IDF**")
            fig = px.bar(x=list(normal.values()), y=list(normal.keys()),
                         orientation='h', color_discrete_sequence=['#5B8DEF'])
            plotly_korean(fig, yaxis={'categoryorder':'total ascending'}, height=400,
                          xaxis_title='TF-IDF 점수', yaxis_title='')
            st.plotly_chart(fig, use_container_width=True)
        with c2:
            st.markdown("**⚖️ 추천수 가중 TF-IDF**")
            fig = px.bar(x=list(weighted.values()), y=list(weighted.keys()),
                         orientation='h', color_discrete_sequence=['#FF6B6B'])
            plotly_korean(fig, yaxis={'categoryorder':'total ascending'}, height=400,
                          xaxis_title='가중 TF-IDF 점수', yaxis_title='')
            st.plotly_chart(fig, use_container_width=True)

        new_kw = set(weighted.keys()) - set(normal.keys())
        if new_kw:
            st.success(f"⚡ **가중치 적용 후 새로 등장한 키워드**: {', '.join(new_kw)}")
            st.markdown("→ '많이 올라오는 불만 ≠ 많이 공감받는 불만'임을 보여줍니다.")

# ============================================================
# 페이지 4: 민원 자동 분류기
# ============================================================
elif page == "🤖 민원 자동 분류기":
    st.title("🤖 민원 자동 분류 시스템")
    st.markdown(f"학습 정확도: **{clf_acc*100:.1f}%** (LogisticRegression + TF-IDF)")
    st.markdown("불만 사항을 입력하면 **카테고리 자동 예측** → **담당 부서 및 연락처 안내** → **유사 민원 검색**까지 제공합니다.")
    st.markdown("---")

    user_input = st.text_area("불만 사항을 입력하세요",
        placeholder="예: 기숙사 와이파이가 자꾸 끊겨서 과제를 못 하겠어요", height=120)

    if st.button("🔍 분석하기", type="primary") and user_input.strip():
        processed = preprocess_input(user_input)
        if processed.strip():
            pred = clf_pipe.predict([processed])[0]
            proba = clf_pipe.predict_proba([processed])[0]
            classes = clf_pipe.classes_

            st.markdown("---")
            st.subheader("📌 예측 결과")
            c1, c2 = st.columns(2)
            c1.metric("예측 카테고리", pred)
            c2.metric("예측 확률", f"{max(proba)*100:.1f}%")

            prob_df = pd.DataFrame({'카테고리':classes, '확률':proba}).sort_values('확률', ascending=True)
            fig = px.bar(prob_df, x='확률', y='카테고리', orientation='h',
                         color='확률', color_continuous_scale='viridis')
            plotly_korean(fig, height=280, coloraxis_showscale=False, title="카테고리별 예측 확률")
            st.plotly_chart(fig, use_container_width=True)

            # 담당 부서
            st.subheader("📞 담당 부서 및 연락처")
            st.caption("📱 일반 전화에서 걸 때: **044-860-** + 내선번호")
            specific = find_specific_dept(user_input + ' ' + processed)
            if specific:
                st.markdown("**🎯 입력 내용 기반 추천 연락처**")
                spec_df = pd.DataFrame(specific).drop_duplicates(subset=['담당부서','내선번호'])
                st.dataframe(spec_df, use_container_width=True, hide_index=True)

            dept_info = DEPT_INFO.get(pred, {})
            if dept_info:
                main = dept_info['main']
                st.markdown(f"**📋 [{pred}] 대표 연락처**")
                st.markdown(f"- **{main['부서']}** — {main['담당']} — 📞 내선 **{main['내선']}**")
                if dept_info.get('sub'):
                    with st.expander("📂 관련 부서 전체 보기"):
                        st.dataframe(pd.DataFrame(dept_info['sub']), use_container_width=True, hide_index=True)

            st.subheader("🔑 추출된 키워드")
            st.markdown(" · ".join([f"`{kw}`" for kw in processed.split()[:10]]))

            st.subheader("📋 유사 민원 Top 5")
            same = df[df['카테고리']==pred].copy()
            vec = TfidfVectorizer(max_features=500)
            cm = vec.fit_transform(same['형태소'].fillna(''))
            im = vec.transform([processed])
            from sklearn.metrics.pairwise import cosine_similarity
            same['유사도'] = cosine_similarity(im, cm).flatten()
            t5 = same.nlargest(5, '유사도')[['제목','카테고리','추천수','유사도']].reset_index(drop=True)
            t5.index = t5.index + 1
            t5['유사도'] = t5['유사도'].apply(lambda x: f"{x:.3f}")
            st.dataframe(t5, use_container_width=True)
        else:
            st.warning("유의미한 키워드를 추출하지 못했습니다. 좀 더 구체적으로 작성해 주세요.")

    st.markdown("---")
    with st.expander("📞 고려대학교 세종캠퍼스 민원 부서 전체 연락처"):
        st.caption("📱 일반 전화: **044-860-** + 내선번호")
        all_d = [
            {'부서':'사무처 / 시설안전관리팀','담당':'전기, 통신, 승강기, 와이파이','내선':'1066'},
            {'부서':'사무처 / 시설안전관리팀','담당':'캠퍼스안전, 산업안전, 킥보드','내선':'1068'},
            {'부서':'사무처 / 행정지원팀','담당':'대관, 경비, 미화, 조경, 차량, 주차, 우편','내선':'1043'},
            {'부서':'교학처 / 교무학사팀','담당':'교과과정, 교무기획, 강사, 성적','내선':'1123'},
            {'부서':'교학처 / 교무학사팀','담당':'학적, 다전공, 소속변경','내선':'1125'},
            {'부서':'교학처 / 교무학사팀','담당':'수업, 학점교류, DSC, 프로젝트학기','내선':'1127'},
            {'부서':'교학처 / 교양교육원','담당':'교양과정 (글로벌잉글리시 등)','내선':'1901'},
            {'부서':'학생복지처 / 학생생활지원팀','담당':'사생, 스카이카페','내선':'1854'},
            {'부서':'학생복지처 / 학생생활지원팀','담당':'시설, 보수','내선':'1855'},
            {'부서':'학생복지처 / 학생생활지원팀','담당':'휘트니스 센터','내선':'1881'},
            {'부서':'학생복지처 / 학생생활지원팀','담당':'복지매장 관리','내선':'1032'},
            {'부서':'총학생회','담당':'학생자치','내선':'1960'},
            {'부서':'학생생활지원팀-시설지원','담당':'자유관 (남)','내선':'40+1***'},
            {'부서':'학생생활지원팀-시설지원','담당':'정의관 (여)','내선':'40+2***'},
            {'부서':'학생생활지원팀-시설지원','담당':'진리관 (남)','내선':'40+3***'},
            {'부서':'학생생활지원팀-시설지원','담당':'진리관 (여)','내선':'40+5***'},
            {'부서':'학생생활지원팀-시설지원','담당':'미래관 (여)','내선':'40+6***'},
            {'부서':'학생생활지원팀-시설지원','담당':'미래관 (남)','내선':'40+7***'},
        ]
        st.dataframe(pd.DataFrame(all_d), use_container_width=True, hide_index=True)
