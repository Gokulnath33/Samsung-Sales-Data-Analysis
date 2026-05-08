import streamlit as st
import pandas as pd
import numpy as np
import sys, os, warnings
import plotly.express as px
import plotly.graph_objects as go
warnings.filterwarnings('ignore')
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Page Configuration
st.set_page_config(
    page_title="Samsung Business Intelligence Dashboard",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── LIGHT THEME CSS ─────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&family=Outfit:wght@300;400;500;600;700;800&display=swap');

:root {
    --primary: #034EA2;
    --accent: #007AFF;
    --bg-main: #FFFFFF;
    --bg-sidebar: #F8F9FA;
    --text-main: #1D1D1F;
    --text-muted: #6E6E73;
    --border: #E5E5E7;
}

html, body, [class*="css"] { 
    font-family: 'Inter', sans-serif !important; 
    color: var(--text-main) !important;
}

.stApp { background: var(--bg-main) !important; }

section[data-testid="stSidebar"] {
    background-color: var(--bg-sidebar) !important;
    border-right: 1px solid var(--border) !important;
}

h1, h2, h3, h4 { 
    font-family: 'Outfit', sans-serif !important; 
    color: var(--text-main) !important;
    font-weight: 700 !important;
}

.stButton > button {
    background: var(--primary) !important;
    color: white !important;
    border-radius: 8px !important;
    padding: 0.6rem 1.2rem !important;
    font-weight: 600 !important;
    transition: all 0.2s ease !important;
}

.metric-card {
    background: #FFFFFF;
    border: 1px solid var(--border);
    border-radius: 12px;
    padding: 1.25rem;
    box-shadow: 0 4px 6px rgba(0,0,0,0.02);
    text-align: center;
}

.hero-section {
    background: linear-gradient(135deg, #034EA2 0%, #007AFF 100%);
    border-radius: 16px;
    padding: 2.5rem;
    color: white !important;
    margin-bottom: 2rem;
}
.hero-section h1, .hero-section p { color: white !important; }

.capability-bar-container {
    height: 24px;
    background: #E5E5E7;
    border-radius: 12px;
    overflow: hidden;
    position: relative;
    margin: 1.5rem 0;
}
.capability-bar-fill {
    height: 100%;
    transition: width 0.5s ease-in-out;
}
</style>
""", unsafe_allow_html=True)

# ── UI UTILITIES ──────────────────────────────────────────────────────────────
def hero(title, sub):
    st.markdown(f'<div class="hero-section"><h1>{title}</h1><p>{sub}</p></div>', unsafe_allow_html=True)

def kpi_card(val, lbl, delta=None):
    d_html = f'<div style="color: {"#28A745" if "+" in delta else "#DC3545"}; font-size: 0.8rem; font-weight: 600;">{delta}</div>' if delta else ''
    st.markdown(f"""
    <div class="metric-card">
        <div style="font-size: 0.75rem; color: #6E6E73; text-transform: uppercase; margin-bottom: 0.5rem; letter-spacing: 0.05em;">{lbl}</div>
        <div style="font-size: 1.8rem; font-weight: 800; color: #034EA2;">{val}</div>
        {d_html}
    </div>""", unsafe_allow_html=True)

# ── DATA LOADING ─────────────────────────────────────────────────────────────
@st.cache_data
def get_data():
    from modules.preprocess import load_data
    return load_data()

# ── SIDEBAR ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown('<div style="padding: 1rem 0;"><div style="font-family: Outfit, sans-serif; font-size: 1.5rem; font-weight: 800; color: #034EA2;">Samsung</div><div style="font-size: 0.75rem; color: #6E6E73; text-transform: uppercase;">Intelligence Platform</div></div>', unsafe_allow_html=True)
    st.markdown("<hr>", unsafe_allow_html=True)
    page = st.radio("Navigation", ["Dashboard Home", "Capability Analysis", "Cohort Intelligence", "Global Analytics"])
    st.markdown("<hr>", unsafe_allow_html=True)

# 🏠 HOME
if page == "Dashboard Home":
    hero("Executive Performance Summary", "Real-time analysis of global mobile device market dynamics.")
    df = get_data()
    if df is not None:
        k1, k2, k3, k4 = st.columns(4)
        with k1: kpi_card(f"${df['Revenue($)'].sum()/1e9:.1f}B", "Global Revenue", "+12.4%")
        with k2: kpi_card(f"{df['Units_Sold'].sum()/1e6:.1f}M", "Units Sold", "+5.2%")
        with k3: kpi_card(f"{(df['5G_Capability']=='Yes').mean()*100:.1f}%", "5G Mix", "+8.1%")
        with k4: kpi_card(f"{df['Market_Share(%)'].mean():.1f}%", "Avg Market Share", "+0.4%")
        st.markdown("<br>", unsafe_allow_html=True)
        st.subheader("Regional Revenue Performance")
        st.plotly_chart(px.bar(df.groupby('Region')['Revenue($)'].sum().reset_index(), x='Region', y='Revenue($)', color_discrete_sequence=['#034EA2']), use_container_width=True)

# 🤖 CAPABILITY ANALYSIS
elif page == "Capability Analysis":
    hero("Model Capability Intelligence", "Interactive assessment of device connectivity potential.")
    c1, c2 = st.columns([1, 1.2])
    with c1:
        st.subheader("Intelligence Parameters")
        u = st.slider("Units Sold", 0, 100000, 35000)
        r = st.slider("Revenue ($)", 0, 100000000, 25000000)
        m = st.slider("Market Share (%)", 0.0, 20.0, 4.5)
        s = st.slider("Target 5G Speed (Mbps)", 0, 1000, 450)
        p = st.slider("Consumer Preference (%)", 0.0, 100.0, 72.0)
        su = st.slider("Regional Subscribers (M)", 0.0, 100.0, 22.5)
        co = st.slider("Coverage Index (%)", 0.0, 100.0, 58.0)
        
    with c2:
        st.subheader("Capability Assessment")
        from modules.predict import predict_5g
        label, proba, pred = predict_5g(u, r, m, s, p, su, co)
        
        # The new ensemble model provides granular percentages (e.g. 74.3%) 
        # that update with every single movement of the sliders.
        cap_index = proba[1] * 100
        
        # UI Feedback
        color = "#034EA2" if cap_index > 70 else ("#F59E0B" if cap_index > 30 else "#6E6E73")
        status_text = "High Performance 5G" if cap_index > 70 else ("Transitionary Capability" if cap_index > 30 else "Legacy connectivity")
        
        st.markdown(f"""
            <div style="padding: 2rem; border: 1px solid var(--border); border-radius: 12px; text-align: center; background: white;">
                <div style="font-size: 0.8rem; color: #8E8E93; text-transform: uppercase; letter-spacing: 0.1em;">Precision Capability Score</div>
                <div style="font-size: 4rem; font-weight: 800; color: {color}; margin: 0.5rem 0; font-family: 'Outfit', sans-serif;">{cap_index:.1f}%</div>
                <div style="font-size: 1.1rem; color: {color}CC; font-weight: 600;">{status_text}</div>
            </div>
        """, unsafe_allow_html=True)
        
        # Capability Spectrum Bar (Like a health bar but for business)
        st.markdown(f"""
            <div style="margin-top: 2rem;">
                <div style="display: flex; justify-content: space-between; font-size: 0.75rem; color: #8E8E93; margin-bottom: 0.5rem;">
                    <span>LEGACY</span>
                    <span>MID-TIER</span>
                    <span>ULTRA-CAPABLE</span>
                </div>
                <div class="capability-bar-container">
                    <div class="capability-bar-fill" style="width: {cap_index}%; background: linear-gradient(90deg, #6E6E73 0%, #F59E0B 50%, #034EA2 100%);"></div>
                </div>
            </div>
        """, unsafe_allow_html=True)
        
        st.info(f"This device model is currently operating at {cap_index:.1f}% of its maximum 5G potential based on regional market adoption and revenue scaling.")

# 👥 COHORT INTELLIGENCE
elif page == "Cohort Intelligence":
    hero("Device Cohort Intelligence", "Strategic identification of market segments using advanced clustering logic.")
    c1, c2 = st.columns([1, 2.5])
    with c1:
        st.subheader("Segmentation Control")
        st.info("The intelligence engine is processing global device records to find natural groupings.")
        k = st.select_slider("Target Segments", options=[2, 3, 4, 5], value=3)
        process = st.button("Generate Market Cohorts")
    
    with c2:
        if process:
            from modules.cluster import run_segmentation
            res_df, n_c, labels, sil, exp, insights = run_segmentation(k)
            st.markdown("<br>", unsafe_allow_html=True)
            cols = st.columns(len(insights))
            for i, (cid, info) in enumerate(insights.items()):
                with cols[i]:
                    st.markdown(f"""<div style="padding: 1rem; border: 1px solid var(--border); border-radius: 8px; border-top: 3px solid #034EA2; background: white;">
                        <div style="font-weight: 700; color: #1D1D1F; font-size: 0.9rem;">{info['label']}</div>
                        <div style="font-size: 1.2rem; font-weight: 800; color: #034EA2; margin: 0.3rem 0;">{info['n']} Models</div>
                        <div style="font-size: 0.7rem; color: #8E8E93;">Adoption Index: {info['pct_5g']:.0f}</div></div>""", unsafe_allow_html=True)
            fig = px.scatter(res_df, x='PC1', y='PC2', color='Cluster', title="Strategic Market Mapping", color_discrete_sequence=px.colors.qualitative.Prism)
            fig.update_layout(plot_bgcolor='white', margin=dict(l=0, r=0, t=40, b=0))
            st.plotly_chart(fig, use_container_width=True)

# 📊 GLOBAL ANALYTICS
elif page == "Global Analytics":
    hero("Global Market Analytics", "In-depth correlation analysis and corporate benchmarking.")
    df = get_data()
    if df is not None:
        tab1, tab2 = st.tabs(["Correlations", "Distribution"])
        with tab1:
            st.plotly_chart(px.scatter(df, x='Units_Sold', y='Revenue($)', color='5G_Capability'), use_container_width=True)
        with tab2:
            st.plotly_chart(px.histogram(df, x='Revenue($)', color_discrete_sequence=['#034EA2']), use_container_width=True)
