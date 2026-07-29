import streamlit as st
import joblib
import pandas as pd
import nltk
from nltk import pos_tag, word_tokenize

# Safe download of required NLTK resources
for resource in ['punkt', 'punkt_tab', 'averaged_perceptron_tagger', 'averaged_perceptron_tagger_eng']:
    try:
        nltk.download(resource, quiet=True)
    except Exception:
        pass

# Page Configuration
st.set_page_config(
    page_title="Linguistic Fake News Detector",
    page_icon="🛡️",
    layout="centered",
    initial_sidebar_state="collapsed"
)

# Custom Styling (Glassmorphism & Clean Typography)
st.markdown("""
<style>
    /* Global Container Adjustments */
    .main .block-container {
        padding-top: 2rem;
        padding-bottom: 3rem;
        max-width: 780px;
    }
    
    /* Header Card */
    .header-card {
        background: linear-gradient(135deg, #1f2937 0%, #111827 100%);
        padding: 2.2rem 2rem;
        border-radius: 16px;
        color: white;
        text-align: center;
        margin-bottom: 2rem;
        box-shadow: 0 10px 25px -5px rgba(0, 0, 0, 0.1);
    }
    .header-title {
        font-size: 2rem;
        font-weight: 700;
        margin-bottom: 0.5rem;
        color: #ffffff;
    }
    .header-subtitle {
        color: #9ca3af;
        font-size: 0.95rem;
        line-height: 1.5;
    }
    
    /* Input Box Focus */
    .stTextArea textarea {
        border-radius: 12px !important;
        border: 1.5px solid #e5e7eb !important;
        font-size: 1rem !important;
    }
    
    /* Result Cards */
    .result-card-reliable {
        background: rgba(16, 185, 129, 0.08);
        border: 1.5px solid #10b981;
        padding: 1.5rem;
        border-radius: 14px;
        margin-top: 1.5rem;
    }
    .result-card-unreliable {
        background: rgba(239, 68, 68, 0.08);
        border: 1.5px solid #ef4444;
        padding: 1.5rem;
        border-radius: 14px;
        margin-top: 1.5rem;
    }
    
    /* Badge styling */
    .badge-reliable {
        background-color: #10b981;
        color: white;
        padding: 4px 12px;
        border-radius: 20px;
        font-weight: 600;
        font-size: 0.85rem;
    }
    .badge-unreliable {
        background-color: #ef4444;
        color: white;
        padding: 4px 12px;
        border-radius: 20px;
        font-weight: 600;
        font-size: 0.85rem;
    }
</style>
""", unsafe_allow_html=True)

# Header Section
st.markdown("""
<div class="header-card">
    <div class="header-title">🛡️ TruthScope AI</div>
    <div class="header-subtitle">Linguistic Stylometry & Deception Detection Pipeline</div>
</div>
""", unsafe_allow_html=True)

# Assets Loading
@st.cache_resource
def load_assets():
    model = joblib.load('fake_news_model.pkl')
    features = joblib.load('feature_names.pkl')
    return model, features

try:
    model, feature_names = load_assets()
except Exception as e:
    st.error("Failed to load model assets. Ensure `fake_news_model.pkl` and `feature_names.pkl` are in the repository root.")
    st.stop()

# Interactive Form
st.markdown("### 📝 Enter Statement to Analyze")

# Preset Sample Selection
if 'input_text' not in st.session_state:
    st.session_state.input_text = ""

st.caption("Try one of these sample statements:")
col_ex1, col_ex2 = st.columns(2)

if col_ex1.button("📄 Sample 1: Formal/Reliable", use_container_width=True):
    st.session_state.input_text = "The committee published the quarterly financial oversight report following the bipartisan audit."

if col_ex2.button("⚠️ Sample 2: Sensational/Unreliable", use_container_width=True):
    st.session_state.input_text = "Look, everybody knows these corrupt politicians are lying constantly and ruining our economy completely!"

user_text = st.text_area(
    label="Statement Input",
    value=st.session_state.input_text,
    placeholder="e.g., The committee published the quarterly financial oversight report following the bipartisan audit...",
    height=130,
    label_visibility="collapsed"
)

col1, col2, col3 = st.columns([1, 2, 1])
with col2:
    analyze_btn = st.button("🚀 Run Linguistic Analysis", use_container_width=True, type="primary")

if analyze_btn:
    if not user_text.strip():
        st.warning("⚠️ Please provide a statement to evaluate.")
    else:
        # 1. Processing Input
        tokens = word_tokenize(user_text)
        tags = pos_tag(tokens)
        tag_counts = pd.Series([tag for _, tag in tags]).value_counts().to_dict()
        
        feature_dict = {col: tag_counts.get(col, 0) for col in feature_names}
        input_df = pd.DataFrame([feature_dict])
        
        # 2. Model Inference
        prediction = model.predict(input_df)[0]
        probabilities = model.predict_proba(input_df)[0]
        
        classes = list(model.classes_)
        reliable_idx = classes.index('Reliable') if 'Reliable' in classes else 0
        unreliable_idx = classes.index('Unreliable') if 'Unreliable' in classes else 1
        
        rel_prob = probabilities[reliable_idx] * 100
        unrel_prob = probabilities[unreliable_idx] * 100

        # 3. Presenting Results
        st.markdown("---")
        st.markdown("### 📊 Classification Results")

        if prediction == "Reliable":
            st.markdown(f"""
            <div class="result-card-reliable">
                <span class="badge-reliable">CLASSIFIED RELIABLE</span>
                <h2 style="color: #065f46; margin-top: 10px; margin-bottom: 5px;">✅ Statement Appears Reliable</h2>
                <p style="color: #047857; margin: 0;">Structural syntax aligns with objective baseline patterns.</p>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class="result-card-unreliable">
                <span class="badge-unreliable">CLASSIFIED UNRELIABLE</span>
                <h2 style="color: #991b1b; margin-top: 10px; margin-bottom: 5px;">🚨 Statement Appears Unreliable</h2>
                <p style="color: #b91c1c; margin: 0;">Linguistic fingerprints indicate subjective/emotive framing patterns.</p>
            </div>
            """, unsafe_allow_html=True)

        # Confidence Metrics
        m1, m2 = st.columns(2)
        m1.metric("Reliable Likelihood", f"{rel_prob:.1f}%")
        m2.metric("Unreliable Likelihood", f"{unrel_prob:.1f}%")

        # Feature Drawer
        with st.expander("🔍 Inspection Drawer: Extracted POS Features"):
            st.write("Live Part-of-Speech counts parsed from your statement:")
            
            # Show non-zero features as table
            extracted_features = {k: v for k, v in tag_counts.items() if v > 0}
            if extracted_features:
                feats_df = pd.DataFrame(list(extracted_features.items()), columns=["POS Tag", "Count"])
                st.dataframe(feats_df, use_container_width=True)
            else:
                st.write("No standard POS tags extracted.")

            st.caption("Model evaluates structural syntax density (Nouns, Verbs, Adverbs) independent of semantic facts.")
