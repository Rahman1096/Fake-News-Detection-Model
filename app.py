import streamlit as st
import joblib
import pandas as pd
import nltk
from nltk import pos_tag, word_tokenize

# Download necessary NLTK data for live POS tagging
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')

# Page Config
st.set_page_config(page_title="Linguistic Fake News Detector", page_icon="🔍")

st.title("🔍 Linguistic Fake News Detector")
st.write("This tool analyzes the **grammatical structure** of a statement to predict its reliability.")

# Load Model & Feature Names
@st.cache_resource
def load_assets():
    model = joblib.load('fake_news_model.pkl')
    features = joblib.load('feature_names.pkl')
    return model, features

model, feature_names = load_assets()

# User Input
user_text = st.text_area("Enter a political statement or news sentence:", "")

if st.button("Analyze Reliability"):
    if user_text.strip() == "":
        st.warning("Please enter text to analyze.")
    else:
        # Extract POS Tag features live from input text
        tokens = word_tokenize(user_text)
        tags = pos_tag(tokens)
        
        # Count POS tags
        tag_counts = pd.Series([tag for _, tag in tags]).value_counts().to_dict()
        
        # Build feature vector matching training columns
        feature_dict = {col: tag_counts.get(col, 0) for col in feature_names}
        input_df = pd.DataFrame([feature_dict])
        
        # Predict
        prediction = model.predict(input_df)[0]
        probabilities = model.predict_proba(input_df)[0]
        confidence = max(probabilities) * 100

        # Display Result
        st.subheader("Analysis Result:")
        if prediction == "Reliable":
            st.success(f"✅ Predicted Category: **Reliable** ({confidence:.1f}% confidence)")
        else:
            st.error(f"🚨 Predicted Category: **Unreliable** ({confidence:.1f}% confidence)")

        st.info("Note: This prediction is generated purely from Part-of-Speech (POS) linguistic stylometry patterns.")
