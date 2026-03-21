




import sys
import types
import streamlit as st
import pickle
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing import text, sequence

# --- 1. VITAL FIX: Trick pickle into finding Keras 3 paths in Keras 2 ---
keras_src = types.ModuleType("keras.src")
sys.modules["keras.src"] = keras_src
keras_src_legacy = types.ModuleType("keras.src.legacy")
keras_src_legacy.__path__ = []
sys.modules["keras.src.legacy"] = keras_src_legacy
keras_src_legacy_pre = types.ModuleType("keras.src.legacy.preprocessing")
keras_src_legacy_pre.__path__ = []
sys.modules["keras.src.legacy.preprocessing"] = keras_src_legacy_pre
sys.modules["keras.src.legacy.preprocessing.text"] = text
sys.modules["keras.src.legacy.preprocessing.sequence"] = sequence

# --- 2. UI Configuration ---
st.set_page_config(page_title="Next Word AI Pro", page_icon="🤖", layout="centered")

st.markdown("""
    <style>
    .main { background-color: #f8f9fa; }
    .stTextInput > div > div > input { border-radius: 8px; }
    .prediction-box {
        padding: 25px;
        background-color: #ffffff;
        border-radius: 12px;
        border-left: 5px solid #007bff;
        box-shadow: 0 4px 12px rgba(0,0,0,0.05);
        margin-top: 20px;
    }
    .result-text { color: #1e293b; font-size: 1.2rem; line-height: 1.6; }
    </style>
    """, unsafe_allow_html=True)

# --- 3. Loading Assets ---
@st.cache_resource
def load_assets():
    with open("tokenizer.pickle", "rb") as f:
        tokenizer = pickle.load(f)
    with open("max_len.pickle", "rb") as f:
        max_len = pickle.load(f)
    # Ensure you ran the 'fix_model.py' script on lstm_model.h5 first!
    model = load_model("lstm_model.h5", compile=False)
    return tokenizer, max_len, model

try:
    tokenizer, max_len, model = load_assets()
except Exception as e:
    st.error(f"Error loading model assets: {e}")
    st.stop()

# --- 4. Prediction Logic ---
def generate_text(seed_text, next_words=5):
    """Generates a sequence of words by looping the prediction."""
    output_text = seed_text
    
    for _ in range(next_words):
        # Tokenize and Pad
        token_list = tokenizer.texts_to_sequences([output_text])[0]
        # Using 745 to match your model's specific requirement
        token_list = pad_sequences([token_list], maxlen=745, padding='pre')
        
        # Predict
        predicted_probs = model.predict(token_list, verbose=0)
        predicted_index = np.argmax(predicted_probs, axis=-1)[0]
        
        # Convert index to word
        output_word = ""
        for word, index in tokenizer.word_index.items():
            if index == predicted_index:
                output_word = word
                break
        
        if not output_word:
            break
            
        output_text += " " + output_word
        
    return output_text

# --- 5. Main UI ---
st.title("🤖 AI Sentence Generator")
st.markdown("Enter a few words and the LSTM model will attempt to complete the thought.")

with st.sidebar:
    st.header("Control Panel")
    word_count = st.slider("Words to generate", 1, 20, 5)
    st.write("---")
    st.caption("Model: LSTM (Keras 2.15)")
    st.info("The model predicts one word at a time, feeding its own prediction back into the input.")

# Input Area
user_input = st.text_input("Start your sentence:", placeholder="e.g., The future of technology is", key="user_input")

col1, col2 = st.columns([1, 4])
with col1:
    generate_btn = st.button("✨ Generate")

if generate_btn:
    if user_input.strip() != "":
        with st.spinner('Generating text...'):
            try:
                result = generate_text(user_input, next_words=word_count)
                
                st.markdown(f"""
                    <div class="prediction-box">
                        <p style="color: #64748b; font-size: 0.9rem; margin-bottom: 10px;">GENERATED SEQUENCE:</p>
                        <div class="result-text">
                            {result}
                        </div>
                    </div>
                """, unsafe_allow_html=True)
                
                st.button("📋 Copy to clipboard", on_click=lambda: st.write("Copied! (This is a placeholder)"))
                
            except Exception as e:
                st.error(f"Generation failed: {e}")
    else:
        st.warning("Please enter at least one word to start.")

# Footer
st.markdown("---")
st.caption("Powered by Streamlit and TensorFlow LSTM")