import pickle
import sys
import keras
import tensorflow.keras.preprocessing as tf_preprocessing

# Map old keras paths to tensorflow.keras equivalents
sys.modules['keras.src'] = keras
sys.modules['keras.src.preprocessing'] = tf_preprocessing
sys.modules['keras.src.preprocessing.text'] = tf_preprocessing.text
sys.modules['keras.src.legacy'] = keras

# Load old tokenizer
with open("tokenizer.pickle", "rb") as f:
    tokenizer = pickle.load(f)

# Save fixed tokenizer
with open("tokenizer_fixed.pickle", "wb") as f:
    pickle.dump(tokenizer, f)

print("Tokenizer fixed successfully")