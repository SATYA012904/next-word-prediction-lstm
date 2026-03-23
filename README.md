# 🤖 Next Word Prediction using LSTM

A Deep Learning-based web application that predicts the next words in a sentence using LSTM (Long Short-Term Memory) networks. The model learns contextual relationships in text data and generates meaningful sequences through an interactive Streamlit interface.

---

## 📌 Project Overview

This project focuses on building a next word prediction system using LSTM, a type of Recurrent Neural Network (RNN) that is effective for sequence modeling. The system takes an initial input sentence and predicts the next sequence of words based on learned patterns from the training data.

---

## 🚀 Features

* Predicts next words based on input text
* Uses LSTM deep learning model for sequence prediction
* Generates multiple words dynamically
* Interactive UI built with Streamlit
* Real-time text generation

---

## 🧠 Model Details

* Model: LSTM (Long Short-Term Memory)
* Framework: TensorFlow / Keras
* Sequence generation using tokenization and padding
* Predicts one word at a time and feeds output back into input

---

## 📊 Workflow

1. Input sentence is tokenized
2. Sequence is padded to fixed length
3. LSTM model predicts next word
4. Predicted word is appended to input
5. Process repeats to generate multiple words

---

## 🛠️ Technologies Used

* Python
* TensorFlow / Keras
* NumPy
* Streamlit
* Pickle

---

## ⚙️ Project Structure

```id="h19g7a"
├── app.py
├── lstm_model.h5
├── tokenizer.pickle
├── max_len.pickle
├── fix_model.py
├── fix_tokenizer.py
├── Next Word Prediction using LSTM.ipynb
└── README.md
```

---

## ▶️ How to Run the Project

### 1. Clone the repository

```id="a1j3k9"
git clone https://github.com/SATYA012904/next-word-prediction-lstm.git
cd next-word-prediction-lstm
```

### 2. Install dependencies

```id="j3l8sd"
pip install -r requirements.txt
```

### 3. Fix model and tokenizer (Important)

```id="8sk2l1"
python fix_model.py
python fix_tokenizer.py
```

### 4. Run the application

```id="p0k29s"
streamlit run app.py
```

---

## 📈 Key Highlights

* Captures contextual relationships in text data
* Uses sequence modeling for prediction
* Handles compatibility issues between Keras versions
* Generates text dynamically word-by-word
* Clean and interactive UI

---

## ⚠️ Important Notes

* The model requires preprocessing fixes for compatibility (handled in `fix_model.py` and `fix_tokenizer.py`)  
* Input sequences are padded to a fixed length before prediction 
* Prediction is based on the highest probability word

---

## 📌 Future Improvements

* Improve model accuracy with larger datasets
* Add beam search for better text generation
* Deploy on cloud platforms
* Add top-k word suggestions instead of single output
* Improve UI with prediction history

---

## 👨‍💻 Author

Satyabrata Sahu
B.Tech Computer Science Student

---

## 📜 License

This project is for educational purposes only.
