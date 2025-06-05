
# 🧠 AI Task Classifier – Multi-Output Neural Network (from Scratch)

[![Python](https://img.shields.io/badge/Python-3.10+-blue?logo=python)](https://www.python.org/)
[![NumPy](https://img.shields.io/badge/Dependencies-Numpy-lightgrey?logo=numpy)](https://numpy.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

A simple, interpretable **multi-output neural network** built entirely from scratch using **Python + NumPy** to classify tasks by both:

- 📌 **Priority**: Low / Medium / High  
- 📂 **Category**: Work / Personal / Health / etc.

This project simulates how a real-world task manager might prioritize and organize to-do items using basic AI methods.

---

## 📁 Project Structure
├── preprocessing.py # Text cleaning & Bag-of-Words feature generation
├── final.py # Neural network model training & evaluation
├── synthetic_dataset.csv # Input task list (task, category, priority, due_date)
├── processed_bow.csv # BoW features and labels (ready for model input)
├── README.md # Project overview and usage

---

## 🚀 Getting Started

### 1. 📊 Preprocess Your Data
Convert task descriptions into Bag-of-Words format:
cmd: python3 preprocessing.py

### 2. Train the Multi-Output Neural Network
cmd : python3 final.py

### 3. To view front end run:
python3 frontend.py

###💡 Key Features
🎯 Multi-task output: Category + Priority

🧹 Custom Preprocessing: Basic NLP and BoW (no external libraries)

🧮 Manual Neural Network:

ReLU + Softmax

Cross-entropy loss

Backpropagation from scratch

🪶 Lightweight: Runs on CPU in < 10MB RAM

###🧠 Concepts Used
Bag-of-Words text encoding

Multi-output feedforward network

Manual forward & backward pass

Accuracy tracking per output

Simple validation/testing split

###🔬 Limitations
Bag-of-Words model ignores context and word order

Small, synthetic dataset (expand to improve!)

No stemming, lemmatization, or tokenization

No saving/loading of trained model yet

📌 Future Improvements
🔄 Add TF-IDF or word embeddings

📈 Use real task datasets

🌐 Web frontend for task input & live classification

🤖 Add LSTM, Transformer or pre-trained BERT (still from scratch)
