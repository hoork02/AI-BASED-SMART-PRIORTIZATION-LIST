# preprocess.py
import csv
import numpy as np
from datetime import datetime

# Define encoding dictionaries
category_map = {}
priority_map = {'Low': 0, 'Medium': 1, 'High': 2}

def tokenize(text):
    return text.lower().replace(',', ' ').split()

def build_vocab(tasks):
    vocab = {}
    for task in tasks:
        for word in tokenize(task):
            if word not in vocab:
                vocab[word] = len(vocab)
    return vocab

def vectorize_task(task, vocab):
    vec = [0] * len(vocab)
    for word in tokenize(task):
        if word in vocab:
            vec[vocab[word]] = 1
    return vec

def normalize_date(due_date):
    target_date = datetime.strptime(due_date, "%Y-%m-%d")
    days_until_due = (target_date - datetime.now()).days
    return days_until_due / 365.0  # normalize to year

def preprocess(filepath):
    tasks, categories, priorities, due_dates = [], [], [], []

    with open(filepath, 'r') as f:
        reader = csv.reader(f)
        next(reader)  # skip header
        for row in reader:
            task, category, priority, due_date = row
            tasks.append(task)
            categories.append(category)
            priorities.append(priority)
            due_dates.append(due_date)

    # Build vocab from tasks
    vocab = build_vocab(tasks)

    # Encode category
    unique_categories = list(set(categories))
    for idx, cat in enumerate(unique_categories):
        category_map[cat] = idx

    features = []
    labels = []

    for i in range(len(tasks)):
        task_vec = vectorize_task(tasks[i], vocab)
        category_id = category_map[categories[i]] / len(category_map)
        priority_id = priority_map[priorities[i]] / 2  # normalize to [0,1]
        due_norm = normalize_date(due_dates[i])

        feat = task_vec + [category_id, priority_id, due_norm]
        label = [1.0 if priority_id == 1.0 or due_norm < 0.1 else 0.0]

        features.append(feat)
        labels.append(label)

    return np.array(features), np.array(labels), len(vocab), vocab

def preprocess_single(task, category, priority, due_date, vocab):
    task_vec = vectorize_task(task, vocab)
    category_id = category_map.get(category, 0) / len(category_map)
    priority_id = priority_map.get(priority, 0) / 2
    due_norm = normalize_date(due_date)
    feat = task_vec + [category_id, priority_id, due_norm]
    return np.array(feat).reshape(1, -1)

if __name__ == "__main__":
    X, Y, vocab_size, vocab = preprocess("synthetic_dataset.csv")
    print(f"Preprocessed {len(X)} samples with input size {X.shape[1]} and vocab size {vocab_size}")
