from tensorflow.keras.datasets import imdb
import csv

# Parameters
vocab_size = 2000
max_len = 100

# Load IMDb dataset
(X_train, y_train), (X_test, y_test) = imdb.load_data(num_words=vocab_size)

# Pad sequences
X_train = [x[:max_len] + [0] * (max_len - len(x)) for x in X_train]
X_test = [x[:max_len] + [0] * (max_len - len(x)) for x in X_test]

# Save to CSV
with open("train_data.csv", "w", newline="") as file:
    writer = csv.writer(file)
    for x, y in zip(X_train, y_train):
        writer.writerow(x + [y])  # 마지막 열에 레이블 추가

with open("test_data.csv", "w", newline="") as file:
    writer = csv.writer(file)
    for x, y in zip(X_test, y_test):
        writer.writerow(x + [y])  # 마지막 열에 레이블 추가