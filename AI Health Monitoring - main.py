# AI Health Monitoring - KNN Classifier
# Candidate: Spoorthi V | gowdaspoorthi888@gmail.com

import random
import math

def generate_dataset(n=100):
    random.seed(42)
    data = []
    for _ in range(n):
        heart_rate = random.randint(55, 110)
        temperature = round(random.uniform(96.0, 103.0), 1)
        spo2 = random.randint(88, 100)
        if heart_rate > 95 or temperature > 100.5 or spo2 < 92:
            label = 1
        else:
            label = 0
        data.append([heart_rate, temperature, spo2, label])
    return data

def split_data(data, ratio=0.8):
    split = int(len(data) * ratio)
    return data[:split], data[split:]

def euclidean_distance(a, b):
    return math.sqrt(sum((a[i] - b[i])**2 for i in range(len(a)-1)))

def knn_predict(train, test_point, k=3):
    distances = []
    for row in train:
        dist = euclidean_distance(row, test_point)
        distances.append((dist, row[-1]))
    distances.sort(key=lambda x: x[0])
    neighbors = distances[:k]
    votes = [n[1] for n in neighbors]
    return max(set(votes), key=votes.count)

def evaluate(train, test):
    correct = 0
    for row in test:
        prediction = knn_predict(train, row)
        if prediction == row[-1]:
            correct += 1
    return round((correct / len(test)) * 100, 2)

def predict_patient(train, heart_rate, temperature, spo2):
    test_point = [heart_rate, temperature, spo2, 0]
    result = knn_predict(train, test_point)
    status = "ABNORMAL - Needs Attention" if result == 1 else "NORMAL - Healthy"
    print(f"Heart Rate: {heart_rate} | Temp: {temperature} | SpO2: {spo2} | {status}")

dataset = generate_dataset(100)
train_data, test_data = split_data(dataset)
accuracy = evaluate(train_data, test_data)

print("=" * 50)
print("AI Health Monitoring - KNN Classifier")
print("Candidate: Spoorthi V")
print(f"Model Accuracy: {accuracy}%")
print("=" * 50)
predict_patient(train_data, 98, 101.2, 90)
predict_patient(train_data, 72, 98.6, 98)
predict_patient(train_data, 105, 99.1, 91)
predict_patient(train_data, 68, 97.8, 99)