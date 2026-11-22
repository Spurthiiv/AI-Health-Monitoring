# AI-Based Health Monitoring Smart T-Shirt

A Python health-data classifier that flags abnormal vital signs using the K-Nearest Neighbours (KNN) algorithm, built as part of a smart T-shirt health monitoring project.

## Overview
The classifier takes heart rate, body temperature and SpO2 as input and labels each reading as normal or abnormal. It is written in plain Python with no external libraries.

## Data
A simulated dataset of 100 readings, generated with a fixed random seed. A reading is labelled abnormal if heart rate > 95, temperature > 100.5, or SpO2 < 92.

## How it works
1. Generate the simulated dataset.
2. Split it into training and test sets (80/20).
3. Classify each test reading with KNN and measure accuracy.

## How to run
python main.py (Python 3, no extra libraries needed)

## Results
Accuracy is printed when the program runs.

## Note
Evaluated on simulated data, not clinical data.

## Author
Spoorthi V
