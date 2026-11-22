# AI-Based Health Monitoring Smart T-Shirt

A wearable-sensor health monitoring project that detects abnormal health patterns using machine learning.

## Overview
Sensor data from a wearable T-shirt is collected through an Arduino data-acquisition layer. A Python ML pipeline classifies the readings and flags abnormal patterns, reaching 90%+ accuracy.

## Tech Stack
- Python (data processing and the ML classifier)
- Arduino (sensor data acquisition)
- Machine Learning (classification of normal vs abnormal readings)

## Sensors / Data
Wearable sensors connected to an Arduino collect health readings, which are then passed to the Python model for classification.

## How it works
1. Sensors send readings through the Arduino.
2. The data is cleaned and prepared in Python.
3. The ML model classifies each reading as normal or abnormal.

## How to run
Run main.py with Python.

## Results
90%+ accuracy in detecting abnormal health patterns.

## Author
Spoorthi V
