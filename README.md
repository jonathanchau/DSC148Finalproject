# Predicting the Virality of Online News Articles

This repository contains the code and live web application for the DSC 148 Final Project.

## 🚀 Live Interactive Demo
**[Click here to access the live web application](INSERT_YOUR_STREAMLIT_URL_HERE)**

## Project Overview
This project predicts whether a published news article will achieve high social media engagement (60+ shares) across Facebook, Google+, and LinkedIn. 

The application utilizes a custom Natural Language Processing (NLP) pipeline and a class-balanced Logistic Regression model. It processes structured temporal/categorical metadata alongside unstructured TF-IDF vector representations of the article's headline.

## Repository Contents
* `Final_Project.ipynb`: The complete data mining pipeline, including EDA, model training, ablation studies, and evaluation.
* `Final_Project_Report.pdf`: The 4-page ACM double-column technical report.
* `app.py`: The Streamlit frontend web application.
* `news_model.pkl`: The exported scikit-learn Logistic Regression and TF-IDF pipeline.

## How to use the Demo
1. Open the live link above.
2. Enter a hypothetical article headline and adjust the categorical metadata.
3. Click "Predict Virality" to run the NLP pipeline and generate a probability score.
