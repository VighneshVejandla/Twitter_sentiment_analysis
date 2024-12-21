# Twitter Sentiment Analysis Using NLTK

This project performs sentiment analysis on Twitter data using the Sentiment140 dataset. The goal is to classify tweets into positive, negative, or neutral sentiments based on their content.

## Dataset

- **Dataset Name**: Sentiment140
- **Source**: [Kaggle - Sentiment140 Dataset](https://www.kaggle.com/datasets/kazanova/sentiment140)
- **Description**: The dataset contains 1.6 million tweets annotated for sentiment (positive or negative). Each row includes the tweet text, polarity, and other metadata.

## Features

- **Polarity**: Sentiment label (0 = negative, 4 = positive).
- **Tweet Text**: The text of the tweet.

## Project Structure

- **Notebook**: `twitter-sentiment-analysis-using-nltk.ipynb`
  - Contains all the code for data preprocessing, analysis, and sentiment classification using NLTK.
- **Dataset**: The Sentiment140 dataset is used for this analysis.
- **Output**: Insights on the sentiment distribution and a trained model to classify sentiment.

## Methodology

1. **Data Preprocessing**:
   - Remove URLs, mentions, hashtags, and special characters.
   - Tokenization and stopword removal.
   - Stemming or lemmatization.

2. **Exploratory Data Analysis**:
   - Visualize the distribution of sentiments.
   - Understand word frequency for positive and negative sentiments.

3. **Model Training**:
   - Feature extraction using techniques like Bag-of-Words or TF-IDF.
   - Classification using logistic regression.

4. **Evaluation**:
   - Evaluate the model's accuracy, precision, recall, and F1-score.

## Requirements

- Python 3.x
- Libraries:
  - `nltk`
  - `pandas`
  - `numpy`
  - `matplotlib`
  - `scikit-learn`

## How to Run

1. Clone this repository or download the files.
2. Install the required Python libraries:
   ```bash
   pip install nltk pandas numpy matplotlib scikit-learn
