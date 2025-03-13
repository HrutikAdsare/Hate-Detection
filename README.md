# Hate Speech Detection

This project is a Hate Speech Detection system that uses Natural Language Processing (NLP) and a Naive Bayes classifier to classify text as either "Hate Speech" or "Not Hate Speech."

## Features

- Preprocessing of text data (lowercasing, removing special characters, tokenization, stopword removal)
- Class balancing using upsampling and downsampling
- Text vectorization using CountVectorizer
- Training a Multinomial Naive Bayes classifier with hyperparameter tuning using GridSearchCV
- Model evaluation using accuracy and classification reports
- A function to classify new text samples

## Installation

1. Clone this repository:
   ```bash
   git clone https://github.com/your-username/hate-speech-detection.git
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Dataset

This project requires `train.csv` and `test.csv` files. Ensure these datasets contain:

- `tweet`: The text of the tweet
- `label`: Binary label (1 for hate speech, 0 for non-hate speech)

## Usage

1. Run the script to train the model:
   ```bash
   python hate_speech_detection.py
   ```
2. To classify a custom text input, modify the script to call `classify_text()`:
   ```python
   text = "i gonna kill you"
   prediction = classify_text(text)
   print("Prediction: ", prediction)
   ```

## Model Training

The model is trained using:

- CountVectorizer for feature extraction
- Multinomial Naive Bayes classifier
- GridSearchCV for hyperparameter tuning
- Train-validation split (80-20%)

## Evaluation

After training, the model prints accuracy scores and classification reports for both training and validation data.


## Author

Hrutik Adsare

