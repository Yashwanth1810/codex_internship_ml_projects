# Spam Mail Detector

A machine learning spam detection system that uses the real UCI SMS Spam Collection dataset to distinguish between spam and legitimate messages.

## What It Does

- Downloads and uses the real UCI SMS dataset (5,574 messages)
- Preprocesses text while preserving spam indicators ($, !, numbers)
- Adds special spam detection features (money, prizes, urgency, free, congratulations)
- Trains Naive Bayes and Logistic Regression models
- Achieves 97%+ accuracy on real data
- Lets you test your own messages

## How It Works

1. **Data Loading**: Automatically downloads UCI SMS dataset (4,827 ham, 747 spam)
2. **Text Preprocessing**: Cleans text, keeps important spam indicators, adds spam features
3. **Feature Extraction**: Uses TF-IDF with 5,000 features, captures 1-4 word patterns
4. **Model Training**: Trains both models on 80% data, tests on 20%
5. **Performance**: Shows accuracy, precision, recall, and F1-score for each model

## Files

- `spam_detector.py` - Main program
- `sms_spam_dataset.csv` - Real dataset (auto-created)
- `requirements.txt` - Python packages needed

## Setup & Usage

1. Create virtual environment: `python -m venv venv`
2. Activate: `venv\Scripts\activate` (Windows) or `source venv/bin/activate` (Mac/Linux)
3. Install packages: `pip install -r requirements.txt`
4. Run: `python spam_detector.py`

## Testing

After training, test any message:
- "Hi, how are you?" → HAM
- "URGENT! You won $5000!" → SPAM
- "Can we meet tomorrow?" → HAM
- "CONGRATS! You won lottery!" → SPAM

Type 'quit' to exit.

## Performance

- **Overall Accuracy**: 97%+
- **Spam Detection**: 80-93% recall
- **False Positives**: Very low (97%+ precision)

## What We Fixed

- Uses real dataset instead of fake data
- Preserves important spam indicators
- Adds spam-specific features
- Much better accuracy and confidence scores

## Learning Outcomes

Text preprocessing, feature engineering, model training, evaluation, and practical NLP application using real-world data.
