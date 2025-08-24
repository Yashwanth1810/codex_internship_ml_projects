import pandas as pd
import numpy as np
import re
import string
import os
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import nltk
from nltk.corpus import stopwords
import warnings
warnings.filterwarnings('ignore')

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

class SpamDetector:
    def __init__(self):
        self.vectorizer = None
        self.model = None
        self.stop_words = set(stopwords.words('english'))
        
    def preprocess_text(self, text):
        """
        Preprocess text by removing special characters, converting to lowercase,
        removing stopwords, and tokenizing
        """
        if pd.isna(text):
            return ""
        
        # Convert to string and lowercase
        text = str(text).lower()
        
        # Keep important spam indicators (numbers, currency symbols, exclamation marks)
        # Only remove punctuation that doesn't carry meaning
        text = re.sub(r'[^\w\s!$£€¥#@%&*]', '', text)
        
        # Remove extra whitespace
        text = ' '.join(text.split())
        
        # Simple tokenization by splitting on whitespace and removing stopwords
        tokens = text.split()
        tokens = [word for word in tokens if word not in self.stop_words and len(word) > 1]
        
        # If all words were stopwords, keep at least one word
        if not tokens and text.strip():
            tokens = [word for word in text.split() if len(word) > 1]
        
        # Add spam-specific features
        spam_indicators = []
        original_text = str(text).lower()
        
        # Check for common spam patterns
        if re.search(r'\$[\d,]+', original_text):
            spam_indicators.append('has_money')
        if re.search(r'won|winner|prize|lottery', original_text):
            spam_indicators.append('has_prize_words')
        if re.search(r'urgent|limited|act now|hurry', original_text):
            spam_indicators.append('has_urgency_words')
        if re.search(r'free|freebie|no cost', original_text):
            spam_indicators.append('has_free_words')
        if re.search(r'congrats|congratulations', original_text):
            spam_indicators.append('has_congrats_words')
        
        # Combine tokens with spam indicators
        final_tokens = tokens + spam_indicators
        
        return ' '.join(final_tokens)
    
    def load_and_prepare_data(self, file_path=None):
        """
        Load data and prepare it for training. If no file is provided, use the local SMS Spam dataset.
        """
        if file_path and os.path.exists(file_path):
            # Load from specified file
            data = pd.read_csv(file_path)
            if 'message' in data.columns and 'label' in data.columns:
                messages = data['message']
                labels = data['label']
            else:
                raise ValueError("File must contain 'message' and 'label' columns")
        else:
            # Use the local SMS Spam dataset
            csv_path = "sms_spam_dataset.csv"
            if os.path.exists(csv_path):
                print(f"Loading local SMS Spam dataset: {csv_path}")
                data = pd.read_csv(csv_path)
                messages = data['message']
                labels = data['label']
                print(f"Dataset loaded successfully!")
                print(f"Total messages: {len(messages)}")
                print(f"Spam messages: {(labels == 'spam').sum()}")
                print(f"Ham messages: {(labels == 'ham').sum()}")
            else:
                raise FileNotFoundError(f"Dataset file {csv_path} not found. Please ensure the SMS Spam dataset is available.")
        
        # Preprocess messages
        print("Preprocessing text data...")
        processed_messages = [self.preprocess_text(msg) for msg in messages]
        
        # Convert labels to binary (spam=1, ham=0)
        binary_labels = [1 if label == 'spam' else 0 for label in labels]
        
        return processed_messages, binary_labels
    
    def extract_features(self, messages, method='tfidf'):
        """
        Extract features from text using either TF-IDF or Bag of Words
        """
        if method == 'tfidf':
            self.vectorizer = TfidfVectorizer(
                max_features=5000,  # Increased for better pattern recognition
                ngram_range=(1, 4),  # Include 4-grams for longer spam patterns
                min_df=1,  # Lower minimum frequency to catch rare spam terms
                max_df=0.98,  # Higher maximum frequency
                stop_words=None  # Keep all words to preserve spam indicators
            )
        else:
            self.vectorizer = CountVectorizer(
                max_features=5000,
                ngram_range=(1, 4),
                min_df=1,
                max_df=0.98,
                stop_words=None
            )
        
        features = self.vectorizer.fit_transform(messages)
        return features
    
    def train_model(self, X_train, y_train, model_type='naive_bayes'):
        """
        Train the specified model
        """
        if model_type == 'naive_bayes':
            self.model = MultinomialNB()
        elif model_type == 'logistic_regression':
            self.model = LogisticRegression(
                random_state=42, 
                max_iter=2000,
                C=0.1,  # Stronger regularization for better generalization
                class_weight='balanced',  # Handle class imbalance
                solver='liblinear'  # Better for small datasets
            )
        else:
            raise ValueError("Model type must be 'naive_bayes' or 'logistic_regression'")
        
        print(f"Training {model_type} model...")
        self.model.fit(X_train, y_train)
        print("Training completed!")
    
    def evaluate_model(self, X_test, y_test):
        """
        Evaluate the model and return metrics
        """
        y_pred = self.model.predict(X_test)
        
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        
        print(f"\nModel Performance:")
        print(f"Accuracy: {accuracy:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall: {recall:.4f}")
        print(f"F1-Score: {f1:.4f}")
        
        print(f"\nClassification Report:")
        print(classification_report(y_test, y_pred, target_names=['Ham', 'Spam']))
        
        return accuracy, precision, recall, f1, y_pred
    
    def plot_confusion_matrix(self, y_test, y_pred):
        """
        Plot confusion matrix
        """
        cm = confusion_matrix(y_test, y_pred)
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                    xticklabels=['Ham', 'Spam'], 
                    yticklabels=['Ham', 'Spam'])
        plt.title('Confusion Matrix')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.show()
    
    def predict_spam(self, message):
        """
        Predict if a single message is spam or ham
        """
        if self.model is None or self.vectorizer is None:
            raise ValueError("Model must be trained first!")
        
        processed_message = self.preprocess_text(message)
        features = self.vectorizer.transform([processed_message])
        prediction = self.model.predict(features)[0]
        probability = self.model.predict_proba(features)[0]
        
        result = "SPAM" if prediction == 1 else "HAM"
        confidence = probability[1] if prediction == 1 else probability[0]
        
        return result, confidence

def main():
    """
    Main function to run the spam detector
    """
    print("=== Spam Mail Detector ===\n")
    
    # Initialize the spam detector
    detector = SpamDetector()
    
    # Load and prepare data
    messages, labels = detector.load_and_prepare_data()
    
    print(f"Dataset loaded: {len(messages)} messages")
    print(f"Spam messages: {sum(labels)}")
    print(f"Ham messages: {len(labels) - sum(labels)}")
    
    # Extract features using TF-IDF
    print("\nExtracting features using TF-IDF...")
    features = detector.extract_features(messages, method='tfidf')
    
    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(
        features, labels, test_size=0.2, random_state=42, stratify=labels
    )
    
    print(f"Training set size: {X_train.shape[0]}")
    print(f"Testing set size: {X_test.shape[0]}")
    
    # Train Naive Bayes model
    print("\n" + "="*50)
    print("Training Naive Bayes Model")
    print("="*50)
    detector.train_model(X_train, y_train, model_type='naive_bayes')
    
    # Evaluate Naive Bayes
    nb_accuracy, nb_precision, nb_recall, nb_f1, nb_pred = detector.evaluate_model(X_test, y_test)
    
    # Plot confusion matrix for Naive Bayes
    detector.plot_confusion_matrix(y_test, nb_pred)
    
    # Train Logistic Regression model
    print("\n" + "="*50)
    print("Training Logistic Regression Model")
    print("="*50)
    detector.train_model(X_train, y_train, model_type='logistic_regression')
    
    # Evaluate Logistic Regression
    lr_accuracy, lr_precision, lr_recall, lr_f1, lr_pred = detector.evaluate_model(X_test, y_test)
    
    # Plot confusion matrix for Logistic Regression
    detector.plot_confusion_matrix(y_test, lr_pred)
    
    # Compare models
    print("\n" + "="*50)
    print("Model Comparison")
    print("="*50)
    comparison_data = {
        'Model': ['Naive Bayes', 'Logistic Regression'],
        'Accuracy': [nb_accuracy, lr_accuracy],
        'Precision': [nb_precision, lr_precision],
        'Recall': [nb_recall, lr_recall],
        'F1-Score': [nb_f1, lr_f1]
    }
    
    comparison_df = pd.DataFrame(comparison_data)
    print(comparison_df.to_string(index=False))
    
    # Interactive testing
    print("\n" + "="*50)
    print("Interactive Testing")
    print("="*50)
    print("Enter messages to test (type 'quit' to exit):")
    
    while True:
        test_message = input("\nEnter a message: ")
        if test_message.lower() == 'quit':
            break
        
        try:
            result, confidence = detector.predict_spam(test_message)
            print(f"Prediction: {result}")
            print(f"Confidence: {confidence:.4f}")
        except Exception as e:
            print(f"Error: {e}")

if __name__ == "__main__":
    main()
