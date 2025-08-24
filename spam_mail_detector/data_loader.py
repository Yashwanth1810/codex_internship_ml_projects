import pandas as pd
import numpy as np
import os
import requests
from sklearn.model_selection import train_test_split
import zipfile
import gzip
import shutil

class SpamDataLoader:
    """
    Class to load and prepare spam datasets for training
    """
    
    def __init__(self):
        self.data_dir = "data"
        self.ensure_data_directory()
    
    def ensure_data_directory(self):
        """Create data directory if it doesn't exist"""
        if not os.path.exists(self.data_dir):
            os.makedirs(self.data_dir)
    
    def download_sms_spam_collection(self):
        """
        Download SMS Spam Collection dataset from UCI
        """
        url = "https://archive.ics.uci.edu/ml/machine-learning-databases/00228/smsspamcollection.zip"
        zip_path = os.path.join(self.data_dir, "sms_spam.zip")
        
        print("Downloading SMS Spam Collection dataset...")
        try:
            response = requests.get(url, stream=True)
            response.raise_for_status()
            
            with open(zip_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
            
            # Extract the zip file
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall(self.data_dir)
            
            # Read the data
            data_path = os.path.join(self.data_dir, "SMSSpamCollection")
            messages = []
            labels = []
            
            with open(data_path, 'r', encoding='utf-8') as f:
                for line in f:
                    parts = line.strip().split('\t')
                    if len(parts) == 2:
                        label, message = parts
                        labels.append(label.lower())
                        messages.append(message)
            
            # Create DataFrame
            df = pd.DataFrame({
                'message': messages,
                'label': labels
            })
            
            # Save as CSV
            csv_path = os.path.join(self.data_dir, "sms_spam_dataset.csv")
            df.to_csv(csv_path, index=False)
            
            print(f"Dataset saved to {csv_path}")
            print(f"Total messages: {len(df)}")
            print(f"Spam messages: {len(df[df['label'] == 'spam'])}")
            print(f"Ham messages: {len(df[df['label'] == 'ham'])}")
            
            # Clean up
            os.remove(zip_path)
            os.remove(data_path)
            
            return csv_path
            
        except Exception as e:
            print(f"Error downloading dataset: {e}")
            return None
    
    def create_sample_dataset(self, size=1000):
        """
        Create a larger sample dataset for demonstration
        """
        print("Creating sample dataset...")
        
        # Spam patterns
        spam_patterns = [
            "URGENT! You have won a FREE prize!",
            "Limited time offer! Click here to claim your reward!",
            "CONGRATULATIONS! You've been selected for a special deal!",
            "Make money fast! Work from home!",
            "FREE Viagra! Click here now!",
            "You've won the lottery! Send your details!",
            "Special discount just for you!",
            "Act now! Limited time offer!",
            "You're the lucky winner!",
            "FREE membership! Join now!"
        ]
        
        # Ham patterns
        ham_patterns = [
            "Hi, how are you doing?",
            "Can we meet tomorrow?",
            "Thanks for your help yesterday.",
            "I'll call you later.",
            "What time is the meeting?",
            "Have a great day!",
            "See you soon!",
            "Thanks for the information.",
            "I'm running late.",
            "Happy birthday!"
        ]
        
        messages = []
        labels = []
        
        # Generate spam messages
        for _ in range(size // 2):
            pattern = np.random.choice(spam_patterns)
            # Add some variation
            if np.random.random() > 0.5:
                pattern += " " + np.random.choice(["URGENT!", "ACT NOW!", "LIMITED TIME!"])
            messages.append(pattern)
            labels.append('spam')
        
        # Generate ham messages
        for _ in range(size // 2):
            pattern = np.random.choice(ham_patterns)
            # Add some variation
            if np.random.random() > 0.5:
                pattern += " " + np.random.choice(["Thanks!", "See you!", "Take care!"])
            messages.append(pattern)
            labels.append('ham')
        
        # Shuffle the data
        data = list(zip(messages, labels))
        np.random.shuffle(data)
        messages, labels = zip(*data)
        
        # Create DataFrame
        df = pd.DataFrame({
            'message': messages,
            'label': labels
        })
        
        # Save as CSV
        csv_path = os.path.join(self.data_dir, "sample_dataset.csv")
        df.to_csv(csv_path, index=False)
        
        print(f"Sample dataset saved to {csv_path}")
        print(f"Total messages: {len(df)}")
        print(f"Spam messages: {len(df[df['label'] == 'spam'])}")
        print(f"Ham messages: {len(df[df['label'] == 'ham'])}")
        
        return csv_path
    
    def load_dataset(self, file_path):
        """
        Load dataset from CSV file
        """
        if not os.path.exists(file_path):
            print(f"File {file_path} not found!")
            return None
        
        try:
            df = pd.read_csv(file_path)
            if 'message' not in df.columns or 'label' not in df.columns:
                print("CSV must contain 'message' and 'label' columns!")
                return None
            
            return df
        except Exception as e:
            print(f"Error loading dataset: {e}")
            return None
    
    def prepare_data_for_training(self, df, test_size=0.2, random_state=42):
        """
        Prepare data for training by splitting into train/test sets
        """
        if df is None:
            return None, None, None, None
        
        # Convert labels to binary
        df['binary_label'] = df['label'].map({'spam': 1, 'ham': 0})
        
        # Split the data
        X_train, X_test, y_train, y_test = train_test_split(
            df['message'], 
            df['binary_label'], 
            test_size=test_size, 
            random_state=random_state, 
            stratify=df['binary_label']
        )
        
        print(f"Training set size: {len(X_train)}")
        print(f"Testing set size: {len(X_test)}")
        
        return X_train, X_test, y_train, y_test

def main():
    """
    Main function to demonstrate data loading
    """
    print("=== Spam Dataset Loader ===\n")
    
    loader = SpamDataLoader()
    
    # Try to download real dataset
    print("1. Attempting to download SMS Spam Collection dataset...")
    real_dataset_path = loader.download_sms_spam_collection()
    
    if real_dataset_path:
        print("Real dataset downloaded successfully!")
        df = loader.load_dataset(real_dataset_path)
        if df is not None:
            X_train, X_test, y_train, y_test = loader.prepare_data_for_training(df)
    else:
        print("Could not download real dataset. Creating sample dataset instead...")
        
        # Create sample dataset
        sample_dataset_path = loader.create_sample_dataset(size=1000)
        df = loader.load_dataset(sample_dataset_path)
        if df is not None:
            X_train, X_test, y_train, y_test = loader.prepare_data_for_training(df)
    
    print("\nDataset preparation completed!")
    print("You can now use this data with the spam_detector.py script.")

if __name__ == "__main__":
    main()
