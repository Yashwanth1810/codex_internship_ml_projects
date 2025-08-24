import nltk

def download_nltk_data():
    """
    Download required NLTK data for the spam detector
    """
    print("Downloading required NLTK data...")
    
    try:
        # Download punkt tokenizer
        print("Downloading punkt tokenizer...")
        nltk.download('punkt')
        
        # Download stopwords
        print("Downloading stopwords...")
        nltk.download('stopwords')
        
        print("NLTK data download completed successfully!")
        
    except Exception as e:
        print(f"Error downloading NLTK data: {e}")

if __name__ == "__main__":
    download_nltk_data()
