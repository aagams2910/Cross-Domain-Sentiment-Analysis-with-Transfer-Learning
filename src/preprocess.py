import os
import pandas as pd
import numpy as np
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import spacy
from transformers import AutoTokenizer
from sklearn.model_selection import train_test_split

# Download NLTK resources
nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)

class DataPreprocessor:
    def __init__(self, max_length=128, tokenizer_name="bert-base-uncased"):
        """
        Initialize the DataPreprocessor.
        
        Args:
            max_length (int): Maximum sequence length for tokenization
            tokenizer_name (str): Name of the pre-trained tokenizer to use
        """
        self.max_length = max_length
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        self.lemmatizer = WordNetLemmatizer()
        self.stop_words = set(stopwords.words('english'))
        
    def clean_text(self, text):
        """
        Clean text by removing HTML tags, URLs, special characters, etc.
        
        Args:
            text (str): Input text to clean
            
        Returns:
            str: Cleaned text
        """
        # Convert to lowercase
        text = text.lower()
        
        # Remove HTML tags
        text = re.sub(r'<.*?>', '', text)
        
        # Remove URLs
        text = re.sub(r'http\S+|www\S+|https\S+', '', text)
        
        # Remove mentions and hashtags
        text = re.sub(r'@\w+|#\w+', '', text)
        
        # Remove numbers and special characters
        text = re.sub(r'\d+', '', text)
        text = re.sub(r'[^\w\s]', '', text)
        
        # Tokenize and lemmatize
        tokens = word_tokenize(text)
        
        # Remove stopwords and lemmatize
        tokens = [self.lemmatizer.lemmatize(token) for token in tokens if token not in self.stop_words]
        
        # Rejoin tokens
        cleaned_text = ' '.join(tokens)
        
        return cleaned_text
    
    def tokenize(self, texts, labels=None):
        """
        Tokenize texts using the BERT tokenizer.
        
        Args:
            texts (list): List of text strings to tokenize
            labels (list, optional): List of labels
            
        Returns:
            dict: Tokenized inputs for model
        """
        encoding = self.tokenizer(
            texts,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='tf'
        )
        
        if labels is not None:
            return {'input_ids': encoding['input_ids'], 
                    'attention_mask': encoding['attention_mask'],
                    'labels': labels}
        else:
            return {'input_ids': encoding['input_ids'], 
                    'attention_mask': encoding['attention_mask']}

    def load_and_preprocess(self, data_path, domain="source", test_size=0.2, val_size=0.1):
        """
        Load and preprocess data from file.
        
        Args:
            data_path (str): Path to the data file
            domain (str): Domain name ('source' or 'target')
            test_size (float): Proportion of data to use for testing
            val_size (float): Proportion of data to use for validation
            
        Returns:
            tuple: Preprocessed train, validation, and test datasets
        """
        # Load data based on file extension
        if data_path.endswith('.csv'):
            df = pd.read_csv(data_path)
        elif data_path.endswith('.tsv'):
            df = pd.read_csv(data_path, sep='\t')
        elif data_path.endswith('.json'):
            df = pd.read_json(data_path)
        else:
            raise ValueError(f"Unsupported file format: {data_path}")
        
        # Make sure we have text and label columns
        text_col = next((col for col in df.columns if col.lower() in 
                         ['text', 'review', 'content', 'tweet', 'comment']), None)
        label_col = next((col for col in df.columns if col.lower() in 
                          ['label', 'sentiment', 'class', 'target']), None)
        
        if text_col is None or label_col is None:
            raise ValueError("Could not identify text and label columns in the dataset.")
        
        # Clean text
        df['cleaned_text'] = df[text_col].astype(str).apply(self.clean_text)
        
        # Map labels to integers if they are not already
        if not pd.api.types.is_numeric_dtype(df[label_col]):
            label_map = {label: i for i, label in enumerate(df[label_col].unique())}
            df['numeric_label'] = df[label_col].map(label_map)
        else:
            df['numeric_label'] = df[label_col]
        
        # Split into train, validation, and test sets
        train_df, test_df = train_test_split(df, test_size=test_size, stratify=df['numeric_label'], random_state=42)
        train_df, val_df = train_test_split(train_df, test_size=val_size/(1-test_size), 
                                            stratify=train_df['numeric_label'], random_state=42)
        
        # Save to processed directory
        os.makedirs(os.path.join('data', f"{domain}_domain", 'processed'), exist_ok=True)
        train_df.to_csv(os.path.join('data', f"{domain}_domain", 'processed', 'train.csv'), index=False)
        val_df.to_csv(os.path.join('data', f"{domain}_domain", 'processed', 'val.csv'), index=False)
        test_df.to_csv(os.path.join('data', f"{domain}_domain", 'processed', 'test.csv'), index=False)
        
        # Tokenize the data
        train_texts = train_df['cleaned_text'].tolist()
        val_texts = val_df['cleaned_text'].tolist()
        test_texts = test_df['cleaned_text'].tolist()
        
        train_labels = train_df['numeric_label'].tolist()
        val_labels = val_df['numeric_label'].tolist()
        test_labels = test_df['numeric_label'].tolist()
        
        train_encodings = self.tokenize(train_texts, train_labels)
        val_encodings = self.tokenize(val_texts, val_labels)
        test_encodings = self.tokenize(test_texts, test_labels)
        
        return train_encodings, val_encodings, test_encodings, label_map

def main():
    """
    Main function to preprocess the datasets.
    """
    processor = DataPreprocessor()
    
    # Process source domain data
    source_path = os.path.join('data', 'source_domain', 'raw', 'source_data.csv')
    if os.path.exists(source_path):
        print("Processing source domain data...")
        processor.load_and_preprocess(source_path, domain="source")
    else:
        print(f"Source domain data not found at {source_path}")
    
    # Process target domain data
    target_path = os.path.join('data', 'target_domain', 'raw', 'target_data.csv')
    if os.path.exists(target_path):
        print("Processing target domain data...")
        processor.load_and_preprocess(target_path, domain="target")
    else:
        print(f"Target domain data not found at {target_path}")

if __name__ == "__main__":
    main() 