import os
import random
import numpy as np
import tensorflow as tf
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_extraction.text import CountVectorizer
from wordcloud import WordCloud
from collections import Counter
from nltk.corpus import stopwords

def set_seed(seed=42):
    """
    Set random seed for reproducibility.
    
    Args:
        seed (int): Random seed
    """
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)

def create_dataset_statistics(data_path, domain_name, output_dir='notebooks'):
    """
    Create and save statistics for a dataset.
    
    Args:
        data_path (str): Path to the data file
        domain_name (str): Name of the domain
        output_dir (str): Directory to save statistics to
        
    Returns:
        pd.DataFrame: Dataset statistics
    """
    # Load data
    if data_path.endswith('.csv'):
        df = pd.read_csv(data_path)
    elif data_path.endswith('.tsv'):
        df = pd.read_csv(data_path, sep='\t')
    elif data_path.endswith('.json'):
        df = pd.read_json(data_path)
    else:
        raise ValueError(f"Unsupported file format: {data_path}")
    
    # Identify text and label columns
    text_col = next((col for col in df.columns if col.lower() in 
                     ['text', 'review', 'content', 'tweet', 'comment']), None)
    label_col = next((col for col in df.columns if col.lower() in 
                      ['label', 'sentiment', 'class', 'target']), None)
    
    if text_col is None or label_col is None:
        raise ValueError("Could not identify text and label columns in the dataset.")
    
    # Text statistics
    df['text_length'] = df[text_col].astype(str).apply(len)
    df['word_count'] = df[text_col].astype(str).apply(lambda x: len(x.split()))
    
    stats = {
        'domain': domain_name,
        'num_samples': len(df),
        'num_classes': len(df[label_col].unique()),
        'class_distribution': dict(df[label_col].value_counts()),
        'avg_text_length': df['text_length'].mean(),
        'std_text_length': df['text_length'].std(),
        'min_text_length': df['text_length'].min(),
        'max_text_length': df['text_length'].max(),
        'avg_word_count': df['word_count'].mean(),
        'std_word_count': df['word_count'].std(),
        'min_word_count': df['word_count'].min(),
        'max_word_count': df['word_count'].max(),
    }
    
    # Save statistics
    pd.DataFrame([stats]).to_csv(os.path.join(output_dir, f"{domain_name}_stats.csv"), index=False)
    
    # Plot class distribution
    plt.figure(figsize=(10, 6))
    sns.countplot(x=label_col, data=df)
    plt.title(f"Class Distribution - {domain_name}")
    plt.xlabel("Class")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"{domain_name}_class_distribution.png"))
    plt.close()
    
    # Plot text length distribution
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    sns.histplot(df['text_length'], kde=True)
    plt.title(f"Text Length Distribution - {domain_name}")
    plt.xlabel("Text Length")
    
    plt.subplot(1, 2, 2)
    sns.histplot(df['word_count'], kde=True)
    plt.title(f"Word Count Distribution - {domain_name}")
    plt.xlabel("Word Count")
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"{domain_name}_text_distributions.png"))
    plt.close()
    
    return stats

def create_vocabulary_analysis(data_path, domain_name, output_dir='notebooks'):
    """
    Create and save vocabulary analysis for a dataset.
    
    Args:
        data_path (str): Path to the data file
        domain_name (str): Name of the domain
        output_dir (str): Directory to save analysis to
        
    Returns:
        dict: Vocabulary statistics
    """
    # Load data
    if data_path.endswith('.csv'):
        df = pd.read_csv(data_path)
    elif data_path.endswith('.tsv'):
        df = pd.read_csv(data_path, sep='\t')
    elif data_path.endswith('.json'):
        df = pd.read_json(data_path)
    else:
        raise ValueError(f"Unsupported file format: {data_path}")
    
    # Identify text and label columns
    text_col = next((col for col in df.columns if col.lower() in 
                     ['text', 'review', 'content', 'tweet', 'comment']), None)
    label_col = next((col for col in df.columns if col.lower() in 
                      ['label', 'sentiment', 'class', 'target']), None)
    
    if text_col is None or label_col is None:
        raise ValueError("Could not identify text and label columns in the dataset.")
    
    # Get stopwords
    stop_words = set(stopwords.words('english'))
    
    # Create vocabulary
    vectorizer = CountVectorizer(stop_words='english')
    X = vectorizer.fit_transform(df[text_col].astype(str))
    
    # Get vocabulary statistics
    vocabulary = vectorizer.get_feature_names_out()
    word_counts = X.sum(axis=0).A1
    
    # Create word count dictionary
    word_count_dict = dict(zip(vocabulary, word_counts))
    
    # Sort by count
    sorted_word_counts = sorted(word_count_dict.items(), key=lambda x: x[1], reverse=True)
    top_words = dict(sorted_word_counts[:100])
    
    # Save vocabulary statistics
    vocab_stats = {
        'domain': domain_name,
        'vocabulary_size': len(vocabulary),
        'top_100_words': top_words
    }
    
    pd.DataFrame([vocab_stats]).to_csv(os.path.join(output_dir, f"{domain_name}_vocab_stats.csv"), index=False)
    
    # Create word cloud
    text = ' '.join(df[text_col].astype(str))
    wordcloud = WordCloud(width=800, height=400, 
                          background_color='white', 
                          stopwords=stop_words, 
                          max_words=100).generate(text)
    
    plt.figure(figsize=(16, 8))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    plt.title(f"Word Cloud - {domain_name}")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"{domain_name}_wordcloud.png"))
    plt.close()
    
    # Plot top words
    plt.figure(figsize=(12, 8))
    words = list(top_words.keys())[:20]  # Take only the top 20 for better visualization
    counts = list(top_words.values())[:20]
    
    sns.barplot(x=counts, y=words)
    plt.title(f"Top 20 Words - {domain_name}")
    plt.xlabel("Count")
    plt.ylabel("Word")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"{domain_name}_top_words.png"))
    plt.close()
    
    return vocab_stats

def compare_domains(source_stats, target_stats, output_dir='notebooks'):
    """
    Compare statistics between source and target domains.
    
    Args:
        source_stats (dict): Statistics for source domain
        target_stats (dict): Statistics for target domain
        output_dir (str): Directory to save comparison to
        
    Returns:
        None (saves plots to disk)
    """
    # Compare sample counts
    domains = ['Source', 'Target']
    sample_counts = [source_stats['num_samples'], target_stats['num_samples']]
    
    plt.figure(figsize=(10, 6))
    sns.barplot(x=domains, y=sample_counts)
    plt.title("Sample Count Comparison")
    plt.ylabel("Number of Samples")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "domain_sample_comparison.png"))
    plt.close()
    
    # Compare text lengths
    text_stats = pd.DataFrame({
        'Domain': domains * 4,
        'Metric': ['Avg Length', 'Avg Length', 'Std Length', 'Std Length', 
                   'Avg Word Count', 'Avg Word Count', 'Std Word Count', 'Std Word Count'],
        'Value': [source_stats['avg_text_length'], target_stats['avg_text_length'],
                  source_stats['std_text_length'], target_stats['std_text_length'],
                  source_stats['avg_word_count'], target_stats['avg_word_count'],
                  source_stats['std_word_count'], target_stats['std_word_count']]
    })
    
    plt.figure(figsize=(12, 8))
    sns.barplot(x='Metric', y='Value', hue='Domain', data=text_stats)
    plt.title("Text Statistics Comparison")
    plt.ylabel("Value")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "domain_text_stats_comparison.png"))
    plt.close()
    
    # Compare vocabulary size
    vocab_sizes = [source_stats.get('vocabulary_size', 0), target_stats.get('vocabulary_size', 0)]
    
    plt.figure(figsize=(10, 6))
    sns.barplot(x=domains, y=vocab_sizes)
    plt.title("Vocabulary Size Comparison")
    plt.ylabel("Vocabulary Size")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "domain_vocab_comparison.png"))
    plt.close()

def analyze_model_weights(model, layer_name, output_dir='notebooks'):
    """
    Analyze weights of a specific layer in the model.
    
    Args:
        model: The model to analyze
        layer_name (str): Name of the layer to analyze
        output_dir (str): Directory to save analysis to
        
    Returns:
        None (saves plots to disk)
    """
    # Get the layer by name
    layer = None
    for l in model.layers:
        if layer_name in l.name:
            layer = l
            break
    
    if layer is None:
        raise ValueError(f"Layer '{layer_name}' not found in the model.")
    
    # Get weights
    weights = layer.get_weights()
    
    if len(weights) == 0:
        raise ValueError(f"Layer '{layer_name}' has no weights.")
    
    # Plot weight distributions
    for i, weight in enumerate(weights):
        plt.figure(figsize=(10, 6))
        sns.histplot(weight.flatten(), kde=True)
        plt.title(f"Weight Distribution - {layer_name} (Weight {i+1})")
        plt.xlabel("Weight Value")
        plt.ylabel("Frequency")
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f"{layer_name}_weight_{i+1}_distribution.png"))
        plt.close()
    
    # If there are exactly two weights (typical for dense layers, weights and biases)
    if len(weights) == 2:
        # Weights are usually the first element, biases the second
        W = weights[0]
        b = weights[1]
        
        # Plot heatmap for weights
        plt.figure(figsize=(12, 8))
        sns.heatmap(W, cmap='coolwarm', center=0)
        plt.title(f"Weight Matrix Heatmap - {layer_name}")
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f"{layer_name}_weight_heatmap.png"))
        plt.close()
        
        # Plot bias values
        plt.figure(figsize=(10, 6))
        plt.bar(range(len(b)), b)
        plt.title(f"Bias Values - {layer_name}")
        plt.xlabel("Neuron Index")
        plt.ylabel("Bias Value")
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f"{layer_name}_bias_values.png"))
        plt.close()

def visualize_attention(model, input_ids, attention_mask, tokenizer, output_dir='notebooks'):
    """
    Visualize attention weights for a given input.
    
    Args:
        model: The model with attention mechanism
        input_ids: Input token IDs
        attention_mask: Attention mask
        tokenizer: Tokenizer for decoding tokens
        output_dir (str): Directory to save visualization to
        
    Returns:
        None (saves plots to disk)
    """
    # Get attention weights
    # This depends on the specific model architecture
    # For transformers, you would typically access the attention outputs
    
    # The following is a simplified example; actual implementation depends on model structure
    outputs = model([input_ids, attention_mask], training=False)
    
    # For BERT models, attention might be accessed as follows
    # This is model-specific
    attention = None
    
    # Try to get attention from the model
    if hasattr(outputs, 'attentions') and outputs.attentions is not None:
        attention = outputs.attentions  # This might be a tuple of attention weights
    
    if attention is None:
        # If attention is not available in outputs, we cannot proceed
        print("Attention weights not available for this model.")
        return
    
    # Convert input_ids to tokens
    tokens = []
    for i in range(input_ids.shape[0]):  # For each example in the batch
        example_tokens = tokenizer.convert_ids_to_tokens(input_ids[i].numpy())
        tokens.append(example_tokens)
    
    # Plot attention
    # This is a simplified version; actual plotting depends on attention structure
    for batch_idx in range(min(3, input_ids.shape[0])):  # Limit to first 3 examples
        # For each layer and attention head
        for layer_idx in range(len(attention)):
            layer_attention = attention[layer_idx][batch_idx]  # Shape: [num_heads, seq_len, seq_len]
            
            # Average across heads for simplicity
            avg_attention = tf.reduce_mean(layer_attention, axis=0).numpy()
            
            # Create attention heatmap
            plt.figure(figsize=(10, 8))
            sns.heatmap(avg_attention, cmap='viridis')
            plt.title(f"Attention Heatmap - Layer {layer_idx+1}, Example {batch_idx+1}")
            plt.xlabel("Token (target)")
            plt.ylabel("Token (source)")
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, f"attention_layer{layer_idx+1}_example{batch_idx+1}.png"))
            plt.close()

def create_learning_curves(history, output_dir='notebooks', model_name='model'):
    """
    Create and save learning curves from training history.
    
    Args:
        history: Training history object
        output_dir (str): Directory to save curves to
        model_name (str): Name of the model
        
    Returns:
        None (saves plots to disk)
    """
    # Create figure
    plt.figure(figsize=(12, 5))
    
    # Plot accuracy
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'])
    if 'val_accuracy' in history.history:
        plt.plot(history.history['val_accuracy'])
        plt.legend(['train', 'validation'], loc='upper left')
    else:
        plt.legend(['train'], loc='upper left')
    plt.title('Model Accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    
    # Plot loss
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'])
    if 'val_loss' in history.history:
        plt.plot(history.history['val_loss'])
        plt.legend(['train', 'validation'], loc='upper right')
    else:
        plt.legend(['train'], loc='upper right')
    plt.title('Model Loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"{model_name}_learning_curves.png"))
    plt.close()
    
    # Save history to CSV
    history_df = pd.DataFrame(history.history)
    history_df.to_csv(os.path.join(output_dir, f"{model_name}_history.csv"), index=False)
    
def main():
    """
    Main function to demonstrate utility functions.
    """
    # Set random seed
    set_seed()
    
    # (Placeholder - you would use the utility functions here)

if __name__ == "__main__":
    main() 