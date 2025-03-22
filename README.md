# Cross-Domain Sentiment Analysis with Transfer Learning

A project that develops a sentiment analysis model using transfer learning techniques for effective cross-domain sentiment classification. The model can be trained on one domain (e.g., product reviews) and then adapted to perform well on a different domain (e.g., social media posts) with minimal labeled target data.

## Project Overview

Sentiment analysis models often perform well in the domain they were trained on but struggle when applied to different domains with unique linguistic characteristics. This project addresses this challenge through transfer learning and domain adaptation techniques.

### Key Features

- Pre-trained BERT-based sentiment analysis model
- Transfer learning for cross-domain adaptation
- Multiple adaptation strategies (fine-tuning, gradual unfreezing, adversarial training)
- Comprehensive evaluation and comparison framework
- Visualization tools for model analysis

## Directory Structure

```
project/
├── data/
│   ├── source_domain/
│   │   ├── raw/         # Raw source domain data
│   │   └── processed/   # Processed source domain data
│   └── target_domain/
│       ├── raw/         # Raw target domain data
│       └── processed/   # Processed target domain data
├── models/
│   ├── baseline/        # Model trained on source domain
│   ├── adapted/         # Model adapted to target domain
│   └── target_only/     # Model trained only on target domain
├── src/
│   ├── preprocess.py    # Data preprocessing utilities
│   ├── train.py         # Model training functions
│   ├── adapt.py         # Domain adaptation techniques
│   ├── evaluate.py      # Model evaluation tools
│   └── utils.py         # General utility functions
├── notebooks/
│   ├── exploratory_analysis.ipynb    # Data exploration
│   └── model_training.ipynb          # Model training and evaluation
├── requirements.txt
└── README.md
```

## Getting Started

### Prerequisites

- Python 3.8+
- TensorFlow 2.x or PyTorch
- Hugging Face Transformers library
- Recommended: CUDA-capable GPU for faster training

### Installation

1. Clone the repository:
   ```
   git clone https://github.com/aagams2910/cross-domain-sentiment-analysis.git
   cd cross-domain-sentiment-analysis
   ```

2. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

3. Download and place datasets in the appropriate directories:
   - Place source domain data in `data/source_domain/raw/`
   - Place target domain data in `data/target_domain/raw/`

### Data Format

The expected format for data files is CSV with at least two columns:
- A text column (can be named: 'text', 'review', 'content', 'tweet', 'comment')
- A label column (can be named: 'label', 'sentiment', 'class', 'target')

For binary sentiment classification, labels should be binary (0/1, positive/negative, etc.).

## Usage

### Data Preprocessing

To preprocess the datasets:

```python
from src.preprocess import DataPreprocessor

preprocessor = DataPreprocessor(max_length=128, tokenizer_name="bert-base-uncased")
preprocessor.load_and_preprocess("data/source_domain/raw/source_data.csv", domain="source")
preprocessor.load_and_preprocess("data/target_domain/raw/target_data.csv", domain="target")
```

### Model Training

To train a model on the source domain:

```python
from src.train import SentimentModelTrainer

trainer = SentimentModelTrainer(model_name="bert-base-uncased", num_labels=2)
model, history = trainer.train_source_model(train_dataset, val_dataset, epochs=3, batch_size=16)
```

### Domain Adaptation

To adapt a source-trained model to the target domain:

```python
from src.train import SentimentModelTrainer

trainer = SentimentModelTrainer(model_name="bert-base-uncased", num_labels=2)
adapted_model, history = trainer.adapt_model(
    source_model, target_train_dataset, target_val_dataset, 
    epochs=3, batch_size=16, strategy='fine_tune'
)
```

### Model Evaluation

To evaluate model performance:

```python
from src.evaluate import ModelEvaluator

evaluator = ModelEvaluator()
metrics, predictions = evaluator.evaluate_model(model, test_dataset, model_name="my_model")
print(f"Accuracy: {metrics['accuracy']}, F1 score: {metrics['f1']}")
```

## Domain Adaptation Strategies

The project implements several transfer learning strategies:

1. **Fine-tuning**: Further train the entire pre-trained model on the target domain data
2. **Gradual Unfreezing**: Gradually unfreeze layers of the model during adaptation
3. **Adversarial Training**: Use domain-adversarial training to learn domain-invariant features

## Notebooks

The project includes Jupyter notebooks for:
- Exploratory data analysis of source and target domains
- Model training, adaptation, and evaluation

## Datasets

Suggested datasets for experimentation:

### Source Domain Options
- IMDb Movie Reviews (sentiment analysis on movie reviews)
- Amazon Product Reviews (sentiment analysis on product reviews)

### Target Domain Options
- Twitter Sentiment140 (sentiment analysis on tweets)
- Reddit Comments (sentiment analysis on social media posts)