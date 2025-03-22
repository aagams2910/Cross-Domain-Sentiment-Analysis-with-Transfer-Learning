import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Dense, Dropout, Input, Lambda
from tensorflow.keras.models import Model
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import seaborn as sns
from transformers import TFAutoModel

class DomainAdaptation:
    def __init__(self, base_model_name="bert-base-uncased", num_labels=2):
        """
        Initialize the domain adaptation module.
        
        Args:
            base_model_name (str): Name of the pre-trained model to use
            num_labels (int): Number of classification labels
        """
        self.base_model_name = base_model_name
        self.num_labels = num_labels
        
    def build_adversarial_model(self, dropout_rate=0.1, hidden_size=768):
        """
        Build a domain-adversarial model based on BERT.
        
        Args:
            dropout_rate (float): Dropout rate
            hidden_size (int): Size of hidden layers
            
        Returns:
            model: The compiled adversarial model
        """
        # Load base BERT model
        bert_model = TFAutoModel.from_pretrained(self.base_model_name)
        
        # Input layers
        input_ids = Input(shape=(None,), dtype=tf.int32, name="input_ids")
        attention_mask = Input(shape=(None,), dtype=tf.int32, name="attention_mask")
        domain_labels = Input(shape=(), dtype=tf.int32, name="domain_labels")
        
        # Get BERT embeddings
        bert_outputs = bert_model([input_ids, attention_mask])
        sequence_output = bert_outputs[0]
        pooled_output = bert_outputs[1]
        
        # Task classifier (sentiment)
        sentiment_dropout = Dropout(dropout_rate)(pooled_output)
        sentiment_output = Dense(self.num_labels, name="sentiment_output")(sentiment_dropout)
        
        # Gradient reversal layer for domain adaptation
        # This layer multiplies the gradient by -1 during backpropagation
        def reverse_gradient(x, gamma=1.0):
            grad_name = "GradientReversal"
            
            @tf.custom_gradient
            def grad_reverse(x):
                def grad(dy):
                    return -gamma * dy
                return x, grad
            
            return Lambda(grad_reverse, name=grad_name)(x)
        
        # Domain classifier with gradient reversal
        reversed_features = Lambda(reverse_gradient)(pooled_output)
        domain_dropout = Dropout(dropout_rate)(reversed_features)
        domain_output = Dense(1, activation='sigmoid', name="domain_output")(domain_dropout)
        
        # Create the model
        model = Model(
            inputs=[input_ids, attention_mask, domain_labels],
            outputs=[sentiment_output, domain_output]
        )
        
        # Compile the model with appropriate losses
        losses = {
            "sentiment_output": tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
            "domain_output": tf.keras.losses.BinaryCrossentropy()
        }
        
        loss_weights = {
            "sentiment_output": 1.0,
            "domain_output": 0.1  # Can be adjusted to control the importance of domain adaptation
        }
        
        metrics = {
            "sentiment_output": tf.keras.metrics.SparseCategoricalAccuracy("sentiment_accuracy"),
            "domain_output": tf.keras.metrics.BinaryAccuracy("domain_accuracy")
        }
        
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=2e-5),
            loss=losses,
            loss_weights=loss_weights,
            metrics=metrics
        )
        
        return model

    def train_adversarial_model(self, source_data, target_data, epochs=3, batch_size=16):
        """
        Train the domain-adversarial model.
        
        Args:
            source_data (dict): Source domain data
            target_data (dict): Target domain data
            epochs (int): Number of training epochs
            batch_size (int): Batch size for training
            
        Returns:
            model: The trained model
            history: Training history
        """
        # Create output directory
        output_dir = os.path.join('models', 'adapted')
        os.makedirs(output_dir, exist_ok=True)
        
        # Build the model
        model = self.build_adversarial_model()
        
        # Prepare source domain data
        source_input_ids = source_data['input_ids']
        source_attention_mask = source_data['attention_mask']
        source_labels = source_data['labels']
        source_domain = np.zeros(len(source_labels))  # 0 for source domain
        
        # Prepare target domain data
        target_input_ids = target_data['input_ids']
        target_attention_mask = target_data['attention_mask']
        target_domain = np.ones(len(target_data['labels']))  # 1 for target domain
        
        # Random target labels for semi-supervised learning
        # (We don't use these for training, but the model expects them)
        target_labels = np.zeros(len(target_data['labels']))
        
        # Combine data (we'll separate them in the generator)
        all_input_ids = np.concatenate([source_input_ids, target_input_ids], axis=0)
        all_attention_mask = np.concatenate([source_attention_mask, target_attention_mask], axis=0)
        all_sentiment_labels = np.concatenate([source_labels, target_labels], axis=0)
        all_domain_labels = np.concatenate([source_domain, target_domain], axis=0)
        
        # Create data generator
        def data_generator():
            indices = np.arange(len(all_input_ids))
            np.random.shuffle(indices)
            
            for start_idx in range(0, len(indices), batch_size):
                end_idx = min(start_idx + batch_size, len(indices))
                batch_indices = indices[start_idx:end_idx]
                
                batch_input_ids = all_input_ids[batch_indices]
                batch_attention_mask = all_attention_mask[batch_indices]
                batch_sentiment_labels = all_sentiment_labels[batch_indices]
                batch_domain_labels = all_domain_labels[batch_indices]
                
                # For target domain samples, we don't use the sentiment labels
                # (masked loss will be implemented in a more complex scenario)
                
                inputs = {
                    "input_ids": batch_input_ids,
                    "attention_mask": batch_attention_mask,
                    "domain_labels": batch_domain_labels
                }
                
                outputs = {
                    "sentiment_output": batch_sentiment_labels,
                    "domain_output": batch_domain_labels
                }
                
                yield inputs, outputs
        
        # Create a tf.data.Dataset from the generator
        dataset = tf.data.Dataset.from_generator(
            data_generator,
            output_signature=(
                {
                    "input_ids": tf.TensorSpec(shape=(None, None), dtype=tf.int32),
                    "attention_mask": tf.TensorSpec(shape=(None, None), dtype=tf.int32),
                    "domain_labels": tf.TensorSpec(shape=(None,), dtype=tf.int32)
                },
                {
                    "sentiment_output": tf.TensorSpec(shape=(None,), dtype=tf.int32),
                    "domain_output": tf.TensorSpec(shape=(None,), dtype=tf.float32)
                }
            )
        ).prefetch(tf.data.AUTOTUNE)
        
        # Set up callbacks
        checkpoint_cb = tf.keras.callbacks.ModelCheckpoint(
            os.path.join(output_dir, 'best_adversarial_model.h5'),
            save_best_only=True,
            monitor='val_sentiment_accuracy',
            mode='max'
        )
        
        early_stopping_cb = tf.keras.callbacks.EarlyStopping(
            monitor='val_sentiment_accuracy',
            patience=2,
            restore_best_weights=True
        )
        
        # Train the model
        steps_per_epoch = len(all_input_ids) // batch_size
        
        history = model.fit(
            dataset,
            epochs=epochs,
            steps_per_epoch=steps_per_epoch,
            callbacks=[checkpoint_cb, early_stopping_cb]
        )
        
        # Save the model
        model.save_weights(os.path.join(output_dir, 'adversarial_model_weights.h5'))
        
        # Plot and save training history
        self._save_training_plots(history, output_dir, is_adversarial=True)
        
        return model, history
    
    def feature_alignment(self, source_model, source_data, target_data):
        """
        Visualize feature alignment between source and target domains.
        
        Args:
            source_model: Model trained on source domain
            source_data (dict): Source domain data
            target_data (dict): Target domain data
            
        Returns:
            None (saves visualization to disk)
        """
        # Create output directory
        output_dir = os.path.join('models', 'analysis')
        os.makedirs(output_dir, exist_ok=True)
        
        # Extract BERT features for source domain
        source_features = self._extract_features(source_model, 
                                                source_data['input_ids'], 
                                                source_data['attention_mask'])
        source_labels = source_data['labels']
        
        # Extract BERT features for target domain
        target_features = self._extract_features(source_model, 
                                                target_data['input_ids'], 
                                                target_data['attention_mask'])
        target_labels = target_data['labels']
        
        # Combine features and create domain labels
        combined_features = np.concatenate([source_features, target_features], axis=0)
        domain_labels = np.concatenate([np.zeros(len(source_features)), 
                                       np.ones(len(target_features))])
        sentiment_labels = np.concatenate([source_labels, target_labels])
        
        # Reduce dimensionality with t-SNE
        tsne = TSNE(n_components=2, random_state=42, perplexity=30)
        reduced_features = tsne.fit_transform(combined_features)
        
        # Create visualization
        plt.figure(figsize=(12, 10))
        
        # Plot by domain
        plt.subplot(2, 1, 1)
        sns.scatterplot(
            x=reduced_features[:, 0],
            y=reduced_features[:, 1],
            hue=domain_labels,
            palette=['blue', 'red'],
            alpha=0.7,
            s=50
        )
        plt.title('Feature Distribution by Domain (Blue: Source, Red: Target)')
        plt.xlabel('t-SNE Dimension 1')
        plt.ylabel('t-SNE Dimension 2')
        
        # Plot by sentiment
        plt.subplot(2, 1, 2)
        sns.scatterplot(
            x=reduced_features[:, 0],
            y=reduced_features[:, 1],
            hue=sentiment_labels,
            palette='viridis',
            alpha=0.7,
            s=50
        )
        plt.title('Feature Distribution by Sentiment')
        plt.xlabel('t-SNE Dimension 1')
        plt.ylabel('t-SNE Dimension 2')
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'feature_alignment.png'))
        plt.close()
    
    def _extract_features(self, model, input_ids, attention_mask):
        """
        Extract features from the model.
        
        Args:
            model: The model to extract features from
            input_ids: Input token IDs
            attention_mask: Attention mask
            
        Returns:
            np.ndarray: Extracted features
        """
        # Get the embedding model (bert base)
        bert_model = model.layers[0]
        
        # Create a feature extraction model
        inputs = [
            tf.keras.layers.Input(shape=(None,), dtype=tf.int32, name='input_ids'),
            tf.keras.layers.Input(shape=(None,), dtype=tf.int32, name='attention_mask')
        ]
        outputs = bert_model(inputs)
        feature_model = tf.keras.Model(inputs=inputs, outputs=outputs[1])
        
        # Extract features in batches (to avoid memory issues)
        features = []
        batch_size = 32
        for i in range(0, len(input_ids), batch_size):
            batch_input_ids = input_ids[i:i+batch_size]
            batch_attention_mask = attention_mask[i:i+batch_size]
            batch_features = feature_model.predict([batch_input_ids, batch_attention_mask])
            features.append(batch_features)
        
        return np.concatenate(features, axis=0)
    
    def _save_training_plots(self, history, output_dir, is_adversarial=False):
        """
        Save training history plots.
        
        Args:
            history: Model training history
            output_dir (str): Directory to save plots to
            is_adversarial (bool): Whether the model is adversarial (has domain output)
        """
        plt.figure(figsize=(15, 5))
        
        if is_adversarial:
            # Plot sentiment accuracy
            plt.subplot(1, 3, 1)
            plt.plot(history.history['sentiment_accuracy'])
            plt.title('Sentiment Accuracy')
            plt.ylabel('Accuracy')
            plt.xlabel('Epoch')
            
            # Plot domain accuracy
            plt.subplot(1, 3, 2)
            plt.plot(history.history['domain_accuracy'])
            plt.title('Domain Classification Accuracy')
            plt.ylabel('Accuracy')
            plt.xlabel('Epoch')
            
            # Plot losses
            plt.subplot(1, 3, 3)
            plt.plot(history.history['sentiment_output_loss'], label='Sentiment')
            plt.plot(history.history['domain_output_loss'], label='Domain')
            plt.title('Losses')
            plt.ylabel('Loss')
            plt.xlabel('Epoch')
            plt.legend()
        else:
            # Plot accuracy
            plt.subplot(1, 2, 1)
            plt.plot(history.history['accuracy'])
            plt.plot(history.history['val_accuracy'])
            plt.title('Model Accuracy')
            plt.ylabel('Accuracy')
            plt.xlabel('Epoch')
            plt.legend(['Train', 'Validation'], loc='upper left')
            
            # Plot loss
            plt.subplot(1, 2, 2)
            plt.plot(history.history['loss'])
            plt.plot(history.history['val_loss'])
            plt.title('Model Loss')
            plt.ylabel('Loss')
            plt.xlabel('Epoch')
            plt.legend(['Train', 'Validation'], loc='upper left')
        
        # Save figure
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'adversarial_training.png' if is_adversarial else 'training_history.png'))
        plt.close()

def main():
    """
    Main function to demonstrate domain adaptation techniques.
    """
    # Initialize domain adaptation
    domain_adapter = DomainAdaptation()
    
    # (Placeholder - you would load data and implement actual adaptation here)
    
if __name__ == "__main__":
    main() 