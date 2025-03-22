import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, TensorBoard
from transformers import TFAutoModelForSequenceClassification, AutoConfig
import matplotlib.pyplot as plt
import time
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

class SentimentModelTrainer:
    def __init__(self, model_name="bert-base-uncased", num_labels=2):
        """
        Initialize the model trainer.
        
        Args:
            model_name (str): Name of the pre-trained model to use
            num_labels (int): Number of classification labels
        """
        self.model_name = model_name
        self.num_labels = num_labels
        
    def build_model(self):
        """
        Build the BERT-based sentiment analysis model.
        
        Returns:
            model: The compiled TensorFlow model
        """
        # Create model configuration
        config = AutoConfig.from_pretrained(self.model_name, num_labels=self.num_labels)
        
        # Create model with the config
        model = TFAutoModelForSequenceClassification.from_pretrained(
            self.model_name, 
            config=config
        )
        
        # Compile the model
        optimizer = tf.keras.optimizers.Adam(learning_rate=2e-5)
        loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
        metrics = [tf.keras.metrics.SparseCategoricalAccuracy('accuracy')]
        
        model.compile(optimizer=optimizer, loss=loss, metrics=metrics)
        
        return model
    
    def train_source_model(self, train_dataset, val_dataset, epochs=3, batch_size=16):
        """
        Train the model on the source domain.
        
        Args:
            train_dataset (dict): Training dataset
            val_dataset (dict): Validation dataset
            epochs (int): Number of training epochs
            batch_size (int): Batch size for training
            
        Returns:
            model: The trained model
            history: Training history
        """
        # Create output directory
        output_dir = os.path.join('models', 'baseline')
        os.makedirs(output_dir, exist_ok=True)
        
        # Build the model
        model = self.build_model()
        
        # Prepare the data
        train_inputs = {
            'input_ids': train_dataset['input_ids'],
            'attention_mask': train_dataset['attention_mask']
        }
        train_labels = train_dataset['labels']
        
        val_inputs = {
            'input_ids': val_dataset['input_ids'],
            'attention_mask': val_dataset['attention_mask']
        }
        val_labels = val_dataset['labels']
        
        # Set up callbacks
        checkpoint_cb = ModelCheckpoint(
            os.path.join(output_dir, 'best_model.h5'),
            save_best_only=True,
            monitor='val_accuracy',
            mode='max'
        )
        
        early_stopping_cb = EarlyStopping(
            monitor='val_accuracy',
            patience=2,
            restore_best_weights=True
        )
        
        # Train the model
        history = model.fit(
            train_inputs,
            train_labels,
            validation_data=(val_inputs, val_labels),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=[checkpoint_cb, early_stopping_cb]
        )
        
        # Save the model
        model.save_pretrained(output_dir)
        
        # Plot and save training history
        self._save_training_plots(history, output_dir)
        
        return model, history
    
    def adapt_model(self, source_model, target_train_dataset, target_val_dataset, 
                    epochs=3, batch_size=16, strategy='fine_tune'):
        """
        Adapt the source model to the target domain.
        
        Args:
            source_model: The model trained on the source domain
            target_train_dataset (dict): Target domain training dataset
            target_val_dataset (dict): Target domain validation dataset
            epochs (int): Number of training epochs
            batch_size (int): Batch size for training
            strategy (str): Adaptation strategy ('fine_tune' or 'gradual_unfreeze')
            
        Returns:
            model: The adapted model
            history: Training history
        """
        # Create output directory
        output_dir = os.path.join('models', 'adapted')
        os.makedirs(output_dir, exist_ok=True)
        
        if strategy == 'fine_tune':
            # For fine-tuning, just use the source model as is
            model = source_model
            model.trainable = True
        elif strategy == 'gradual_unfreeze':
            # For gradual unfreezing, start with all layers frozen
            # and gradually unfreeze them
            model = source_model
            # Freeze all layers first
            model.trainable = False
            
            # Unfreeze only the last layer initially
            for layer in model.layers[-1:]:
                layer.trainable = True
        else:
            raise ValueError(f"Unknown adaptation strategy: {strategy}")
        
        # Recompile the model with a lower learning rate
        optimizer = tf.keras.optimizers.Adam(learning_rate=5e-6)
        loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
        metrics = [tf.keras.metrics.SparseCategoricalAccuracy('accuracy')]
        
        model.compile(optimizer=optimizer, loss=loss, metrics=metrics)
        
        # Prepare the data
        train_inputs = {
            'input_ids': target_train_dataset['input_ids'],
            'attention_mask': target_train_dataset['attention_mask']
        }
        train_labels = target_train_dataset['labels']
        
        val_inputs = {
            'input_ids': target_val_dataset['input_ids'],
            'attention_mask': target_val_dataset['attention_mask']
        }
        val_labels = target_val_dataset['labels']
        
        # Set up callbacks
        checkpoint_cb = ModelCheckpoint(
            os.path.join(output_dir, 'best_model.h5'),
            save_best_only=True,
            monitor='val_accuracy',
            mode='max'
        )
        
        early_stopping_cb = EarlyStopping(
            monitor='val_accuracy',
            patience=2,
            restore_best_weights=True
        )
        
        # Train the model
        history = model.fit(
            train_inputs,
            train_labels,
            validation_data=(val_inputs, val_labels),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=[checkpoint_cb, early_stopping_cb]
        )
        
        # Save the model
        model.save_pretrained(output_dir)
        
        # Plot and save training history
        self._save_training_plots(history, output_dir)
        
        return model, history
    
    def train_target_only(self, target_train_dataset, target_val_dataset, epochs=3, batch_size=16):
        """
        Train a model directly on the target domain only.
        
        Args:
            target_train_dataset (dict): Target domain training dataset
            target_val_dataset (dict): Target domain validation dataset
            epochs (int): Number of training epochs
            batch_size (int): Batch size for training
            
        Returns:
            model: The trained model
            history: Training history
        """
        # Create output directory
        output_dir = os.path.join('models', 'target_only')
        os.makedirs(output_dir, exist_ok=True)
        
        # Build a fresh model
        model = self.build_model()
        
        # Prepare the data
        train_inputs = {
            'input_ids': target_train_dataset['input_ids'],
            'attention_mask': target_train_dataset['attention_mask']
        }
        train_labels = target_train_dataset['labels']
        
        val_inputs = {
            'input_ids': target_val_dataset['input_ids'],
            'attention_mask': target_val_dataset['attention_mask']
        }
        val_labels = target_val_dataset['labels']
        
        # Set up callbacks
        checkpoint_cb = ModelCheckpoint(
            os.path.join(output_dir, 'best_model.h5'),
            save_best_only=True,
            monitor='val_accuracy',
            mode='max'
        )
        
        early_stopping_cb = EarlyStopping(
            monitor='val_accuracy',
            patience=2,
            restore_best_weights=True
        )
        
        # Train the model
        history = model.fit(
            train_inputs,
            train_labels,
            validation_data=(val_inputs, val_labels),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=[checkpoint_cb, early_stopping_cb]
        )
        
        # Save the model
        model.save_pretrained(output_dir)
        
        # Plot and save training history
        self._save_training_plots(history, output_dir)
        
        return model, history
    
    def evaluate_model(self, model, test_dataset):
        """
        Evaluate the model on the test dataset.
        
        Args:
            model: The model to evaluate
            test_dataset (dict): Test dataset
            
        Returns:
            dict: Evaluation metrics
        """
        # Prepare the data
        test_inputs = {
            'input_ids': test_dataset['input_ids'],
            'attention_mask': test_dataset['attention_mask']
        }
        test_labels = test_dataset['labels']
        
        # Get predictions
        predictions = model.predict(test_inputs)
        predicted_labels = np.argmax(predictions.logits, axis=1)
        
        # Calculate metrics
        accuracy = accuracy_score(test_labels, predicted_labels)
        precision, recall, f1, _ = precision_recall_fscore_support(
            test_labels, predicted_labels, average='weighted'
        )
        
        metrics = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1
        }
        
        return metrics, predicted_labels
    
    def _save_training_plots(self, history, output_dir):
        """
        Save training history plots.
        
        Args:
            history: Model training history
            output_dir (str): Directory to save plots to
        """
        # Plot accuracy
        plt.figure(figsize=(12, 4))
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
        plt.savefig(os.path.join(output_dir, 'training_history.png'))
        plt.close()

def main():
    """
    Main function to train the models.
    """
    # Load preprocessed datasets
    # (Placeholder - you would load the actual datasets here)
    
    # Initialize trainer
    trainer = SentimentModelTrainer()
    
    # Train source model
    # (Placeholder - you would train the actual model here)
    
    # Adapt to target domain
    # (Placeholder - you would adapt the model here)
    
    # Train target-only model
    # (Placeholder - you would train the target-only model here)
    
    # Evaluate models
    # (Placeholder - you would evaluate the models here)

if __name__ == "__main__":
    main() 