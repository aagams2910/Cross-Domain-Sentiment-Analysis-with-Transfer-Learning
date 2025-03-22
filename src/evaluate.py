import os
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support, confusion_matrix,
    classification_report, roc_curve, auc
)
from transformers import TFAutoModelForSequenceClassification

class ModelEvaluator:
    def __init__(self):
        """
        Initialize the model evaluator.
        """
        # Create output directory for evaluation results
        self.output_dir = os.path.join('models', 'evaluation')
        os.makedirs(self.output_dir, exist_ok=True)
    
    def load_model(self, model_path):
        """
        Load a saved model.
        
        Args:
            model_path (str): Path to the saved model
            
        Returns:
            model: The loaded model
        """
        model = TFAutoModelForSequenceClassification.from_pretrained(model_path)
        return model
    
    def evaluate_model(self, model, test_data, model_name="model"):
        """
        Evaluate the model on test data.
        
        Args:
            model: The model to evaluate
            test_data (dict): Test dataset
            model_name (str): Name of the model for saving results
            
        Returns:
            dict: Evaluation metrics
        """
        # Prepare inputs
        test_inputs = {
            "input_ids": test_data["input_ids"],
            "attention_mask": test_data["attention_mask"]
        }
        
        # Get predictions
        predictions = model.predict(test_inputs)
        predicted_probs = tf.nn.softmax(predictions.logits, axis=1).numpy()
        predicted_labels = np.argmax(predicted_probs, axis=1)
        true_labels = test_data["labels"]
        
        # Calculate metrics
        accuracy = accuracy_score(true_labels, predicted_labels)
        precision, recall, f1, _ = precision_recall_fscore_support(
            true_labels, predicted_labels, average='weighted'
        )
        
        # Generate and save evaluation report
        report = classification_report(true_labels, predicted_labels, output_dict=True)
        report_df = pd.DataFrame(report).transpose()
        report_df.to_csv(os.path.join(self.output_dir, f"{model_name}_classification_report.csv"))
        
        # Plot and save confusion matrix
        self._plot_confusion_matrix(true_labels, predicted_labels, model_name)
        
        # Plot ROC curve for binary classification
        if predicted_probs.shape[1] == 2:
            self._plot_roc_curve(true_labels, predicted_probs[:, 1], model_name)
        
        # Save evaluation metrics
        metrics = {
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1": f1
        }
        
        # Save metrics to file
        metrics_df = pd.DataFrame([metrics])
        metrics_df.to_csv(os.path.join(self.output_dir, f"{model_name}_metrics.csv"), index=False)
        
        return metrics, predicted_labels
    
    def compare_models(self, model_metrics, model_names=None):
        """
        Compare multiple models based on their evaluation metrics.
        
        Args:
            model_metrics (list): List of dictionaries containing model metrics
            model_names (list): List of model names
            
        Returns:
            None (saves comparison plots to disk)
        """
        if model_names is None:
            model_names = [f"Model {i+1}" for i in range(len(model_metrics))]
        
        # Create a DataFrame for metrics
        metrics_df = pd.DataFrame(model_metrics)
        metrics_df.index = model_names
        
        # Save comparison to CSV
        metrics_df.to_csv(os.path.join(self.output_dir, "model_comparison.csv"))
        
        # Plot comparison
        plt.figure(figsize=(12, 6))
        metrics_df.plot(kind='bar')
        plt.title('Model Comparison')
        plt.ylabel('Score')
        plt.xlabel('Model')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, "model_comparison.png"))
        plt.close()
        
        # Radar chart for visualization
        self._plot_radar_chart(metrics_df, model_names)
    
    def analyze_errors(self, true_labels, predicted_labels, texts, model_name="model"):
        """
        Analyze model errors.
        
        Args:
            true_labels (np.ndarray): True labels
            predicted_labels (np.ndarray): Predicted labels
            texts (list): Corresponding texts
            model_name (str): Name of the model for saving results
            
        Returns:
            None (saves error analysis to disk)
        """
        # Find errors
        errors = predicted_labels != true_labels
        error_indices = np.where(errors)[0]
        
        # Create error analysis DataFrame
        error_df = pd.DataFrame({
            "text": [texts[i] for i in error_indices],
            "true_label": true_labels[error_indices],
            "predicted_label": predicted_labels[error_indices]
        })
        
        # Save error analysis
        error_df.to_csv(os.path.join(self.output_dir, f"{model_name}_errors.csv"), index=False)
        
        # Count error types
        error_types = pd.crosstab(
            error_df["true_label"], 
            error_df["predicted_label"],
            rownames=["True Label"],
            colnames=["Predicted Label"]
        )
        
        # Plot error distribution
        plt.figure(figsize=(10, 8))
        sns.heatmap(error_types, annot=True, cmap="YlGnBu", fmt="d")
        plt.title(f"Error Distribution for {model_name}")
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, f"{model_name}_error_distribution.png"))
        plt.close()
    
    def cross_domain_analysis(self, source_metrics, target_metrics, model_names=None):
        """
        Analyze model performance across domains.
        
        Args:
            source_metrics (list): List of metrics on source domain
            target_metrics (list): List of metrics on target domain
            model_names (list): List of model names
            
        Returns:
            None (saves cross-domain analysis to disk)
        """
        if model_names is None:
            model_names = [f"Model {i+1}" for i in range(len(source_metrics))]
        
        # Create DataFrames
        source_df = pd.DataFrame(source_metrics)
        source_df.index = model_names
        source_df['domain'] = 'Source'
        
        target_df = pd.DataFrame(target_metrics)
        target_df.index = model_names
        target_df['domain'] = 'Target'
        
        # Combine data
        combined_df = pd.concat([source_df, target_df])
        
        # Reset index and convert to long format for plotting
        combined_df = combined_df.reset_index().rename(columns={'index': 'model'})
        plot_df = pd.melt(
            combined_df, 
            id_vars=['model', 'domain'], 
            value_vars=['accuracy', 'precision', 'recall', 'f1'],
            var_name='metric', 
            value_name='score'
        )
        
        # Plot cross-domain performance
        plt.figure(figsize=(14, 10))
        sns.barplot(x='model', y='score', hue='domain', col='metric', data=plot_df)
        plt.title('Cross-Domain Performance Comparison')
        plt.ylabel('Score')
        plt.xlabel('Model')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, "cross_domain_comparison.png"))
        plt.close()
        
        # Calculate domain adaptation gain
        for model_idx, model_name in enumerate(model_names):
            source_perf = source_metrics[model_idx]
            target_perf = target_metrics[model_idx]
            
            adaptation_gain = {
                metric: target_perf[metric] - source_perf[metric]
                for metric in ['accuracy', 'precision', 'recall', 'f1']
            }
            
            # Save adaptation gain
            gain_df = pd.DataFrame([adaptation_gain])
            gain_df.index = [model_name]
            gain_df.to_csv(os.path.join(self.output_dir, f"{model_name}_adaptation_gain.csv"))
    
    def _plot_confusion_matrix(self, true_labels, predicted_labels, model_name):
        """
        Plot confusion matrix.
        
        Args:
            true_labels (np.ndarray): True labels
            predicted_labels (np.ndarray): Predicted labels
            model_name (str): Name of the model for saving results
            
        Returns:
            None (saves plot to disk)
        """
        # Compute confusion matrix
        cm = confusion_matrix(true_labels, predicted_labels)
        
        # Plot confusion matrix
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
        plt.title(f"Confusion Matrix - {model_name}")
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, f"{model_name}_confusion_matrix.png"))
        plt.close()
    
    def _plot_roc_curve(self, true_labels, predicted_probs, model_name):
        """
        Plot ROC curve for binary classification.
        
        Args:
            true_labels (np.ndarray): True labels
            predicted_probs (np.ndarray): Predicted probabilities for positive class
            model_name (str): Name of the model for saving results
            
        Returns:
            None (saves plot to disk)
        """
        # Compute ROC curve and ROC area
        fpr, tpr, _ = roc_curve(true_labels, predicted_probs)
        roc_auc = auc(fpr, tpr)
        
        # Plot ROC curve
        plt.figure(figsize=(10, 8))
        plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(f'ROC Curve - {model_name}')
        plt.legend(loc="lower right")
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, f"{model_name}_roc_curve.png"))
        plt.close()
    
    def _plot_radar_chart(self, metrics_df, model_names):
        """
        Plot radar chart for model comparison.
        
        Args:
            metrics_df (pd.DataFrame): DataFrame containing model metrics
            model_names (list): List of model names
            
        Returns:
            None (saves plot to disk)
        """
        # Prepare data for radar chart
        categories = list(metrics_df.columns)
        N = len(categories)
        
        # Create angles for radar chart
        angles = [n / float(N) * 2 * np.pi for n in range(N)]
        angles += angles[:1]  # Close the loop
        
        # Create radar chart
        fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(polar=True))
        
        # Add lines and points for each model
        for i, model in enumerate(model_names):
            values = metrics_df.loc[model].values.tolist()
            values += values[:1]  # Close the loop
            ax.plot(angles, values, linewidth=2, linestyle='solid', label=model)
            ax.fill(angles, values, alpha=0.1)
        
        # Set category labels
        plt.xticks(angles[:-1], categories)
        
        # Add legend
        plt.legend(loc='upper right', bbox_to_anchor=(0.1, 0.1))
        plt.title('Model Comparison - Radar Chart')
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, "model_comparison_radar.png"))
        plt.close()

def main():
    """
    Main function to evaluate models.
    """
    evaluator = ModelEvaluator()
    
    # (Placeholder - you would load models and test data here)
    
    # Example:
    # baseline_model = evaluator.load_model('models/baseline')
    # adapted_model = evaluator.load_model('models/adapted')
    # target_only_model = evaluator.load_model('models/target_only')
    
    # source_test_data = load_source_test_data()
    # target_test_data = load_target_test_data()
    
    # Evaluate models on target domain
    # baseline_metrics, _ = evaluator.evaluate_model(baseline_model, target_test_data, "baseline")
    # adapted_metrics, _ = evaluator.evaluate_model(adapted_model, target_test_data, "adapted")
    # target_only_metrics, _ = evaluator.evaluate_model(target_only_model, target_test_data, "target_only")
    
    # Compare models
    # evaluator.compare_models(
    #     [baseline_metrics, adapted_metrics, target_only_metrics],
    #     ["Baseline (Source)", "Adapted", "Target-Only"]
    # )

if __name__ == "__main__":
    main() 