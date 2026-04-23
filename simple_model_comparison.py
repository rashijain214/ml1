#!/usr/bin/env python3
"""
Simplified Model Comparison for MLBlocker
This script runs a basic model comparison without heavy dependencies.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix
)
import matplotlib.pyplot as plt
import seaborn as sns
import time
import os

# Create output directory
OUTPUT_DIR = "output"
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

def load_sample_data():
    """Create sample data for demonstration with realistic performance"""
    print("Creating sample dataset for demonstration...")
    
    # Generate sample data similar to MLBlocker features
    np.random.seed(42)
    n_samples = 1000
    
    # Create features similar to MLBlocker with more overlap between classes
    data = {
        'url_length': np.random.exponential(50, n_samples) + np.random.normal(0, 10, n_samples),
        'is_third_party': np.random.binomial(1, 0.45, n_samples),  # More balanced
        'num_requests_sent': np.random.poisson(5, n_samples) + np.random.normal(0, 2, n_samples),
        'num_get_storage': np.random.poisson(2, n_samples) + np.random.normal(0, 1, n_samples),
        'num_set_storage': np.random.poisson(2, n_samples) + np.random.normal(0, 1, n_samples),
        'keyword_raw_present': np.random.binomial(1, 0.35, n_samples),  # More overlap
        'content_policy_type': np.random.choice([0, 1, 2, 3], n_samples),
        'brackettodot': np.random.binomial(1, 0.15, n_samples),
        'avg_ident': np.random.normal(8, 4, n_samples),  # More variance
        'avg_charperline': np.random.normal(25, 15, n_samples),  # More variance
    }
    
    # Add some n-gram features with noise
    for i in range(10):
        data[f'ng_{i}_{i}_{i}'] = np.random.exponential(2, n_samples) + np.random.normal(0, 1, n_samples)
    
    # Create labels with much less predictability and more noise
    X = pd.DataFrame(data)
    
    # Create a learnable relationship for labels targeting ~0.8 accuracy
    # Stronger signal but still with noise to keep it challenging
    base_signal = (
        0.5 +  # Base probability (middle point)
        0.15 * X['is_third_party'] +  # Moderate influence
        0.12 * X['keyword_raw_present'] +  # Moderate influence
        0.08 * (X['url_length'] > 80).astype(int) +  # Lower threshold
        0.06 * (X['num_requests_sent'] > 8).astype(int) +  # Lower threshold
        0.04 * X['brackettodot'] +
        0.03 * (X['avg_ident'] > 10).astype(int) +
        0.02 * (X['avg_charperline'] > 30).astype(int) +
        np.random.normal(0, 0.18, n_samples)  # Moderate Gaussian noise
    )
    
    # Add some non-linear interactions but not too complex
    interaction_noise = 0.04 * np.sin(X['url_length'] / 20) * np.random.normal(0, 0.5, n_samples)
    base_signal += interaction_noise
    
    # Create labels with threshold that creates balanced classes
    threshold = np.percentile(base_signal, 50)  # Create balanced classes
    y = (base_signal > threshold).astype(int)
    
    # Add moderate label noise (flip 5% of labels randomly)
    flip_indices = np.random.choice(n_samples, size=int(0.05 * n_samples), replace=False)
    y[flip_indices] = 1 - y[flip_indices]
    
    print(f"Created dataset with {X.shape[1]} features and {n_samples} samples")
    print(f"Label distribution: {y.value_counts().to_dict()}")
    print(f"Expected accuracy for random guessing: {max(y.value_counts()) / len(y):.3f}")
    
    return X, y

def train_random_forest(X_train, y_train, X_val, y_val):
    """Train Random Forest model"""
    print("Training Random Forest...")
    
    start_time = time.time()
    
    rf = RandomForestClassifier(
        n_estimators=80,  # Increased for better learning
        max_depth=8,      # Increased depth
        min_samples_split=8,   # Reduced to allow more learning
        min_samples_leaf=4,    # Reduced to allow more learning
        max_features='sqrt',   # Reduce features per tree
        random_state=42,
        n_jobs=-1
    )
    
    rf.fit(X_train, y_train)
    
    training_time = time.time() - start_time
    
    # Predictions
    start_time = time.time()
    y_pred = rf.predict(X_val)
    y_pred_proba = rf.predict_proba(X_val)[:, 1]
    inference_time = time.time() - start_time
    
    return rf, y_pred, y_pred_proba, training_time, inference_time

def train_decision_tree(X_train, y_train, X_val, y_val):
    """Train Decision Tree model"""
    print("Training Decision Tree...")
    
    from sklearn.tree import DecisionTreeClassifier
    
    start_time = time.time()
    
    dt = DecisionTreeClassifier(
        max_depth=6,      # Increased depth for better learning
        min_samples_split=12,  # Reduced threshold
        min_samples_leaf=6,     # Reduced threshold
        max_features='sqrt',    # Reduce features
        random_state=42
    )
    
    dt.fit(X_train, y_train)
    
    training_time = time.time() - start_time
    
    # Predictions
    start_time = time.time()
    y_pred = dt.predict(X_val)
    y_pred_proba = dt.predict_proba(X_val)[:, 1]
    inference_time = time.time() - start_time
    
    return dt, y_pred, y_pred_proba, training_time, inference_time

def train_logistic_regression(X_train, y_train, X_val, y_val):
    """Train Logistic Regression model"""
    print("Training Logistic Regression...")
    
    from sklearn.linear_model import LogisticRegression
    
    start_time = time.time()
    
    lr = LogisticRegression(
        random_state=42,
        max_iter=1000
    )
    
    lr.fit(X_train, y_train)
    
    training_time = time.time() - start_time
    
    # Predictions
    start_time = time.time()
    y_pred = lr.predict(X_val)
    y_pred_proba = lr.predict_proba(X_val)[:, 1]
    inference_time = time.time() - start_time
    
    return lr, y_pred, y_pred_proba, training_time, inference_time

def train_svm(X_train, y_train, X_val, y_val):
    """Train SVM model"""
    print("Training SVM...")
    
    from sklearn.svm import SVC
    
    start_time = time.time()
    
    svm = SVC(
        probability=True,
        random_state=42,
        kernel='rbf'
    )
    
    svm.fit(X_train, y_train)
    
    training_time = time.time() - start_time
    
    # Predictions
    start_time = time.time()
    y_pred = svm.predict(X_val)
    y_pred_proba = svm.predict_proba(X_val)[:, 1]
    inference_time = time.time() - start_time
    
    return svm, y_pred, y_pred_proba, training_time, inference_time

def calculate_metrics(y_true, y_pred, y_pred_proba):
    """Calculate comprehensive metrics"""
    metrics = {
        'accuracy': accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred),
        'recall': recall_score(y_true, y_pred),
        'f1_score': f1_score(y_true, y_pred),
        'roc_auc': roc_auc_score(y_true, y_pred_proba)
    }
    
    # Confusion matrix
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    metrics['true_negatives'] = tn
    metrics['false_positives'] = fp
    metrics['false_negatives'] = fn
    metrics['true_positives'] = tp
    metrics['false_positive_rate'] = fp / (fp + tn) if (fp + tn) > 0 else 0
    metrics['false_negative_rate'] = fn / (tp + fn) if (tp + fn) > 0 else 0
    
    return metrics

def create_visualizations(results, training_times, inference_times):
    """Create comparison visualizations"""
    
    # Create results DataFrame
    results_df = pd.DataFrame(results).T
    
    # 1. Performance Metrics Comparison
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle('Model Performance Comparison', fontsize=16)
    
    metrics = ['accuracy', 'precision', 'recall', 'f1_score', 'roc_auc']
    positions = [(0,0), (0,1), (0,2), (1,0), (1,1)]
    
    for metric, pos in zip(metrics, positions):
        ax = axes[pos[0], pos[1]]
        results_df[metric].plot(kind='bar', ax=ax, color='skyblue')
        ax.set_title(metric.title())
        ax.set_ylim(0, 1)
        ax.tick_params(axis='x', rotation=45)
    
    # Add training times
    ax = axes[1, 2]
    pd.Series(training_times).plot(kind='bar', ax=ax, color='lightcoral')
    ax.set_title('Training Time (s)')
    ax.tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'model_performance_comparison.png'), dpi=300, bbox_inches='tight')
    plt.show()
    
    # 2. Training vs Inference Time
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    model_names = list(training_times.keys())
    train_times = list(training_times.values())
    inf_times = list(inference_times.values())
    
    ax1.bar(model_names, train_times, color='blue', alpha=0.7)
    ax1.set_title('Training Time Comparison')
    ax1.set_ylabel('Time (seconds)')
    ax1.tick_params(axis='x', rotation=45)
    
    ax2.bar(model_names, [t*1000 for t in inf_times], color='red', alpha=0.7)
    ax2.set_title('Inference Time Comparison')
    ax2.set_ylabel('Time (milliseconds)')
    ax2.tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'time_comparison.png'), dpi=300, bbox_inches='tight')
    plt.show()
    
    # 3. Error Rate Analysis
    fig, ax = plt.subplots(figsize=(10, 6))
    
    error_metrics = ['false_positive_rate', 'false_negative_rate']
    results_df[error_metrics].plot(kind='bar', ax=ax, color=['orange', 'purple'])
    ax.set_title('Error Rate Analysis')
    ax.set_ylabel('Error Rate')
    ax.tick_params(axis='x', rotation=45)
    ax.legend(['False Positive Rate', 'False Negative Rate'])
    
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'error_rate_analysis.png'), dpi=300, bbox_inches='tight')
    plt.show()
    
    return results_df

def main():
    """Main function to run simplified model comparison"""
    print("="*60)
    print("MLBlocker Simplified Model Comparison")
    print("="*60)
    print("Comparing 4 different ML models for ad blocking performance:")
    print("1. RandomForest")
    print("2. Decision Tree")
    print("3. Logistic Regression")
    print("4. SVM")
    print("="*60)
    
    # Load data
    X, y = load_sample_data()
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train, test_size=0.25, random_state=42, stratify=y_train
    )
    
    print(f"Train set: {X_train.shape}")
    print(f"Validation set: {X_val.shape}")
    print(f"Test set: {X_test.shape}")
    
    # Models to train
    models_to_train = {
        'RandomForest': train_random_forest,
        'DecisionTree': train_decision_tree,
        'LogisticRegression': train_logistic_regression,
        'SVM': train_svm
    }
    
    results = {}
    training_times = {}
    inference_times = {}
    
    # Train and evaluate each model
    for model_name, train_func in models_to_train.items():
        print(f"\n{'='*50}")
        print(f"Training {model_name}")
        print(f"{'='*50}")
        
        try:
            model, y_pred, y_pred_proba, train_time, inf_time = train_func(X_train, y_train, X_val, y_val)
            
            # Calculate metrics
            metrics = calculate_metrics(y_val, y_pred, y_pred_proba)
            
            results[model_name] = metrics
            training_times[model_name] = train_time
            inference_times[model_name] = inf_time
            
            print(f"Training Time: {train_time:.2f}s")
            print(f"Inference Time: {inf_time:.4f}s")
            print(f"Accuracy: {metrics['accuracy']:.4f}")
            print(f"Precision: {metrics['precision']:.4f}")
            print(f"Recall: {metrics['recall']:.4f}")
            print(f"F1 Score: {metrics['f1_score']:.4f}")
            print(f"ROC AUC: {metrics['roc_auc']:.4f}")
            
        except Exception as e:
            print(f"Error training {model_name}: {str(e)}")
            continue
    
    # Create visualizations
    print(f"\n{'='*50}")
    print("Creating Visualizations")
    print(f"{'='*50}")
    
    results_df = create_visualizations(results, training_times, inference_times)
    
    # Create summary table
    summary_df = results_df[['accuracy', 'precision', 'recall', 'f1_score', 'roc_auc']].copy()
    summary_df['Training Time (s)'] = pd.Series(training_times)
    summary_df['Inference Time (ms)'] = pd.Series(inference_times) * 1000
    
    # Rank models by F1 score
    summary_df['Rank'] = summary_df['f1_score'].rank(ascending=False)
    summary_df = summary_df.sort_values('Rank')
    
    # Save summary table
    summary_df.to_csv(os.path.join(OUTPUT_DIR, "simplified_model_comparison.csv"))
    
    print(f"\n{'='*50}")
    print("MODEL COMPARISON SUMMARY")
    print(f"{'='*50}")
    print(summary_df.to_string())
    
    print(f"\nBest model: {summary_df.index[0]}")
    print(f"F1 Score: {summary_df.iloc[0]['f1_score']:.4f}")
    print(f"Accuracy: {summary_df.iloc[0]['accuracy']:.4f}")
    
    print(f"\nGenerated files in '{OUTPUT_DIR}' directory:")
    print("- model_performance_comparison.png")
    print("- time_comparison.png")
    print("- error_rate_analysis.png")
    print("- simplified_model_comparison.csv")
    
    return summary_df

if __name__ == "__main__":
    summary = main()
