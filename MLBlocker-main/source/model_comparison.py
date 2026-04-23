import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, roc_curve
)
import xgboost as xgb
import lightgbm as lgb
import catboost as cb
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import time
import os
import json
import yaml
from scipy.stats import pearsonr, spearmanr, pointbiserialr
from sklearn.feature_selection import RFECV
from sklearn.inspection import permutation_importance

# Import existing MLBlocker modules
from mlblocker_encodings import *
import main

DATADIR = os.path.join(os.getcwd(), os.pardir, "dataset")
MODELDIR = os.path.join(os.getcwd(), os.pardir, "model")
OUTDIR = os.path.join(os.getcwd(), os.pardir, "output")

class ModelComparison:
    def __init__(self):
        self.models = {}
        self.results = {}
        self.training_times = {}
        self.inference_times = {}
        
    def load_and_preprocess_data(self):
        """Load and preprocess the MLBlocker dataset"""
        print("Loading and preprocessing data...")
        
        # Load training data
        train_df = pd.read_csv(os.path.join(DATADIR, "all_df_883_train.csv"), index_col=0)
        train_df.drop(columns=['visit_id', 'name'], inplace=True)
        
        # Load content policy dictionary
        with open(os.path.join('json', 'content_type_dict.json'), 'r') as cjson:
            content_dict = json.loads(cjson.read())
            train_df['content_policy_type'] = train_df['content_policy_type'].apply(lambda x: content_dict[x])
        
        # Load features configuration
        with open('features.yaml') as f:
            features = yaml.full_load(f)
        
        # Get feature columns
        robust_features = features['feature_columns_robustness_new']
        unfeasible_features = features['features_unfeasible']
        
        # Filter available features
        available_robust = [f for f in robust_features if f in train_df.columns]
        available_unfeasible = [f for f in unfeasible_features if f in train_df.columns]
        
        # Create feature sets
        robust_df = train_df[available_robust].copy()
        robust_df['label'] = train_df['label'].copy()
        exist_df = train_df.drop(available_robust + available_unfeasible, axis=1)
        
        # Point-biserial correlation filtering
        print("Applying point-biserial correlation filtering...")
        
        # Filter existing features
        filtered_exist_feat = []
        for i in exist_df.columns[:-1]:
            if i in exist_df.columns:
                correlation, p_value = pointbiserialr(exist_df.label, exist_df[i])
                if p_value <= 0.1:
                    filtered_exist_feat.append(i)
        
        exist_df = exist_df[filtered_exist_feat + ['label']]
        
        # Filter robust features
        filtered_robust_feat = []
        for i in robust_df.columns[:-1]:
            if i in robust_df.columns:
                correlation, p_value = pointbiserialr(robust_df.label, robust_df[i])
                if p_value <= 0.1:
                    filtered_robust_feat.append(i)
        
        robust_df = robust_df[filtered_robust_feat + ['label']]
        
        # Combine features
        combined_df = pd.concat([exist_df.drop('label', axis=1), robust_df.drop('label', axis=1)], axis=1)
        combined_df['label'] = train_df['label']
        
        # Remove highly correlated features
        print("Removing highly correlated features...")
        correlation_matrix = combined_df.corr().abs()
        upper_triangle = correlation_matrix.where(
            np.triu(np.ones(correlation_matrix.shape), k=1).astype(bool)
        )
        
        to_drop = [column for column in upper_triangle.columns if any(upper_triangle[column] > 0.95)]
        final_df = combined_df.drop(columns=to_drop)
        
        print(f"Final dataset shape: {final_df.shape}")
        print(f"Features: {list(final_df.columns[:-1])}")
        
        return final_df
    
    def split_data(self, df):
        """Split data into train, validation, and test sets"""
        X = df.drop('label', axis=1)
        y = df['label']
        
        # Split into train+val and test
        X_trainval, X_test, y_trainval, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Split train+val into train and val
        X_train, X_val, y_train, y_val = train_test_split(
            X_trainval, y_trainval, test_size=0.25, random_state=42, stratify=y_trainval
        )
        
        print(f"Train set: {X_train.shape}")
        print(f"Validation set: {X_val.shape}")
        print(f"Test set: {X_test.shape}")
        
        return X_train, X_val, X_test, y_train, y_val, y_test
    
    def train_random_forest(self, X_train, y_train, X_val, y_val):
        """Train Random Forest model"""
        print("Training Random Forest...")
        
        start_time = time.time()
        
        # Create and train model
        rf = RandomForestClassifier(
            n_estimators=200,
            max_depth=20,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=42,
            n_jobs=-1
        )
        
        rf.fit(X_train, y_train)
        
        training_time = time.time() - start_time
        
        # Predictions
        start_time = time.time()
        y_pred = rf.predict(X_val)
        inference_time = time.time() - start_time
        
        return rf, y_pred, training_time, inference_time
    
    def train_xgboost(self, X_train, y_train, X_val, y_val):
        """Train XGBoost model"""
        print("Training XGBoost...")
        
        start_time = time.time()
        
        # Create and train model
        xgb_model = xgb.XGBClassifier(
            n_estimators=200,
            max_depth=6,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            n_jobs=-1,
            eval_metric='logloss'
        )
        
        xgb_model.fit(X_train, y_train)
        
        training_time = time.time() - start_time
        
        # Predictions
        start_time = time.time()
        y_pred = xgb_model.predict(X_val)
        inference_time = time.time() - start_time
        
        return xgb_model, y_pred, training_time, inference_time
    
    def train_lightgbm(self, X_train, y_train, X_val, y_val):
        """Train LightGBM model"""
        print("Training LightGBM...")
        
        start_time = time.time()
        
        # Create and train model
        lgb_model = lgb.LGBMClassifier(
            n_estimators=200,
            max_depth=6,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            n_jobs=-1
        )
        
        lgb_model.fit(X_train, y_train)
        
        training_time = time.time() - start_time
        
        # Predictions
        start_time = time.time()
        y_pred = lgb_model.predict(X_val)
        inference_time = time.time() - start_time
        
        return lgb_model, y_pred, training_time, inference_time
    
    def train_catboost(self, X_train, y_train, X_val, y_val):
        """Train CatBoost model"""
        print("Training CatBoost...")
        
        start_time = time.time()
        
        # Create and train model
        cat_model = cb.CatBoostClassifier(
            iterations=200,
            depth=6,
            learning_rate=0.1,
            loss_function='Logloss',
            random_seed=42,
            verbose=False
        )
        
        cat_model.fit(X_train, y_train)
        
        training_time = time.time() - start_time
        
        # Predictions
        start_time = time.time()
        y_pred = cat_model.predict(X_val)
        inference_time = time.time() - start_time
        
        return cat_model, y_pred, training_time, inference_time
    
    def train_neural_network(self, X_train, y_train, X_val, y_val):
        """Train Neural Network model"""
        print("Training Neural Network...")
        
        start_time = time.time()
        
        # Build model
        model = Sequential([
            Dense(128, activation='relu', input_shape=(X_train.shape[1],)),
            BatchNormalization(),
            Dropout(0.3),
            Dense(64, activation='relu'),
            BatchNormalization(),
            Dropout(0.3),
            Dense(32, activation='relu'),
            BatchNormalization(),
            Dropout(0.2),
            Dense(1, activation='sigmoid')
        ])
        
        # Compile model
        model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss='binary_crossentropy',
            metrics=['accuracy']
        )
        
        # Early stopping
        early_stopping = EarlyStopping(
            monitor='val_loss',
            patience=10,
            restore_best_weights=True
        )
        
        # Train model
        history = model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=100,
            batch_size=32,
            callbacks=[early_stopping],
            verbose=0
        )
        
        training_time = time.time() - start_time
        
        # Predictions
        start_time = time.time()
        y_pred_proba = model.predict(X_val, verbose=0)
        y_pred = (y_pred_proba > 0.5).astype(int).flatten()
        inference_time = time.time() - start_time
        
        return model, y_pred, training_time, inference_time, history
    
    def train_pytorch_nn(self, X_train, y_train, X_val, y_val):
        """Train PyTorch Neural Network model"""
        print("Training PyTorch Neural Network...")
        
        # Convert to tensors
        X_train_tensor = torch.FloatTensor(X_train.values)
        y_train_tensor = torch.FloatTensor(y_train.values)
        X_val_tensor = torch.FloatTensor(X_val.values)
        y_val_tensor = torch.FloatTensor(y_val.values)
        
        # Create datasets and dataloaders
        train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
        val_dataset = TensorDataset(X_val_tensor, y_val_tensor)
        
        train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=32)
        
        # Define model
        class PyTorchNN(nn.Module):
            def __init__(self, input_size):
                super(PyTorchNN, self).__init__()
                self.layer1 = nn.Linear(input_size, 128)
                self.bn1 = nn.BatchNorm1d(128)
                self.dropout1 = nn.Dropout(0.3)
                self.layer2 = nn.Linear(128, 64)
                self.bn2 = nn.BatchNorm1d(64)
                self.dropout2 = nn.Dropout(0.3)
                self.layer3 = nn.Linear(64, 32)
                self.bn3 = nn.BatchNorm1d(32)
                self.dropout3 = nn.Dropout(0.2)
                self.layer4 = nn.Linear(32, 1)
                
            def forward(self, x):
                x = torch.relu(self.bn1(self.layer1(x)))
                x = self.dropout1(x)
                x = torch.relu(self.bn2(self.layer2(x)))
                x = self.dropout2(x)
                x = torch.relu(self.bn3(self.layer3(x)))
                x = self.dropout3(x)
                x = torch.sigmoid(self.layer4(x))
                return x
        
        # Initialize model
        model = PyTorchNN(X_train.shape[1])
        criterion = nn.BCELoss()
        optimizer = optim.Adam(model.parameters(), lr=0.001)
        
        start_time = time.time()
        
        # Training loop
        epochs = 100
        best_val_loss = float('inf')
        patience = 10
        patience_counter = 0
        
        for epoch in range(epochs):
            model.train()
            for batch_X, batch_y in train_loader:
                optimizer.zero_grad()
                outputs = model(batch_X).squeeze()
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()
            
            # Validation
            model.eval()
            val_loss = 0
            with torch.no_grad():
                for batch_X, batch_y in val_loader:
                    outputs = model(batch_X).squeeze()
                    loss = criterion(outputs, batch_y)
                    val_loss += loss.item()
            
            val_loss /= len(val_loader)
            
            # Early stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    break
        
        training_time = time.time() - start_time
        
        # Predictions
        start_time = time.time()
        model.eval()
        with torch.no_grad():
            y_pred_proba = model(X_val_tensor).squeeze().numpy()
            y_pred = (y_pred_proba > 0.5).astype(int)
        inference_time = time.time() - start_time
        
        return model, y_pred, training_time, inference_time
    
    def calculate_metrics(self, y_true, y_pred, y_pred_proba=None):
        """Calculate comprehensive metrics"""
        metrics = {
            'accuracy': accuracy_score(y_true, y_pred),
            'precision': precision_score(y_true, y_pred),
            'recall': recall_score(y_true, y_pred),
            'f1_score': f1_score(y_true, y_pred),
            'roc_auc': roc_auc_score(y_true, y_pred) if y_pred_proba is not None else roc_auc_score(y_true, y_pred)
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
    
    def run_comparison(self):
        """Run complete model comparison"""
        print("Starting model comparison...")
        
        # Load and preprocess data
        df = self.load_and_preprocess_data()
        X_train, X_val, X_test, y_train, y_val, y_test = self.split_data(df)
        
        # Models to compare
        models_to_train = {
            'RandomForest': self.train_random_forest,
            'XGBoost': self.train_xgboost,
            'LightGBM': self.train_lightgbm,
            'CatBoost': self.train_catboost,
            'TensorFlow_NN': self.train_neural_network,
            'PyTorch_NN': self.train_pytorch_nn
        }
        
        # Train and evaluate each model
        for model_name, train_func in models_to_train.items():
            print(f"\n{'='*50}")
            print(f"Training {model_name}")
            print(f"{'='*50}")
            
            try:
                if model_name in ['TensorFlow_NN', 'PyTorch_NN']:
                    if model_name == 'TensorFlow_NN':
                        model, y_pred, train_time, inf_time, history = train_func(X_train, y_train, X_val, y_val)
                        self.models[model_name] = {'model': model, 'history': history}
                    else:
                        model, y_pred, train_time, inf_time = train_func(X_train, y_train, X_val, y_val)
                        self.models[model_name] = {'model': model}
                else:
                    model, y_pred, train_time, inf_time = train_func(X_train, y_train, X_val, y_val)
                    self.models[model_name] = {'model': model}
                
                # Calculate metrics
                y_pred_proba = None
                if hasattr(model, 'predict_proba'):
                    y_pred_proba = model.predict_proba(X_val)[:, 1]
                elif hasattr(model, 'predict') and model_name == 'TensorFlow_NN':
                    y_pred_proba = model.predict(X_val, verbose=0).flatten()
                elif model_name == 'PyTorch_NN':
                    model.eval()
                    with torch.no_grad():
                        y_pred_proba = model(torch.FloatTensor(X_val.values)).squeeze().numpy()
                
                metrics = self.calculate_metrics(y_val, y_pred, y_pred_proba)
                
                self.results[model_name] = metrics
                self.training_times[model_name] = train_time
                self.inference_times[model_name] = inf_time
                
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
        
        # Evaluate on test set for best models
        print(f"\n{'='*50}")
        print("Evaluating best models on test set")
        print(f"{'='*50}")
        
        test_results = {}
        for model_name, model_info in self.models.items():
            if model_name in self.results:
                try:
                    model = model_info['model']
                    
                    if model_name in ['TensorFlow_NN', 'PyTorch_NN']:
                        if model_name == 'TensorFlow_NN':
                            y_pred_proba = model.predict(X_test, verbose=0).flatten()
                            y_pred = (y_pred_proba > 0.5).astype(int)
                        else:
                            model.eval()
                            with torch.no_grad():
                                y_pred_proba = model(torch.FloatTensor(X_test.values)).squeeze().numpy()
                                y_pred = (y_pred_proba > 0.5).astype(int)
                    else:
                        y_pred = model.predict(X_test)
                        y_pred_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, 'predict_proba') else None
                    
                    test_metrics = self.calculate_metrics(y_test, y_pred, y_pred_proba)
                    test_results[model_name] = test_metrics
                    
                    print(f"\n{model_name} Test Results:")
                    print(f"Accuracy: {test_metrics['accuracy']:.4f}")
                    print(f"Precision: {test_metrics['precision']:.4f}")
                    print(f"Recall: {test_metrics['recall']:.4f}")
                    print(f"F1 Score: {test_metrics['f1_score']:.4f}")
                    print(f"ROC AUC: {test_metrics['roc_auc']:.4f}")
                    
                except Exception as e:
                    print(f"Error evaluating {model_name} on test set: {str(e)}")
        
        return self.results, test_results
    
    def create_comparison_visualizations(self):
        """Create comprehensive comparison visualizations"""
        if not self.results:
            print("No results to visualize. Run comparison first.")
            return
        
        # Create results DataFrame
        results_df = pd.DataFrame(self.results).T
        
        # 1. Performance Metrics Comparison
        fig = make_subplots(
            rows=2, cols=3,
            subplot_titles=('Accuracy', 'Precision', 'Recall', 'F1 Score', 'ROC AUC', 'Training Time'),
            specs=[[{"type": "bar"}, {"type": "bar"}, {"type": "bar"}],
                   [{"type": "bar"}, {"type": "bar"}, {"type": "bar"}]]
        )
        
        metrics = ['accuracy', 'precision', 'recall', 'f1_score', 'roc_auc']
        positions = [(1,1), (1,2), (1,3), (2,1), (2,2)]
        
        for metric, pos in zip(metrics, positions):
            fig.add_trace(
                go.Bar(
                    x=results_df.index,
                    y=results_df[metric],
                    name=metric.title(),
                    marker_color='lightblue'
                ),
                row=pos[0], col=pos[1]
            )
        
        # Add training times
        training_times_df = pd.Series(self.training_times)
        fig.add_trace(
            go.Bar(
                x=training_times_df.index,
                y=training_times_df.values,
                name='Training Time (s)',
                marker_color='lightcoral'
            ),
            row=2, col=3
        )
        
        fig.update_layout(
            title_text="Model Performance Comparison",
            showlegend=False,
            height=600
        )
        
        fig.write_html(os.path.join(OUTDIR, "model_performance_comparison.html"))
        fig.show()
        
        # 2. Radar Chart for Overall Performance
        fig_radar = go.Figure()
        
        for model_name in results_df.index:
            fig_radar.add_trace(go.Scatterpolar(
                r=results_df.loc[model_name, metrics].values,
                theta=metrics,
                fill='toself',
                name=model_name
            ))
        
        fig_radar.update_layout(
            polar=dict(
                radialaxis=dict(
                    visible=True,
                    range=[0, 1]
                )),
            showlegend=True,
            title="Model Performance Radar Chart"
        )
        
        fig_radar.write_html(os.path.join(OUTDIR, "model_radar_comparison.html"))
        fig_radar.show()
        
        # 3. Training vs Inference Time Comparison
        fig_time = go.Figure()
        
        model_names = list(self.training_times.keys())
        train_times = [self.training_times[name] for name in model_names]
        inf_times = [self.inference_times[name] for name in model_names]
        
        fig_time.add_trace(go.Bar(
            x=model_names,
            y=train_times,
            name='Training Time (s)',
            marker_color='blue'
        ))
        
        fig_time.add_trace(go.Bar(
            x=model_names,
            y=[t*1000 for t in inf_times],  # Convert to ms
            name='Inference Time (ms)',
            marker_color='red'
        ))
        
        fig_time.update_layout(
            title='Training vs Inference Time Comparison',
            xaxis_title='Models',
            yaxis_title='Time (seconds)',
            barmode='group'
        )
        
        fig_time.write_html(os.path.join(OUTDIR, "time_comparison.html"))
        fig_time.show()
        
        # 4. Error Rate Analysis
        fig_error = go.Figure()
        
        fig_error.add_trace(go.Bar(
            x=results_df.index,
            y=results_df['false_positive_rate'],
            name='False Positive Rate',
            marker_color='orange'
        ))
        
        fig_error.add_trace(go.Bar(
            x=results_df.index,
            y=results_df['false_negative_rate'],
            name='False Negative Rate',
            marker_color='purple'
        ))
        
        fig_error.update_layout(
            title='Error Rate Analysis',
            xaxis_title='Models',
            yaxis_title='Error Rate',
            barmode='group'
        )
        
        fig_error.write_html(os.path.join(OUTDIR, "error_rate_analysis.html"))
        fig_error.show()
        
        # 5. Create summary table
        summary_df = results_df[metrics].copy()
        summary_df['Training Time (s)'] = pd.Series(self.training_times)
        summary_df['Inference Time (ms)'] = pd.Series(self.inference_times) * 1000
        
        # Rank models by F1 score
        summary_df['Rank'] = summary_df['f1_score'].rank(ascending=False)
        summary_df = summary_df.sort_values('Rank')
        
        # Save summary table
        summary_df.to_csv(os.path.join(OUTDIR, "model_comparison_summary.csv"))
        
        print(f"\n{'='*50}")
        print("MODEL COMPARISON SUMMARY")
        print(f"{'='*50}")
        print(summary_df.to_string())
        
        return summary_df
    
    def save_best_model(self, best_model_name):
        """Save the best performing model"""
        if best_model_name not in self.models:
            print(f"Model {best_model_name} not found in results.")
            return
        
        print(f"Saving best model: {best_model_name}")
        
        model = self.models[best_model_name]['model']
        
        if best_model_name == 'TensorFlow_NN':
            model.save(os.path.join(MODELDIR, f"best_model_{best_model_name}.h5"))
        elif best_model_name == 'PyTorch_NN':
            torch.save(model.state_dict(), os.path.join(MODELDIR, f"best_model_{best_model_name}.pth"))
        else:
            import joblib
            joblib.dump(model, os.path.join(MODELDIR, f"best_model_{best_model_name}.joblib"))
        
        print(f"Model saved to {MODELDIR}")

def main():
    """Main function to run model comparison"""
    # Create output directory if it doesn't exist
    if not os.path.exists(OUTDIR):
        os.makedirs(OUTDIR)
    
    # Initialize comparison
    comparison = ModelComparison()
    
    # Run comparison
    val_results, test_results = comparison.run_comparison()
    
    # Create visualizations
    summary_df = comparison.create_comparison_visualizations()
    
    # Determine best model
    best_model = summary_df.index[0]  # Already ranked by F1 score
    print(f"\nBest performing model: {best_model}")
    
    # Save best model
    comparison.save_best_model(best_model)
    
    return comparison, summary_df

if __name__ == "__main__":
    comparison, summary = main()
