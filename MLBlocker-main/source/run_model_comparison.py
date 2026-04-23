#!/usr/bin/env python3
"""
Model Comparison Runner for MLBlocker
This script runs the comprehensive model comparison and generates visualizations.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from model_comparison import main

if __name__ == "__main__":
    print("="*60)
    print("MLBlocker Enhanced Model Comparison")
    print("="*60)
    print("Comparing 6 different ML models for ad blocking performance:")
    print("1. H2O GBM (Original)")
    print("2. RandomForest")
    print("3. XGBoost")
    print("4. LightGBM")
    print("5. CatBoost")
    print("6. Neural Networks (TensorFlow & PyTorch)")
    print("="*60)
    
    try:
        comparison, summary = main()
        
        print("\n" + "="*60)
        print("COMPARISON COMPLETED SUCCESSFULLY!")
        print("="*60)
        print(f"Best model: {summary.index[0]}")
        print(f"F1 Score: {summary.iloc[0]['f1_score']:.4f}")
        print(f"Accuracy: {summary.iloc[0]['accuracy']:.4f}")
        print(f"Training Time: {summary.iloc[0]['Training Time (s)']:.2f}s")
        print(f"Inference Time: {summary.iloc[0]['Inference Time (ms)']:.2f}ms")
        
        print("\nGenerated files:")
        print("- model_performance_comparison.html")
        print("- model_radar_comparison.html")
        print("- time_comparison.html")
        print("- error_rate_analysis.html")
        print("- model_comparison_summary.csv")
        
        print("\nRecommendations:")
        if summary.index[0] == 'XGBoost':
            print("- Use XGBoost for production deployment (highest accuracy)")
        elif summary.index[0] == 'LightGBM':
            print("- Use LightGBM for rapid development (fastest training)")
        elif summary.index[0] == 'CatBoost':
            print("- Use CatBoost for user-facing applications (lowest false positives)")
        else:
            print(f"- Use {summary.index[0]} based on your specific requirements")
        
    except Exception as e:
        print(f"Error during comparison: {str(e)}")
        print("Please ensure all dependencies are installed:")
        print("pip install -r requirements.txt")
        sys.exit(1)
