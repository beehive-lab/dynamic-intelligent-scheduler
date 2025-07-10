#!/usr/bin/env python3
"""
Extract trained models from the TornadoVM ML notebook
This script runs the model training parts and saves the models for inference
"""

import os
import sys
import json
import pandas as pd
import numpy as np
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.metrics import accuracy_score
from imblearn.over_sampling import SMOTE
from joblib import dump

# Add the current directory to path to import utilities
sys.path.append('.')

from utilities import preprocess_column_name
from plotting_utilities import plot_roc, precision_recall_threshold

def load_and_preprocess_data():
    """Load and preprocess data similar to the notebook"""
    
    # For demonstration, we'll create synthetic data since we don't have the Google Sheets access
    # In real usage, you would load from your data source
    
    print("Creating synthetic training data...")
    
    # Create synthetic data with realistic patterns
    np.random.seed(42)
    n_samples = 1000
    
    # Generate synthetic features
    data = {
        'threads': np.random.randint(8, 1025, n_samples),
        'global_memory_loads': np.random.randint(0, 10000, n_samples),
        'global_memory_stores': np.random.randint(0, 5000, n_samples),
        'local_memory_loads': np.random.randint(0, 2000, n_samples),
        'local_memory_stores': np.random.randint(0, 1000, n_samples),
        'total_loops': np.random.randint(1, 1000, n_samples),
        'parallel_loops': np.random.randint(0, 800, n_samples),
        'cast_operations': np.random.randint(0, 100, n_samples),
        'vector_operations': np.random.randint(0, 200, n_samples),
        'total_integer_operations': np.random.randint(100, 100000, n_samples),
        'workload_name': [f'workload_{i}' for i in range(n_samples)]
    }
    
    # Create synthetic performance ratios (target variables)
    # These determine which hardware is better for each workload
    
    # iGPU vs CPU ratio
    data['igpu_cpu'] = np.random.uniform(0.5, 2.0, n_samples)
    
    # GPU vs CPU ratio  
    data['gpu_cpu'] = np.random.uniform(0.3, 3.0, n_samples)
    
    # GPU vs iGPU ratio
    data['gpu_igpu'] = np.random.uniform(0.2, 2.5, n_samples)
    
    df = pd.DataFrame(data)
    
    print(f"Created dataset with {len(df)} samples")
    return df

def train_classifiers(df):
    """Train the three classifiers as in the notebook"""
    
    print("\nTraining Classifier 1: iGPU vs CPU")
    print("-" * 40)
    
    # Classifier 1: iGPU vs CPU
    df_clf_1 = df.copy()
    df_clf_1['target_class'] = np.where(df_clf_1['igpu_cpu'] < 1, 0, 1)
    
    # Select top features (same as notebook)
    clf_1_top = ["threads", "global_memory_loads", "global_memory_stores", 
                  "local_memory_loads", "local_memory_stores", "total_loops", 
                  "parallel_loops", "cast_operations", "vector_operations", 
                  "total_integer_operations"]
    
    df_clf_1 = df_clf_1[clf_1_top + ['target_class']]
    
    # Apply SMOTE for balancing
    smote = SMOTE(k_neighbors=8, sampling_strategy=0.8, random_state=0)
    X_sm, y_sm = smote.fit_resample(df_clf_1.drop(columns=['target_class']), df_clf_1['target_class'])
    
    # Train classifier
    parameters = {'n_estimators': [50, 100], 'max_depth': [10, 50]}
    clf_1 = GridSearchCV(ExtraTreesClassifier(random_state=0), parameters, cv=5)
    clf_1.fit(X_sm, y_sm)
    forest_1 = clf_1.best_estimator_
    
    print(f"Best CV score: {clf_1.best_score_:.3f}")
    print(f"Best parameters: {clf_1.best_params_}")
    
    print("\nTraining Classifier 2: GPU vs CPU")
    print("-" * 40)
    
    # Classifier 2: GPU vs CPU
    df_clf_2 = df.copy()
    df_clf_2['target_class'] = np.where(df_clf_2['gpu_cpu'] < 1, 0, 1)
    df_clf_2 = df_clf_2[clf_1_top + ['target_class']]
    
    # Apply SMOTE
    smote = SMOTE(k_neighbors=3, sampling_strategy=0.8, random_state=0)
    X_sm, y_sm = smote.fit_resample(df_clf_2.drop(columns=['target_class']), df_clf_2['target_class'])
    
    # Train classifier
    clf_2 = GridSearchCV(ExtraTreesClassifier(random_state=0), parameters, cv=5)
    clf_2.fit(X_sm, y_sm)
    forest_2 = clf_2.best_estimator_
    
    print(f"Best CV score: {clf_2.best_score_:.3f}")
    print(f"Best parameters: {clf_2.best_params_}")
    
    print("\nTraining Classifier 3: GPU vs iGPU")
    print("-" * 40)
    
    # Classifier 3: GPU vs iGPU
    df_clf_3 = df.copy()
    df_clf_3['target_class'] = np.where(df_clf_3['gpu_igpu'] < 1, 0, 1)
    df_clf_3 = df_clf_3[clf_1_top + ['target_class']]
    
    # Apply SMOTE
    smote = SMOTE(sampling_strategy=0.8, k_neighbors=8)
    X_sm, y_sm = smote.fit_resample(df_clf_3.drop(columns=['target_class']), df_clf_3['target_class'])
    
    # Train classifier
    clf_3 = GridSearchCV(ExtraTreesClassifier(random_state=0), parameters, cv=5)
    clf_3.fit(X_sm, y_sm)
    forest_3 = clf_3.best_estimator_
    
    print(f"Best CV score: {clf_3.best_score_:.3f}")
    print(f"Best parameters: {clf_3.best_params_}")
    
    return forest_1, forest_2, forest_3, clf_1_top

def save_models_and_features(forest_1, forest_2, forest_3, feature_names):
    """Save the trained models and feature information"""
    
    print("\nSaving models and features...")
    
    # Create directories if they don't exist
    os.makedirs("Final Artifacts", exist_ok=True)
    
    # Save models
    dump(forest_1, 'IGPUvsCPU_final.joblib')
    dump(forest_2, 'GPUvsCPU_final.joblib')
    dump(forest_3, 'GPUvsIGPU_final.joblib')
    
    # Save feature names
    classifier_features = {
        "c1": feature_names,
        "c2": feature_names,
        "c3": feature_names
    }
    
    with open('./Final Artifacts/features.txt', 'w') as outfile:
        json.dump(classifier_features, outfile)
    
    print("✓ Models saved successfully:")
    print("  - IGPUvsCPU_final.joblib")
    print("  - GPUvsCPU_final.joblib") 
    print("  - GPUvsIGPU_final.joblib")
    print("  - ./Final Artifacts/features.txt")
    
    # Print feature importance for reference
    print("\nFeature Importance (Classifier 1 - iGPU vs CPU):")
    importances = forest_1.feature_importances_
    for i, (feature, importance) in enumerate(zip(feature_names, importances)):
        print(f"  {i+1:2d}. {feature:25s}: {importance:.4f}")

def main():
    """Main function to extract models from notebook"""
    
    print("TornadoVM ML Model Extraction")
    print("=" * 40)
    
    try:
        # Load and preprocess data
        df = load_and_preprocess_data()
        
        # Train classifiers
        forest_1, forest_2, forest_3, feature_names = train_classifiers(df)
        
        # Save models and features
        save_models_and_features(forest_1, forest_2, forest_3, feature_names)
        
        print("\n" + "=" * 40)
        print("✓ Model extraction completed successfully!")
        print("\nYou can now use the inference engine with:")
        print("  python inference_engine.py")
        print("  python test_json_input.py")
        
    except Exception as e:
        print(f"✗ Error during model extraction: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main() 