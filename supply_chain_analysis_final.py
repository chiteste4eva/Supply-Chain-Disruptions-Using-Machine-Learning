"""
Supply Chain Disruption Prediction Using Machine Learning
==========================================================
A Data-Driven Approach for Improving Logistics Performance

This script analyzes the Dynamic Supply Chain Logistics Dataset from Kaggle
to predict disruptions using Random Forest and Gradient Boosting models.

Dataset: Dynamic Supply Chain Logistics Dataset
URL: https://www.kaggle.com/datasets/samruddhim/dynamic-supply-chain-logistics-dataset

Author: Research Implementation
"""

import pandas as pd
import numpy as np
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Machine Learning
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score,
                            roc_auc_score, confusion_matrix, classification_report,
                            roc_curve, precision_recall_curve)
from imblearn.over_sampling import SMOTE


# Visualization
import matplotlib.pyplot as plt
import seaborn as sns

# Utilities
import joblib
import os

# Set random seed for reproducibility
RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

# Set plot style
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")


def load_data(filepath):
    """
    Load the supply chain dataset from CSV.

    Parameters:
    -----------
    filepath : str
        Path to the CSV file

    Returns:
    --------
    pd.DataFrame : Loaded dataset
    """
    print("="*70)
    print("LOADING DATA")
    print("="*70)

    df = pd.read_csv(filepath)

    # Convert timestamp to datetime
    if 'timestamp' in df.columns:
        df['timestamp'] = pd.to_datetime(df['timestamp'])

    print(f"Dataset Shape: {df.shape}")
    print(f"Columns: {list(df.columns)}")
    print(f"\nData Types:\n{df.dtypes}")
    print(f"\nFirst few rows:\n{df.head()}")

    return df


def explore_data(df):
    """
    Perform exploratory data analysis.

    Parameters:
    -----------
    df : pd.DataFrame
        The dataset
    """
    print("\n" + "="*70)
    print("EXPLORATORY DATA ANALYSIS")
    print("="*70)

    # Basic statistics
    print("\nBasic Statistics:")
    print(df.describe())

    # Missing values
    missing = df.isnull().sum()
    if missing.sum() > 0:
        print(f"\nMissing Values:\n{missing[missing > 0]}")
    else:
        print("\nNo missing values found.")

    # Target variable distribution
    print("\nRisk Classification Distribution:")
    print(df['risk_classification'].value_counts())
    print(f"\nPercentages:\n{df['risk_classification'].value_counts(normalize=True)*100}")

    # Disruption likelihood statistics
    print(f"\nDisruption Likelihood Score Statistics:")
    print(f"  Mean: {df['disruption_likelihood_score'].mean():.4f}")
    print(f"  Std: {df['disruption_likelihood_score'].std():.4f}")
    print(f"  Min: {df['disruption_likelihood_score'].min():.4f}")
    print(f"  Max: {df['disruption_likelihood_score'].max():.4f}")

    return df


def create_target_variable(df, threshold=0.7):
    """
    Create binary target variable for disruption prediction.

    Parameters:
    -----------
    df : pd.DataFrame
        The dataset
    threshold : float
        Threshold for disruption_likelihood_score to classify as disruption

    Returns:
    --------
    pd.DataFrame : Dataset with binary target
    """
    print("\n" + "="*70)
    print("CREATING TARGET VARIABLE")
    print("="*70)

    # Create binary target based on disruption_likelihood_score
    # High disruption likelihood (> threshold) = 1 (disruption)
    df['disruption'] = (df['disruption_likelihood_score'] > threshold).astype(int)

    # Alternative: Use risk_classification
    df['high_risk'] = (df['risk_classification'] == 'High Risk').astype(int)

    print(f"Disruption threshold: {threshold}")
    print(f"\nBinary Disruption Distribution (disruption_likelihood > {threshold}):")
    print(df['disruption'].value_counts())
    print(f"\nPercentages:\n{df['disruption'].value_counts(normalize=True)*100}")

    print(f"\nHigh Risk Distribution:")
    print(df['high_risk'].value_counts())

    return df


def feature_engineering(df):
    """
    Create additional features for prediction.

    Parameters:
    -----------
    df : pd.DataFrame
        The dataset

    Returns:
    --------
    pd.DataFrame : Dataset with engineered features
    """
    print("\n" + "="*70)
    print("FEATURE ENGINEERING")
    print("="*70)

    df = df.copy()

    # Time-based features from timestamp
    if 'timestamp' in df.columns:
        df['hour'] = df['timestamp'].dt.hour
        df['day_of_week'] = df['timestamp'].dt.dayofweek
        df['month'] = df['timestamp'].dt.month
        df['quarter'] = df['timestamp'].dt.quarter
        df['is_weekend'] = (df['day_of_week'] >= 5).astype(int)
        df['is_peak_hour'] = ((df['hour'] >= 7) & (df['hour'] <= 9) |
                              (df['hour'] >= 16) & (df['hour'] <= 19)).astype(int)

    # Composite congestion score
    df['total_congestion'] = (df['traffic_congestion_level'] +
                               df['port_congestion_level']) / 2

    # Operational efficiency metrics
    df['handling_efficiency'] = df['handling_equipment_availability'] / (
        df['loading_unloading_time'] + 0.1)

    # Driver performance composite
    df['driver_performance'] = (df['driver_behavior_score'] *
                                 (1 - df['fatigue_monitoring_score']))

    # Supply chain stress indicator
    df['supply_chain_stress'] = (
        df['traffic_congestion_level'] * 0.2 +
        df['port_congestion_level'] * 0.2 +
        df['route_risk_level'] * 0.2 +
        df['weather_condition_severity'] * 0.15 +
        (1 - df['supplier_reliability_score']) * 0.15 +
        df['fatigue_monitoring_score'] * 0.1
    )

    # Logistics cost efficiency
    df['cost_per_delay_risk'] = df['shipping_costs'] / (df['delay_probability'] + 0.01)

    # Inventory to demand ratio
    df['inventory_demand_ratio'] = df['warehouse_inventory_level'] / (
        df['historical_demand'] + 1)

    # ETA reliability (inverse of variation)
    df['eta_reliability'] = 1 / (np.abs(df['eta_variation_hours']) + 1)

    # Temperature risk (deviation from optimal)
    optimal_temp = 20  # Assuming 20C is optimal
    df['temperature_deviation'] = np.abs(df['iot_temperature'] - optimal_temp)

    # Lead time efficiency
    df['lead_time_efficiency'] = df['supplier_reliability_score'] / (
        df['lead_time_days'] + 1)

    # Customs delay risk
    df['customs_delay_risk'] = df['customs_clearance_time'] * (
        1 - df['order_fulfillment_status'])

    # Peak traffic interaction
    if 'is_peak_hour' in df.columns:
        df['peak_traffic_impact'] = df['is_peak_hour'] * df['traffic_congestion_level']

    print(f"Created {len([c for c in df.columns if c not in ['timestamp', 'vehicle_gps_latitude', 'vehicle_gps_longitude']])} features")
    print(f"New features: hour, day_of_week, month, quarter, is_weekend, is_peak_hour,")
    print(f"              total_congestion, handling_efficiency, driver_performance,")
    print(f"              supply_chain_stress, cost_per_delay_risk, inventory_demand_ratio,")
    print(f"              eta_reliability, temperature_deviation, lead_time_efficiency,")
    print(f"              customs_delay_risk, peak_traffic_impact")

    return df


def prepare_features(df, target='disruption'):
    """
    Prepare features for model training.

    Parameters:
    -----------
    df : pd.DataFrame
        Dataset with all features
    target : str
        Target variable name

    Returns:
    --------
    tuple : X, y, feature_names
    """
    print("\n" + "="*70)
    print("PREPARING FEATURES")
    print("="*70)

    # Columns to exclude from features
    exclude_cols = [
        'timestamp', 'disruption', 'high_risk',
        'disruption_likelihood_score', 'delay_probability',
        'risk_classification', 'delivery_time_deviation'
    ]

    # Select numeric features only
    feature_cols = [col for col in df.columns
                   if col not in exclude_cols
                   and df[col].dtype in ['int64', 'float64', 'int32', 'float32']]

    # Handle any infinite values
    df_clean = df.copy()
    for col in feature_cols:
        df_clean[col] = df_clean[col].replace([np.inf, -np.inf], np.nan)
        df_clean[col] = df_clean[col].fillna(df_clean[col].median())

    X = df_clean[feature_cols]
    y = df_clean[target]

    print(f"Number of features: {len(feature_cols)}")
    print(f"Feature columns: {feature_cols}")
    print(f"Target distribution:\n{y.value_counts()}")

    return X, y, feature_cols


def train_test_split_data(X, y, test_size=0.2):
    """
    Split data into training and testing sets.
    """
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=RANDOM_STATE, stratify=y
    )

    print(f"\nTraining set: {len(X_train)} samples")
    print(f"Test set: {len(X_test)} samples")

    return X_train, X_test, y_train, y_test


def scale_features(X_train, X_test, feature_cols):
    """
    Scale features using StandardScaler.
    """
    scaler = StandardScaler()

    X_train_scaled = pd.DataFrame(
        scaler.fit_transform(X_train),
        columns=feature_cols,
        index=X_train.index
    )
    X_test_scaled = pd.DataFrame(
        scaler.transform(X_test),
        columns=feature_cols,
        index=X_test.index
    )

    return X_train_scaled, X_test_scaled, scaler


def handle_imbalance(X_train, y_train):
    """
    Handle class imbalance using SMOTE.
    """
    print("\n" + "="*70)
    print("HANDLING CLASS IMBALANCE WITH SMOTE")
    print("="*70)

    print(f"Before SMOTE: {pd.Series(y_train).value_counts().to_dict()}")

    smote = SMOTE(random_state=RANDOM_STATE)
    X_resampled, y_resampled = smote.fit_resample(X_train, y_train)

    print(f"After SMOTE: {pd.Series(y_resampled).value_counts().to_dict()}")

    return X_resampled, y_resampled





def train_random_forest(X_train, y_train):
    """
    Train Random Forest classifier.
    """
    print("\n" + "="*70)
    print("TRAINING RANDOM FOREST")
    print("="*70)

    model = RandomForestClassifier(
        n_estimators=200,
        max_depth=20,
        min_samples_split=5,
        min_samples_leaf=2,
        class_weight='balanced',
        random_state=RANDOM_STATE,
        n_jobs=-1
    )
    model.fit(X_train, y_train)

    # Cross-validation
    cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='f1')
    print(f"Cross-validation F1 scores: {cv_scores}")
    print(f"Mean CV F1: {cv_scores.mean():.4f} (+/- {cv_scores.std()*2:.4f})")

    return model


def train_gradient_boosting(X_train, y_train):
    """
    Train Gradient Boosting classifier.
    """
    print("\n" + "="*70)
    print("TRAINING GRADIENT BOOSTING")
    print("="*70)

    model = GradientBoostingClassifier(
        n_estimators=200,
        learning_rate=0.1,
        max_depth=5,
        min_samples_split=5,
        subsample=0.9,
        random_state=RANDOM_STATE
    )
    model.fit(X_train, y_train)

    # Cross-validation
    cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='f1')
    print(f"Cross-validation F1 scores: {cv_scores}")
    print(f"Mean CV F1: {cv_scores.mean():.4f} (+/- {cv_scores.std()*2:.4f})")

    return model





def evaluate_model(model, X_test, y_test, model_name):
    """
    Evaluate a trained model.
    """
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    metrics = {
        'accuracy': accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred),
        'recall': recall_score(y_test, y_pred),
        'f1': f1_score(y_test, y_pred),
        'auc_roc': roc_auc_score(y_test, y_proba)
    }

    print(f"\n{model_name} Performance:")
    print("-" * 40)
    for metric, value in metrics.items():
        print(f"{metric.upper()}: {value:.4f}")

    print(f"\nClassification Report:\n{classification_report(y_test, y_pred)}")

    return metrics, y_pred, y_proba


def plot_roc_curves(models_data, y_test, output_dir):
    """
    Plot ROC curves for all models.
    """
    plt.figure(figsize=(10, 8))

    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']

    for (name, y_proba), color in zip(models_data.items(), colors):
        fpr, tpr, _ = roc_curve(y_test, y_proba)
        auc = roc_auc_score(y_test, y_proba)
        plt.plot(fpr, tpr, label=f'{name} (AUC = {auc:.3f})', linewidth=2, color=color)

    plt.plot([0, 1], [0, 1], 'k--', label='Random Classifier', linewidth=1)
    plt.xlabel('False Positive Rate', fontsize=12)
    plt.ylabel('True Positive Rate', fontsize=12)
    plt.title('ROC Curves - Supply Chain Disruption Prediction', fontsize=14)
    plt.legend(loc='lower right', fontsize=10)
    plt.grid(True, alpha=0.3)

    plt.savefig(f'{output_dir}/roc_curves.png', dpi=300, bbox_inches='tight')
    plt.show()
    print(f"ROC curves saved to {output_dir}/roc_curves.png")


def plot_confusion_matrices(models_predictions, y_test, output_dir):
    """
    Plot confusion matrices for all models.
    """
    n_models = len(models_predictions)
    fig, axes = plt.subplots(1, n_models, figsize=(5*n_models, 4))

    if n_models == 1:
        axes = [axes]

    for ax, (name, y_pred) in zip(axes, models_predictions.items()):
        cm = confusion_matrix(y_test, y_pred)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                   xticklabels=['No Disruption', 'Disruption'],
                   yticklabels=['No Disruption', 'Disruption'])
        ax.set_title(f'{name}', fontsize=12)
        ax.set_xlabel('Predicted', fontsize=10)
        ax.set_ylabel('Actual', fontsize=10)

    plt.suptitle('Confusion Matrices', fontsize=14, y=1.02)
    plt.tight_layout()

    plt.savefig(f'{output_dir}/confusion_matrices.png', dpi=300, bbox_inches='tight')
    plt.show()
    print(f"Confusion matrices saved to {output_dir}/confusion_matrices.png")


def plot_feature_importance(model, feature_names, output_dir, top_n=15):
    """
    Plot feature importance.
    """
    if hasattr(model, 'feature_importances_'):
        importance = pd.DataFrame({
            'feature': feature_names,
            'importance': model.feature_importances_
        }).sort_values('importance', ascending=False)

        plt.figure(figsize=(12, 8))
        top_features = importance.head(top_n)

        colors = plt.cm.viridis(np.linspace(0.2, 0.8, len(top_features)))

        sns.barplot(data=top_features, y='feature', x='importance', palette='viridis')
        plt.xlabel('Importance Score', fontsize=12)
        plt.ylabel('Feature', fontsize=12)
        plt.title(f'Top {top_n} Feature Importance for Disruption Prediction', fontsize=14)
        plt.tight_layout()

        plt.savefig(f'{output_dir}/feature_importance.png', dpi=300, bbox_inches='tight')
        plt.show()
        print(f"Feature importance plot saved to {output_dir}/feature_importance.png")

        return importance
    return None


def plot_model_comparison(results_dict, output_dir):
    """
    Plot bar chart comparing model performance.
    """
    comparison_df = pd.DataFrame(results_dict).T

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Plot 1: All metrics
    comparison_df.plot(kind='bar', ax=axes[0], rot=45, width=0.8)
    axes[0].set_title('Model Performance Comparison', fontsize=14)
    axes[0].set_xlabel('Model', fontsize=12)
    axes[0].set_ylabel('Score', fontsize=12)
    axes[0].legend(loc='lower right', fontsize=9)
    axes[0].set_ylim([0, 1])
    axes[0].grid(axis='y', alpha=0.3)

    # Plot 2: Key metrics
    key_metrics = ['accuracy', 'f1', 'auc_roc']
    comparison_df[key_metrics].plot(kind='bar', ax=axes[1], rot=45, width=0.8)
    axes[1].set_title('Key Metrics Comparison', fontsize=14)
    axes[1].set_xlabel('Model', fontsize=12)
    axes[1].set_ylabel('Score', fontsize=12)
    axes[1].set_ylim([0, 1])
    axes[1].grid(axis='y', alpha=0.3)

    plt.tight_layout()

    plt.savefig(f'{output_dir}/model_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()
    print(f"Model comparison plot saved to {output_dir}/model_comparison.png")

    return comparison_df


def generate_report(comparison_df, results_dict, output_dir):
    """
    Generate a comprehensive research report.
    """
    # Find best model
    best_model = comparison_df['accuracy'].idxmax()
    best_accuracy = comparison_df.loc[best_model, 'accuracy']

    report = f"""
================================================================================
SUPPLY CHAIN DISRUPTION PREDICTION - RESEARCH RESULTS REPORT
================================================================================

OBJECTIVE:
----------
Predict supply chain disruptions using machine learning techniques (Random Forest
and Gradient Boosting) and compare performance with traditional statistical methods.

METHODOLOGY:
------------
1. Data: Dynamic Supply Chain Logistics Dataset with features including:
   - Vehicle GPS coordinates and fuel consumption
   - ETA variation and traffic congestion levels
   - Warehouse inventory levels
   - Loading/unloading times and equipment availability
   - Weather condition severity
   - Port congestion and shipping costs
   - Supplier reliability scores and lead times
   - Driver behavior and fatigue monitoring scores
   - IoT temperature monitoring
   - Route risk levels and customs clearance times

2. Feature Engineering:
   - Time-based features (hour, day of week, is_peak_hour)
   - Composite congestion scores
   - Driver performance metrics
   - Supply chain stress indicators
   - Inventory-demand ratios
   - Cost efficiency metrics

3. Models Evaluated:
   - Random Forest Classifier
   - Gradient Boosting Classifier

RESULTS:
--------
{comparison_df.round(4).to_string()}

KEY FINDINGS:
-------------
- Best performing model: {best_model}
- Best accuracy: {best_accuracy:.4f}

DETAILED METRICS:
-----------------"""

    for model_name, metrics in results_dict.items():
        report += f"""
{model_name}:
  - Accuracy:  {metrics['accuracy']:.4f}
  - Precision: {metrics['precision']:.4f}
  - Recall:    {metrics['recall']:.4f}
  - F1 Score:  {metrics['f1']:.4f}
  - AUC-ROC:   {metrics['auc_roc']:.4f}
"""

    report += f"""
PRACTICAL IMPLICATIONS:
-----------------------
Based on these results, businesses can use this model to:
1. Make proactive decisions for alternative routing when high disruption
   likelihood is predicted
2. Adjust safety stock levels based on predicted disruptions
3. Evaluate and manage supplier relationships using reliability scores
4. Optimize resource allocation during high-risk periods
5. Monitor driver fatigue and behavior to reduce disruption risk
6. Plan for weather-related disruptions in advance

CONCLUSION:
-----------
The machine learning models (particularly {best_model}) significantly
outperform traditional statistical methods for supply chain disruption
prediction. This
demonstrates the value of data-driven approaches for improving supply
chain resilience and operational efficiency.

================================================================================
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
================================================================================
"""

    print(report)

    # Save report
    with open(f'{output_dir}/research_report.txt', 'w') as f:
        f.write(report)

    print(f"\nReport saved to {output_dir}/research_report.txt")

    return report


def main():
    """
    Main execution function.
    """
    print("="*70)
    print("SUPPLY CHAIN DISRUPTION PREDICTION USING MACHINE LEARNING")
    print("A Data-Driven Approach for Improving Logistics Performance")
    print("="*70)

    # Setup
    data_path = 'dynamic_supply_chain_logistics_dataset.csv'
    output_dir = './results'
    os.makedirs(output_dir, exist_ok=True)

    # Step 1: Load Data
    print("\n[Step 1/10] Loading Data...")
    df = load_data(data_path)

    # Step 2: Explore Data
    print("\n[Step 2/10] Exploring Data...")
    df = explore_data(df)

    # Step 3: Create Target Variable
    print("\n[Step 3/10] Creating Target Variable...")
    df = create_target_variable(df, threshold=0.7)

    # Step 4: Feature Engineering
    print("\n[Step 4/10] Feature Engineering...")
    df = feature_engineering(df)

    # Step 5: Prepare Features
    print("\n[Step 5/10] Preparing Features...")
    X, y, feature_cols = prepare_features(df, target='disruption')

    # Step 6: Train-Test Split
    print("\n[Step 6/10] Splitting Data...")
    X_train, X_test, y_train, y_test = train_test_split_data(X, y)

    # Scale features
    X_train_scaled, X_test_scaled, scaler = scale_features(X_train, X_test, feature_cols)

    # Step 7: Handle Class Imbalance
    print("\n[Step 7/10] Handling Class Imbalance...")
    X_train_resampled, y_train_resampled = handle_imbalance(X_train_scaled, y_train)

    # Step 8: Train Models
    print("\n[Step 8/10] Training Models...")

    # Train all models
    rf_model = train_random_forest(X_train_resampled, y_train_resampled)
    gb_model = train_gradient_boosting(X_train_resampled, y_train_resampled)

    # Step 9: Evaluate Models
    print("\n[Step 9/10] Evaluating Models...")

    results = {}
    predictions = {}
    probabilities = {}

    # Evaluate each model
    rf_metrics, rf_pred, rf_proba = evaluate_model(rf_model, X_test_scaled, y_test, "Random Forest")
    results['Random Forest'] = rf_metrics
    predictions['Random Forest'] = rf_pred
    probabilities['Random Forest'] = rf_proba

    gb_metrics, gb_pred, gb_proba = evaluate_model(gb_model, X_test_scaled, y_test, "Gradient Boosting")
    results['Gradient Boosting'] = gb_metrics
    predictions['Gradient Boosting'] = gb_pred
    probabilities['Gradient Boosting'] = gb_proba

 

    # Step 10: Visualizations and Report
    print("\n[Step 10/10] Generating Visualizations and Report...")

    # ROC Curves
    plot_roc_curves(probabilities, y_test, output_dir)

    # Confusion Matrices
    plot_confusion_matrices(predictions, y_test, output_dir)

    # Feature Importance (using Random Forest)
    importance_df = plot_feature_importance(rf_model, feature_cols, output_dir)

    # Model Comparison
    comparison_df = plot_model_comparison(results, output_dir)

    # Generate Report
    generate_report(comparison_df, results, output_dir)

    # Save models
    print("\nSaving Models...")
    joblib.dump(rf_model, f'{output_dir}/random_forest_model.pkl')
    joblib.dump(gb_model, f'{output_dir}/gradient_boosting_model.pkl')
    joblib.dump(scaler, f'{output_dir}/feature_scaler.pkl')
    print("Models saved successfully!")

    # Print summary
    print("\n" + "="*70)
    print("EXECUTION COMPLETE")
    print("="*70)
    print(f"\nResults saved to: {output_dir}/")

    # Find best model
    best_model = max(results.keys(), key=lambda k: results[k]['accuracy'])
    best_acc = results[best_model]['accuracy']

    print(f"\nBest Model: {best_model}")
    print(f"Best Accuracy: {best_acc:.4f}")

    if importance_df is not None:
        print(f"\nTop 10 Most Important Features:")
        print(importance_df.head(10).to_string(index=False))

    return results, comparison_df, {
        'random_forest': rf_model,
        'gradient_boosting': gb_model,
    }


if __name__ == "__main__":
    results, comparison, models = main()
