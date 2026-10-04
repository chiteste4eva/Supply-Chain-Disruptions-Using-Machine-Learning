# Supply-Chain-Disruptions-Using-Machine-Learning
# Predicting Supply Chain Disruptions Using Machine Learning

## Overview
This project explores how machine learning can predict supply chain disruptions using logistics operational data. It compares Random Forest and Gradient Boosting to support proactive logistics decisions.

## Dataset
The project uses the Dynamic Supply Chain Logistics Dataset.

Features include:
- Transportation metrics and estimated arrival time variations
- Traffic and port congestion
- Inventory levels and warehouse handling times
- Weather conditions
- Supplier reliability and lead times
- Driver behavior and fatigue
- Shipping costs and order fulfillment status

## Methodology
- Handle missing values using median imputation.
- Scale numerical features using StandardScaler.
- Address class imbalance using SMOTE.
- Use a stratified 80% training and 20% testing split.
- Train and compare Random Forest and Gradient Boosting.

Engineered features include congestion scores, driver performance indices, supply chain stress indicators, and operational efficiency metrics.

## Reported Results

| Model | Accuracy | Precision | Recall | F1 Score | ROC-AUC |
|---|---:|---:|---:|---:|---:|
| Random Forest | 66.71% | 75.11% | 82.88% | 0.7880 | 0.5026 |
| Gradient Boosting | 73.87% | 74.62% | 98.52% | 0.8492 | 0.4920 |

Gradient Boosting achieved higher accuracy, recall, and F1 score in the reported experiment.

## Limitations
Both ROC-AUC scores are close to 0.50, indicating limited ability to distinguish disruptions from non-disruptions across thresholds. Further validation and comparison with a baseline model are needed before operational deployment.

## Potential Business Applications
- Identify shipments that may require intervention.
- Support alternative routing decisions.
- Inform inventory and safety stock planning.
- Monitor supplier reliability.
- Improve planning for congestion, weather, and driver-related risks.

## Author
Nwankwo Chibuzo Joseph
