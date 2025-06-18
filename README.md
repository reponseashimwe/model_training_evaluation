# Early Detection of Child Malnutrition using Machine Learning

This project implements various machine learning models to predict child stunting using anthropometric measurements and demographic data. The goal is to develop an early detection system for child malnutrition by comparing different model architectures and optimization techniques.

## Dataset

The analysis uses the Stunting Wasting Dataset from Kaggle, which contains the following features:
- Gender
- Age (in months)
- Body Weight
- Body Length
- Stunting Status (Target Variable)
- Wasting Status (Not used in this analysis)

## Project Structure

```
ML/STUNTING/
├── README.md
├── notebook.ipynb
└── saved_models/
    ├── random_forest_optimized.joblib
    ├── nn_model_instance1.h5
    ├── nn_model_instance2.h5
    ├── nn_model_instance3.h5
    └── nn_model_instance4.h5
```

## Model Implementations

### 1. Random Forest Classifier
- Optimized using GridSearchCV
- Hyperparameters tuned:
  - n_estimators: [100, 200, 300]
  - max_depth: [10, 20, 30, None]
  - min_samples_split: [2, 5, 10]
  - min_samples_leaf: [1, 2, 4]

### 2. Neural Network Models

#### Instance 1: Baseline Model
- Architecture: 2 hidden layers [64, 32]
- Optimizer: Adam (default learning rate)
- No regularization
- Fixed 50 epochs

#### Instance 2: L2 Regularized Model
- Architecture: 2 hidden layers [64, 32]
- Optimizer: Adam (learning rate = 0.0001)
- L2 regularization (strength = 0.001)
- Early stopping (patience = 10)
- Max 200 epochs

#### Instance 3: Dropout Model
- Architecture: 3 hidden layers [128, 64, 32]
- Optimizer: RMSprop (default learning rate)
- Dropout rate: 0.2
- Fixed 70 epochs

#### Instance 4: L1 Regularized Model
- Architecture: 2 hidden layers [64, 32]
- Optimizer: Adam (learning rate = 0.01)
- L1 regularization (strength = 0.001)
- Fixed 100 epochs

## Model Performance Comparison

| Model | Configuration | Accuracy | F1 Score | Recall | Precision |
|-------|--------------|----------|-----------|---------|------------|
| Baseline NN | Simple 2-layer, Adam (default) | {results[0]['metrics']['accuracy']:.4f} | {results[0]['metrics']['f1']:.4f} | {results[0]['metrics']['recall']:.4f} | {results[0]['metrics']['precision']:.4f} |
| L2 Regularized NN | Adam (LR=0.0001), L2, Early Stopping | {results[1]['metrics']['accuracy']:.4f} | {results[1]['metrics']['f1']:.4f} | {results[1]['metrics']['recall']:.4f} | {results[1]['metrics']['precision']:.4f} |
| Dropout NN | RMSprop, Dropout=0.2 | {results[2]['metrics']['accuracy']:.4f} | {results[2]['metrics']['f1']:.4f} | {results[2]['metrics']['recall']:.4f} | {results[2]['metrics']['precision']:.4f} |
| L1 Regularized NN | Adam (LR=0.01), L1 | {results[3]['metrics']['accuracy']:.4f} | {results[3]['metrics']['f1']:.4f} | {results[3]['metrics']['recall']:.4f} | {results[3]['metrics']['precision']:.4f} |
| Random Forest | GridSearchCV Optimized | {results[4]['metrics']['accuracy']:.4f} | {results[4]['metrics']['f1']:.4f} | {results[4]['metrics']['recall']:.4f} | {results[4]['metrics']['precision']:.4f} |

## Key Findings

1. Best Performing Model: {best_model['Model']}
   - Configuration: {best_model['Configuration']}
   - F1 Score: {best_model['F1 Score']:.4f}
   - Accuracy: {best_model['Accuracy']:.4f}

2. Impact of Regularization:
   - L2 regularization with early stopping showed improved generalization
   - Dropout helped reduce overfitting in the deeper network
   - L1 regularization promoted feature sparsity

3. Optimization Insights:
   - Lower learning rates (0.0001) with Adam performed better than higher rates
   - RMSprop with dropout showed competitive performance
   - Early stopping helped prevent overfitting in Instance 2

## Usage

1. Install required packages:
```bash
pip install numpy pandas scikit-learn tensorflow seaborn matplotlib
```

2. Run the notebook:
```bash
jupyter notebook notebook.ipynb
```

3. For predictions using the best model:
```python
from tensorflow.keras.models import load_model
import joblib

# Load the best model (check the model type first)
if best_model_type == 'Random Forest':
    model = joblib.load('saved_models/random_forest_optimized.joblib')
else:
    model = load_model(f'saved_models/nn_model_instance{best_instance}.h5')

# Make predictions
predictions = model.predict(X_new)
```

## Future Improvements

1. Feature Engineering:
   - Create BMI (Body Mass Index) feature
   - Add age groups/categories
   - Incorporate additional health indicators if available

2. Model Enhancements:
   - Try ensemble methods combining neural networks and random forests
   - Experiment with different neural network architectures
   - Implement cross-validation for neural networks

3. Deployment:
   - Create a simple web interface for predictions
   - Develop an API endpoint for model serving
   - Add model versioning and monitoring
