"""
Iris Model Evaluation & Visualization
-------------------------------------
This script accompanies 'train_iris.py'. It runs sanity tests to ensure
models are performing above a baseline and generates visualizations
for better interpretability.

Usage:
    Run this script after configuring 'train_iris.py' to see graphical results.
"""

import unittest
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
import sys
import os

# Ensure we can import from the current directory
sys.path.append(os.getcwd())

try:
    from train_iris import load_and_preprocess_data, train_svm, train_rf
except ImportError:
    # If running from a different directory, try relative import adjustment
    sys.path.append(os.path.dirname(__file__))
    from train_iris import load_and_preprocess_data, train_svm, train_rf

# ==========================================
#              UNIT TESTS
# ==========================================

class TestModelPerformance(unittest.TestCase):
    """
    Sanity checks for model performance.
    """
    
    @classmethod
    def setUpClass(cls):
        print("[-] Setting up test environment (training models)...")
        cls.X_train, cls.X_test, cls.y_train, cls.y_test, cls.class_names = load_and_preprocess_data()
        cls.svm_model = train_svm(cls.X_train, cls.y_train)
        cls.rf_model = train_rf(cls.X_train, cls.y_train)

    def test_svm_accuracy_baseline(self):
        """Test if SVM accuracy is acceptable (> 85%)"""
        acc = self.svm_model.score(self.X_test, self.y_test)
        self.assertGreater(acc, 0.85, f"SVM Accuracy too low: {acc:.2f}")

    def test_rf_accuracy_baseline(self):
        """Test if Random Forest accuracy is acceptable (> 85%)"""
        acc = self.rf_model.score(self.X_test, self.y_test)
        self.assertGreater(acc, 0.85, f"RF Accuracy too low: {acc:.2f}")

# ==========================================
#           VISUALIZATION LOGIC
# ==========================================

def plot_confusion_matrix(model, X_test, y_test, class_names, title, ax):
    """
    Plots a confusion matrix using Seaborn on a specific axis.
    """
    y_pred = model.predict(X_test)
    cm = confusion_matrix(y_test, y_pred)
    acc = model.score(X_test, y_test)
    
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names, ax=ax, cbar=False)
    ax.set_xlabel('Predicted')
    ax.set_ylabel('Actual')
    ax.set_title(f'CM: {title}\n(Accuracy: {acc:.6f})')

def plot_feature_importance(model, feature_names, ax):
    """
    Plots feature importance for Tree-based models on a specific axis.
    """
    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1]
    
    ax.set_title("Random Forest Feature Importances")
    ax.bar(range(len(importances)), importances[indices], align="center", color='skyblue')
    ax.set_xticks(range(len(importances)))
    ax.set_xticklabels([feature_names[i] for i in indices], rotation=45)
    ax.set_ylabel("Importance Score")

def run_visualizations():
    print("\n[-] Generating Visualizations Dashboard...")
    
    X_train, X_test, y_train, y_test, class_names = load_and_preprocess_data()
    svm_model = train_svm(X_train, y_train)
    rf_model = train_rf(X_train, y_train)
    
    feature_names = ['sepal length', 'sepal width', 'petal length', 'petal width']

    # Create a 1x3 dashboard for a side-by-side view
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    # 1. SVM Confusion Matrix
    plot_confusion_matrix(svm_model, X_test, y_test, class_names, "SVM", axes[0])
    
    # 2. RF Confusion Matrix
    plot_confusion_matrix(rf_model, X_test, y_test, class_names, "Random Forest", axes[1])
    
    # 3. RF Feature Importance
    plot_feature_importance(rf_model, feature_names, axes[2])
    
    plt.suptitle("Iris Classification: Model Comparison Dashboard", fontsize=16)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    print("[-] Dashboard ready. Displaying window...")
    plt.show()

if __name__ == "__main__":
    # Run Tests First
    print("Running Tests...")
    # Load tests from TestCase
    suite = unittest.TestLoader().loadTestsFromTestCase(TestModelPerformance)
    test_result = unittest.TextTestRunner(verbosity=2).run(suite)
    
    if test_result.wasSuccessful():
        print("\nAll tests passed! Proceeding to visualizations...")
        try:
            run_visualizations()
        except Exception as e:
            print(f"Visualization failed: {e}")
            print("Note: If you are running in a headless environment (no display), plots might not show.")
    else:
        print("\nTests failed. Skipping visualizations.")
