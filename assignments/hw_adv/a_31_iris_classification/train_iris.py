"""
Iris Classification Training Script
-----------------------------------
This script trains Support Vector Machine (SVM) and Random Forest classifiers
on the classic Iris dataset.

It is designed with configuration knobs at the top to allow easy experimentation
with hyperparameters and training settings.

Dependencies:
    - scikit-learn
    - pandas
    - numpy
"""

from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
import joblib
import os

# Set working directory to the script's location
os.chdir(os.path.dirname(os.path.abspath(__file__)))

# ==========================================
#        CONFIGURATION KNOBS
# ==========================================

# -- Data Splitting --
TEST_SIZE = 0.2          # Proportion of the dataset to include in the test split (0.0 to 1.0)
RANDOM_STATE = 42        # Seed for reproducibility

# -- SVM Hyperparameters --
SVM_C = 1.0              # Regularization parameter. The strength of the regularization is inversely proportional to C.
SVM_KERNEL = 'rbf'       # Specifies the kernel type to be used in the algorithm ('linear', 'poly', 'rbf', 'sigmoid')
SVM_GAMMA = 'scale'      # Kernel coefficient for 'rbf', 'poly' and 'sigmoid'.

# -- Random Forest Hyperparameters --
RF_N_ESTIMATORS = 100    # The number of trees in the forest.
RF_MAX_DEPTH = None      # The maximum depth of the tree. If None, nodes are expanded until all leaves are pure.
RF_MIN_SAMPLES_SPLIT = 2 # The minimum number of samples required to split an internal node.

# -- Output Paths --
MODEL_DIR = 'models'
SVM_MODEL_PATH = os.path.join(MODEL_DIR, 'svm_iris_model.joblib')
RF_MODEL_PATH = os.path.join(MODEL_DIR, 'rf_iris_model.joblib')

# ==========================================
#           MAIN EXECUTION
# ==========================================

def load_and_preprocess_data():
    """
    Loads Iris data, splits it into train/test sets, and scales features.
    """
    print("[-] Loading Iris dataset...")
    iris = datasets.load_iris()
    X = iris.data
    y = iris.target
    class_names = iris.target_names

    # Split data
    print(f"[-] Splitting data (Test Size: {TEST_SIZE})...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )

    # Scale data (Important for SVM)
    print("[-] Scaling features...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    return X_train_scaled, X_test_scaled, y_train, y_test, class_names

def train_svm(X_train, y_train):
    """
    Trains a Support Vector Machine classifier.
    """
    print(f"[-] Training SVM (C={SVM_C}, Kernel={SVM_KERNEL})...")
    svm_clf = SVC(C=SVM_C, kernel=SVM_KERNEL, gamma=SVM_GAMMA, random_state=RANDOM_STATE)
    svm_clf.fit(X_train, y_train)
    return svm_clf

def train_rf(X_train, y_train):
    """
    Trains a Random Forest classifier.
    """
    print(f"[-] Training Random Forest (Trees={RF_N_ESTIMATORS}, Max Depth={RF_MAX_DEPTH})...")
    rf_clf = RandomForestClassifier(
        n_estimators=RF_N_ESTIMATORS, 
        max_depth=RF_MAX_DEPTH, 
        min_samples_split=RF_MIN_SAMPLES_SPLIT,
        random_state=RANDOM_STATE
    )
    rf_clf.fit(X_train, y_train)
    return rf_clf

def evaluate_model(model, X_test, y_test, class_names, model_name="Model"):
    """
    Evaluates a trained model and prints the report.
    """
    print(f"\n--- Evaluation Results: {model_name} ---")
    y_pred = model.predict(X_test)
    
    acc = accuracy_score(y_test, y_pred)
    print(f"Accuracy: {acc:.4f}")
    
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=class_names))
    
    return acc

def save_models(svm_model, rf_model):
    """
    Saves trained models to disk.
    """
    if not os.path.exists(MODEL_DIR):
        os.makedirs(MODEL_DIR)
        
    joblib.dump(svm_model, SVM_MODEL_PATH)
    joblib.dump(rf_model, RF_MODEL_PATH)
    print(f"[-] Models saved to '{MODEL_DIR}/' directory.")

def main():
    # 1. Prepare Data
    X_train, X_test, y_train, y_test, class_names = load_and_preprocess_data()
    
    # 2. Train Models
    svm_model = train_svm(X_train, y_train)
    rf_model = train_rf(X_train, y_train)
    
    # 3. Evaluate
    evaluate_model(svm_model, X_test, y_test, class_names, "Support Vector Machine")
    evaluate_model(rf_model, X_test, y_test, class_names, "Random Forest")

    # 4. Save
    save_models(svm_model, rf_model)

if __name__ == "__main__":
    main()
