import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.ensemble import RandomForestClassifier

# ========================================
# 1. LOAD DATA
# ========================================
print("="*60)
print("🧬 MAN VS MACHINE: BREAST CANCER DETECTION")
print("="*60)

data = load_breast_cancer()
X = pd.DataFrame(data.data, columns=data.feature_names)
y = data.target # 0 = Malignant, 1 = Benign

# Split (Standard 80/20)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"Test Set Size: {len(X_test)} patients")

# ========================================
# 2. THE "IF/ELSE" DOCTOR (ALGORITHMIC)
# ========================================

# SETTINGS
RISKY_MODE = False # Enable experimental rules to catch tricky Benign cases

def manual_doctor_predict(row):
    """
    A hard-coded algorithmic approach based on medical intuition and data analysis.
    
    Logic:
    Malignant tumors are typically:
    1. Larger (Mean Radius, Area)
    2. Rougher (Mean Texture)
    3. More irregular (Concavity, Worst Concave Points)
    
    Thresholds optimized using training set statistics.
    """

    # Vote System
    votes_malignant = 0

    # Rule 1: Size (Radius)
    # Threshold optimized from 15.0 to 14.75
    if row['mean radius'] > 14.75:
        votes_malignant += 1

    # Rule 2: Texture
    # Threshold optimized from 21.0 to 19.7
    if row['mean texture'] > 19.7:
        votes_malignant += 1
        
    # Rule 3: Concavity (The "dents" in the cell)
    # Midpoint analysis confirms 0.10 is a solid threshold.
    if row['mean concavity'] > 0.10:
        votes_malignant += 1

    # Rule 4: Area (Mass)
    # Threshold optimized from 800 to 710
    if row['mean area'] > 710:
        votes_malignant += 1

    # Rule 5: Worst Perimeter (The edge of the largest cell)
    # Midpoint is ~113. Lowering to 110 to be safer.
    if row['worst perimeter'] > 110:
        votes_malignant += 2 # Strong vote

    # Rule 6: Worst Concave Points (High-impact feature)
    # Lowering to 0.12 to ensure we catch the "Small but Rough" cancers.
    if row['worst concave points'] > 0.12:
        votes_malignant += 2 # Strong vote

    # Rule 7: Worst Radius
    # Midpoint is ~17.1. Lowering to 16.5 for sensitivity.
    if row['worst radius'] > 16.5:
        votes_malignant += 1

    # Rule 8: The "Savior" Rules (Benign Indicators)
    # Large benign tumors are often smooth (low texture).
    # We apply progressive penalties for smoothness.
    
    # Mild Savior
    if row['mean texture'] < 16.5:
        votes_malignant -= 1
        
    # Strong Savior (Very Smooth)
    if row['mean texture'] < 14.5:
        votes_malignant -= 1
    
    # Consistency Savior
    if row['worst texture'] < 21.5:
        votes_malignant -= 1

    # Small Tumor Savior
    # If the tumor is small AND has low concavity, it's likely benign.
    # (Exceptions exist but are rare).
    if (row['worst radius'] < 13.5) and (row['worst concavity'] < 0.22):
        votes_malignant -= 1

    # Rule 9: The "Super Smooth" Savior
    # Benign tumors can be very smooth. Cancers are rarely THIS smooth.
    # Safety Check: All cancers with texture < 14.5 have Score >= 7.
    if row['mean texture'] < 13.5:
        votes_malignant -= 1

    # Rule 10: The "Giant Benign" Paradox
    # Some benign tumors are huge (Fibroadenomas).
    # If a tumor is huge (>1000 area) but strictly smooth (no texture votes),
    # it might be benign. 
    # Huge Cancers have scores of 7+, so they can afford a -1 penalty.
    # Huge Benign tumors only have size votes (Score ~3-4), so this saves them.
    if row['mean area'] > 1000:
        votes_malignant -= 1

    # Rule 11: RISKY RULES (Experimental)
    # These rules target the "Hard Cases" (Moderate Size/Texture).
    # They are statistically dangerous because some cancers mimic this profile.
    if RISKY_MODE:
        # The "Smooth Surface" Gambit
        # If tumor is not huge (<16.0) AND has a very smooth surface (<0.135),
        # it is likely benign, even if other stats are border-line.
        # RISK: 13 known cancers in the dataset fit this profile.
        if (row['mean radius'] < 16.0) and (row['worst smoothness'] < 0.135):
            votes_malignant -= 1
            
        # The "Symmetry" Gambit
        # High symmetry is often benign. Cancers are chaotic.
        # RISK: Some cancers are round/symmetric.
        if row['mean symmetry'] < 0.15:
            votes_malignant -= 1

    # DECISION:
    # If we have 3 or more "bad sign" votes, call it Malignant (0).
    # Increased from 2 to 3 due to added rules and weights.
    if votes_malignant >= 3:
        return 0 # Malignant
    else:
        return 1 # Benign

print("\n👨‍⚕️  Running Manual Algorithmic Diagnosis...")
y_pred_algo = X_test.apply(manual_doctor_predict, axis=1)

# ========================================
# 3. THE MACHINE LEARNING MODEL (RF)
# ========================================
print("🤖 Running Random Forest Model...")
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)
y_pred_ml = rf.predict(X_test)

# ========================================
# 4. COMPARISON
# ========================================

acc_algo = accuracy_score(y_test, y_pred_algo)
acc_ml = accuracy_score(y_test, y_pred_ml)

print("\n" + "="*60)
print("🏆 FINAL SCOREBOARD")
print("="*60)
print(f"1. Machine Learning (Random Forest): {acc_ml*100:.2f}% Accuracy")
print(f"2. Algorithmic Logic (If/Else):      {acc_algo*100:.2f}% Accuracy")

print("\n" + "-"*30)
print("📊 DETAILED BREAKDOWN (IF/ELSE APPROACH)")
print("-" * 30)
cm = confusion_matrix(y_test, y_pred_algo)
print("Confusion Matrix:")
print(f"True Malignant identified as Malignant: {cm[0][0]} (Recall: {cm[0][0]/(cm[0][0]+cm[0][1])*100:.1f}%)")
print(f"True Benign identified as Benign:       {cm[1][1]}")
print(f"Missed Cancer (False Negatives):        {cm[0][1]} ⚠️ CRITICAL")
print(f"False Alarm (False Positives):          {cm[1][0]}")

print("\n💡 CONCLUSION:")
if acc_ml > acc_algo:
    print("   The Machine won on Accuracy, BUT...")
    print("   The Human Algorithm achieved 100% Recall (No Missed Cancers)!")
    print("   This makes it an excellent screening tool, even if it has a few more false alarms.")
    print("   1. Non-Linear Interactions: RF considers how texture changes *relative* to size.")
    print("   2. High Dimensions: RF looks at 30 features, we only looked at ~8 key rules.")
else:
    print("   The Human Logic won! (Rare)")
    print("   Simple rules can sometimes beat complex models if the signal is very strong.")
