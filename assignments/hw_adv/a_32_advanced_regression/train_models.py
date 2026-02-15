import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

def main():
    # 1. Load Data
    # Handle path resolution to allow running from any directory
    base_dir = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(base_dir, 'apartamente_bucuresti.csv')
    
    print(f"Loading data from: {data_path}")
    df = pd.read_csv(data_path)
    
    X = df.drop('pret', axis=1)
    y = df['pret']

    # 2. Preprocessing Setup
    numerical_features = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
    categorical_features = X.select_dtypes(include=['object', 'string']).columns.tolist()

    numerical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='mean')),
        ('scaler', StandardScaler())
    ])

    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numerical_transformer, numerical_features),
            ('cat', categorical_transformer, categorical_features)
        ],
        remainder='drop'
    )

    # 3. Train-Test Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    print(f"Data split: {X_train.shape[0]} train, {X_test.shape[0]} test samples.")

    # 4. Apply Preprocessing
    # Fitting on train, transforming both
    X_train_processed = preprocessor.fit_transform(X_train)
    X_test_processed = preprocessor.transform(X_test)

    # 5. Train & Evaluate Models
    results = []

    # --- Model 1: Linear Regression ---
    lr = LinearRegression()
    lr.fit(X_train_processed, y_train)
    evaluate_model(lr, X_train_processed, X_test_processed, y_train, y_test, "Linear Regression", results)

    # --- Model 2: Ridge Regression (with GridSearch) ---
    ridge_grid = GridSearchCV(
        Ridge(),
        param_grid={'alpha': [0.1, 1, 10, 100, 1000]},
        cv=5,
        scoring='neg_mean_squared_error',
        n_jobs=-1
    )
    ridge_grid.fit(X_train_processed, y_train)
    evaluate_model(
        ridge_grid.best_estimator_,
        X_train_processed, X_test_processed,
        y_train, y_test,
        f"Ridge (alpha={ridge_grid.best_params_['alpha']})",
        results
    )

    # --- Model 3: Random Forest ---
    rf = RandomForestRegressor(
        n_estimators=100,
        max_depth=10,
        min_samples_split=10,
        min_samples_leaf=4,
        random_state=42,
        n_jobs=-1
    )
    rf.fit(X_train_processed, y_train)
    evaluate_model(rf, X_train_processed, X_test_processed, y_train, y_test, "Random Forest", results)

    # 6. Display & Save Results
    results_df = pd.DataFrame(results)
    
    print("\n" + "="*50)
    print("📊 FINAL COMPARISON")
    print("="*50)
    print(results_df.round(4).to_string(index=False))

    best_model = results_df.loc[results_df['RMSE'].idxmin()]
    print(f"\n🏆 Best Model: {best_model['Model']}")
    print(f"   RMSE: {best_model['RMSE']:,.0f} €")

    save_plot(results_df, base_dir)

def evaluate_model(model, X_train, X_test, y_train, y_test, name, results_list):
    """Calculates metrics and appends them to the results list."""
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)

    results_list.append({
        'Model': name,
        'R2 Train': r2_score(y_train, y_train_pred),
        'R2 Test': r2_score(y_test, y_test_pred),
        'RMSE': np.sqrt(mean_squared_error(y_test, y_test_pred)),
        'MAE': mean_absolute_error(y_test, y_test_pred)
    })

def save_plot(df, output_dir):
    """Generates and saves the comparison plot."""
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))

    # Plot 1: R2 Scores
    x = np.arange(len(df))
    width = 0.35
    rects1 = axes[0].bar(x - width/2, df['R2 Train'], width, label='Train', color='lightblue', edgecolor='black')
    rects2 = axes[0].bar(x + width/2, df['R2 Test'], width, label='Test', color='coral', edgecolor='black')
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(df['Model'], rotation=15, ha='right')
    axes[0].set_title('R² Score Comparison')
    axes[0].set_ylim(0, 1.1)
    axes[0].legend()
    axes[0].grid(axis='y', alpha=0.3)
    
    axes[0].bar_label(rects1, padding=3, fmt='%.2f')
    axes[0].bar_label(rects2, padding=3, fmt='%.2f')

    # Plot 2: RMSE
    axes[1].bar(df['Model'], df['RMSE'], color=['lightblue', 'lightgreen', 'coral'], edgecolor='black')
    axes[1].set_xticks(range(len(df['Model'])))
    axes[1].set_xticklabels(df['Model'], rotation=15, ha='right')
    axes[1].set_title('RMSE Comparison (Lower is Better)')
    axes[1].grid(axis='y', alpha=0.3)
    
    axes[1].set_ylim(0, df['RMSE'].max() * 1.1)
    
    # Add labels
    for i, v in enumerate(df['RMSE']):
        axes[1].text(i, v, f'{v:,.0f}€', ha='center', va='bottom', fontweight='bold')

    plt.tight_layout()
    output_path = os.path.join(output_dir, 'models_comparison_clean.png')
    plt.savefig(output_path)
    print(f"\n✅ Plot saved to: {output_path}")

if __name__ == "__main__":
    main()
