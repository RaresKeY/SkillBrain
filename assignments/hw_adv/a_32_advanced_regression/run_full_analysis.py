import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
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
    base_dir = os.path.dirname(os.path.abspath(__file__))
    
    # 1. Load Data & EDA
    print("\n" + "="*50)
    print("STEP 1: Loading Data & Exploratory Analysis")
    print("="*50)
    df = load_and_explore_data(base_dir)
    
    # 2. Preprocessing
    print("\n" + "="*50)
    print("STEP 2: Preprocessing")
    print("="*50)
    X_train_proc, X_test_proc, y_train, y_test, preprocessor, feature_names = preprocess_data(df)

    # 3. Model Training
    print("\n" + "="*50)
    print("STEP 3: Training Models")
    print("="*50)
    models = train_models(X_train_proc, y_train)

    # 4. Evaluation & Comparison
    print("\n" + "="*50)
    print("STEP 4: Evaluation & Comparison")
    print("="*50)
    results_df = evaluate_models(models, X_train_proc, X_test_proc, y_train, y_test)
    save_comparison_plot(results_df, base_dir)

    # 5. Detailed Visualizations
    print("\n" + "="*50)
    print("STEP 5: Detailed Visualizations")
    print("="*50)
    generate_detailed_plots(models, X_test_proc, y_test, feature_names, base_dir)

def load_and_explore_data(output_dir):
    # Load existing CSV
    csv_path = os.path.join(output_dir, 'apartamente_bucuresti.csv')
    try:
        df = pd.read_csv(csv_path)
        print(f"Dataset loaded: {csv_path}")
        print(f"Shape: {df.shape}")
        print("Columns:", df.columns.tolist())
    except FileNotFoundError:
        print(f"Error: {csv_path} not found. Please run update_dataset.py first.")
        exit(1)

    # --- EDA Plots ---
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))

    # 1. Price Distribution
    axes[0, 0].hist(df['pret'].dropna(), bins=50, color='skyblue', edgecolor='black', alpha=0.7)
    axes[0, 0].axvline(df['pret'].mean(), color='red', linestyle='--', linewidth=2, label='Mean')
    axes[0, 0].set_title('Price Distribution')
    axes[0, 0].legend()

    # 2. Price vs Area
    if 'suprafata' in df.columns:
        axes[0, 1].scatter(df['suprafata'], df['pret'], alpha=0.5, color='coral')
        axes[0, 1].set_xlabel('Area (mp)')
        axes[0, 1].set_ylabel('Price (€)')
        axes[0, 1].set_title('Price vs Area')
    else:
        axes[0, 1].text(0.5, 0.5, 'Column "suprafata" missing', ha='center')

    # 3. Avg Price per Zone
    if 'zona' in df.columns:
        zone_avg = df.groupby('zona')['pret'].mean().sort_values()
        axes[1, 0].barh(zone_avg.index, zone_avg.values, color='lightgreen', edgecolor='black')
        axes[1, 0].set_title('Avg Price per Zone')
    else:
        axes[1, 0].text(0.5, 0.5, 'Column "zona" missing', ha='center')

    # 4. Missing Values
    missing_data = df.isnull().sum().sort_values(ascending=False)
    # Filter only columns with missing values or top N to avoid clutter if many columns
    missing_data = missing_data[missing_data > 0]
    if not missing_data.empty:
        missing_percent = (missing_data / len(df) * 100).round(1)
        axes[1, 1].barh(missing_data.index, missing_percent.values, color='indianred', edgecolor='black')
        axes[1, 1].set_title('Missing Values (%)')
    else:
        axes[1, 1].text(0.5, 0.5, 'No missing values', ha='center')

    plt.tight_layout()
    plot_path = os.path.join(output_dir, 'real_estate_exploration.png')
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"EDA plot saved: {plot_path}")
    plt.close()

    return df

def preprocess_data(df):
    X = df.drop('pret', axis=1)
    y = df['pret']

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

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Fit & Transform
    X_train_processed = preprocessor.fit_transform(X_train)
    X_test_processed = preprocessor.transform(X_test)

    # Get feature names
    feature_names = numerical_features.copy()
    cat_encoder = preprocessor.named_transformers_['cat']['onehot']
    feature_names.extend(cat_encoder.get_feature_names_out(categorical_features))

    return X_train_processed, X_test_processed, y_train, y_test, preprocessor, feature_names

def train_models(X_train, y_train):
    models = {}

    # 1. Linear Regression
    lr = LinearRegression()
    lr.fit(X_train, y_train)
    models['Linear Regression'] = lr

    # 2. Ridge
    ridge_grid = GridSearchCV(
        Ridge(),
        param_grid={'alpha': [0.1, 1, 10, 100, 1000]},
        cv=5,
        scoring='neg_mean_squared_error',
        n_jobs=-1
    )
    ridge_grid.fit(X_train, y_train)
    models['Ridge Regression'] = ridge_grid.best_estimator_
    print(f"  Best Ridge Alpha: {ridge_grid.best_params_['alpha']}")

    # 3. Random Forest
    rf = RandomForestRegressor(
        n_estimators=100,
        max_depth=10,
        min_samples_split=10,
        min_samples_leaf=4,
        random_state=42,
        n_jobs=-1
    )
    rf.fit(X_train, y_train)
    models['Random Forest'] = rf

    return models

def evaluate_models(models, X_train, X_test, y_train, y_test):
    results = []
    for name, model in models.items():
        y_train_pred = model.predict(X_train)
        y_test_pred = model.predict(X_test)
        
        results.append({
            'Model': name,
            'R2 Train': r2_score(y_train, y_train_pred),
            'R2 Test': r2_score(y_test, y_test_pred),
            'RMSE': np.sqrt(mean_squared_error(y_test, y_test_pred)),
            'MAE': mean_absolute_error(y_test, y_test_pred)
        })
    
    results_df = pd.DataFrame(results)
    print("\n" + results_df.round(4).to_string(index=False))
    return results_df

def save_comparison_plot(df, output_dir):
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))

    # R2 Scores
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

    # RMSE
    axes[1].bar(df['Model'], df['RMSE'], color=['lightblue', 'lightgreen', 'coral'], edgecolor='black')
    axes[1].set_xticks(range(len(df['Model'])))
    axes[1].set_xticklabels(df['Model'], rotation=15, ha='right')
    axes[1].set_title('RMSE Comparison (Lower is Better)')
    axes[1].grid(axis='y', alpha=0.3)
    
    axes[1].set_ylim(0, df['RMSE'].max() * 1.1)
    
    for i, v in enumerate(df['RMSE']):
        axes[1].text(i, v, f'{v:,.0f}€', ha='center', va='bottom', fontweight='bold')

    plt.tight_layout()
    output_path = os.path.join(output_dir, 'models_comparison.png')
    plt.savefig(output_path)
    print(f"Comparison plot saved: {output_path}")
    plt.close()

def generate_detailed_plots(models, X_test, y_test, feature_names, output_dir):
    # 1. Prediction vs Actual
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    colors = ['blue', 'purple', 'green']
    
    for idx, (name, model) in enumerate(models.items()):
        ax = axes[idx]
        predictions = model.predict(X_test)
        
        ax.scatter(y_test, predictions, alpha=0.5, color=colors[idx], edgecolors='black', s=50)
        
        # Perfect line
        min_val = min(y_test.min(), predictions.min())
        max_val = max(y_test.max(), predictions.max())
        ax.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2)
        
        ax.set_xlabel('Actual Price (€)')
        ax.set_ylabel('Predicted Price (€)')
        ax.set_title(f'{name}\nPrediction vs Actual')
        ax.grid(alpha=0.3)
        
        r2 = r2_score(y_test, predictions)
        ax.text(0.05, 0.95, f'R² = {r2:.3f}', transform=ax.transAxes,
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8), va='top')

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'prediction_vs_actual.png'), dpi=300, bbox_inches='tight')
    plt.close()

    # 2. Residuals Analysis
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    
    for idx, (name, model) in enumerate(models.items()):
        predictions = model.predict(X_test)
        residuals = y_test - predictions
        
        # Residuals vs Predicted
        ax1 = axes[0, idx]
        ax1.scatter(predictions, residuals, alpha=0.5, color=colors[idx], edgecolors='black', s=50)
        ax1.axhline(y=0, color='red', linestyle='--')
        ax1.set_xlabel('Predicted Price (€)')
        ax1.set_ylabel('Residuals (€)')
        ax1.set_title(f'{name}\nResiduals vs Predicted')
        ax1.grid(alpha=0.3)
        
        # Residuals Distribution
        ax2 = axes[1, idx]
        ax2.hist(residuals, bins=30, color=colors[idx], alpha=0.7, edgecolor='black', density=True)
        
        mu, sigma = residuals.mean(), residuals.std()
        x = np.linspace(residuals.min(), residuals.max(), 100)
        ax2.plot(x, stats.norm.pdf(x, mu, sigma), 'r-', linewidth=2)
        ax2.axvline(0, color='black', linestyle='--')
        
        ax2.set_title(f'{name}\nResiduals Distribution')
        ax2.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'residuals_analysis.png'), dpi=300, bbox_inches='tight')
    plt.close()

    # 3. Feature Importance (RF only)
    if 'Random Forest' in models:
        rf = models['Random Forest']
        importances = rf.feature_importances_
        
        feat_df = pd.DataFrame({
            'feature': feature_names,
            'importance': importances
        }).sort_values('importance', ascending=False).head(15)
        
        plt.figure(figsize=(12, 8))
        colors_grad = plt.cm.RdYlGn(np.linspace(0.3, 0.9, len(feat_df)))
        plt.barh(feat_df['feature'], feat_df['importance'], color=colors_grad, edgecolor='black')
        plt.gca().invert_yaxis()
        plt.title('Top 15 Feature Importance - Random Forest')
        plt.grid(axis='x', alpha=0.3)
        
        plt.xlim(0, feat_df['importance'].max() * 1.15)
        
        for i, (idx, row) in enumerate(feat_df.iterrows()):
            plt.text(row['importance'], i, f" {row['importance']:.3f}", va='center')
            
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'feature_importance.png'), dpi=300, bbox_inches='tight')
        plt.close()
    
    print("Detailed visualization plots saved.")

if __name__ == "__main__":
    main()
