import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_diabetes, fetch_california_housing, make_regression
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    mean_squared_error, mean_absolute_error, r2_score,
    explained_variance_score, mean_absolute_percentage_error
)

# Import various regressors
from sklearn.linear_model import (
    LinearRegression, Ridge, Lasso, ElasticNet,
    BayesianRidge, HuberRegressor
)
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import (
    RandomForestRegressor, GradientBoostingRegressor,
    AdaBoostRegressor, ExtraTreesRegressor
)
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neural_network import MLPRegressor

# For warnings
import warnings
warnings.filterwarnings('ignore')

# Set style for better visualizations
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

def load_and_display_dataset(dataset_name):
    """
    Load a regression dataset from sklearn and display its information
    """
    datasets = {}

    # Load available datasets
    datasets['diabetes'] = load_diabetes()
    datasets['california_housing'] = fetch_california_housing()

    if dataset_name not in datasets:
        print(f"Dataset {dataset_name} not found. Available: {list(datasets.keys())}")
        return None

    data = datasets[dataset_name]

    # Create DataFrame
    df = pd.DataFrame(data.data, columns=data.feature_names)
    df['target'] = data.target

    # For California housing, add target name
    if dataset_name == 'california_housing':
        df['target_name'] = 'House Value (in $100,000)'
    elif dataset_name == 'diabetes':
        df['target_name'] = 'Disease Progression'

    # Display basic information
    print("="*80)
    print(f"REGRESSION DATASET: {dataset_name.upper()}")
    print("="*80)
    print(f"\nShape: {df.shape}")
    print(f"Number of Features: {data.data.shape[1]}")
    print(f"Feature Names: {list(data.feature_names)}")
    print(f"Target Range: [{data.target.min():.2f}, {data.target.max():.2f}]")
    print(f"Target Mean: {data.target.mean():.2f}")
    print(f"Target Std Dev: {data.target.std():.2f}")

    print(f"\nFirst 5 rows:")
    print(df.head())

    print(f"\nStatistical Summary:")
    print(df.describe())

    return df, data

def visualize_regression_dataset(df, dataset_name):
    """
    Create comprehensive visualizations for regression dataset
    """
    fig = plt.figure(figsize=(20, 16))

    feature_cols = df.columns[:-2] if 'target_name' in df.columns else df.columns[:-1]
    n_features = len(feature_cols)

    # 1. Target Distribution
    ax1 = plt.subplot(3, 3, 1)
    df['target'].hist(bins=30, color='skyblue', edgecolor='black', ax=ax1)
    ax1.set_title(f'Target Variable Distribution - {dataset_name}', fontsize=12, fontweight='bold')
    ax1.set_xlabel('Target Value')
    ax1.set_ylabel('Frequency')
    ax1.axvline(df['target'].mean(), color='red', linestyle='--', label=f'Mean: {df["target"].mean():.2f}')
    ax1.axvline(df['target'].median(), color='green', linestyle='--', label=f'Median: {df["target"].median():.2f}')
    ax1.legend()

    # 2. Feature Distributions (first 6 features)
    ax2 = plt.subplot(3, 3, 2)
    n_features_hist = min(6, n_features)
    for i in range(n_features_hist):
        df[feature_cols[i]].hist(alpha=0.5, label=feature_cols[i], ax=ax2, bins=20)
    ax2.set_title(f'Feature Distributions (First {n_features_hist} features)', fontsize=12, fontweight='bold')
    ax2.set_xlabel('Value')
    ax2.set_ylabel('Frequency')
    ax2.legend(fontsize=8)

# 3. Correlation Heatmap
    ax3 = plt.subplot(3, 3, 3)
    features_to_plot = min(10, n_features)
    all_cols = list(feature_cols[:features_to_plot]) + ['target']
    corr_matrix = df[all_cols].corr()
    sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='coolwarm',
                center=0, ax=ax3, square=True, cbar_kws={"shrink": 0.8})
    ax3.set_title('Feature Correlation Matrix (including target)', fontsize=12, fontweight='bold')

    # 4. Top features correlation with target
    ax4 = plt.subplot(3, 3, 4)
    target_corr = df[feature_cols].corrwith(df['target']).sort_values(ascending=False)
    colors = ['green' if x > 0 else 'red' for x in target_corr.values]
    target_corr.head(10).plot(kind='bar', ax=ax4, color=colors, edgecolor='black')
    ax4.set_title('Top Features Correlated with Target', fontsize=12, fontweight='bold')
    ax4.set_xlabel('Features')
    ax4.set_ylabel('Correlation with Target')
    ax4.tick_params(axis='x', rotation=45)
    ax4.axhline(y=0, color='black', linestyle='-', linewidth=0.5)

    # 5. Scatter plot of best feature vs target
    ax5 = plt.subplot(3, 3, 5)
    if len(target_corr) > 0:
        best_feature = target_corr.index[0]
        ax5.scatter(df[best_feature], df['target'], alpha=0.5, s=30)
        # Add trend line
        z = np.polyfit(df[best_feature], df['target'], 1)
        p = np.poly1d(z)
        ax5.plot(np.sort(df[best_feature]), p(np.sort(df[best_feature])),
                "r--", alpha=0.8, label=f'Trend (r={target_corr.iloc[0]:.2f})')
        ax5.set_xlabel(best_feature)
        ax5.set_ylabel('Target')
        ax5.set_title(f'Best Feature vs Target: {best_feature}', fontsize=12, fontweight='bold')
        ax5.legend()
        ax5.grid(True, alpha=0.3)

    # 6. Box plots for top 5 features
    ax6 = plt.subplot(3, 3, 6)
    top_5_features = target_corr.head(5).index
    df_top5 = df[top_5_features]
    df_top5.boxplot(ax=ax6, rot=45)
    ax6.set_title('Box Plots - Top 5 Features', fontsize=12, fontweight='bold')
    ax6.set_ylabel('Value')

    # 7. Scatter matrix for top 4 features
    ax7 = plt.subplot(3, 3, 7)
    if len(target_corr) >= 4:
        top_4_features = target_corr.head(4).index
        # Create a scatter matrix using pandas
        pd.plotting.scatter_matrix(df[top_4_features], alpha=0.5,
                                  figsize=(8, 8), diagonal='hist', ax=ax7)
        ax7.set_title('Scatter Matrix - Top 4 Features', fontsize=12, fontweight='bold')
    else:
        ax7.text(0.5, 0.5, 'Insufficient features for scatter matrix',
                ha='center', va='center', transform=ax7.transAxes, fontsize=12)
        ax7.set_title('Scatter Matrix', fontsize=12, fontweight='bold')
        ax7.axis('off')

    # 8. Feature importance placeholder
    ax8 = plt.subplot(3, 3, 8)
    ax8.text(0.5, 0.5, 'Feature Importance\n(Will be shown after model training)',
            ha='center', va='center', transform=ax8.transAxes, fontsize=12)
    ax8.set_title('Feature Importance (from Random Forest)', fontsize=12, fontweight='bold')
    ax8.axis('off')

    # 9. Target vs index (time series view)
    ax9 = plt.subplot(3, 3, 9)
    ax9.plot(df.index, df['target'], alpha=0.6, linewidth=0.5)
    ax9.set_title('Target Values Over Index', fontsize=12, fontweight='bold')
    ax9.set_xlabel('Sample Index')
    ax9.set_ylabel('Target Value')
    ax9.grid(True, alpha=0.3)

    plt.suptitle(f'Regression Dataset Visualization: {dataset_name.upper()}',
                fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.show()


def get_regressors():
    """
    Define a dictionary of regressors to evaluate
    """
    regressors = {
        'Linear Regression': LinearRegression(),
        'Ridge Regression': Ridge(alpha=1.0, random_state=42),
        'Lasso Regression': Lasso(alpha=1.0, random_state=42),
        'Elastic Net': ElasticNet(alpha=1.0, l1_ratio=0.5, random_state=42),
        'Decision Tree': DecisionTreeRegressor(random_state=42),
        'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42),
        'Gradient Boosting': GradientBoostingRegressor(n_estimators=100, random_state=42),
        'AdaBoost': AdaBoostRegressor(n_estimators=100, random_state=42),
        'Extra Trees': ExtraTreesRegressor(n_estimators=100, random_state=42),
        'SVR': SVR(kernel='rbf'),
        'K-Neighbors': KNeighborsRegressor(n_neighbors=5),
        'Bayesian Ridge': BayesianRidge(),
        'Huber Regressor': HuberRegressor(),
        'MLP Regressor': MLPRegressor(hidden_layer_sizes=(100,), max_iter=1000, random_state=42)
    }
    return regressors

def train_and_evaluate_regressors(X_train, X_test, y_train, y_test, regressors, dataset_name):
    """
    Train multiple regressors and evaluate their performance
    """
    results = []

    print("\n" + "="*100)
    print(f"REGRESSOR PERFORMANCE COMPARISON - {dataset_name.upper()}")
    print("="*100)

    for reg_name, reg in regressors.items():
        print(f"\n{'='*50}")
        print(f"Training {reg_name}...")
        print(f"{'='*50}")

        try:
            # Train the regressor
            reg.fit(X_train, y_train)

            # Make predictions
            y_pred = reg.predict(X_test)

            # Calculate metrics
            mse = mean_squared_error(y_test, y_pred)
            rmse = np.sqrt(mse)
            mae = mean_absolute_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)
            evs = explained_variance_score(y_test, y_pred)

            # Cross-validation score (using negative MSE for compatibility)
            cv_scores = cross_val_score(reg, X_train, y_train, cv=5,
                                       scoring='neg_mean_squared_error')
            cv_rmse = np.sqrt(-cv_scores.mean())

            # Store results
            results.append({
                'Regressor': reg_name,
                'RMSE': rmse,
                'MSE': mse,
                'MAE': mae,
                'R2': r2,
                'Explained Variance': evs,
                'CV RMSE': cv_rmse
            })

            # Display results
            print(f"RMSE: {rmse:.4f}")
            print(f"MAE: {mae:.4f}")
            print(f"R² Score: {r2:.4f}")
            print(f"Explained Variance: {evs:.4f}")
            print(f"Cross-validation RMSE: {cv_rmse:.4f}")

            # Feature importance for tree-based models
            if hasattr(reg, 'feature_importances_'):
                print(f"\nTop 3 Important Features:")
                importances = reg.feature_importances_
                indices = np.argsort(importances)[::-1][:3]
                for i, idx in enumerate(indices):
                    print(f"  {i+1}. Feature {idx}: {importances[idx]:.4f}")

        except Exception as e:
            print(f"Error training {reg_name}: {str(e)}")
            results.append({
                'Regressor': reg_name,
                'RMSE': np.nan,
                'MSE': np.nan,
                'MAE': np.nan,
                'R2': np.nan,
                'Explained Variance': np.nan,
                'CV RMSE': np.nan
            })

    return pd.DataFrame(results)

def plot_regression_results(results_df, dataset_name):
    """
    Create comparison plots for all regressors
    """
    # Remove any rows with NaN values for plotting
    plot_df = results_df.dropna()

    if len(plot_df) == 0:
        print("No valid results to plot")
        return

    fig, axes = plt.subplots(2, 2, figsize=(15, 12))

# 1. RMSE Comparison (lower is better)
    axes[0, 0].barh(plot_df['Regressor'], plot_df['RMSE'], color='coral', edgecolor='black')
    axes[0, 0].set_xlabel('RMSE (Root Mean Square Error)', fontsize=12)
    axes[0, 0].set_title('RMSE Comparison Across Regressors (Lower is Better)',
                         fontsize=12, fontweight='bold')
    axes[0, 0].grid(axis='x', alpha=0.3)

    # 2. R² Score Comparison (higher is better)
    axes[0, 1].barh(plot_df['Regressor'], plot_df['R2'], color='lightgreen', edgecolor='black')
    axes[0, 1].set_xlabel('R² Score', fontsize=12)
    axes[0, 1].set_title('R² Score Comparison Across Regressors (Higher is Better)',
                         fontsize=12, fontweight='bold')
    axes[0, 1].set_xlim(-1, 1)
    axes[0, 1].axvline(x=0, color='red', linestyle='--', alpha=0.5)
    axes[0, 1].grid(axis='x', alpha=0.3)

    # 3. Multi-metric Comparison (normalized)
    metrics = ['RMSE', 'MAE', 'R2', 'Explained Variance']
    # Normalize metrics for visualization
    normalized_df = plot_df.copy()
    for metric in ['RMSE', 'MAE']:
        # For error metrics, lower is better, so we invert for visualization
        max_val = normalized_df[metric].max()
        if max_val > 0:
            normalized_df[f'Norm_{metric}'] = 1 - (normalized_df[metric] / max_val)
    for metric in ['R2', 'Explained Variance']:
        min_val = normalized_df[metric].min()
        max_val = normalized_df[metric].max()
        if max_val > min_val:
            normalized_df[f'Norm_{metric}'] = (normalized_df[metric] - min_val) / (max_val - min_val)
        else:
            normalized_df[f'Norm_{metric}'] = normalized_df[metric]

    x = np.arange(len(normalized_df['Regressor']))
    width = 0.2
    norm_metrics = ['Norm_RMSE', 'Norm_MAE', 'Norm_R2', 'Norm_Explained Variance']
    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4']

    for i, (metric, color) in enumerate(zip(norm_metrics, colors)):
        if metric in normalized_df.columns:
            axes[1, 0].bar(x + i*width, normalized_df[metric], width,
                          label=metric.replace('Norm_', ''), color=color)

    axes[1, 0].set_xlabel('Regressors', fontsize=12)
    axes[1, 0].set_ylabel('Normalized Score (1 is best)', fontsize=12)
    axes[1, 0].set_title('Multi-Metric Comparison (Normalized)', fontsize=12, fontweight='bold')
    axes[1, 0].set_xticks(x + width*1.5)
    axes[1, 0].set_xticklabels(normalized_df['Regressor'], rotation=45, ha='right')
    axes[1, 0].legend()
    axes[1, 0].set_ylim(0, 1)
    axes[1, 0].grid(axis='y', alpha=0.3)

    # 4. Cross-validation Performance
    axes[1, 1].barh(plot_df['Regressor'], plot_df['CV RMSE'], color='salmon', edgecolor='black')
    axes[1, 1].set_xlabel('Cross-validation RMSE (5-fold)', fontsize=12)
    axes[1, 1].set_title('Cross-validation Performance (Lower is Better)',
                         fontsize=12, fontweight='bold')
    axes[1, 1].grid(axis='x', alpha=0.3)

    plt.suptitle(f'Regressor Performance Comparison - {dataset_name.upper()}',
                fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.show()

    # Display the results table
    print("\n" + "="*100)
    print("COMPLETE RESULTS SUMMARY")
    print("="*100)
    print(plot_df.round(4).to_string(index=False))

def plot_predictions_vs_actual(y_test, y_pred, regressor_name, dataset_name):
    """
    Create predictions vs actual plot for the best regressor
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # 1. Predictions vs Actual
    axes[0].scatter(y_test, y_pred, alpha=0.5, s=30)
    axes[0].plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()],
                'r--', lw=2, label='Perfect Prediction')
    axes[0].set_xlabel('Actual Values')
    axes[0].set_ylabel('Predicted Values')
    axes[0].set_title(f'{regressor_name}\nPredictions vs Actual', fontsize=12, fontweight='bold')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    residuals = y_test - y_pred
    axes[1].scatter(y_pred, residuals, alpha=0.5, s=30)
    axes[1].axhline(y=0, color='r', linestyle='--', linewidth=2)
    axes[1].set_xlabel('Predicted Values')
    axes[1].set_ylabel('Residuals')
    axes[1].set_title(f'{regressor_name}\nResiduals Plot', fontsize=12, fontweight='bold')
    axes[1].grid(True, alpha=0.3)

    plt.suptitle(f'Model Diagnostics - {dataset_name.upper()}', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.show()



def run_complete_assignment():
    """
    Main function to run the complete regression assignment for multiple datasets
    """
    print("="*100)
    print("MACHINE LEARNING REGRESSION ASSIGNMENT")
    print("Using Multiple Regressors on scikit-learn Datasets")
    print("="*100)

    # List of datasets to analyze
    datasets_to_analyze = ['diabetes', 'california_housing']

    all_results = {}
    best_models = {}

    for dataset_name in datasets_to_analyze:
        print(f"\n{'#'*100}")
        print(f"# PROCESSING DATASET: {dataset_name.upper()}")
        print(f"{'#'*100}")

        # Load and explore the dataset
        df, data = load_and_display_dataset(dataset_name)

        if df is None:
            continue

        # Visualize the dataset
        visualize_regression_dataset(df, dataset_name)

        # Prepare features and target
        X = data.data
        y = data.target

        # Split the data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        print(f"\nData Split:")
        print(f"Training set size: {X_train.shape[0]} samples")
        print(f"Test set size: {X_test.shape[0]} samples")

        # Feature scaling (standardization)
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        print(f"\nFeature Scaling: Applied StandardScaler")

        # Get regressors
        regressors = get_regressors()

        # Train and evaluate regressors
        results_df = train_and_evaluate_regressors(
            X_train_scaled, X_test_scaled, y_train, y_test,
            regressors, dataset_name
        )

        # Plot comparison results
        plot_regression_results(results_df, dataset_name)

        # Store results
        all_results[dataset_name] = results_df

        # Identify best regressor (based on R2 score)
        results_df_clean = results_df.dropna()
        if len(results_df_clean) > 0:
            best_regressor = results_df_clean.loc[results_df_clean['R2'].idxmax(), 'Regressor']
            best_r2 = results_df_clean['R2'].max()
            best_models[dataset_name] = {'name': best_regressor, 'r2': best_r2}

            print(f"\n{'='*50}")
            print(f"BEST PERFORMING REGRESSOR FOR {dataset_name.upper()}:")
            print(f"{best_regressor} with R² Score: {best_r2:.4f}")
            print(f"{'='*50}")

            # Train the best regressor again for visualization
            best_reg = regressors[best_regressor]
            best_reg.fit(X_train_scaled, y_train)
            y_pred_best = best_reg.predict(X_test_scaled)

            # Plot predictions vs actual for the best regressor
            plot_predictions_vs_actual(y_test, y_pred_best,
                                      best_regressor, dataset_name)
        else:
            print(f"\nNo successful models for {dataset_name}")

        if dataset_name != datasets_to_analyze[-1]:
            input("\nPress Enter to continue to the next dataset...")

    # Final comparison across datasets
    print("\n" + "="*100)
    print("FINAL SUMMARY - BEST PERFORMANCE ACROSS DATASETS")
    print("="*100)

    final_summary = []
    for dataset, results in all_results.items():
        results_clean = results.dropna()
        if len(results_clean) > 0:
            best = results_clean.loc[results_clean['R2'].idxmax()] 
            final_summary.append({
                'Dataset': dataset.upper(),
                'Best Regressor': best['Regressor'],
                'Best R² Score': best['R2'], 
                'Best RMSE': best['RMSE'],
                'Best MAE': best['MAE']
            })

    if final_summary:
        final_df = pd.DataFrame(final_summary)
        print(final_df.to_string(index=False))

        # Create summary visualization
        plt.figure(figsize=(12, 6))
        x = np.arange(len(final_df))
        width = 0.35

        plt.bar(x - width/2, final_df['Best R² Score'], width, label='R² Score', color='lightgreen')

        # Normalize RMSE for visualization
        if final_df['Best RMSE'].max() > 0:
            norm_rmse = final_df['Best RMSE'] / final_df['Best RMSE'].max()
            plt.bar(x + width/2, norm_rmse, width, label='Normalized RMSE', color='coral')

        plt.xlabel('Datasets')
        plt.ylabel('Score')
        plt.title('Best Model Performance Across Datasets', fontsize=14, fontweight='bold')
        plt.xticks(x, final_df['Dataset'])
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()
    else:
        print("No successful models to summarize")

    print("\n" + "="*100)
    print("ASSIGNMENT COMPLETED SUCCESSFULLY!")
    print("="*100)



if __name__ == "__main__":
    run_complete_assignment()