import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.feature_selection import mutual_info_classif, mutual_info_regression
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm  # Optional, for progress bar


# 1. Data Loading
def load_data(file_path):
    df = pd.read_csv(file_path)
    # Extract target variable (5th column) and features (columns 11,12,14-36)
    target = df.iloc[:, 4]  # 6th column (index 5)
    features = df.iloc[:, list(range(10, 36))]  # Columns 11,12,14-36
    # features = df.iloc[:, list(range(10, 12)) + list(range(13, 36))]

    # Standardize features
    scaler = StandardScaler()
    features_scaled = pd.DataFrame(scaler.fit_transform(features),
                                   columns=features.columns)

    return features_scaled, target, features.columns


# 2. Correlation Filtering
def filter_features(features, target, min_relevance=0.1, max_redundancy=0.8):
    # Calculate correlation between features and target
    target_corr = features.corrwith(target).abs()
    relevant_features = target_corr[target_corr > min_relevance].index.tolist()

    # Filter out features with low correlation to target
    features_filtered = features[relevant_features]

    # Calculate inter-feature correlation matrix
    corr_matrix = features_filtered.corr().abs()

    # Find highly correlated feature pairs
    high_corr_pairs = np.where(corr_matrix > max_redundancy)
    high_corr_pairs = [(corr_matrix.columns[x], corr_matrix.columns[y])
                       for x, y in zip(*high_corr_pairs) if x != y and x < y]

    # Collect features to remove (keep the one with higher correlation to target)
    to_remove = set()
    for pair in high_corr_pairs:
        if target_corr[pair[0]] > target_corr[pair[1]]:
            to_remove.add(pair[1])
        else:
            to_remove.add(pair[0])

    # Final retained features
    final_features = [f for f in relevant_features if f not in to_remove]

    return features[final_features], final_features, target_corr


# 3. Manual MRMR Feature Selection
def manual_mrmr(features, target, n_features=8, task_type='regression'):
    """
    Manual implementation of MRMR feature selection
    :param features: Feature DataFrame
    :param target: Target variable
    :param n_features: Number of features to select
    :param task_type: 'regression' or 'classification'
    :return: Selected feature list
    """
    feature_names = features.columns.tolist()
    selected_features = []
    remaining_features = feature_names.copy()

    # Select mutual information function based on task type
    mi_func = mutual_info_regression if task_type == 'regression' else mutual_info_classif

    # First round: Select feature with maximum mutual information with target
    print("\n=== Initial Feature Selection ===")
    mi_scores = mi_func(features, target)
    first_feature = feature_names[np.argmax(mi_scores)]
    print(f"Selected first feature '{first_feature}', Mutual information with target: {mi_scores.max():.4f}")
    selected_features.append(first_feature)
    remaining_features.remove(first_feature)

    # Subsequent rounds: Select feature that maximizes [mutual information with target - average mutual information with selected features]
    for i in range(n_features - 1):
        print(f"\n=== Round {i + 2} Feature Selection ===")
        print(f"Currently selected features: {selected_features}")
        print(f"Remaining candidate features: {remaining_features}")

        scores = []
        relevance_scores = []
        redundancy_scores = []

        # Calculate MRMR score for each remaining feature
        for feat in remaining_features:
            # Mutual information with target
            relevance = mi_func(features[[feat]], target)[0]
            relevance_scores.append(relevance)

            # Average mutual information with selected features (redundancy)
            redundancy = 0
            if selected_features:
                redundancy = np.mean([mi_func(features[[feat, sf]],
                                              features[sf])[0]
                                      for sf in selected_features])
            redundancy_scores.append(redundancy)

            # MRMR score
            # mrmr_score = relevance - redundancy
            # MRMR score (using MIQ form)
            mrmr_score = relevance / (redundancy + 1e-12)  # Add small value to prevent division by zero
            scores.append(mrmr_score)

            # Print calculation results for current feature
            print(f"Feature '{feat}': Relevance={relevance:.4f}, Redundancy={redundancy:.4f}, MRMR score={mrmr_score:.4f}")

        # Select feature with highest score
        best_idx = np.argmax(scores)
        best_feature = remaining_features[best_idx]
        selected_features.append(best_feature)
        remaining_features.remove(best_feature)

        print(f"\nThis round selected feature '{best_feature}'")
        print(f"Highest MRMR score: {scores[best_idx]:.4f}")
        print(f"Relevance: {relevance_scores[best_idx]:.4f}")
        print(f"Redundancy: {redundancy_scores[best_idx]:.4f}")

        # Early termination if no more features available
        if not remaining_features:
            print("\nNo more features available, terminating selection process early")
            break

    return selected_features


def visualize_results(features, target, selected_features, target_corr):
    # Set color theme
    corr_palette = sns.color_palette("crest", as_cmap=True)
    mrmr_palette = sns.color_palette("YlOrRd", as_cmap=True)

    # 1. All features correlation with target
    plt.figure(figsize=(10, 8))
    all_features = features.columns
    all_corr = target_corr[all_features].sort_values()
    # Fix: Convert palette to list and apply correctly
    colors = [corr_palette(x) for x in all_corr.values]
    ax = sns.barplot(x=all_corr.values, y=all_corr.index, hue=all_corr.index,
                palette=colors, legend=False)
    plt.axvline(x=0.1, color='r', linestyle='--')
    ax.tick_params(axis='x', labelsize=12)  # x-axis data (numerical labels)
    ax.tick_params(axis='y', labelsize=12)  # y-axis data (feature name labels)
    plt.xlabel('Correlation Coefficient', fontsize=14)
    plt.ylabel('Variables', fontsize=14)
    plt.rcParams['font.family'] = 'Arial'  # Set global font
    plt.tight_layout()
    plt.show()

    # 2. Selected features correlation with target
    plt.figure(figsize=(10, 6))
    selected_corr = target_corr[selected_features].sort_values()
    # Fix: Convert palette to list and apply correctly
    colors = [corr_palette(x) for x in selected_corr.values]
    ax = sns.barplot(x=selected_corr.values, y=selected_corr.index,
                hue=selected_corr.index, palette=colors, legend=False)
    plt.axvline(x=0.1, color='r', linestyle='--')
    ax.tick_params(axis='x', labelsize=12)  # x-axis data (numerical labels)
    ax.tick_params(axis='y', labelsize=12)  # y-axis data (feature name labels)
    plt.xlabel('Correlation Coefficient', fontsize=14)
    plt.ylabel('Variables', fontsize=14)
    plt.rcParams['font.family'] = 'Arial'  # Set global font
    plt.tight_layout()
    plt.show()

    # 3. Correlation heatmap among selected features
    plt.figure(figsize=(10, 8))
    corr_matrix = features[selected_features].corr()
    ax = sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap='coolwarm',
                center=0, vmin=-1, vmax=1, linewidths=0.5)
    ax.tick_params(axis='x', labelsize=12)  # x-axis data (numerical labels)
    ax.tick_params(axis='y', labelsize=12)  # y-axis data (feature name labels)
    plt.rcParams['font.family'] = 'Arial'  # Set global font
    plt.tight_layout()
    plt.show()

    # 4. Calculate and display MRMR scores
    mi_func = mutual_info_regression if len(target.unique()) > 20 else mutual_info_classif
    mrmr_scores = []
    relevance_scores = []
    redundancy_scores = []

    for feat in selected_features:
        relevance = mi_func(features[[feat]], target)[0]
        relevance_scores.append(relevance)

        redundancy = 0
        if len(selected_features) > 1:
            other_features = [f for f in selected_features if f != feat]
            redundancy = np.mean([mi_func(features[[feat, of]], features[of])[0]
                                  for of in other_features])
        redundancy_scores.append(redundancy)
        mrmr_scores.append(relevance / (redundancy + 1e-12))

    # Create DataFrame for visualization
    mrmr_df = pd.DataFrame({
        'Feature': selected_features,
        'Relevance': relevance_scores,
        'Redundancy': redundancy_scores,
        'MRMR_Score': mrmr_scores
    }).sort_values('MRMR_Score', ascending=True)

    # 4a. MRMR total scores
    plt.figure(figsize=(10, 6))
    # Fix: Convert palette to list and apply correctly
    mrmr_scores_normalized = (mrmr_df['MRMR_Score'] - mrmr_df['MRMR_Score'].min()) / \
                             (mrmr_df['MRMR_Score'].max() - mrmr_df['MRMR_Score'].min())
    colors = [mrmr_palette(x) for x in mrmr_scores_normalized]

    ax = sns.barplot(
        x='MRMR_Score',
        y='Feature',
        data=mrmr_df,
        hue='Feature',
        palette=colors,
        legend=False
    )

    ax.tick_params(axis='x', labelsize=12)  # x-axis data (numerical labels)
    ax.tick_params(axis='y', labelsize=12)  # y-axis data (feature name labels)
    plt.xlabel('MRMR Scores', fontsize=14)
    plt.ylabel('Variables', fontsize=14)
    plt.rcParams['font.family'] = 'Arial'  # Set global font
    plt.tight_layout()
    plt.show()

    # 4b. Relevance and redundancy decomposition
    plt.figure(figsize=(10, 6))
    mrmr_df.set_index('Feature')[['Relevance', 'Redundancy']].plot(kind='barh',
                                                                   stacked=False, color=['#3498db', '#e74c3c'])
    plt.title('Relevance and Redundancy Components of MRMR')
    plt.xlabel('Score')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.show()


# Main function
def main(file_path):
    # 1. Load data
    features, target, feature_names = load_data(file_path)

    # 2. Correlation filtering
    features_filtered, filtered_features, target_corr = filter_features(features, target)

    print(f"Initial number of features: {features.shape[1]}")
    print(f"Number of features after correlation filtering: {features_filtered.shape[1]}")

    # Automatically determine if classification or regression problem
    task_type = 'classification' if len(target.unique()) <= 20 else 'regression'
    print(f"\nDetected task type: {task_type}")

    # 3. MRMR feature selection
    selected_features = manual_mrmr(features_filtered, target,
                                    n_features=8, task_type=task_type)

    print("\nFinal selected features:")
    for i, feat in enumerate(selected_features, 1):
        print(f"{i}. {feat} (Correlation with target: {target_corr[feat]:.3f})")

    # Check inter-feature correlation
    corr_matrix = features_filtered[selected_features].corr().abs()
    max_corr = corr_matrix.values[np.triu_indices_from(corr_matrix, k=1)].max()
    print(f"\nMaximum correlation among selected features: {max_corr:.3f}")

    # 4. Visualize results
    visualize_results(features_filtered, target, selected_features, target_corr)

    return selected_features


# Usage example
if __name__ == "__main__":
    csv_file_path = "PFT/forest.csv"  # Replace with your CSV file path
    selected_features = main(csv_file_path)

