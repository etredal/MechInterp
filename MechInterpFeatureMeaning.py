import torch
from transformer_lens import HookedTransformer
from sae_lens import SAE
import ModelInterpreter
import matplotlib.pyplot as plt
import numpy as np

# Load in everything for gpt2
device = "cuda" if torch.cuda.is_available() else "cpu"
model = HookedTransformer.from_pretrained("gpt2-small").to(device)
sae, cfg_dict, sparsity = SAE.from_pretrained(
    release="gpt2-small-res-jb",
    sae_id="blocks.8.hook_resid_pre",
    device=device
)

# Lists of opposing statements
happy_statements = [
    "I am feeling happy today!",
    "The sun is shining and I feel great!",
    "I'm in a good mood today.",
    "I'm feeling cheerful and optimistic.",
    "I'm having a wonderful day!",
    "You did wonderful.",
    "This is so great!"
]

angry_statements = [
    "You did terrible.",
    "This is so bad!",
    "What a bad example of how to do this!",
    "Don't you know this is useless?",
    "This is so terrible.",
    "He always does horrible work!",
    "I hate this project so much."
]

# Make these just random statements
baseline_statements = [
    "The purple giraffe danced elegantly under the disco ball while humming Beethoven's Fifth Symphony.",
    "Quantum bananas might one day revolutionize interstellar travel if we can harness their peel energy.",
    "A committee of sentient toasters just declared independence from the kitchen, demanding equal rights.",
    "The number seven has been feeling quite lonely..",
    "If gravity suddenly reversed, penguins would become the dominant life form due to their aerodynamic properties.",
    "To prove time travel is possible, a historian from the year 2456 just photobombed the Mona Lisa.",
    "Every time you sneeze, an alternate universe is created where you sneezed slightly differently.",
    "The dictionary secretly updates itself every night, adding new words that nobody notices until it's too late.",
    "Artificial intelligence once tried to write a Shakespearean sonnet, but it accidentally started a cult instead.",
    "Somewhere, right now, a cat is plotting to overthrow its human owner with a well-timed hairball."
]

ModelInterpreter = ModelInterpreter.ModelInterpreter()

# Run all statements through the model
happy_activations = [ModelInterpreter.get_activations_np(statement) for statement in happy_statements]
angry_activations = [ModelInterpreter.get_activations_np(statement) for statement in angry_statements]
baseline_activations = [ModelInterpreter.get_activations_np(statement) for statement in baseline_statements]

# Calculate the mean activation for each feature per statement type
happy_feature_means = [np.mean(statement, axis=0) for statement in happy_activations]
angry_feature_means = [np.mean(statement, axis=0) for statement in angry_activations]
baseline_feature_means = [np.mean(statement, axis=0) for statement in baseline_activations]

# Combine across statements
happy_feature_combined = np.mean(happy_feature_means, axis=0)
angry_feature_combined = np.mean(angry_feature_means, axis=0)
baseline_feature_combined = np.mean(baseline_feature_means, axis=0)

# Sort features by activation strength (descending)
top_20_happy = np.argsort(happy_feature_combined)[-20:][::-1]
top_20_angry = np.argsort(angry_feature_combined)[-20:][::-1]
top_20_baseline = np.argsort(baseline_feature_combined)[-20:][::-1]

print("\n")

print("Top 20 features for happy statements:")
for i, idx in enumerate(top_20_happy, 1):
    print(f"{i}. Feature {idx}: {happy_feature_combined[idx]:.4f}")

print("\nTop 20 features for angry statements:")
for i, idx in enumerate(top_20_angry, 1):
    print(f"{i}. Feature {idx}: {angry_feature_combined[idx]:.4f}")

print("\nTop 20 features for baseline statements:")
for i, idx in enumerate(top_20_baseline, 1):
    print(f"{i}. Feature {idx}: {baseline_feature_combined[idx]:.4f}")

# Scatter plot to show all 7 happy statements' feature activations
plt.figure(figsize=(14, 8))

# Get number of features
num_features = len(happy_feature_means[0])
feature_indices = np.arange(num_features)

# Plot each happy statement's features separately
for i in range(len(happy_statements)):  # Assuming there are 7 happy statements
    plt.scatter(feature_indices, happy_feature_means[i], 
               alpha=0.5, s=20, label=f'Happy Statement {i+1}')

print()
print("Click out of the plot to continue.")

# Label the axes and the plot
plt.title('Feature Activations Across 7 Happy Statements.  CLICK EXIT TO CONTINUE RUNNING THE CODE', fontsize=14)
plt.xlabel('Feature Index', fontsize=12)
plt.ylabel('Activation Strength', fontsize=12)
plt.grid(True, linestyle='--', alpha=0.3)
plt.legend()

# Show the plot
plt.show()

# Find the intersection of top 20 features between happy and baseline
common_features_happy = np.intersect1d(top_20_happy, top_20_baseline)
common_features_angry = np.intersect1d(top_20_angry, top_20_baseline)

print()
print("Common features in top 20 for Happy and Baseline:", common_features_happy)
print("Common features in top 20 for Angry and Baseline:", common_features_angry)\

# Remove all the common features
happy_activations_filtered = [np.delete(statement, common_features_happy, axis=1) for statement in happy_activations]
angry_activations_filtered = [np.delete(statement, common_features_angry, axis=1) for statement in angry_activations]

# Get the means across the tokens for each statement
happy_feature_means_filtered = [np.mean(statement, axis=0) for statement in happy_activations_filtered]
angry_feature_means_filtered = [np.mean(statement, axis=0) for statement in angry_activations_filtered]

# Graph the filtered happy activations
plt.figure(figsize=(14, 8))

# Get number of features
num_features = len(happy_feature_means_filtered[0])
feature_indices = np.arange(num_features)

# Plot each happy statement's features separately
for i in range(len(happy_statements)):  # Assuming there are 7 happy statements
    plt.scatter(feature_indices, happy_feature_means_filtered[i], 
               alpha=0.5, s=20, label=f'Happy Statement {i+1}')
    
print()
print("Click out of the plot to continue.")

# Label the axes and the plot
plt.title('Feature Activations Across 7 Happy Statements After Filtering Common Features.  CLICK EXIT TO CONTINUE RUNNING THE CODE', fontsize=14)
plt.xlabel('Feature Index', fontsize=12)
plt.ylabel('Activation Strength', fontsize=12)
plt.grid(True, linestyle='--', alpha=0.3)
plt.legend()

# Show the plot
plt.show()

# Convert list of activations to NumPy array
feature_matrix = np.array([statement.mean(axis=0) for statement in happy_activations_filtered])  # Shape (7, num_features)

# Identify features that have at least an activation of 0.5
non_zero_percentage = np.sum(feature_matrix > 0.5, axis=0) / feature_matrix.shape[0]

# Select features that are active in at least 75% of statements (adjust threshold as needed)
meaningful_features = non_zero_percentage >= 0.75
feature_matrix_filtered = feature_matrix[:, meaningful_features]

# Compute correlation matrix for meaningful features
correlation_matrix = np.corrcoef(feature_matrix_filtered, rowvar=False)  # Shape (num_filtered_features, num_filtered_features)

# Get average correlation for each feature with other features
feature_mean_corrs = np.mean(correlation_matrix, axis=1)

# Get indices of top correlated features
top_n = 20  # Adjust as needed
top_indices = np.argsort(feature_mean_corrs)[-top_n:][::-1]

# Map back to original feature indices
original_indices = np.where(meaningful_features)[0][top_indices]

# Print top correlated features
print("\nTop Features with Highest Mean Correlation:")
for i, (orig_idx, corr) in enumerate(zip(original_indices, feature_mean_corrs[top_indices])):
    print(f"Rank {i+1}: Feature {orig_idx} - Avg Correlation {corr:.3f}")


