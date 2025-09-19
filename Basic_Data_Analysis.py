# Import necessary libraries
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_iris
import numpy as np

# Set style for better looking plots
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

# Task 1: Load and Explore the Dataset
try:
    # Load Iris dataset
    iris = load_iris()
    df = pd.DataFrame(iris.data, columns=iris.feature_names)
    df['species'] = iris.target
    df['species'] = df['species'].map({0: 'setosa', 1: 'versicolor', 2: 'virginica'})
    
    print("Dataset loaded successfully!")
    print(f"Dataset shape: {df.shape}")
    
    # Display first few rows
    print("\nFirst 5 rows:")
    print(df.head())
    
    # Explore structure
    print("\nDataset info:")
    print(df.info())
    
    # Check for missing values
    print("\nMissing values:")
    print(df.isnull().sum())
    
    # No missing values in Iris dataset, but for demonstration:
    if df.isnull().sum().sum() > 0:
        df = df.dropna()  # or df.fillna(method='ffill')
        print("Missing values handled!")
    
except Exception as e:
    print(f"Error loading dataset: {e}")

# Task 2: Basic Data Analysis
print("\n" + "="*50)
print("BASIC DATA ANALYSIS")
print("="*50)

# Basic statistics
print("\nBasic statistics:")
print(df.describe())

# Group by species and compute mean of numerical columns
print("\nMean values by species:")
species_stats = df.groupby('species').mean()
print(species_stats)

# Additional analysis - find patterns
print("\nInteresting findings:")
max_sepal_length = df.loc[df['sepal length (cm)'].idxmax()]
print(f"Max sepal length: {max_sepal_length['sepal length (cm)']}cm ({max_sepal_length['species']})")

# Task 3: Data Visualization
print("\n" + "="*50)
print("DATA VISUALIZATION")
print("="*50)

# Create figure with subplots
fig, axes = plt.subplots(2, 2, figsize=(15, 12))
fig.suptitle('Iris Dataset Analysis', fontsize=16, fontweight='bold')

# 1. Line chart (simulating trends - since Iris doesn't have time data)
# We'll use index as x-axis for demonstration
axes[0, 0].plot(df.index[:50], df['sepal length (cm)'][:50], 'b-', label='Sepal Length')
axes[0, 0].plot(df.index[:50], df['petal length (cm)'][:50], 'r-', label='Petal Length')
axes[0, 0].set_title('Trend Analysis (First 50 Samples)')
axes[0, 0].set_xlabel('Sample Index')
axes[0, 0].set_ylabel('Length (cm)')
axes[0, 0].legend()
axes[0, 0].grid(True, alpha=0.3)

# 2. Bar chart - average measurements by species
species_means = df.groupby('species').mean()
x_pos = np.arange(len(species_means.index))
width = 0.2

axes[0, 1].bar(x_pos - width, species_means['sepal length (cm)'], width, label='Sepal Length')
axes[0, 1].bar(x_pos, species_means['sepal width (cm)'], width, label='Sepal Width')
axes[0, 1].bar(x_pos + width, species_means['petal length (cm)'], width, label='Petal Length')
axes[0, 1].set_title('Average Measurements by Species')
axes[0, 1].set_xlabel('Species')
axes[0, 1].set_ylabel('Measurement (cm)')
axes[0, 1].set_xticks(x_pos)
axes[0, 1].set_xticklabels(species_means.index)
axes[0, 1].legend()
axes[0, 1].grid(True, alpha=0.3)

# 3. Histogram - distribution of sepal length
axes[1, 0].hist(df['sepal length (cm)'], bins=15, alpha=0.7, edgecolor='black')
axes[1, 0].set_title('Distribution of Sepal Length')
axes[1, 0].set_xlabel('Sepal Length (cm)')
axes[1, 0].set_ylabel('Frequency')
axes[1, 0].grid(True, alpha=0.3)

# 4. Scatter plot - sepal length vs petal length
colors = {'setosa': 'red', 'versicolor': 'green', 'virginica': 'blue'}
for species in df['species'].unique():
    subset = df[df['species'] == species]
    axes[1, 1].scatter(subset['sepal length (cm)'], subset['petal length (cm)'], 
                      label=species, alpha=0.7, s=50)
axes[1, 1].set_title('Sepal Length vs Petal Length')
axes[1, 1].set_xlabel('Sepal Length (cm)')
axes[1, 1].set_ylabel('Petal Length (cm)')
axes[1, 1].legend()
axes[1, 1].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# Additional insights
print("\nKey Insights:")
print("1. Setosa species has the smallest petals but wider sepals")
print("2. Virginica has the largest sepals and petals")
print("3. Strong positive correlation between sepal and petal length")
print("4. Clear separation between species in the scatter plot")

# Save the dataframe to CSV for future reference (optional)
df.to_csv('iris_analysis.csv', index=False)
print("\nAnalysis completed and results saved!")