import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

def visualize_data(filepath="data/component_data.csv"):
    df = pd.read_csv(filepath)
    sns.set(style="whitegrid")

    plt.figure(figsize=(8,5))
    sns.countplot(x="failure", data=df, palette="coolwarm")
    plt.title("Component Failure Distribution")
    plt.show()

    plt.figure(figsize=(10,6))
    sns.heatmap(df.corr(), annot=True, cmap="coolwarm", fmt=".2f")
    plt.title("Feature Correlation Heatmap")
    plt.show()
