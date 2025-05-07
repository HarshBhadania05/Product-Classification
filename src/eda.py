import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from src import config

def run_eda(df):
    print("Data Head:")
    print(df.head())

    print("\nData Info:")
    print(df.info())

    print("\nMissing Values:")
    print(df.isnull().sum())

    print("\nCategory Distribution:")
    category_counts = df['Category Label'].value_counts()
    print(category_counts)

    # Plot category distribution
    plt.figure(figsize=(12, 6))
    sns.countplot(data=df, y='Category Label', order=category_counts.index)
    plt.title("Category Label Distribution")
    plt.tight_layout()
    plt.savefig(f"{config.FIGURE_DIR}/category_distribution.png")
    plt.close()

    # Plot histogram of product title lengths
    df['title_length'] = df['Product Title'].apply(len)
    plt.figure(figsize=(10, 6))
    sns.histplot(df['title_length'], bins=50, kde=True)
    plt.title("Distribution of Product Title Lengths")
    plt.xlabel("Character Count")
    plt.ylabel("Frequency")
    plt.tight_layout()
    plt.savefig(f"{config.FIGURE_DIR}/title_length_distribution.png")
    plt.close()

    # Plot histogram of word counts in product titles
    df['word_count'] = df['Product Title'].apply(lambda x: len(x.split()))
    plt.figure(figsize=(10, 6))
    sns.histplot(df['word_count'], bins=20, kde=True)
    plt.title("Distribution of Word Counts in Product Titles")
    plt.xlabel("Word Count")
    plt.ylabel("Frequency")
    plt.tight_layout()
    plt.savefig(f"{config.FIGURE_DIR}/word_count_distribution.png")
    plt.close()
