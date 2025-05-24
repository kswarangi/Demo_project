import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Load data
df = pd.read_csv("./Data/train.csv") 

# Basic info
print(df.info())
print(df.describe())

# Missing values
print(df.isnull().sum())

# Fill missing Age
df["Age"].fillna(df["Age"].median(), inplace=True)

# Visualization: Survival rate by gender
sns.barplot(x="Sex", y="Survived", data=df)
plt.title("Survival Rate by Gender")
plt.show()

# Correlation heatmap
sns.heatmap(df.corr(), annot=True, cmap="coolwarm")
plt.show()
