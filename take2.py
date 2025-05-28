# Import Libraries
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px

# Load Dataset
df = pd.read_csv('https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv')

# Basic Information
print("🔹 Dataset Info:")
df.info()

print("\n🔹 Missing Values:\n", df.isnull().sum())

print("\n🔹 Summary Statistics:")
print(df.describe())

# Data Cleaning: Fill missing Age with median
df['Age'].fillna(df['Age'].median(), inplace=True)

# Fill Embarked missing values
df['Embarked'].fillna(df['Embarked'].mode()[0], inplace=True)

# Drop Cabin (too many missing)
df.drop(columns=['Cabin'], inplace=True)

# Histogram: Age Distribution
plt.figure(figsize=(6,4))
sns.histplot(df['Age'], bins=30, kde=True, color='teal')
plt.title("Age Distribution")
plt.xlabel("Age")
plt.ylabel("Count")
plt.show()

# Boxplot: Age vs Survival
plt.figure(figsize=(6,4))
sns.boxplot(x='Survived', y='Age', data=df)
plt.title("Age vs Survival")
plt.show()

plt.figure(figsize=(8,6))
sns.heatmap(df.select_dtypes(include='number').corr(), annot=True, cmap='coolwarm')
plt.title("Feature Correlation Matrix")
plt.show()

# Pairplot (Selected Columns)
sns.pairplot(df[['Age', 'Fare', 'Pclass', 'Survived']], hue='Survived')
plt.show()

# Countplot: Survival vs Sex
plt.figure(figsize=(6,4))
sns.countplot(x='Sex', hue='Survived', data=df)
plt.title("Survival by Gender")
plt.show()

# Boxplot: Fare by Class
plt.figure(figsize=(6,4))
sns.boxplot(x='Pclass', y='Fare', data=df)
plt.title("Fare by Passenger Class")
plt.show()

# Pie Chart: Embarkation Points
plt.figure(figsize=(5,5))
df['Embarked'].value_counts().plot.pie(autopct='%1.1f%%', colors=['#66b3ff','#99ff99','#ffcc99'])
plt.title("Embarkation Distribution")
plt.ylabel("")
plt.show()

# Plotly Interactive: Age vs Fare
fig = px.scatter(df, x='Age', y='Fare', color='Survived',
                 hover_data=['Sex', 'Pclass'],
                 title="Age vs Fare (Interactive)")
fig.show()