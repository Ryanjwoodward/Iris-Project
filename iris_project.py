
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

# Load the dataset
iris = load_iris()

# Features (input data)
X = iris.data  # Sepal length, sepal width, petal length, petal width

# Target (output data)
y = iris.target  # Species: 0 = Setosa, 1 = Versicolor, 2 = Virginica

# Feature names
feature_names = iris.feature_names  # ['sepal length (cm)', 'sepal width (cm)', ...]

# Target names
target_names = iris.target_names  # ['setosa', 'versicolor', 'virginica']

# Print dataset overview
print("Features:\n", feature_names)
print("Target Classes:\n", target_names)
print("First 5 samples:\n", X[:5])


#! Data Preprocessing Step 1. Check for missing, incomplete, or duplicate data
# Load the dataset
data = pd.DataFrame(iris.data, columns=feature_names)

# Check for missing values
print("\nMissing values per feature:\n", data.isnull().sum())

# Check for duplicates
print("Number of duplicate rows:", data.duplicated().sum())

# Drop the duplicates from the existing data 
data.drop_duplicates(inplace=True)

# Check for missing values
print("\nMissing values per feature:\n", data.isnull().sum())

# Check for duplicates
print("Number of duplicate rows:", data.duplicated().sum())

#! Data Preprocessing Step 2. Standardize Feature Values
# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize the StandardScaler
scaler = StandardScaler()

# Fit the scaler on the training data and transform both train and test sets
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Print the first few scaled values
print("\nFirst 5 rows of scaled training data:\n", X_train_scaled[:5])

#! Step 3: Train and Evaluate the Logistic Regression Model
# Initialize and train the Logistic Regression model
model = LogisticRegression()
model.fit(X_train_scaled, y_train)

# Make predictions on the test data
y_pred = model.predict(X_test_scaled)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print(f"\nAccuracy: {accuracy * 100:.2f}%")

# Generate a classification report
print("\nClassification Report:\n", classification_report(y_test, y_pred, target_names=target_names))

# Confusion matrix A confusion matrix is a table used to evaluate the performance of a classification model. 
# It provides a detailed breakdown of the model's predictions compared to the actual outcomes, 
cm = confusion_matrix(y_test, y_pred)

# Visualize the confusion matrix
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=target_names, yticklabels=target_names)
plt.xlabel("Predicted")
plt.ylabel("True")
plt.title("Confusion Matrix")
plt.show()


