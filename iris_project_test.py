import unittest
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix

class TestIrisPipeline(unittest.TestCase):
    def __init__(self):
        # Load the Iris dataset
        self.iris = load_iris()
        self.X = self.iris.data
        self.y = self.iris.target
        self.feature_names = self.iris.feature_names
        self.target_names = self.iris.target_names

    def test_data_loading(self):
        # Test if data loads correctly
        self.assertEqual(self.X.shape, (150, 4))  # 150 samples, 4 features
        self.assertEqual(self.y.shape, (150,))
        self.assertEqual(len(self.feature_names), 4)
        self.assertEqual(len(self.target_names), 3)

    def test_missing_and_duplicates(self):
        data = pd.DataFrame(self.X, columns=self.feature_names)
        self.assertTrue(data.isnull().sum().sum() == 0)  # No missing values
    
        duplicate_count = data.duplicated().sum()
        if duplicate_count > 0:
            print(f"Number of duplicate rows: {duplicate_count}")
            data.drop_duplicates(inplace=True)
    
        self.assertEqual(data.duplicated().sum(), 0)  # Ensure no duplicates after removal


    def test_data_splitting(self):
        # Test the data splitting
        X_train, X_test, y_train, y_test = train_test_split(self.X, self.y, test_size=0.2, random_state=42)
        self.assertEqual(len(X_train), 120)  # 80% of 150
        self.assertEqual(len(X_test), 30)   # 20% of 150

    def test_scaling(self):
        # Test data scaling
        X_train, X_test, _, _ = train_test_split(self.X, self.y, test_size=0.2, random_state=42)
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        self.assertAlmostEqual(X_train_scaled.mean(), 0, delta=0.1)  # Scaled data mean should be approx 0
        self.assertAlmostEqual(X_train_scaled.std(), 1, delta=0.1)   # Scaled data std should be approx 1

    def test_model_training_and_evaluation(self):
        # Test model training and evaluation
        X_train, X_test, y_train, y_test = train_test_split(self.X, self.y, test_size=0.2, random_state=42)
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        model = LogisticRegression()
        model.fit(X_train_scaled, y_train)
        y_pred = model.predict(X_test_scaled)

        # Test model accuracy
        accuracy = accuracy_score(y_test, y_pred)
        self.assertGreaterEqual(accuracy, 0.9)  # Ensure accuracy is at least 90%

        # Test confusion matrix
        cm = confusion_matrix(y_test, y_pred)
        self.assertEqual(cm.shape, (3, 3))  # Should be 3x3 for 3 classes

if __name__ == '__main__':
    unittest.main()
