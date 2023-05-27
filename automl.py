pip install tpot
from tpot import TPOTClassifier
import sklearn.datasets
import sklearn.model_selection
import sklearn.metrics

# Load a classification dataset
X, y = sklearn.datasets.load_digits(return_X_y=True)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(X, y, random_state=42)

# Define the TPOT classifier
tpot = TPOTClassifier(generations=5, population_size=20, verbosity=2, random_state=42)

# Fit the TPOT classifier to the training data
tpot.fit(X_train, y_train)

# Evaluate the performance of the TPOT model on the testing data
accuracy = tpot.score(X_test, y_test)
print("Accuracy:", accuracy)
