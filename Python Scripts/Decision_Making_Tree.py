import pandas as pd
from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report, accuracy_score
from sklearn.pipeline import Pipeline
import re
import numpy as np
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score
import time

# Load your dataset
data_path = r"C:\Users\khadi\OneDrive\Desktop\AI Assignment Dataset\customer_support_tickets.csv"
df = pd.read_csv(data_path)

# Check the first few rows of the dataset
print(df.head())

# Check the column names in the dataset
print("Column names in dataset:")
print(df.columns)

# Define primary and alternative target column names
primary_target_column = 'Predicted_Category'
alternative_target_column = 'Ticket Type'  # Replace this with any possible column that might contain category information

# Check if either target column is present in the dataset
if primary_target_column in df.columns:
    target_column = primary_target_column
    print(f"Target column '{target_column}' found in the dataset.")
elif alternative_target_column in df.columns:
    target_column = alternative_target_column
    print(f"Using alternative target column '{target_column}' found in the dataset.")
else:
    # Prompt for manual labeling if no target column is found
    raise ValueError(f"Neither '{primary_target_column}' nor '{alternative_target_column}' found in the dataset. "
                     "Please provide a target column with category labels.")

# Check for label distribution to ensure balance
label_counts = df[target_column].value_counts()
print(f"Label distribution in '{target_column}':\n{label_counts}")

# Ensure that 'Ticket Description' (or your chosen text column) is specified correctly
text_column = 'Ticket Description'  # Replace with the correct text column if different

# Text preprocessing function to clean the text data
def preprocess_text(text):
    text = text.lower()  # Convert to lowercase
    text = re.sub(r'\d+', '', text)  # Remove numbers
    text = re.sub(r'\b\w{1,2}\b', '', text)  # Remove short words (1-2 characters)
    text = re.sub(r'[^\w\s]', '', text)  # Remove punctuation
    text = re.sub(r'\s+', ' ', text).strip()  # Remove extra spaces
    return text

# Apply text preprocessing to the text column
df[text_column] = df[text_column].apply(preprocess_text)

# Features and target variable
X = df[text_column]  # Extract text data
y = df[target_column]  # Extract target variable

# Define a pipeline with TfidfVectorizer and DecisionTreeClassifier
pipeline = Pipeline([
    ('tfidf', TfidfVectorizer(max_df=0.8, min_df=5, ngram_range=(1, 2))),
    ('classifier', DecisionTreeClassifier(random_state=42))
])

# Hyperparameter tuning for Decision Tree
param_grid = {
    'classifier__max_depth': [10, 20, 30, None],
    'classifier__min_samples_split': [2, 5, 10],
    'classifier__min_samples_leaf': [1, 2, 4]
}

# Using GridSearchCV for hyperparameter tuning
print("Starting GridSearchCV fitting process...")
grid_search = GridSearchCV(pipeline, param_grid, cv=StratifiedKFold(n_splits=5), scoring='accuracy', n_jobs=-1)
start_time = time.time()
try:
    grid_search.fit(X, y)
    print("GridSearchCV completed.")
except Exception as e:
    print(f"Error during GridSearchCV: {e}")
grid_search_time = (time.time() - start_time) * 1000  # Convert to milliseconds

# Print the best parameters and best score from grid search
print("Best parameters:", grid_search.best_params_)
print("Best cross-validated accuracy:", grid_search.best_score_)

# Use the best estimator from GridSearchCV
best_model = grid_search.best_estimator_

# Splitting data into training and testing sets for further evaluation
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
print("Model fitted successfully!")

# Measure training time
start_time = time.time()
best_model.fit(X_train, y_train)
training_time = (time.time() - start_time) * 1000  # Convert to milliseconds

# Make predictions
start_time = time.time()
y_pred = best_model.predict(X_test)
prediction_time = (time.time() - start_time) * 1000  # Convert to milliseconds

print(f"Predicted categories for test set: {y_pred[:10]}")  # Display the first 10 predictions

# Accuracy on the test set
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy on test set: {accuracy:.2f}")

# Classification report
print("Classification Report:")
report = classification_report(y_test, y_pred)
print(report)

# Confusion Matrix
conf_matrix = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:")
print(conf_matrix)

# Additional metrics (precision, recall, F1 score)
precision = precision_score(y_test, y_pred, average='weighted')
recall = recall_score(y_test, y_pred, average='weighted')
f1 = f1_score(y_test, y_pred, average='weighted')

# Print the percentages for each metric
print(f"\nPrecision: {precision * 100:.2f}%")
print(f"Recall: {recall * 100:.2f}%")
print(f"F1 Score: {f1 * 100:.2f}%")
print(f"Accuracy: {accuracy * 100:.2f}%")
print(f"GridSearchCV Processing Speed: {grid_search_time:.2f} ms")
print(f"Training Speed: {training_time:.2f} ms")
print(f"Prediction Speed: {prediction_time:.2f} ms")

# Example of predicting a new ticket
new_ticket = ['I want to know about my last invoice']  # Replace with an actual example
new_ticket_processed = [preprocess_text(new_ticket[0])]  # Preprocess the new ticket text

# Ensure the new ticket is transformed using the pipeline
predicted_category = best_model.predict(new_ticket_processed)

print("Predicted category for new ticket:", predicted_category[0])
