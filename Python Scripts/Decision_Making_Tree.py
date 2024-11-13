import pandas as pd
from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix, precision_score, recall_score, f1_score
from sklearn.pipeline import Pipeline
import matplotlib.pyplot as plt
import seaborn as sns
import re
import time

# Load the dataset
data_path = r"C:\Users\khadi\OneDrive\Desktop\AI Assignment Dataset\customer_support_tickets.csv"
df = pd.read_csv(data_path)

# Display basic dataset information
print("Dataset Preview:\n", df.head())
print("\nColumn names in dataset:\n", df.columns)

# Select the target column
target_column = 'Ticket Type' if 'Ticket Type' in df.columns else 'Predicted_Category'
print(f"\nUsing target column '{target_column}'.")

# Show label distribution
label_counts = df[target_column].value_counts()
print(f"\nLabel distribution in '{target_column}':\n{label_counts}")

# Text preprocessing function
def preprocess_text(text):
    text = text.lower()  # Convert text to lowercase
    text = re.sub(r'\d+', '', text)  # Remove numbers
    text = re.sub(r'\b\w{1,2}\b', '', text)  # Remove short words
    text = re.sub(r'[^\w\s]', '', text)  # Remove punctuation
    text = re.sub(r'\s+', ' ', text).strip()  # Remove extra spaces
    return text

# Apply text preprocessing
text_column = 'Ticket Description'
df[text_column] = df[text_column].apply(preprocess_text)

# Define features and target
X = df[text_column]  # Feature: preprocessed text
y = df[target_column]  # Target: ticket category

# Set up pipeline with TF-IDF Vectorizer and Random Forest Classifier
pipeline = Pipeline([
    ('tfidf', TfidfVectorizer(max_df=0.85, min_df=5, ngram_range=(1, 2))),
    ('classifier', RandomForestClassifier(class_weight='balanced', random_state=42))
])

# Parameter grid for RandomForest
param_grid = {
    'classifier__n_estimators': [100, 200],
    'classifier__max_depth': [10, 20, None],
    'classifier__min_samples_split': [2, 5]
}

# Set up GridSearchCV for hyperparameter tuning
grid_search = GridSearchCV(pipeline, param_grid, cv=StratifiedKFold(n_splits=5), scoring='accuracy', n_jobs=-1)

# Track GridSearchCV runtime
print("\nStarting GridSearchCV for hyperparameter tuning...")
start_time = time.time()
grid_search.fit(X, y)
grid_search_time = (time.time() - start_time) * 1000  # Convert to milliseconds
print("GridSearchCV completed.")

# Best model and parameters
best_model = grid_search.best_estimator_
print("Best parameters found:", grid_search.best_params_)
print(f"Best cross-validated accuracy: {grid_search.best_score_:.2f}")

# Split data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Measure training time for the best model
print("\nTraining the best model...")
start_time = time.time()
best_model.fit(X_train, y_train)
training_time = (time.time() - start_time) * 1000  # Convert to milliseconds

# Prediction on test data
start_time = time.time()
y_pred = best_model.predict(X_test)
prediction_time = (time.time() - start_time) * 1000  # Convert to milliseconds

# Display accuracy and first few predictions
accuracy = accuracy_score(y_test, y_pred)
print(f"\nTest set accuracy: {accuracy:.2f}")
print("First 10 predictions on test set:", y_pred[:10])

# Detailed classification report
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# Confusion Matrix
conf_matrix = confusion_matrix(y_test, y_pred)
print("\nConfusion Matrix:\n", conf_matrix)

# Additional metrics: Precision, Recall, F1 score
precision = precision_score(y_test, y_pred, average='weighted')
recall = recall_score(y_test, y_pred, average='weighted')
f1 = f1_score(y_test, y_pred, average='weighted')

# Display metrics and timings
print(f"\nPrecision: {precision * 100:.2f}%")
print(f"Recall: {recall * 100:.2f}%")
print(f"F1 Score: {f1 * 100:.2f}%")
print(f"Accuracy: {accuracy * 100:.2f}%")
print(f"GridSearchCV Time: {grid_search_time:.2f} ms")
print(f"Training Time: {training_time:.2f} ms")
print(f"Prediction Time: {prediction_time:.2f} ms")

# Visualization
categories = y_test.unique()

# Chart 1: Precision, Recall, and F1 Score by Category
precision_per_category = precision_score(y_test, y_pred, average=None, labels=categories)
recall_per_category = recall_score(y_test, y_pred, average=None, labels=categories)
f1_per_category = f1_score(y_test, y_pred, average=None, labels=categories)

plt.figure(figsize=(12, 6))
x = range(len(categories))

# Plotting bar charts for each metric
plt.bar(x, precision_per_category, width=0.25, label='Precision', color='skyblue', align='center')
plt.bar([i + 0.25 for i in x], recall_per_category, width=0.25, label='Recall', color='salmon', align='center')
plt.bar([i + 0.5 for i in x], f1_per_category, width=0.25, label='F1 Score', color='lightgreen', align='center')

# Adding labels and formatting
plt.xticks(ticks=[i + 0.25 for i in x], labels=categories, rotation=45)
plt.xlabel('Categories')
plt.ylabel('Scores')
plt.ylim(0, 1)
plt.title('Precision, Recall, and F1 Score by Category')
plt.legend()
plt.tight_layout()
plt.show()

# Chart 2: Confusion Matrix Heatmap
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt="d", cmap='Blues', xticklabels=categories, yticklabels=categories)
plt.title('Confusion Matrix Heatmap')
plt.xlabel('Predicted Category')
plt.ylabel('Actual Category')
plt.show()

# Chart 3: Distribution of Predictions by Category
predicted_counts = pd.Series(y_pred).value_counts().reindex(categories, fill_value=0)
actual_counts = pd.Series(y_test).value_counts().reindex(categories, fill_value=0)

plt.figure(figsize=(10, 6))
bar_width = 0.35
index = range(len(categories))

# Plot bars for actual and predicted counts
plt.bar(index, actual_counts, bar_width, label='Actual Counts', color='slateblue')
plt.bar([i + bar_width for i in index], predicted_counts, bar_width, label='Predicted Counts', color='coral')

# Customize the plot
plt.xticks([i + bar_width / 2 for i in index], categories, rotation=45)
plt.xlabel('Categories')
plt.ylabel('Count')
plt.title('Distribution of Actual vs. Predicted Counts by Category')
plt.legend()
plt.tight_layout()
plt.show()

# Predicting the category for a new ticket
new_ticket = ['I want to know about my last invoice']
new_ticket_processed = [preprocess_text(new_ticket[0])]
predicted_category = best_model.predict(new_ticket_processed)
print("\nPredicted category for new ticket:", predicted_category[0])
