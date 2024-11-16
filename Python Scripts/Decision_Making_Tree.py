# Import necessary libraries
import pandas as pd  # For data manipulation and analysis
from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV  # For splitting data and performing cross-validation/grid search
from sklearn.ensemble import RandomForestClassifier  # RandomForest Classifier model
from sklearn.feature_extraction.text import TfidfVectorizer  # For text vectorization using TF-IDF
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix, precision_score, recall_score, f1_score, roc_curve, auc  # For model evaluation metrics
from sklearn.pipeline import Pipeline  # For creating machine learning pipelines
import matplotlib.pyplot as plt  # For plotting graphs
import seaborn as sns  # For advanced plotting, especially heatmaps
import re  # For text preprocessing (regular expressions)
import time  # For tracking execution time
from sklearn.preprocessing import LabelBinarizer  # For converting labels to binary format for ROC curve

# Load the dataset from the specified path
data_path = r"C:\Users\khadi\OneDrive\Desktop\AI Assignment Dataset\customer_support_tickets.csv"
df = pd.read_csv(data_path)  # Read the CSV file into a DataFrame

# Display basic dataset information
print("Dataset Preview:\n", df.head())  # Display the first few rows of the dataset
print("\nColumn names in dataset:\n", df.columns)  # Show the column names of the dataset

# Select the target column based on available columns
target_column = 'Ticket Type' if 'Ticket Type' in df.columns else 'Predicted_Category'
print(f"\nUsing target column '{target_column}'.")  # Output the selected target column name

# Show label distribution for the selected target column
label_counts = df[target_column].value_counts()  # Count the occurrences of each label in the target column
print(f"\nLabel distribution in '{target_column}':\n{label_counts}")  # Output the label distribution

# Define a function to preprocess the text data
def preprocess_text(text):
    text = text.lower()  # Convert text to lowercase
    text = re.sub(r'\d+', '', text)  # Remove numbers from text
    text = re.sub(r'\b\w{1,2}\b', '', text)  # Remove short words (1-2 characters)
    text = re.sub(r'[^\w\s]', '', text)  # Remove punctuation characters
    text = re.sub(r'\s+', ' ', text).strip()  # Remove extra spaces and strip leading/trailing spaces
    return text  # Return the cleaned text

# Apply the text preprocessing function to the 'Ticket Description' column
text_column = 'Ticket Description'  # Specify the column name for the text data
df[text_column] = df[text_column].apply(preprocess_text)  # Apply the preprocessing function to each entry

# Define features and target variables
X = df[text_column]  # Feature: 'Ticket Description' column (preprocessed text)
y = df[target_column]  # Target: The selected target column (e.g., 'Ticket Type')

# Set up a pipeline that includes a TF-IDF vectorizer and a Random Forest classifier
pipeline = Pipeline([
    ('tfidf', TfidfVectorizer(max_df=0.85, min_df=5, ngram_range=(1, 2))),  # TF-IDF vectorizer to convert text into numerical features
    ('classifier', RandomForestClassifier(class_weight='balanced', random_state=42))  # Random Forest classifier with balanced class weights
])

# Define the parameter grid for RandomForest hyperparameter tuning
param_grid = {
    'classifier__n_estimators': [100, 200],  # Number of trees in the forest
    'classifier__max_depth': [10, 20, None],  # Maximum depth of the trees
    'classifier__min_samples_split': [2, 5]  # Minimum samples required to split an internal node
}

# Set up GridSearchCV for hyperparameter tuning with cross-validation
grid_search = GridSearchCV(pipeline, param_grid, cv=StratifiedKFold(n_splits=5), scoring='accuracy', n_jobs=-1)  # Perform cross-validation and hyperparameter tuning

# Track GridSearchCV runtime
print("\nStarting GridSearchCV for hyperparameter tuning...")
start_time = time.time()  # Record the start time
grid_search.fit(X, y)  # Fit the GridSearchCV to the data
grid_search_time = (time.time() - start_time) * 1000  # Calculate the time taken in milliseconds
print("GridSearchCV completed.")  # Notify when GridSearchCV is done

# Output the best model and parameters found by GridSearchCV
best_model = grid_search.best_estimator_  # Get the best model after grid search
print("Best parameters found:", grid_search.best_params_)  # Print the best parameters
print(f"Best cross-validated accuracy: {grid_search.best_score_:.2f}")  # Print the best accuracy score from cross-validation

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)  # Split data into train and test (80-20 split)

# Measure the time taken to train the best model
print("\nTraining the best model...")
start_time = time.time()  # Start time for training
best_model.fit(X_train, y_train)  # Train the model on the training data
training_time = (time.time() - start_time) * 1000  # Calculate the training time in milliseconds

# Make predictions on the test data
start_time = time.time()  # Start time for prediction
y_pred = best_model.predict(X_test)  # Predict the categories on the test set
prediction_time = (time.time() - start_time) * 1000  # Calculate prediction time in milliseconds

# Display the accuracy and first few predictions
accuracy = accuracy_score(y_test, y_pred)  # Calculate accuracy of the model on the test set
print(f"\nTest set accuracy: {accuracy:.2f}")  # Print the accuracy score
print("First 10 predictions on test set:", y_pred[:10])  # Display the first 10 predictions

# Generate and display a classification report (precision, recall, F1-score, etc.)
print("\nClassification Report:\n", classification_report(y_test, y_pred))  # Detailed classification metrics

# Calculate and print confusion matrix
conf_matrix = confusion_matrix(y_test, y_pred)  # Generate confusion matrix
print("\nConfusion Matrix:\n", conf_matrix)  # Print the confusion matrix

# Calculate additional metrics (precision, recall, F1-score) for the test set
precision = precision_score(y_test, y_pred, average='weighted')  # Weighted precision
recall = recall_score(y_test, y_pred, average='weighted')  # Weighted recall
f1 = f1_score(y_test, y_pred, average='weighted')  # Weighted F1 score

# Display the evaluation metrics and execution times
print(f"\nPrecision: {precision * 100:.2f}%")  # Print precision in percentage
print(f"Recall: {recall * 100:.2f}%")  # Print recall in percentage
print(f"F1 Score: {f1 * 100:.2f}%")  # Print F1 score in percentage
print(f"Accuracy: {accuracy * 100:.2f}%")  # Print accuracy in percentage
print(f"GridSearchCV Time: {grid_search_time:.2f} ms")  # Print the time taken by GridSearchCV
print(f"Training Time: {training_time:.2f} ms")  # Print the time taken to train the model
print(f"Prediction Time: {prediction_time:.2f} ms")  # Print the time taken to make predictions

# --- ROC Curve Section ---
# Binarize the true labels for multi-class ROC curve calculation
lb = LabelBinarizer()  # Initialize the label binarizer
y_test_binarized = lb.fit_transform(y_test)  # Binarize the true labels for multi-class ROC
y_pred_prob = best_model.predict_proba(X_test)  # Get the predicted probabilities for each class

# Initialize plot for ROC curve
plt.figure(figsize=(10, 8))  # Set the size of the ROC curve plot

# Loop through each class and plot the ROC curve
for i in range(len(lb.classes_)):  # Iterate over all classes
    fpr, tpr, thresholds = roc_curve(y_test_binarized[:, i], y_pred_prob[:, i])  # Calculate FPR and TPR for each class
    roc_auc = auc(fpr, tpr)  # Compute AUC for each class
    plt.plot(fpr, tpr, label=f'{lb.classes_[i]} (AUC = {roc_auc:.2f})')  # Plot ROC curve for each class

# Plot the random classifier diagonal (AUC = 0.5)
plt.plot([0, 1], [0, 1], 'k--', label='Random Classifier (AUC = 0.50)')  # Add the random classifier diagonal

# Finalize the plot appearance
plt.xlabel('False Positive Rate')  # Label for X-axis
plt.ylabel('True Positive Rate')  # Label for Y-axis
plt.title('ROC Curve for Multi-Class Classification')  # Title of the plot
plt.legend(loc='lower right')  # Position the legend
plt.grid(True)  # Enable gridlines
plt.tight_layout()  # Adjust the layout for tightness
plt.show()  # Display the ROC curve plot

# Visualization
categories = y_test.unique()  # Get the unique categories from the test set

# Chart 1: Precision, Recall, and F1 Score by Category
precision_per_category = precision_score(y_test, y_pred, average=None, labels=categories)  # Calculate precision for each category
recall_per_category = recall_score(y_test, y_pred, average=None, labels=categories)  # Calculate recall for each category
f1_per_category = f1_score(y_test, y_pred, average=None, labels=categories)  # Calculate F1 score for each category

plt.figure(figsize=(12, 6))  # Set figure size for the bar chart
x = range(len(categories))  # Create an index for categories
width = 0.2  # Set the width of the bars for better visualization
# Plot the bars for precision, recall, and F1 scores
plt.bar(x, precision_per_category, width=width, label='Precision', color='skyblue', align='center')
plt.bar([i + width for i in x], recall_per_category, width=width, label='Recall', color='salmon', align='center')
plt.bar([i + 2 * width for i in x], f1_per_category, width=width, label='F1 Score', color='lightgreen', align='center')
plt.xticks(ticks=[i + width for i in x], labels=categories, rotation=45)  # Set category labels
plt.xlabel('Categories')  # Label for X-axis
plt.ylabel('Scores')  # Label for Y-axis
plt.ylim(0, 1)  # Set the Y-axis range
plt.title('Precision, Recall, and F1 Score by Category')  # Title of the bar chart
plt.legend(loc='upper right')  # Position the legend
plt.tight_layout()  # Adjust layout
plt.show()  # Display the bar chart

# Chart 2: Confusion Matrix Heatmap
plt.figure(figsize=(8, 6))  # Set the size of the heatmap plot
sns.heatmap(conf_matrix, annot=True, fmt="d", cmap='Blues', xticklabels=categories, yticklabels=categories)  # Create a heatmap for the confusion matrix
plt.title('Confusion Matrix Heatmap')  # Title of the heatmap
plt.xlabel('Predicted Category')  # Label for X-axis
plt.ylabel('Actual Category')  # Label for Y-axis
plt.tight_layout()  # Adjust layout
plt.show()  # Display the heatmap

# Chart 3: Distribution of Actual vs. Predicted Counts by Category
predicted_counts = pd.Series(y_pred).value_counts().reindex(categories, fill_value=0)  # Count predicted categories
actual_counts = pd.Series(y_test).value_counts().reindex(categories, fill_value=0)  # Count actual categories

plt.figure(figsize=(10, 6))  # Set figure size for bar chart
bar_width = 0.35  # Set bar width
index = range(len(categories))  # Create index for categories

# Plot bars for actual and predicted counts
plt.bar(index, actual_counts, bar_width, label='Actual Counts', color='slateblue')
plt.bar([i + bar_width for i in index], predicted_counts, bar_width, label='Predicted Counts', color='coral')

# Customize the plot appearance
plt.xticks([i + bar_width / 2 for i in index], categories, rotation=45)  # Set category labels
plt.xlabel('Categories')  # Label for X-axis
plt.ylabel('Count')  # Label for Y-axis
plt.title('Distribution of Actual vs. Predicted Counts by Category')  # Title of the bar chart
plt.legend(loc='upper right')  # Position the legend
plt.tight_layout()  # Adjust layout
plt.show()  # Display the chart

# Predicting the category for a new ticket
new_ticket = ['I want to know about my last invoice']  # A sample new ticket for prediction
new_ticket_processed = [preprocess_text(new_ticket[0])]  # Preprocess the new ticket text
predicted_category = best_model.predict(new_ticket_processed)  # Predict the category for the new ticket
print("\nPredicted category for new ticket:", predicted_category[0])  # Display the predicted category
