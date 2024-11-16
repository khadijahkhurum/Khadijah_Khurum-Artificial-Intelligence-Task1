import pandas as pd  # Import the pandas library for data manipulation
import re  # Import the regular expression library for text preprocessing
from sklearn.feature_extraction.text import TfidfVectorizer  # Import TF-IDF vectorizer for text feature extraction
from sklearn.model_selection import train_test_split, RandomizedSearchCV  # Import functions for splitting data and hyperparameter tuning
from sklearn.svm import SVC  # Import Support Vector Classifier for the machine learning model
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc  # Import metrics for evaluation
import seaborn as sns  # Import seaborn for creating statistical plots
import matplotlib.pyplot as plt  # Import matplotlib for plotting
import numpy as np  # Import numpy for numerical operations
from sklearn.preprocessing import label_binarize  # Import label_binarize for multi-class ROC curve
from sklearn.metrics import roc_auc_score  # Import AUC for evaluating ROC curve

# Load the dataset
data_path = r"Dataset\customer_support_tickets.csv"  # Set the file path for the dataset
df = pd.read_csv(data_path)  # Read the dataset into a pandas DataFrame

# Select the target column and text column
target_column = 'Ticket Type'  # The column that contains the target labels (ticket types)
text_column = 'Ticket Description'  # The column that contains the text (ticket descriptions)

# Preprocess text function
def preprocess_text(text):
    text = text.lower()  # Convert text to lowercase
    text = re.sub(r'\d+', '', text)  # Remove all digits
    text = re.sub(r'\b\w{1,2}\b', '', text)  # Remove short words (1-2 characters)
    text = re.sub(r'[^\w\s]', '', text)  # Remove all punctuation
    text = re.sub(r'\s+', ' ', text).strip()  # Remove extra spaces and strip leading/trailing spaces
    return text  # Return the cleaned text

df[text_column] = df[text_column].apply(preprocess_text)  # Apply text preprocessing to the entire text column

# Split the dataset into training and testing sets
X = df[text_column]  # Features (text descriptions)
y = df[target_column]  # Target labels (ticket types)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)  # Split data into training (80%) and testing (20%) sets

# TF-IDF with parameter tuning
tfidf_vectorizer = TfidfVectorizer(stop_words='english', ngram_range=(1, 2), max_features=2000)  # Initialize TF-IDF vectorizer with stop words, n-grams, and max features

# Fit and transform the training data, and transform the test data
X_train_tfidf = tfidf_vectorizer.fit_transform(X_train)  # Fit and transform the training data into TF-IDF features
X_test_tfidf = tfidf_vectorizer.transform(X_test)  # Transform the test data into TF-IDF features

# SVM Model with hyperparameter tuning using RandomizedSearchCV
svm_model = SVC(class_weight='balanced', random_state=42, probability=True)  # Initialize SVM model with balanced class weights and probability estimation

# Parameters for tuning (smaller grid for speed)
parameters = {
    'C': [0.1, 1, 10],  # Regularization parameter
    'kernel': ['linear', 'rbf'],  # Kernel types for the SVM
    'gamma': ['scale', 'auto']  # Kernel coefficient for 'rbf' kernel
}

# Perform RandomizedSearchCV to find the best hyperparameters with parallelization
random_search = RandomizedSearchCV(svm_model, parameters, cv=3, n_iter=5, scoring='accuracy', n_jobs=-1, random_state=42)  # Perform randomized search with cross-validation
random_search.fit(X_train_tfidf, y_train)  # Fit the randomized search to the training data

# Best parameters from random search
print("Best parameters found: ", random_search.best_params_)  # Print the best parameters found by RandomizedSearchCV

# Train the best model
best_svm_model = random_search.best_estimator_  # Select the best model from the random search

# Predict on the test set
y_pred = best_svm_model.predict(X_test_tfidf)  # Make predictions on the test data

# Evaluate the model
accuracy = (y_pred == y_test).mean()  # Calculate accuracy by comparing predictions to true labels
print(f"\nTF-IDF with SVM Accuracy: {accuracy:.2f}\n")  # Print the accuracy

# Classification report and confusion matrix
print("Classification Report:")  # Print the classification report
report = classification_report(y_test, y_pred, output_dict=True)  # Generate the classification report as a dictionary

# Print the classification report in the desired format
print(f"{'':<25} {'precision':<10} {'recall':<10} {'f1-score':<10} {'support':<10}")
for class_name, metrics in report.items():  # Loop through each class's metrics in the report
    if class_name not in ['accuracy', 'macro avg', 'weighted avg']:  # Skip overall accuracy metrics
        print(f"{class_name:<25} {metrics['precision']:<10.2f} {metrics['recall']:<10.2f} {metrics['f1-score']:<10.2f} {metrics['support']:<10}")

# Print overall metrics (accuracy, macro avg, weighted avg)
print(f"\n{'accuracy':<25} {accuracy * 100:.2f}%")  # Print accuracy as a percentage
print(f"{'macro avg':<25} {report['macro avg']['precision'] * 100:.2f}% {report['macro avg']['recall'] * 100:.2f}% {report['macro avg']['f1-score'] * 100:.2f}%")  # Print macro averages
print(f"{'weighted avg':<25} {report['weighted avg']['precision'] * 100:.2f}% {report['weighted avg']['recall'] * 100:.2f}% {report['weighted avg']['f1-score'] * 100:.2f}%")  # Print weighted averages
        
# Calculate and print average metrics (precision, recall, F1-score, accuracy)
precision_avg = sum([report[class_name]['precision'] for class_name in report.keys() if class_name not in ['accuracy', 'macro avg', 'weighted avg']]) / len(report)  # Calculate average precision
recall_avg = sum([report[class_name]['recall'] for class_name in report.keys() if class_name not in ['accuracy', 'macro avg', 'weighted avg']]) / len(report)  # Calculate average recall
f1_avg = sum([report[class_name]['f1-score'] for class_name in report.keys() if class_name not in ['accuracy', 'macro avg', 'weighted avg']]) / len(report)  # Calculate average F1-score

# Print the averages (Precision, Recall, F1-Score, Accuracy)
print(f"\nAverage Precision: {precision_avg * 100:.2f}%")  # Print average precision
print(f"Average Recall: {recall_avg * 100:.2f}%")  # Print average recall
print(f"Average F1-Score: {f1_avg * 100:.2f}%")  # Print average F1-score
print(f"Average Accuracy: {accuracy * 100:.2f}%")  # Print average accuracy

# Confusion matrix
conf_matrix = confusion_matrix(y_test, y_pred)  # Generate the confusion matrix

# Confusion matrix heatmap
plt.figure(figsize=(8, 6))  # Set the figure size for the heatmap
sns.heatmap(conf_matrix, annot=True, fmt="d", cmap='Blues', xticklabels=best_svm_model.classes_, yticklabels=best_svm_model.classes_)  # Plot the confusion matrix as a heatmap
plt.title('Confusion Matrix Heatmap')  # Set the title for the heatmap
plt.xlabel('Predicted Category')  # Label for the x-axis
plt.ylabel('Actual Category')  # Label for the y-axis
plt.tight_layout()  # Adjust layout for tightness
plt.show()  # Display the heatmap

# ROC Curve

# Binarize the output labels for multi-class ROC (one-vs-rest)
y_test_bin = label_binarize(y_test, classes=best_svm_model.classes_)  # Binarize the test labels
y_pred_prob = best_svm_model.predict_proba(X_test_tfidf)  # Get the predicted probabilities for each class

# Plot ROC curve for each class
plt.figure(figsize=(12, 8))  # Set the figure size for the ROC curve plot
for i, class_name in enumerate(best_svm_model.classes_):  # Loop through each class
    fpr, tpr, _ = roc_curve(y_test_bin[:, i], y_pred_prob[:, i])  # Calculate false positive rate and true positive rate for ROC curve
    roc_auc = auc(fpr, tpr)  # Calculate AUC (Area Under the Curve)
    plt.plot(fpr, tpr, label=f'{class_name} (AUC = {roc_auc:.2f})')  # Plot the ROC curve for each class

# Plot the diagonal line (Random guess)
plt.plot([0, 1], [0, 1], 'k--')  # Plot a diagonal line indicating random guessing

# Labeling the plot
plt.title('Receiver Operating Characteristic (ROC) Curve')  # Title for the ROC curve plot
plt.xlabel('False Positive Rate')  # Label for the x-axis
plt.ylabel('True Positive Rate')  # Label for the y-axis
plt.legend(loc='lower right')  # Display the legend in the lower-right corner
plt.tight_layout()  # Adjust layout for tightness
plt.show()  # Display the ROC curve

# Predicting the category for a new ticket
new_ticket = ['I want to know about my last invoice']  # Sample new ticket for prediction
new_ticket_processed = preprocess_text(new_ticket[0])  # Preprocess the new ticket text
new_ticket_tfidf = tfidf_vectorizer.transform([new_ticket_processed])  # Transform the new ticket using the TF-IDF vectorizer
predicted_category = best_svm_model.predict(new_ticket_tfidf)  # Predict the category of the new ticket
print("\nPredicted category for new ticket:", predicted_category[0])  # Print the predicted category
