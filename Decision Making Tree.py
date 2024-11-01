# %%
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.tree import DecisionTreeClassifier
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import classification_report, accuracy_score

# Load your dataset
data_path = r"C:\Users\khadi\OneDrive\Desktop\AI Assignment Dataset\customer_support_tickets.csv"
df = pd.read_csv(data_path)

# Check the first few rows of the dataset
print(df.head())

# Check the column names in the dataset
print("Column names in dataset:")
print(df.columns)

# Ensure that 'Ticket Description' (or your chosen text column) and target category column are specified correctly
text_column = 'Ticket Description'  # Replace with the correct text column
target_column = 'Predicted_Category'  # Replace with your actual target column name

# Ensure the target column is available
if target_column not in df.columns:
    print(f"Target column '{target_column}' not found in the dataset.")
else:
    # Features and target variable
    X = df[text_column]  # Extract text data
    y = df[target_column]  # Extract target variable

    # Vectorizing text data
    vectorizer = CountVectorizer()
    X_vectorized = vectorizer.fit_transform(X)

    # Create and train the Decision Tree classifier
    clf = DecisionTreeClassifier(random_state=42)

    # Cross-validation to evaluate model performance
    cv = StratifiedKFold(n_splits=5)  # Using StratifiedKFold for class balance
    scores = cross_val_score(clf, X_vectorized, y, cv=cv)  # 5-fold cross-validation
    print("Cross-validated accuracy scores:", scores)
    print("Mean accuracy:", scores.mean())

    # Splitting data into training and testing sets for further evaluation
    X_train, X_test, y_train, y_test = train_test_split(X_vectorized, y, test_size=0.2, random_state=42)
    clf.fit(X_train, y_train)

    # Making predictions
    y_pred = clf.predict(X_test)

    # Evaluating the model
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print("Classification Report:\n", classification_report(y_test, y_pred))

    # Example of predicting a new ticket
    new_ticket = ['I want to know about my last invoice']  # Replace with an actual example
    new_ticket_vectorized = vectorizer.transform(new_ticket)
    predicted_category = clf.predict(new_ticket_vectorized)

    print("Predicted category for new ticket:", predicted_category[0])


