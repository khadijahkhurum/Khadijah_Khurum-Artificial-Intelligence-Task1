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


# %%
import pandas as pd  # Import the pandas library for data manipulation

# Load your dataset
data_path = r"C:\Users\khadi\OneDrive\Desktop\AI Assignment Dataset\customer_support_tickets.csv"  # Specify the path to your dataset
df = pd.read_csv(data_path)  # Load the CSV file into a DataFrame

# Check the first few rows of the dataset
print(df.head())  # Print the first five rows of the DataFrame to understand its structure

# Define a function to classify the tickets based on Ticket Description
def categorize_ticket(description):
    description = description.lower()  # Convert the ticket description to lowercase for uniformity
    # Check if certain keywords are present in the description and return corresponding category
    if 'billing' in description or 'invoice' in description:
        return 'Billing'  # Categorize as Billing if keywords are found
    elif 'technical' in description or 'issue' in description or 'problem' in description:
        return 'Technical Support'  # Categorize as Technical Support if relevant keywords are present
    elif 'account' in description:
        return 'Account Issues'  # Categorize as Account Issues if the keyword 'account' is found
    elif 'general' in description or 'inquiry' in description:
        return 'General Inquiry'  # Categorize as General Inquiry for related keywords
    else:
        return 'Other'  # Default category if no keywords match

# Create the target column using the function
df['Predicted_Category'] = df['Ticket Description'].apply(categorize_ticket)  # Apply the categorization function to the Ticket Description column

# Define another function to classify tickets with specific keywords
def classify_ticket(ticket_description):
    # Define keywords for each category
    billing_keywords = ['invoice', 'billing', 'payment', 'charge', 
                        'billing statement', 'debit', 'credit', 'receipt', 
                        'subscription', 'fee']  # List of keywords for Billing category
    technical_support_keywords = ['issue', 'problem', 'not working', 
                                   'error', 'support', 'help', 'assistance']  # List for Technical Support
    account_issues_keywords = ['account', 'login', 'password', 'access']  # List for Account Issues

    # Check for billing keywords first
    if any(keyword in ticket_description.lower() for keyword in billing_keywords):
        return 'Billing'  # Return 'Billing' if any keyword is found
    
    # Then check for technical support keywords
    if any(keyword in ticket_description.lower() for keyword in technical_support_keywords):
        return 'Technical Support'  # Return 'Technical Support' if any keyword is found
    
    # Lastly check for account issues keywords
    if any(keyword in ticket_description.lower() for keyword in account_issues_keywords):
        return 'Account Issues'  # Return 'Account Issues' if any keyword is found
    
    # If no keywords matched, return 'Other'
    return 'Other'  # Default return value if no keywords matched

# Make sure to check if the target column is created successfully
print(df[['Ticket Description', 'Predicted_Category']].head())  # Print the Ticket Description and its predicted category

# Features and target variable
X = df['Ticket Description']  # Extract the text data from the Ticket Description column
y = df['Predicted_Category']  # Define the target variable based on the predicted categories

# Vectorizing text data
from sklearn.feature_extraction.text import CountVectorizer  # Import the CountVectorizer for converting text to numerical data

vectorizer = CountVectorizer()  # Initialize CountVectorizer
X_vectorized = vectorizer.fit_transform(X)  # Transform the text data into a numerical format

# Creating and training the Decision Tree classifier
from sklearn.model_selection import train_test_split, cross_val_score  # Import functions for model evaluation
from sklearn.tree import DecisionTreeClassifier  # Import the Decision Tree classifier
from sklearn.metrics import classification_report, accuracy_score  # Import metrics for evaluation

clf = DecisionTreeClassifier(random_state=42)  # Create an instance of the Decision Tree classifier

# Cross-validation to evaluate model performance
from sklearn.model_selection import StratifiedKFold  # Import StratifiedKFold for cross-validation
cv = StratifiedKFold(n_splits=5)  # Define 5 splits for cross-validation
scores = cross_val_score(clf, X_vectorized, y, cv=cv)  # Perform cross-validation and get accuracy scores
print("Cross-validated accuracy scores:", scores)  # Print the cross-validated scores
print("Mean accuracy:", scores.mean())  # Print the mean accuracy across the folds

# Splitting data into training and testing sets for further evaluation
X_train, X_test, y_train, y_test = train_test_split(X_vectorized, y, test_size=0.2, random_state=42)  # Split the data into training (80%) and testing (20%) sets
clf.fit(X_train, y_train)  # Train the classifier on the training set

# Making predictions
y_pred = clf.predict(X_test)  # Use the classifier to make predictions on the test set

# Evaluating the model
print("Accuracy:", accuracy_score(y_test, y_pred))  # Print the accuracy of the model
print("Classification Report:\n", classification_report(y_test, y_pred))  # Print a detailed classification report

# Example of predicting a new ticket
new_ticket = ['I want to know about my last invoice']  # Define a new ticket description to predict its category
new_ticket_vectorized = vectorizer.transform(new_ticket)  # Transform the new ticket description using the trained vectorizer
predicted_category = clf.predict(new_ticket_vectorized)  # Predict the category of the new ticket

print("Predicted category for new ticket:", predicted_category[0])  # Print the predicted category for the new ticket



