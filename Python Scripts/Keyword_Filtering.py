# Import necessary libraries
import pandas as pd  # For data manipulation and analysis
import re  # For regular expressions (text processing)
import matplotlib.pyplot as plt  # For plotting visualizations
import seaborn as sns  # For creating aesthetically pleasing statistical graphics
from sklearn.metrics import confusion_matrix  # To generate a confusion matrix
import numpy as np  # For numerical computations

# Load the dataset
data_path = r"Dataset\customer_support_tickets.csv"  # Path to the dataset (adjust if necessary)
df = pd.read_csv(data_path)  # Read the dataset into a DataFrame

# Display the dataset preview
print("Dataset Preview:")  # Print a message
print(df.head(), "\n")  # Display the first 5 rows of the dataset for a quick overview

# Display column names
print("Column names in dataset:\n", df.columns, "\n")  # Print all column names in the dataset

# Define the target column and text column
target_column = 'Ticket Type'  # Column containing the actual labels/categories
text_column = 'Ticket Description'  # Column containing text descriptions to classify

# Print label distribution
print(f"Using target column '{target_column}'.\n")  # Print the selected target column
label_distribution = df[target_column].value_counts()  # Count the frequency of each category
print("Label distribution in 'Ticket Type':")  # Print a message
print(label_distribution, "\n")  # Display the distribution of categories

# Plot label distribution (Fix FutureWarning)
plt.figure(figsize=(10, 6))  # Set the figure size
sns.barplot(x=label_distribution.index, y=label_distribution.values, hue=label_distribution.index, palette="viridis", legend=False)  # Create a bar plot for category distribution
plt.title("Label Distribution in Ticket Type", fontsize=16)  # Add a title to the plot
plt.ylabel("Count", fontsize=12)  # Label the y-axis
plt.xlabel("Ticket Type", fontsize=12)  # Label the x-axis
plt.xticks(rotation=45, fontsize=10)  # Rotate x-axis labels for better readability
plt.tight_layout()  # Adjust the layout to avoid overlapping
plt.show()  # Display the plot

# Preprocess text
def preprocess_text(text):
    if pd.isna(text):  # Check if the text is NaN (missing value)
        return ""  # Return an empty string for missing values
    text = text.lower()  # Convert text to lowercase
    text = re.sub(r'[^\w\s]', '', text)  # Remove punctuation
    return text.strip()  # Remove leading/trailing spaces

# Apply text preprocessing to the specified column
df[text_column] = df[text_column].apply(preprocess_text)

# Rule-based keyword filtering
def classify_ticket(description):
    # Define keyword mappings for each category
    keyword_map = {
        'Refund request': ['refund', 'return', 'money back', 'refund request'],
        'Technical issue': ['technical', 'error', 'not working', 'crash', 'bug'],
        'Cancellation request': ['cancel', 'termination', 'close account'],
        'Product inquiry': ['price', 'spec', 'details', 'buy', 'purchase'],
        'Billing inquiry': ['bill', 'invoice', 'charge', 'payment', 'account balance'],
        'Shipping': ['ship', 'delivery', 'track', 'shipping', 'dispatch'],
        'General Inquiry': ['question', 'help', 'general', 'information', 'assistance']
    }
    for category, keywords in keyword_map.items():  # Loop through each category and its keywords
        if any(keyword in description for keyword in keywords):  # Check if any keyword matches the description
            return category  # Return the matching category
    return 'General Inquiry'  # Default category if no match is found

# Apply classification to each ticket description
df['Predicted_Category'] = df[text_column].apply(classify_ticket)

# Display predicted categories for the first few rows
print("Predicted categories:")  # Print a message
print(df[['Predicted_Category']].head(), "\n")  # Show the predicted categories for the first 5 rows

# Confusion Matrix
conf_matrix = confusion_matrix(df[target_column], df['Predicted_Category'], labels=df[target_column].unique())  # Compute the confusion matrix

# Plot Confusion Matrix Heatmap
plt.figure(figsize=(10, 6))  # Set the figure size
sns.heatmap(conf_matrix, annot=True, fmt='g', cmap="Blues", xticklabels=df[target_column].unique(), yticklabels=df[target_column].unique())  # Create a heatmap of the confusion matrix
plt.title("Confusion Matrix Heatmap", fontsize=16)  # Add a title
plt.ylabel("True Labels", fontsize=12)  # Label the y-axis
plt.xlabel("Predicted Labels", fontsize=12)  # Label the x-axis
plt.xticks(rotation=45, fontsize=10)  # Rotate x-axis labels for better readability
plt.yticks(rotation=0, fontsize=10)  # Ensure y-axis labels are horizontal
plt.tight_layout()  # Adjust the layout
plt.show()  # Display the heatmap

# Visualize predicted categories (Pie Chart)
predicted_distribution = df['Predicted_Category'].value_counts()  # Count the frequency of predicted categories

# Pie chart for predicted categories
plt.figure(figsize=(8, 8))  # Set the figure size
plt.pie(predicted_distribution, labels=predicted_distribution.index, autopct='%1.1f%%', colors=sns.color_palette("pastel", len(predicted_distribution)))  # Create a pie chart
plt.title("Distribution of Predicted Categories", fontsize=16)  # Add a title
plt.tight_layout()  # Adjust the layout
plt.show()  # Display the pie chart

# Simulate predicting a new ticket
new_ticket = "I have an issue with my invoice and charges"  # Define a new ticket description
new_ticket_processed = preprocess_text(new_ticket)  # Preprocess the new ticket description
predicted_category = classify_ticket(new_ticket_processed)  # Predict the category of the new ticket
print("Predicted category for new ticket:", predicted_category)  # Print the predicted category
