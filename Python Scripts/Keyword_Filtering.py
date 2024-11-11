import pandas as pd
import re
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score

# Load the dataset
data_path = r"C:\Users\khadi\OneDrive\Desktop\AI Assignment Dataset\customer_support_tickets.csv"
df = pd.read_csv(data_path)
df.rename(columns={'Ticket Type': 'Actual_Category'}, inplace=True)
print(df['Actual_Category'].unique())



# Check the first few rows of the dataset
print("Dataset Preview:\n", df.head())

# Define a keyword dictionary with priority
# Higher numbers indicate higher priority for category matching
keywords = {
    "technical issue": (["issue", "problem", "technical", "error", "support"], 2),
    "billing inquiry": (["billing", "invoice", "charge", "refund", "overcharge"], 1),
    "cancellation request": (["cancel", "cancellation", "request"], 0),
    "product inquiry": (["product", "purchase", "order", "details", "info"], 1),
    "refund request": (["refund", "money back", "chargeback", "reimbursement"], 1)
}


# Initialize an empty column for predicted category
df["Predicted_Category"] = "other"
df['Predicted_Category'] = df['Predicted_Category'].str.strip().str.lower()
df['Actual_Category'] = df['Actual_Category'].str.strip().str.lower()
print(df[['Ticket ID', 'Actual_Category', 'Predicted_Category']].head(10))



# Keyword-based filtering function with prioritization
def categorize_ticket(description):
    if pd.isna(description):
        return "other"  # Return "other" if description is NaN
    
    description = description.lower().strip()  # Lowercase and strip whitespace for uniformity
    matched_category = "other"
    highest_priority = -1  # Initialize with low priority

    # Check each category and its associated keywords and priority
    for category, (words, priority) in keywords.items():
        # If priority is higher than the highest found, look for keyword matches
        if priority > highest_priority:
            for word in words:
                # Search for keyword using regex for partial matches, word boundaries
                if re.search(r"\b" + re.escape(word) + r"\b", description):
                    matched_category = category
                    highest_priority = priority  # Update priority
                    break  # Stop if a match is found within this priority level
    return matched_category

# Apply categorization function to each ticket description
df["Predicted_Category"] = df["Ticket Description"].apply(categorize_ticket)

# Display a sample of categorized data
print("\nSample Categorized Data:\n", df[["Ticket ID", "Ticket Description", "Predicted_Category"]].head())

# Check if 'Actual_Category' column exists in the dataframe
if 'Actual_Category' in df.columns:
    # Calculate metrics
    precision = precision_score(df['Actual_Category'], df['Predicted_Category'], average='weighted', zero_division=0)
    recall = recall_score(df['Actual_Category'], df['Predicted_Category'], average='weighted', zero_division=0)
    f1 = f1_score(df['Actual_Category'], df['Predicted_Category'], average='weighted', zero_division=0)
    accuracy = accuracy_score(df['Actual_Category'], df['Predicted_Category'])

    # Print results
    print(f"\nPrecision: {precision:.2f}")
    print(f"Recall: {recall:.2f}")
    print(f"F1 Score: {f1:.2f}")
    print(f"Accuracy: {accuracy:.2f}")
else:
    print("The dataset does not contain an 'Actual_Category' column for evaluation.")

# Save the categorized data to a new CSV file
output_path = r"C:\Users\khadi\OneDrive\Desktop\AI Assignment Dataset\customer_support_tickets_categorized.csv"
df.to_csv(output_path, index=False)
print(f"Categorized data saved to {output_path}")
