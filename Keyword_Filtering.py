# %%
import pandas as pd
import re

# Load the dataset
data_path = r"C:\Users\khadi\OneDrive\Desktop\AI Assignment Dataset\customer_support_tickets.csv"
df = pd.read_csv(data_path)

# Check the first few rows of the dataset
print("Dataset Preview:\n", df.head())

# Define a keyword dictionary with priority
# Higher numbers indicate higher priority for category matching
keywords = {
    "complaint": (["issue", "problem", "not working", "faulty", "complaint"], 2),
    "billing": (["invoice", "billing", "charge", "overcharge", "refund"], 1),
    "technical_support": (["error", "bug", "technical", "support"], 2),
    "general_inquiry": (["info", "information", "details", "query"], 0)
}

# Initialize an empty column for predicted category
df["Predicted_Category"] = "other"

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

# Save the categorized data to a new CSV file
output_path = r"C:\Users\khadi\OneDrive\Desktop\AI Assignment Dataset\customer_support_tickets_categorized.csv"
df.to_csv(output_path, index=False)
print(f"Categorized data saved to {output_path}")
# %%
