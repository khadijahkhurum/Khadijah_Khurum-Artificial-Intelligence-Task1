import pandas as pd
import re
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import seaborn as sns

# Load the dataset
data_path = r"C:\Users\khadi\OneDrive\Desktop\AI Assignment Dataset\customer_support_tickets.csv"
df = pd.read_csv(data_path)

# Rename columns for clarity
df.rename(columns={'Ticket Type': 'Actual_Category'}, inplace=True)

# Check for required columns
if 'Ticket Description' not in df.columns or 'Actual_Category' not in df.columns:
    raise ValueError("The dataset must contain 'Ticket Description' and 'Actual_Category' columns.")

# Preview dataset and show unique categories
print(f"Unique Actual Categories: {df['Actual_Category'].unique()}")
print("Dataset Preview:\n", df.head())

# Expanded keyword dictionary with associated priority levels
keywords = {
    "technical issue": (["issue", "problem", "technical", "error", "support", "not working", "system"], 2),
    "billing inquiry": (["billing", "invoice", "charge", "refund", "overcharge", "payment"], 1),
    "cancellation request": (["cancel", "cancellation", "terminate", "stop subscription", "end service"], 0),
    "product inquiry": (["product", "purchase", "order", "details", "info", "availability"], 1),
    "refund request": (["refund", "money back", "chargeback", "reimbursement", "return payment"], 1)
}

# Initialize columns and normalize
df["Predicted_Category"] = "other"
df['Actual_Category'] = df['Actual_Category'].str.strip().str.lower()

# Keyword-based categorization function
def categorize_ticket(description):
    if pd.isna(description):
        return "other"
    description = description.lower().strip()
    matched_category = "other"
    highest_priority = -1

    for category, (words, priority) in keywords.items():
        if priority > highest_priority:
            for word in words:
                if re.search(r"\b" + re.escape(word) + r"\b", description):
                    matched_category = category
                    highest_priority = priority
                    break
    return matched_category

# Apply the categorization function
df["Predicted_Category"] = df["Ticket Description"].apply(categorize_ticket)

# Display categorized data sample
print("\nSample Categorized Data:\n", df[["Ticket ID", "Ticket Description", "Predicted_Category"]].head())

# Metrics Calculation and Checking for Categories with No Predictions
precision = precision_score(df['Actual_Category'], df['Predicted_Category'], average='weighted', zero_division=0)
recall = recall_score(df['Actual_Category'], df['Predicted_Category'], average='weighted', zero_division=0)
f1 = f1_score(df['Actual_Category'], df['Predicted_Category'], average='weighted', zero_division=0)
accuracy = accuracy_score(df['Actual_Category'], df['Predicted_Category'])

# Print overall metrics
print(f"\nPrecision: {precision * 100:.2f}%")
print(f"Recall: {recall * 100:.2f}%")
print(f"F1 Score: {f1 * 100:.2f}%")
print(f"Accuracy: {accuracy * 100:.2f}%")

# Check if any categories had no predictions
categories = df['Actual_Category'].unique()
no_predictions = [category for category in categories if category not in df['Predicted_Category'].values]
print(f"\nCategories with no predictions: {no_predictions}" if no_predictions else "\nAll categories have predictions.")

# Visualization: Accuracy Comparison
accuracy_data = pd.DataFrame({
    'Algorithm': ['Keyword Filtering'],
    'Accuracy': [accuracy]
})
plt.figure(figsize=(8, 6))
sns.barplot(x='Algorithm', y='Accuracy', data=accuracy_data, palette='viridis', hue='Algorithm', dodge=False, legend=False)
plt.title('Accuracy of Keyword Filtering Algorithm')
plt.xlabel('Algorithm')
plt.ylabel('Accuracy')
plt.ylim(0, 1)
plt.show()

# Confusion Matrix for Keyword Filtering Model
cm = confusion_matrix(df['Actual_Category'], df['Predicted_Category'], labels=categories)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=categories)
disp.plot(cmap='Blues')
plt.title('Confusion Matrix of Keyword Filtering')
plt.show()

# Precision and Recall by Category
precision_per_category = precision_score(df['Actual_Category'], df['Predicted_Category'], average=None, labels=categories, zero_division=0)
recall_per_category = recall_score(df['Actual_Category'], df['Predicted_Category'], average=None, labels=categories, zero_division=0)

plt.figure(figsize=(10, 6))
x = range(len(categories))
plt.bar(x, precision_per_category, width=0.4, label='Precision', color='skyblue', align='center')
plt.bar(x, recall_per_category, width=0.4, label='Recall', color='salmon', align='edge')
plt.xticks(ticks=x, labels=categories, rotation=45)
plt.xlabel('Categories')
plt.ylabel('Scores')
plt.ylim(0, 1)
plt.title('Precision and Recall per Category')
plt.legend()
plt.show()

# Save the categorized data to a new CSV file
output_path = r"C:\Users\khadi\OneDrive\Desktop\AI Assignment Dataset\customer_support_tickets_categorized.csv"
df.to_csv(output_path, index=False)
print(f"Categorized data saved to {output_path}")
