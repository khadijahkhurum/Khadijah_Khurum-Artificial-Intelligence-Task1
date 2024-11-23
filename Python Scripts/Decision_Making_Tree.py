import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load the dataset
data_path = r"C:\Users\khadi\OneDrive\Desktop\AI Assignment Dataset\customer_support_tickets.csv"
df = pd.read_csv(data_path)

# Display dataset info
print("Dataset Preview:\n", df.head())
print("\nColumn names in dataset:\n", df.columns)

# Define the target and text columns
target_column = 'Ticket Type' if 'Ticket Type' in df.columns else 'Predicted_Category'
text_column = 'Ticket Description'
print(f"\nUsing target column '{target_column}'.")

# Label distribution
print(f"\nLabel distribution in '{target_column}':\n{df[target_column].value_counts()}")

# Text preprocessing function
def preprocess_text(text):
    import re
    text = text.lower()
    text = re.sub(r'\d+', '', text)
    text = re.sub(r'\b\w{1,2}\b', '', text)
    text = re.sub(r'[^\w\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

# Apply preprocessing
df[text_column] = df[text_column].apply(preprocess_text)

# Rule-based AI logic
def predict_category(description):
    description = description.lower()
    if 'payment' in description or 'invoice' in description:
        return 'Billing'
    elif 'login' in description or 'password' in description:
        return 'Technical Support'
    elif 'delivery' in description or 'shipping' in description:
        return 'Shipping'
    elif 'refund' in description or 'return' in description:
        return 'Returns'
    else:
        return 'General Inquiry'

# Predict categories for the dataset
df['Predicted_Category'] = df[text_column].apply(predict_category)

# Show predictions
print("\nPredicted categories:\n", df[['Predicted_Category']].head())

# Evaluation: Calculate Actual and Predicted Counts
actual_counts = df[target_column].value_counts()
predicted_counts = pd.Series(df['Predicted_Category']).value_counts()

# Ensure both series align on categories
all_categories = actual_counts.index.union(predicted_counts.index)
actual_counts = actual_counts.reindex(all_categories, fill_value=0)
predicted_counts = predicted_counts.reindex(all_categories, fill_value=0)

# Chart 1: Precision of Predicted Categories (Pie Chart)
plt.figure(figsize=(8, 8))
plt.pie(
    predicted_counts,
    labels=predicted_counts.index,
    autopct='%1.1f%%',
    startangle=90,
    colors=sns.color_palette("pastel"),
    wedgeprops={'edgecolor': 'black'}
)
plt.title('Predicted Category Distribution', fontsize=14)
plt.tight_layout()
plt.show()

# Chart 2: Actual vs. Predicted Counts (Bar Chart)
plt.figure(figsize=(10, 6))
bar_width = 0.35
index = range(len(all_categories))

plt.bar(index, actual_counts, bar_width, label='Actual Counts', color='skyblue')
plt.bar([i + bar_width for i in index], predicted_counts, bar_width, label='Predicted Counts', color='salmon')

plt.xticks([i + bar_width / 2 for i in index], all_categories, rotation=45)
plt.xlabel('Categories')
plt.ylabel('Counts')
plt.title('Distribution of Actual vs Predicted Counts by Category')
plt.legend()
plt.tight_layout()
plt.show()

# Sample prediction
new_ticket = "I want to know about my last invoice"
new_ticket_processed = preprocess_text(new_ticket)
predicted_category = predict_category(new_ticket_processed)
print("\nPredicted category for new ticket:", predicted_category)
