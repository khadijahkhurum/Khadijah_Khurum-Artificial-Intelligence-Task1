Customer Support Classification using Rule-Based AI:

This project implements two rule-based AI algorithms for classifying customer support tickets. The goal is to compare their performance and assess their suitability for solving the problem of automatic customer support categorization. This README provides setup instructions, usage guidelines, and an overview of the project structure.

Project Structure
Keyword Filtering.py: The main Python script that processes the CSV data, applies keyword-based filtering, and categorizes each ticket based on its content.
customer_support_tickets.csv: The input dataset containing customer support tickets. This dataset includes columns like Ticket Description, Customer Name, Customer Age, and more.
customer_support_tickets_categorized.csv: The output dataset with an additional Predicted_Category column containing the predicted category for each ticket.
Decision_Making.py: The Python script that processes the csv data, applies decision making tree algorithm. 
   
Research Goal
To evaluate and compare the effectiveness of Keyword Filtering and Decision Tree approaches in automatically classifying customer support tickets based on categories such as "Billing," "Technical," "Account," etc.

Key Features
1. Keyword Filtering Algorithm:
   A rule-based approach for classifying tickets. Specific keywords are associated with categories, and a ticket is categorized if it contains keywords matching a category.
2. Customization:
   You can easily customize the keywords to fit different ticket categories or add new categories based on project needs.
3. Dataset Preview and Sample Output:
   The script previews the dataset and sample categorized data for user validation before saving.

   
