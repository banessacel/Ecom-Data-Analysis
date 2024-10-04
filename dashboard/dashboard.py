import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import datetime
from sklearn.linear_model import LinearRegression
review_score = pd.read_csv("dashboard/review_score.csv")
rfm_df = pd.read_csv("dashboard/rfm_data.csv")

# Convert 'review_year_day' to datetime
review_score['review_year_day'] = pd.to_datetime(review_score['review_year_day'])

# Streamlit app layout
st.title("E-Commerce Review Analysis")

# Sidebar for date range selection
st.sidebar.header("Average Review Score")
min_date = review_score['review_year_day'].min().date()
max_date = review_score['review_year_day'].max().date()

# Single date selection
selected_date = st.sidebar.date_input(
    "Select a date:",
    min_value=min_date,
    max_value=max_date
)

# Filter data for the selected date
filtered_data = review_score[review_score['review_year_day'] == pd.to_datetime(selected_date)]

# Rename columns for better presentation
filtered_data.rename(columns={
    'review_year_day': 'Review Date',
    'avg_review_score': 'Average Review Score'
}, inplace=True)

# Display results
if not filtered_data.empty:
    st.sidebar.write(f"Average Review Scores for {selected_date}:")
    # Display only the relevant columns
    st.sidebar.dataframe(filtered_data[['Review Date', 'Average Review Score']].reset_index(drop=True))
else:
    st.sidebar.write("No data available for the selected date.")

st.header("Question 1: How is the correlation between the frequency of reviews and the average review score for sellers?")

# Define custom bins for Frequency
frequency_bins = [0, 50, 100, 200, 500, float('inf')]
frequency_labels = ['0-50', '51-100', '101-200', '201-500', '500+']
rfm_df['F_Segment'] = pd.cut(rfm_df['Frequency'], bins=frequency_bins, labels=frequency_labels)

# Define custom bins for Monetary (Review Score)
monetary_bins = [0, 1, 2, 3, 4, 5]
monetary_labels = ['0-1', '1-2', '2-3', '3-4', '4-5']
rfm_df['M_Segment'] = pd.cut(rfm_df['Monetary'], bins=monetary_bins, labels=monetary_labels)

# Frequency Segmentation Plot
fig_freq, ax_freq = plt.subplots(figsize=(10, 6))
sns.countplot(data=rfm_df, x='F_Segment', palette='viridis', ax=ax_freq)
ax_freq.set_title('Frequency Segmentation')
ax_freq.set_xlabel('Frequency Segments')
ax_freq.set_ylabel('Number of Sellers')
ax_freq.tick_params(axis='x', rotation=45)
st.pyplot(fig_freq)

# Review Score Segmentation Plot
fig_monetary, ax_monetary = plt.subplots(figsize=(10, 6))
sns.countplot(data=rfm_df, x='M_Segment', palette='viridis', ax=ax_monetary)
ax_monetary.set_title('Review Score Segmentation')
ax_monetary.set_xlabel('Review Score Segments')
ax_monetary.set_ylabel('Number of Sellers')
ax_monetary.tick_params(axis='x', rotation=45)
st.pyplot(fig_monetary)


X = np.array(range(len(review_score))).reshape(-1, 1)  # Day numbers as X
y = review_score['avg_review_score'].values  # Average review scores as y

# Create model
model = LinearRegression()
model.fit(X, y)
review_score['trend_line'] = model.predict(X)

# Visualization
plt.figure(figsize=(12, 6))
sns.lineplot(x='review_year_day', y='avg_review_score', data=review_score, marker='o', label='Average Review Score', color='blue')
sns.lineplot(x='review_year_day', y='trend_line', data=review_score, label='Trend Line', color='green', linestyle='-')

# Average score fill
plt.fill_between(review_score['review_year_day'],
                 review_score['avg_review_score'],
                 color='blue', alpha=0.1)

# Plot customization
plt.title('Average Review Score per Day', fontsize=14)
plt.xlabel('Time', fontsize=12)
plt.ylabel('Average Review Score', fontsize=12)
plt.xticks(rotation=45)  # Rotate x-ticks for better readability
plt.grid(True, linestyle='--', alpha=0.5)
plt.legend()

# Display the plot in Streamlit
st.pyplot(plt)

st.markdown("""
The average review score shows no correlation with the number of sales made by the seller. This observation implies that consumers in the e-commerce sector are not significantly influenced by the sales figures of a seller when deciding which products to purchase. The majority of the frequency of selling is clustered on the left side of the graph, indicating that most sellers have a low frequency of sales, specifically between 0 and 50 sales. As a result, sellers seeking to enhance their sales performance should focus primarily on analyzing their review scores, particularly those with lower ratings. The majority of review scores fall within the range of 4 to 5, suggesting that while customer experiences are generally positive, there remains a significant number of sellers who are not reaching their full sales potential. By identifying areas where customer feedback indicates dissatisfaction, sellers can implement targeted improvements in their services, product quality, or customer interactions. This approach not only helps in addressing the issues highlighted in low reviews but also fosters customer loyalty and can lead to an increase in sales over time.""")


st.header("Question 2: How is the performance of review scores as time series data?")

# Visualization: Scatter Plot with Linear Regression Line
plt.figure(figsize=(10, 6))
sns.scatterplot(data=rfm_df, x='Frequency', y='Monetary', hue='RFM_Score', palette='viridis', size='Frequency', sizes=(20, 200))
sns.regplot(data=rfm_df, x='Frequency', y='Monetary', scatter=False, color='red', line_kws={'label': 'Linear Fit'})

# Adding titles and labels
plt.title('Frequency vs. Monetary Score with Linear Regression Line')
plt.xlabel('Frequency of Reviews')
plt.ylabel('Average Review Score')
plt.legend(title='RFM Score', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()

# Display the plot in Streamlit
st.pyplot(plt)
st.markdown("""
- **(2)** Overall, customer satisfaction within the e-commerce landscape is witnessing a steady improvement each day. This trend suggests that e-commerce platforms are actively working to enhance their services year after year, contributing to a higher level of customer satisfaction. The insights drawn from seller review scores are vital in this context, as they serve as a direct indicator of consumer perceptions and experiences. Therefore, it is crucial for sellers to take note of those who consistently receive low review ratings. By doing so, they can gain valuable insights into the specific aspects of their offerings or services that may require enhancement. Ultimately, focusing on improving customer experience based on feedback will not only elevate individual seller performance but also strengthen the overall reputation of the e-commerce marketplace.
""")
