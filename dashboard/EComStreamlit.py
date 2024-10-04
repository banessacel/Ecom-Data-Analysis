import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import datetime
from sklearn.linear_model import LinearRegression
review_score = pd.read_csv("review_score.csv")
rfm_df = pd.read_csv("rfm_data.csv")

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
