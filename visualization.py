import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st

def visualize_sentiment_distribution(results_df):
    sentiment_counts = results_df['Predicted Sentiment'].value_counts()
    plt.figure(figsize=(8, 6))
    sns.barplot(x=sentiment_counts.index, y=sentiment_counts.values)
    plt.title('Sentiment Distribution')
    plt.xlabel('Sentiment')
    plt.ylabel('Count')
    st.pyplot(plt.gcf())
