import streamlit as st
import pandas as pd
from facebook_api import fetch_facebook_posts, extract_comments
from model import load_model, perform_sentiment_analysis, evaluate_model
from visualization import visualize_sentiment_distribution

st.set_page_config(layout="wide")

def main():
    st.title("Social Media Analysis")

    with st.form(key='input_form'):
        page_id = st.text_input("Enter Facebook Page ID:")
        access_token = st.text_input("Enter Facebook Access Token:")
        submit_button = st.form_submit_button(label='Analyze')

    left_column, right_column = st.columns(2)

    if submit_button and page_id and access_token:
        data = fetch_facebook_posts(page_id, access_token)
        comment_texts, post_ids = extract_comments(data)
        model, vectorizer, reverse_label_mapping, X_test, y_test = load_model()
        predicted_sentiments = perform_sentiment_analysis(comment_texts, model, vectorizer, reverse_label_mapping)
        results_df = pd.DataFrame({
            'Post ID': post_ids,
            'Comment': comment_texts,
            'Predicted Sentiment': predicted_sentiments
        })

        y_pred = model.predict(X_test)
        accuracy, f1, precision, recall = evaluate_model(y_test, y_pred)

        with left_column:
            st.write("Sentiment analysis results:")
            st.write(results_df)
            st.download_button(
                "Download Results",
                results_df.to_csv(index=False),
                "facebook_comments_sentiment.csv",
                "text/csv"
            )
            
            st.write("Model Evaluation Metrics:")
            metrics_df = pd.DataFrame({
                'Metric': ['Accuracy', 'F1 Score', 'Precision', 'Recall'],
                'Value': [accuracy, f1, precision, recall]
            })
            st.table(metrics_df)

        with right_column:
            visualize_sentiment_distribution(results_df)
    else:
        st.error("Please enter both page ID and access token.")

if __name__ == "__main__":
    main()
