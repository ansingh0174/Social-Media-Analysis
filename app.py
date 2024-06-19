import streamlit as st
import requests
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from emoji import demojize
import matplotlib.pyplot as plt
import seaborn as sns

st.set_page_config(layout="wide")

kaggle_df = pd.read_csv('sentiment.csv', encoding='latin1')
kaggle_df = kaggle_df[['text', 'sentiment']].dropna()
label_mapping = {'positive': 1, 'negative': 0, 'neutral': 2}
reverse_label_mapping = {1: 'positive', 0: 'negative', 2: 'neutral'}
kaggle_df['sentiment'] = kaggle_df['sentiment'].map(label_mapping)
kaggle_df = kaggle_df.dropna()

vectorizer = TfidfVectorizer(max_features=1000)
X = vectorizer.fit_transform(kaggle_df['text'])
y = kaggle_df['sentiment']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

def fetch_facebook_posts(page_id, access_token):
    url = f'https://graph.facebook.com/v11.0/{page_id}/posts'
    params = {
        'access_token': access_token,
        'fields': 'id,created_time,message,comments{message}'
    }
    response = requests.get(url, params=params)
    data = response.json()
    return data

def extract_comments(data):
    comment_texts = []
    post_ids = []
    for post in data['data']:
        post_id = post['id']
        if 'comments' in post:
            for comment in post['comments']['data']:
                comment_message = comment['message']
                comment_texts.append(demojize(comment_message))
                post_ids.append(post_id)
    return comment_texts, post_ids

def perform_sentiment_analysis(comment_texts):
    X_new = vectorizer.transform(comment_texts)
    predicted_sentiments = model.predict(X_new)
    return [reverse_label_mapping[sentiment] for sentiment in predicted_sentiments]

def evaluate_model(y_test, y_pred):
    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average='weighted')
    precision = precision_score(y_test, y_pred, average='weighted')
    recall = recall_score(y_test, y_pred, average='weighted')
    return accuracy, f1, precision, recall

def visualize_sentiment_distribution(results_df):
    sentiment_counts = results_df['Predicted Sentiment'].value_counts()
    plt.figure(figsize=(8, 6))
    sns.barplot(x=sentiment_counts.index, y=sentiment_counts.values)
    plt.title('Sentiment Distribution')
    plt.xlabel('Sentiment')
    plt.ylabel('Count')
    st.pyplot(plt.gcf())

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
        predicted_sentiments = perform_sentiment_analysis(comment_texts)
        results_df = pd.DataFrame({
            'Post ID': post_ids,
            'Comment': comment_texts,
            'Predicted Sentiment': predicted_sentiments
        })

        with left_column:
            st.write("Sentiment analysis results:")
            st.write(results_df)
            st.download_button(
                "Download Results",
                results_df.to_csv(index=False),
                "facebook_comments_sentiment.csv",
                "text/csv"
            )

        with right_column:
            visualize_sentiment_distribution(results_df)
    else:
        st.error("Please enter both page ID and access token.")

if __name__ == "__main__":
    main()
