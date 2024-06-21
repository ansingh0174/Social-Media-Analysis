import requests
from emoji import demojize

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
