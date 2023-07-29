import pandas as pd
import ast
from collections import defaultdict
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

DATA = pd.read_csv("user_informations.csv")

DATA['liked_fields'] = DATA['liked_fields'].apply(ast.literal_eval)
DATA['liked_articles'] = DATA['liked_articles'].apply(ast.literal_eval)

# Convert time_consumption to a dictionary of user-article-time mappings
DATA['time_consumption_on_the_article'] = DATA.apply(lambda row: {
    row['liked_articles'][i]: row['time_consumption_on_the_article'][i]
    for i in range(len(row['liked_articles']))
}, axis=1)

# user-article interaction matrix for time consumption
user_article_time_matrix = defaultdict(lambda: defaultdict(int))
for _, row in DATA.iterrows():
    user_id = row['id']
    for article, time_consumption in row['time_consumption_on_the_article'].items():
        user_article_time_matrix[user_id][article] = time_consumption

#cosine similarity between users based on time consumption
user_ids = list(user_article_time_matrix.keys())
user_time_matrix = np.array([list(user_article_time_matrix[user].values()) for user in user_ids])
user_similarity_matrix = cosine_similarity(user_time_matrix)

#dictionary of user similarities for faster lookup
user_similarities = {user_ids[i]: user_similarity_matrix[i] for i in range(len(user_ids))}

def get_user_data(user_id):
    user_data = DATA.loc[DATA['id'] == user_id]
    liked_fields = user_data['liked_fields'].values[0]
    liked_articles = user_data['liked_articles'].values[0]
    return liked_fields, liked_articles

def get_recommendations(user_id):
    liked_fields, liked_articles = get_user_data(user_id)
    user_recommendations = defaultdict(float)

    for other_user_id, similarity_score in sorted(
        user_similarities[user_id].items(), key=lambda x: x[1], reverse=True
    ):
        if other_user_id != user_id:
            other_liked_articles = DATA.loc[DATA['id'] == other_user_id, 'liked_articles'].values[0]
            for article in other_liked_articles:
                if article not in liked_articles:
                    user_recommendations[article] += similarity_score

    # Sort recommendations depend on scores
    sorted_recommendations = sorted(user_recommendations.items(), key=lambda x: x[1], reverse=True)
    return sorted_recommendations
################################################################################################################
def update_recommendation_system(new_data):
    global DATA, user_article_time_matrix, user_similarities
    # NEW DATA with same .csv format
    DATA = pd.concat([DATA, new_data], ignore_index=True)

    # Update user-article interaction matrix for time consumption(again)
    for _, row in new_data.iterrows():
        user_id = row['id']
        for article, time_consumption in row['time_consumption_on_the_article'].items():
            user_article_time_matrix[user_id][article] = time_consumption

    #cosine similarity for new users based on time consumption(again)
    user_ids = list(user_article_time_matrix.keys())
    user_time_matrix = np.array([list(user_article_time_matrix[user].values()) for user in user_ids])
    user_similarity_matrix = cosine_similarity(user_time_matrix)

    # Updating user similarities
    user_similarities = {user_ids[i]: user_similarity_matrix[i] for i in range(len(user_ids))}


if __name__ == "__main__":
    #for user id=1 
    user_id_to_recommend = 1
    recommendations = get_recommendations(user_id_to_recommend)
    print("Recommendations for user {}:".format(user_id_to_recommend))
    for rec_article, score in recommendations[:5]:  # Show top 5 recommendations
        print("- Article: {}, Score: {}".format(rec_article, score))
