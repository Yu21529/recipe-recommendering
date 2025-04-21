import streamlit as st
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import MinMaxScaler
from surprise import Dataset, Reader, SVD
from surprise.model_selection import train_test_split

# Load data
data = pd.read_csv("cleaned_data.csv")
final_features = np.load("final_features.npy")
similarity_matrix = cosine_similarity(final_features)
similarity_matrix = MinMaxScaler().fit_transform(similarity_matrix)

# Normalize feedback data
data[['clicks_on_recipe', 'time_spent_on_recipe']] = data[['clicks_on_recipe', 'time_spent_on_recipe']].fillna(0)
feedback_norm = MinMaxScaler().fit_transform(data[['clicks_on_recipe', 'time_spent_on_recipe']])
data['feedback_score'] = feedback_norm.mean(axis=1)

# Collaborative Filtering
reader = Reader(rating_scale=(0, 5))
ratings = data[['AuthorId', 'RecipeId', 'Rating']].dropna()
ratings.columns = ['userId', 'recipeId', 'rating']
dataset = Dataset.load_from_df(ratings, reader)
trainset, _ = train_test_split(dataset, test_size=0.2)
model = SVD()
model.fit(trainset)

# Streamlit UI
st.set_page_config(page_title="Smart Recipe Recommender", layout="centered")
st.title("🍲 Smart Recipe Recommender")

user_id = st.number_input("Enter User ID:", min_value=1, step=1)
veg = st.checkbox("Vegetarian")
nonveg = st.checkbox("Non-Vegetarian")
sub_pref = ""
if veg:
    sub_pref = st.selectbox("Veg Type", ["none", "paneer", "tofu", "mushroom"])
elif nonveg:
    sub_pref = st.selectbox("Non-Veg Type", ["none", "chicken", "beef", "mutton", "fish"])

low_sugar = st.checkbox("Diabetic-friendly (Low Sugar)")
low_fat = st.checkbox("Low Fat")
low_sodium = st.checkbox("Low Sodium")
max_time = st.slider("Max Cook Time (minutes):", 5, 120, 30)

if st.button("🔍 Get Recommendations"):
    user_recipes = ratings[ratings.userId == user_id].recipeId.tolist()
    candidates = data[~data["RecipeId"].isin(user_recipes)]
    candidates = candidates[candidates["TotalTime1"] <= max_time]

    if low_sugar:
        candidates = candidates[candidates["SugarContent"] <= 10]
    if low_fat:
        candidates = candidates[candidates["FatContent"] <= 10]
    if low_sodium:
        candidates = candidates[candidates["SodiumContent"] <= 140]
    if veg:
        candidates = candidates[~candidates["Keywords_clean"].str.contains("meat|chicken|beef|mutton|fish", na=False, case=False)]
        if sub_pref != "none":
            candidates = candidates[candidates["Keywords_clean"].str.contains(sub_pref, na=False, case=False)]
    if nonveg:
        candidates = candidates[candidates["Keywords_clean"].str.contains("meat|chicken|beef|mutton|fish", na=False, case=False)]
        if sub_pref != "none":
            candidates = candidates[candidates["Keywords_clean"].str.contains(sub_pref, na=False, case=False)]

    recs = []
    for _, row in candidates.iterrows():
        idx = data[data["RecipeId"] == row["RecipeId"]].index[0]
        try:
            pred = model.predict(user_id, row["RecipeId"]).est
        except:
            pred = 3.5
        rating_score = (pred - 1) / 4
        content_score = similarity_matrix[idx][idx]
        feedback_score = row['feedback_score']
        final_score = 0.4 * rating_score + 0.3 * content_score + 0.3 * feedback_score
        recs.append({
            "name": row["Name"],
            "score": final_score,
            "ingredients": row["RecipeIngredientParts"],
            "instructions": row["RecipeInstructions"]
        })
    recs = sorted(recs, key=lambda x: x['score'], reverse=True)[:5]
    st.subheader("📌 Top Personalized Recommendations")
    for i, r in enumerate(recs, 1):
        st.markdown(f"**{i}. {r['name']} — Score: {r['score']:.2f}**")
        st.markdown(f"*Ingredients:* {r['ingredients']}")
        st.markdown(f"*Instructions:* {r['instructions']}")
        st.markdown("---")
