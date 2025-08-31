import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from collections import Counter
import nltk
from nltk.corpus import stopwords
from sentence_transformers import SentenceTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
import re

# ---------------------------
# Config & Setup
# ---------------------------
nltk.download("stopwords")
stop_words = set(stopwords.words("english"))
st.set_page_config(page_title="AI Echo - Sentiment Dashboard", layout="wide")
st.title("🤖 AI Echo: Sentiment Analysis Dashboard")
st.write("Analyze ChatGPT user reviews, explore insights, and predict sentiment in real time.")

# ---------------------------
# Load Data
# ---------------------------
df = pd.read_csv("chatgpt_style_reviews_dataset.xlsx - Sheet1.csv")

if "clean_review" not in df.columns:
    df["clean_review"] = df["review"].astype(str).str.lower()

# Sentiment labels from ratings
def map_sentiment(rating):
    if rating <= 2: return "Negative"
    elif rating == 3: return "Neutral"
    else: return "Positive"

df["sentiment"] = df["rating"].apply(map_sentiment)
df["date"] = pd.to_datetime(df["date"])
df["review_length"] = df["clean_review"].apply(lambda x: len(str(x).split()))

# ---------------------------
# Train Sentiment Classifier (BERT embeddings + Logistic Regression)
# ---------------------------
le = LabelEncoder()
df["label"] = le.fit_transform(df["sentiment"])

X_train, X_test, y_train, y_test = train_test_split(
    df["clean_review"], df["label"], test_size=0.2, random_state=42, stratify=df["label"]
)

model_st = SentenceTransformer("all-MiniLM-L6-v2")
X_train_emb = model_st.encode(X_train.tolist())
X_test_emb = model_st.encode(X_test.tolist())

clf = LogisticRegression(max_iter=200)
clf.fit(X_train_emb, y_train)

# Evaluate model
y_pred = clf.predict(X_test_emb)
acc = accuracy_score(y_test, y_pred)

# ---------------------------
# Dashboard Sections
# ---------------------------

# Q1: Overall Sentiment
st.subheader("1️⃣ Overall Sentiment of User Reviews")
fig, ax = plt.subplots()
sns.countplot(x="sentiment", data=df, palette="Set2", ax=ax)
st.pyplot(fig)

# Q2: Sentiment vs Rating
st.subheader("2️⃣ Sentiment vs Rating")
fig, ax = plt.subplots()
sns.countplot(x="rating", hue="sentiment", data=df, palette="Set1", ax=ax)
st.pyplot(fig)

# Q3: Keywords by Sentiment (Word Clouds)
st.subheader("3️⃣ Common Keywords per Sentiment")
col1, col2, col3 = st.columns(3)
for sentiment, col in zip(["Positive", "Neutral", "Negative"], [col1,col2,col3]):
    with col:
        st.markdown(f"**{sentiment} Reviews**")
        text = df[df["sentiment"]==sentiment]["clean_review"]
        wc = WordCloud(width=400, height=300, background_color="white").generate(" ".join(text))
        st.image(wc.to_array())

# Q4: Sentiment Trend Over Time
st.subheader("4️⃣ Sentiment Over Time")
fig, ax = plt.subplots()
df.groupby(df["date"].dt.to_period("M"))["rating"].mean().plot(ax=ax, marker="o")
plt.ylabel("Average Rating")
st.pyplot(fig)

# Q5: Verified vs Non-Verified
st.subheader("5️⃣ Verified Users vs Non-Verified")
fig, ax = plt.subplots()
sns.countplot(x="verified_purchase", hue="sentiment", data=df, palette="muted", ax=ax)
st.pyplot(fig)

# Q6: Review Length vs Sentiment
st.subheader("6️⃣ Review Length vs Sentiment")
fig, ax = plt.subplots()
sns.boxplot(x="sentiment", y="review_length", data=df, palette="pastel", ax=ax)
st.pyplot(fig)

# Q7: Sentiment by Location
st.subheader("7️⃣ Sentiment by Location")
fig, ax = plt.subplots(figsize=(10,5))
sns.countplot(x="location", hue="sentiment", data=df, ax=ax)
plt.xticks(rotation=90)
st.pyplot(fig)

# Q8: Platform Differences
st.subheader("8️⃣ Sentiment by Platform (Web vs Mobile)")
fig, ax = plt.subplots()
sns.countplot(x="platform", hue="sentiment", data=df, palette="coolwarm", ax=ax)
st.pyplot(fig)

# Q9: Sentiment by Version
st.subheader("9️⃣ Sentiment by ChatGPT Version")
fig, ax = plt.subplots()
sns.barplot(x="version", y="rating", data=df, ci=None, palette="mako", ax=ax)
plt.xticks(rotation=45)
st.pyplot(fig)

# Q10: Common Negative Themes
st.subheader("🔟 Common Themes in Negative Reviews")
neg_texts = " ".join(df[df["sentiment"]=="Negative"]["clean_review"])
wc_neg = WordCloud(width=800, height=400, background_color="white").generate(neg_texts)
st.image(wc_neg.to_array(), caption="Negative Feedback WordCloud")

# ---------------------------
# Insights & Recommendations
# ---------------------------
st.subheader("💡 Insights & Recommendations")
st.write(f"""
- **Model Validation Accuracy:** {acc:.2f}  
- Most reviews are **positive**, but negatives often mention accuracy & outdated info.  
- **Mobile reviews** trend slightly lower than Web → improve mobile UX.  
- **Verified users** give higher ratings → premium users are more satisfied.  
- **Recent versions** generally improved satisfaction → updates are working.  
""")

# ---------------------------
# Predict Sentiment (Interactive)
# ---------------------------
st.subheader("🧪 Try It Yourself: Predict Sentiment")
user_review = st.text_area("Enter a review text here:")

if st.button("Predict Sentiment"):
    if user_review.strip():
        user_emb = model_st.encode([user_review])
        pred = clf.predict(user_emb)
        sentiment = le.inverse_transform(pred)[0]
        st.success(f"Predicted Sentiment: **{sentiment}**")
    else:
        st.warning("Please enter some text to predict sentiment.")
