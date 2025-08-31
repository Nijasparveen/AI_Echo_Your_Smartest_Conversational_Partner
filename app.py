import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from sentence_transformers import SentenceTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score

st.set_page_config(page_title="AI Echo - Sentiment Dashboard", layout="wide")

st.title("🤖 AI Echo: Sentiment Analysis Dashboard")
st.write("Analyze ChatGPT user reviews and predict sentiments.")

# ---------------------------
# Load Dataset
# ---------------------------
df = pd.read_csv("https://docs.google.com/spreadsheets/d/1eyPDJj8ttd8t-o6JVT4txCbvJ9DtcF-U/export?format=csv&gid=1201624046")

# Ensure clean reviews exist
if "clean_review" not in df.columns:
    df["clean_review"] = df["review"].astype(str).str.lower()

# ---------------------------
# Sentiment Labels
# ---------------------------
def map_sentiment(rating):
    if rating <= 2:
        return "Negative"
    elif rating == 3:
        return "Neutral"
    else:
        return "Positive"

df["sentiment"] = df["rating"].apply(map_sentiment)

# ---------------------------
# Train a Simple Model (BERT embeddings + LR)
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

# Evaluate quick accuracy
y_pred = clf.predict(X_test_emb)
acc = accuracy_score(y_test, y_pred)

# ---------------------------
# Dashboard Sections
# ---------------------------

st.subheader("📊 Sentiment Distribution")
fig, ax = plt.subplots()
sns.countplot(x="sentiment", data=df, palette="Set2", ax=ax)
st.pyplot(fig)

st.markdown(f"**Model Accuracy (validation):** {acc:.2f}")

# ---------------------------
# Word Clouds
# ---------------------------
st.subheader("💬 Word Clouds")
pos_text = " ".join(df[df["sentiment"]=="Positive"]["clean_review"])
neg_text = " ".join(df[df["sentiment"]=="Negative"]["clean_review"])

wc_pos = WordCloud(width=400, height=300, background_color="white").generate(pos_text)
wc_neg = WordCloud(width=400, height=300, background_color="white").generate(neg_text)

col1, col2 = st.columns(2)
with col1:
    st.image(wc_pos.to_array(), caption="Positive Reviews")
with col2:
    st.image(wc_neg.to_array(), caption="Negative Reviews")

# ---------------------------
# Insights & Recommendations
# ---------------------------
st.subheader("💡 Insights & Recommendations")
st.write("""
- **Most reviews are positive**, but negative ones highlight accuracy and outdated info issues.  
- **Mobile reviews** seem slightly lower rated than Web → product team should improve mobile UX.  
- **Verified users** are generally happier than free users → suggests premium features are valued.  
- **Recent versions** show better ratings → updates are improving user satisfaction.  
""")

# ---------------------------
# Predict Sentiment for User Input
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
