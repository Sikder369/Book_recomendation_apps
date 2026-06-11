import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# =====================================================
# PAGE SETUP
# =====================================================

st.set_page_config(
    page_title="BookWise Recommender",
    page_icon="📚",
    layout="wide"
)

# =====================================================
# CUSTOM CSS
# =====================================================

st.markdown("""
<style>
.stApp {
    background-color: #FFF8E7;
}

/* Main heading box */
.title-box {
    background: linear-gradient(90deg, #FF9933, #FFB347);
    padding: 32px;
    border-radius: 22px;
    color: white;
    text-align: center;
    box-shadow: 0px 5px 20px rgba(0,0,0,0.18);
    margin-bottom: 25px;
}

/* Section heading box */
.section-box {
    background-color: #1E3A8A;
    color: white;
    padding: 12px 18px;
    border-radius: 14px;
    font-size: 22px;
    font-weight: bold;
    margin-bottom: 15px;
}

/* Sidebar */
[data-testid="stSidebar"] {
    background-color: #FF9933;
}

[data-testid="stSidebar"] h1,
[data-testid="stSidebar"] h2,
[data-testid="stSidebar"] h3,
[data-testid="stSidebar"] label,
[data-testid="stSidebar"] p {
    color: white !important;
}

/* Keep dropdown text visible */
[data-testid="stSidebar"] input,
[data-testid="stSidebar"] textarea,
[data-testid="stSidebar"] div[data-baseweb="select"] * {
    color: black !important;
}

/* Metric cards */
.metric-card {
    background: white;
    border-radius: 18px;
    padding: 20px;
    text-align: center;
    box-shadow: 0px 5px 15px rgba(0,0,0,0.12);
}

/* Recommendation card - old better style */
.book-card {
    background-color: white;
    padding: 18px;
    border-radius: 15px;
    margin-bottom: 15px;
    border-left: 7px solid #6C63FF;
    box-shadow: 0px 4px 12px rgba(0,0,0,0.08);
}

.book-title {
    font-size: 20px;
    font-weight: bold;
    color: #3B3663;
}

.small-text {
    color: #555;
    font-size: 14px;
    line-height: 1.7;
}

.stButton > button {
    background-color: #FF9933;
    color: white;
    border-radius: 12px;
    border: none;
    font-weight: bold;
}

.stButton > button:hover {
    background-color: #E67E22;
    color: white;
}
</style>
""", unsafe_allow_html=True)

# =====================================================
# HEADER
# =====================================================

st.markdown("""
<div class="title-box">
    <h1>📚 BookWise Recommendation System</h1>
    <h4>Discover your next favorite book using smart content-based recommendations</h4>
</div>
""", unsafe_allow_html=True)

st.info("""
👋 Welcome to **BookWise**!

Use the Sidebar to filter books by **source, category, rating, or title**.  
Then select a book and get similar recommendations .
""")

# =====================================================
# LOAD DATA
# =====================================================

df = pd.read_csv("final_book_recommender_dataset.csv")
df = df.fillna("")

required_columns = [
    "title",
    "authors",
    "subjects",
    "category",
    "rating",
    "ratings_count",
    "price",
    "year",
    "source",
    "content"
]

for col in required_columns:
    if col not in df.columns:
        df[col] = ""

# Convert numeric columns safely
df["rating_numeric"] = pd.to_numeric(
    df["rating"],
    errors="coerce"
)

df["ratings_count_numeric"] = pd.to_numeric(
    df["ratings_count"],
    errors="coerce"
)

# Create content column again for safety
df["content"] = (
    df["title"].astype(str) + " " +
    df["authors"].astype(str) + " " +
    df["subjects"].astype(str) + " " +
    df["category"].astype(str)
)

# =====================================================
# DASHBOARD METRICS
# =====================================================

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.markdown(f"""
    <div class="metric-card">
        <h2>{len(df)}</h2>
        <p>📚 Total Books</p>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown(f"""
    <div class="metric-card">
        <h2>{df["category"].nunique()}</h2>
        <p>🏷 Categories</p>
    </div>
    """, unsafe_allow_html=True)

with col3:
    st.markdown(f"""
    <div class="metric-card">
        <h2>{df["authors"].nunique()}</h2>
        <p>✍ Authors</p>
    </div>
    """, unsafe_allow_html=True)

with col4:
    avg_rating = df["rating_numeric"].mean()

    if pd.isna(avg_rating):
        avg_rating = "N/A"
    else:
        avg_rating = round(avg_rating, 2)

    st.markdown(f"""
    <div class="metric-card">
        <h2>{avg_rating}</h2>
        <p>⭐ Average Rating</p>
    </div>
    """, unsafe_allow_html=True)

st.markdown("---")

# =====================================================
# SIDEBAR FILTERS
# =====================================================

st.sidebar.title("🔎 Filter Dashboard")

source_options = ["All"] + sorted(
    [src for src in df["source"].unique().tolist() if src != ""]
)

selected_source = st.sidebar.selectbox(
    "📌 Select Source",
    source_options
)

category_options = ["All"] + sorted(
    [cat for cat in df["category"].unique().tolist() if cat != ""]
)

selected_category = st.sidebar.selectbox(
    "🏷 Select Category",
    category_options
)

min_rating = st.sidebar.slider(
    "⭐ Minimum Rating",
    min_value=0.0,
    max_value=5.0,
    value=0.0,
    step=0.5
)

title_search = st.sidebar.text_input(
    "🔍 Search by Book Title"
)

# =====================================================
# APPLY FILTERS
# =====================================================

filtered_df = df.copy()

if selected_source != "All":
    filtered_df = filtered_df[
        filtered_df["source"] == selected_source
    ]

if selected_category != "All":
    filtered_df = filtered_df[
        filtered_df["category"] == selected_category
    ]

if min_rating > 0:
    filtered_df = filtered_df[
        filtered_df["rating_numeric"].fillna(0) >= min_rating
    ]

if title_search:
    filtered_df = filtered_df[
        filtered_df["title"].str.contains(
            title_search,
            case=False,
            na=False
        )
    ]

# =====================================================
# BUILD RECOMMENDATION MODEL
# =====================================================

tfidf = TfidfVectorizer(
    stop_words="english",
    max_features=5000
)

tfidf_matrix = tfidf.fit_transform(df["content"])

cosine_sim = cosine_similarity(
    tfidf_matrix,
    tfidf_matrix
)

def recommend_books(book_title, filtered_data, num_recommendations=5):
    matches = df[df["title"] == book_title]

    if matches.empty:
        return pd.DataFrame()

    index = matches.index[0]

    scores = list(enumerate(cosine_sim[index]))

    scores = sorted(
        scores,
        key=lambda x: x[1],
        reverse=True
    )

    recommended_indices = [i[0] for i in scores[1:]]

    recommendations = df.iloc[recommended_indices]

    # Apply current filters to recommended results
    recommendations = recommendations[
        recommendations.index.isin(filtered_data.index)
    ]

    return recommendations.head(num_recommendations)

# =====================================================
# MAIN LAYOUT
# =====================================================

left_col, right_col = st.columns([1.25, 1])

# -------------------------------
# LEFT SIDE: FILTERED BOOKS
# -------------------------------

with left_col:
    st.subheader("📊 Filtered Book Collection")

    st.write(f"Showing **{len(filtered_df)}** books based on your filters.")

    st.dataframe(
        filtered_df[[
            "title",
            "authors",
            "category",
            "rating",
            "ratings_count",
            "year",
            "source"
        ]],
        use_container_width=True,
        height=450
    )

# -------------------------------
# RIGHT SIDE: RECOMMENDER
# -------------------------------

with right_col:
    st.subheader("🤖 Get Book Recommendations")

    if len(filtered_df) == 0:
        st.warning("No books found. Please change your filters.")
    else:
        selected_book = st.selectbox(
            "📖 Choose a book",
            sorted(filtered_df["title"].unique())
        )

        num_recommendations = st.slider(
            "Number of recommendations",
            min_value=1,
            max_value=10,
            value=5
        )

        recommend_button = st.button(
            "✨ Recommend Books",
            use_container_width=True
        )

        if recommend_button:
            results = recommend_books(
                selected_book,
                filtered_df,
                num_recommendations
            )

            if results.empty:
                st.warning("No similar books found with the current filters.")
            else:
                st.success("Here are your recommended books:")

                for _, row in results.iterrows():

                    rating = row["rating"] if row["rating"] != "" else "Not Available"
                    ratings_count = row["ratings_count"] if row["ratings_count"] != "" else "Not Available"
                    year = row["year"] if row["year"] != "" else "Not Available"
                    author = row["authors"] if row["authors"] != "" else "Unknown"
                    category = row["category"] if row["category"] != "" else "Unknown"

                    st.markdown(f"""
                    <div class="book-card">
                        <div class="book-title">📚 {row["title"]}</div>
                        <div class="small-text">✍ <b>Author:</b> {author}</div>
                        <div class="small-text">🏷 <b>Category:</b> {category}</div>
                        <div class="small-text">⭐ <b>Rating:</b> {rating}</div>
                        <div class="small-text">📊 <b>Ratings Count:</b> {ratings_count}</div>
                        <div class="small-text">📅 <b>Year:</b> {year}</div>
                        <div class="small-text">🔗 <b>Source:</b> {row["source"]}</div>
                    </div>
                    """, unsafe_allow_html=True)

# =====================================================
# EXTRA SUMMARY SECTION
# =====================================================

st.markdown("---")

st.subheader("📌 Dataset Summary")

summary_col1, summary_col2 = st.columns(2)

with summary_col1:
    st.write("### Books by Source")
    st.bar_chart(df["source"].value_counts())

with summary_col2:
    st.write("### Top Categories")
    st.bar_chart(df["category"].value_counts().head(10))

# =====================================================
# FOOTER
# =====================================================

st.markdown("---")

st.markdown("""
<center>
    <h4 style='color:#1E3A8A;'>📚 BookWise Recommendation System</h4>
    <p style='color:#555;'>
        Built with Streamlit · TF-IDF · Cosine Similarity · Open Library API · Goodreads Scraping
    </p>
</center>
""", unsafe_allow_html=True)