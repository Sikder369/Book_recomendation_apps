# 📚 BookWise - Book Recommendation System

## 📖 Project Overview

BookWise is a content-based book recommendation system that helps users discover books based on their interests. The project combines data collected from APIs and web scraping, performs data analysis, and generates recommendations using machine learning techniques.

## 🚀 Features

- Search books by title
- Filter books by category
- Filter books by rating
- View similar book recommendations
- Interactive Streamlit dashboard

## 📊 Data Sources

- Open Library API
- Goodreads Web Scraping

## ⚙️ Technologies Used

- Python
- Pandas
- Scikit-learn
- TF-IDF Vectorization
- Cosine Similarity
- Matplotlib
- Streamlit
- BeautifulSoup

## 🧠 Recommendation Method

The recommendation system uses:

1. Feature Engineering
   - Combined title, author, subject, and category into a single content feature.

2. TF-IDF Vectorization
   - Converts text into numerical vectors.

3. Cosine Similarity
   - Measures similarity between books and generates recommendations.

## 📂 Project Workflow

Data Collection → Data Cleaning → EDA → Feature Engineering → TF-IDF → Cosine Similarity → Recommendation Engine → Streamlit Dashboard

## ▶️ Run the Application

Install dependencies:

```bash
pip install -r requirements.txt
```

Run Streamlit:

```bash
streamlit run app.py
```

## 📌 Project Outcome

Successfully developed an interactive book recommendation system using real-world data collected from APIs and web scraping, providing personalized book suggestions through a user-friendly dashboard.

---
Developed by Pollob Sikder
