"""
CineMatch - Movie Recommendation Dashboard
--------------------------------------------
A Streamlit dashboard built on top of the MovieLens 100K dataset.
Users can search for any movie and instantly get similar-movie
recommendations from an item-item cosine similarity model, browse
popular / top-rated movies, and explore dataset insights.

Run locally with:  streamlit run app.py
"""

import pandas as pd
import plotly.express as px
import streamlit as st

from src.data_utils import (
    GENRE_COLS,
    build_similarity,
    closest_titles,
    get_most_rated,
    get_recommendations,
    get_top_movies,
    load_data,
    movie_stats,
    search_titles,
)

# ----------------------------------------------------------------------------
# Page config & styling
# ----------------------------------------------------------------------------
st.set_page_config(
    page_title="CineMatch | Movie Recommender",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
        .main > div {padding-top: 1.5rem;}
        .movie-card {
            background: linear-gradient(135deg, #1f2937 0%, #111827 100%);
            border-radius: 14px;
            padding: 18px 20px;
            margin-bottom: 12px;
            border: 1px solid #2d3748;
        }
        .movie-card h4 {margin: 0 0 6px 0; color: #f9fafb;}
        .movie-card span {color: #9ca3af; font-size: 0.85rem;}
        .metric-badge {
            display: inline-block; padding: 3px 10px; border-radius: 20px;
            background: #374151; color: #e5e7eb; font-size: 0.78rem; margin-right: 6px;
        }
        .hero {
            padding: 22px 26px; border-radius: 16px; margin-bottom: 22px;
            background: linear-gradient(120deg, #7c3aed 0%, #2563eb 100%);
            color: white;
        }
        .hero h1 {margin: 0; font-size: 1.9rem;}
        .hero p {margin: 6px 0 0 0; opacity: 0.9;}
    </style>
    """,
    unsafe_allow_html=True,
)


# ----------------------------------------------------------------------------
# Data / model loading (cached so this only runs once per session)
# ----------------------------------------------------------------------------
@st.cache_resource(show_spinner=False)
def load_pipeline():
    df, movies = load_data()
    movie_matrix, similarity_df = build_similarity(df)
    stats = movie_stats(df)
    return df, movies, similarity_df, stats


with st.spinner("Loading movies & building recommendation model..."):
    df, movies, similarity_df, stats_df = load_pipeline()

all_titles = list(similarity_df.columns)

if "selected_movie" not in st.session_state:
    st.session_state.selected_movie = None


# ----------------------------------------------------------------------------
# Sidebar navigation
# ----------------------------------------------------------------------------
with st.sidebar:
    st.markdown("## 🎬 CineMatch")
    st.caption("Movie recommendations powered by MovieLens 100K")
    page = st.radio(
        "Navigate",
        ["🔍 Search & Recommend", "🔥 Popular Movies", "📊 Insights", "ℹ️ About"],
        label_visibility="collapsed",
    )
    st.divider()
    st.markdown("### Settings")
    top_n = st.slider("Number of recommendations", 5, 20, 10)
    weighted = st.toggle(
        "Popularity-boosted ranking",
        value=True,
        help="Blends raw similarity with how many people rated the movie, "
        "so results are less skewed by one-off coincidental matches.",
    )
    st.divider()
    st.caption(f"📚 {df['movie_id'].nunique():,} movies · "
               f"👥 {df['user_id'].nunique():,} users · "
               f"⭐ {len(df):,} ratings")


# ----------------------------------------------------------------------------
# Helper to render a recommendation card
# ----------------------------------------------------------------------------
def render_recommendation_table(rec_df: pd.DataFrame):
    if rec_df.empty:
        st.warning("No recommendations found for this movie yet.")
        return

    chart_df = rec_df.sort_values("match_score")
    fig = px.bar(
        chart_df,
        x="match_score",
        y="title",
        orientation="h",
        color="match_score",
        color_continuous_scale="Purples",
        labels={"match_score": "Match score", "title": ""},
        height=max(320, 34 * len(chart_df)),
    )
    fig.update_layout(
        showlegend=False,
        coloraxis_showscale=False,
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=10, r=10, t=10, b=10),
    )
    st.plotly_chart(fig, use_container_width=True)

    with st.expander("View as table"):
        display_df = rec_df[["title", "similarity", "avg_rating", "num_ratings"]].copy()
        display_df.columns = ["Movie", "Similarity", "Avg Rating", "# Ratings"]
        display_df["Similarity"] = display_df["Similarity"].round(3)
        display_df["Avg Rating"] = display_df["Avg Rating"].round(2)
        st.dataframe(display_df, use_container_width=True, hide_index=True)


# ----------------------------------------------------------------------------
# PAGE: Search & Recommend
# ----------------------------------------------------------------------------
if page == "🔍 Search & Recommend":
    st.markdown(
        """
        <div class="hero">
            <h1>Find your next favorite movie 🎬</h1>
            <p>Search any movie from the catalog and get instant, similarity-based recommendations.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    query = st.text_input(
        "Search for a movie",
        placeholder="Start typing a title... e.g. Toy Story, Star Wars, Titanic",
    )

    matches = search_titles(query, all_titles, limit=15) if query else []

    if query and not matches:
        suggestions = closest_titles(query, all_titles)
        if suggestions:
            st.info("No exact match found. Did you mean:")
            cols = st.columns(min(len(suggestions), 5))
            for c, s in zip(cols, suggestions):
                if c.button(s, key=f"sugg_{s}"):
                    st.session_state.selected_movie = s
        else:
            st.warning("No movies found matching that search.")

    if matches:
        chosen = st.selectbox(f"{len(matches)} match(es) found — pick one", matches)
        st.session_state.selected_movie = chosen

    st.divider()

    selected = st.session_state.selected_movie
    if selected and selected in all_titles:
        row = stats_df[stats_df["title"] == selected]
        avg_rating = float(row["avg_rating"].iloc[0]) if not row.empty else 0
        num_ratings = int(row["num_ratings"].iloc[0]) if not row.empty else 0
        genres = movies.loc[movies["title"] == selected, "genres"]
        genre_list = genres.iloc[0] if not genres.empty else []

        st.markdown(
            f"""
            <div class="movie-card">
                <h4>🎞️ {selected}</h4>
                <span class="metric-badge">⭐ {avg_rating:.2f} / 5</span>
                <span class="metric-badge">🗳️ {num_ratings} ratings</span>
                <span class="metric-badge">🎭 {', '.join(genre_list) if genre_list else 'N/A'}</span>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.subheader(f"Because you're interested in *{selected}*, you might also like:")
        recs = get_recommendations(selected, similarity_df, stats_df, n=top_n, weighted=weighted)
        render_recommendation_table(recs)
    else:
        st.info("👆 Search for a movie above to get personalized recommendations.")


# ----------------------------------------------------------------------------
# PAGE: Popular Movies
# ----------------------------------------------------------------------------
elif page == "🔥 Popular Movies":
    st.markdown(
        """
        <div class="hero">
            <h1>What's trending 🔥</h1>
            <p>Browse the most-rated and top-rated movies in the catalog.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    genre_options = ["All genres"] + [g for g in GENRE_COLS if g != "unknown"]
    selected_genre = st.selectbox("Filter by genre", genre_options)

    if selected_genre != "All genres":
        genre_titles = movies[movies["genres"].apply(lambda g: selected_genre in g)]["title"]
        filtered_stats = stats_df[stats_df["title"].isin(genre_titles)]
    else:
        filtered_stats = stats_df

    tab1, tab2 = st.tabs(["⭐ Top Rated", "🗳️ Most Rated"])

    with tab1:
        min_ratings = st.slider("Minimum number of ratings to qualify", 5, 100, 20, key="minr")
        top_rated = get_top_movies(filtered_stats, min_ratings=min_ratings, n=15)
        if top_rated.empty:
            st.warning("No movies match this filter combination.")
        else:
            fig = px.bar(
                top_rated.sort_values("avg_rating"),
                x="avg_rating", y="title", orientation="h",
                color="avg_rating", color_continuous_scale="Oranges",
                labels={"avg_rating": "Average rating", "title": ""},
                height=max(320, 34 * len(top_rated)),
            )
            fig.update_layout(showlegend=False, coloraxis_showscale=False,
                               plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)")
            st.plotly_chart(fig, use_container_width=True)

    with tab2:
        most_rated = get_most_rated(filtered_stats, n=15)
        fig = px.bar(
            most_rated.sort_values("num_ratings"),
            x="num_ratings", y="title", orientation="h",
            color="num_ratings", color_continuous_scale="Blues",
            labels={"num_ratings": "Number of ratings", "title": ""},
            height=max(320, 34 * len(most_rated)),
        )
        fig.update_layout(showlegend=False, coloraxis_showscale=False,
                           plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(fig, use_container_width=True)


# ----------------------------------------------------------------------------
# PAGE: Insights
# ----------------------------------------------------------------------------
elif page == "📊 Insights":
    st.markdown(
        """
        <div class="hero">
            <h1>Dataset insights 📊</h1>
            <p>A quick look at rating patterns across the MovieLens 100K dataset.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    c1, c2, c3 = st.columns(3)
    c1.metric("Total Ratings", f"{len(df):,}")
    c2.metric("Unique Movies", f"{df['movie_id'].nunique():,}")
    c3.metric("Unique Users", f"{df['user_id'].nunique():,}")

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("#### Rating distribution")
        rating_counts = df["rating"].value_counts().sort_index()
        fig = px.bar(
            x=rating_counts.index, y=rating_counts.values,
            labels={"x": "Rating", "y": "Count"}, color=rating_counts.values,
            color_continuous_scale="Viridis",
        )
        fig.update_layout(showlegend=False, coloraxis_showscale=False,
                           plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.markdown("#### Ratings per genre")
        genre_counts = {}
        for genres in movies["genres"]:
            for g in genres:
                genre_counts[g] = genre_counts.get(g, 0) + 1
        genre_series = pd.Series(genre_counts).sort_values(ascending=True)
        fig = px.bar(
            x=genre_series.values, y=genre_series.index, orientation="h",
            labels={"x": "Number of movies", "y": ""}, color=genre_series.values,
            color_continuous_scale="Tealgrn",
        )
        fig.update_layout(showlegend=False, coloraxis_showscale=False,
                           plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
                           height=450)
        st.plotly_chart(fig, use_container_width=True)


# ----------------------------------------------------------------------------
# PAGE: About
# ----------------------------------------------------------------------------
else:
    st.markdown(
        """
        <div class="hero">
            <h1>About CineMatch ℹ️</h1>
            <p>How this recommendation system works.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.markdown(
        """
- **Dataset**: [MovieLens 100K](https://grouplens.org/datasets/movielens/100k/) — 100,000 ratings from 943 users on 1,682 movies.
- **Approach**: Item-based collaborative filtering. Each movie is represented as a vector of ratings
  given by every user; cosine similarity between these vectors measures how similarly two movies
  were rated across the community.
- **Popularity-boosted ranking**: An optional toggle blends raw similarity with the number of ratings
  a candidate movie has received, reducing noisy matches from movies rated by very few people.
- **Tech stack**: Python, pandas, scikit-learn (cosine similarity), Streamlit, Plotly.

Built from the original MovieLens-100K exploration notebook and turned into an interactive dashboard.
        """
    )
