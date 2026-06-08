# ============================================================
# AI MUSIC INTELLIGENCE DASHBOARD
# ============================================================
#
# INSTALL:
#
# pip install streamlit pandas numpy plotly pillow
#
# OPTIONAL:
#
# pip install rapidfuzz
#
# RUN:
#
# streamlit run dashboard.py
#
# ============================================================

import random
from collections import Counter
from difflib import SequenceMatcher
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

from music_analysis.features import Features

# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="AI Music Intelligence", layout="wide", initial_sidebar_state="expanded"
)

# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown(
    """
<style>

html, body, [class*="css"] {
    font-family: Inter, sans-serif;
}

.stApp {
    background:
        radial-gradient(circle at top left, #1f1f1f 0%, #0d0d0d 40%),
        linear-gradient(135deg, #0a0a0a 0%, #121212 100%);
    color: white;
}

/* Sidebar */
section[data-testid="stSidebar"] {
    background: rgba(17,17,17,0.92);
    backdrop-filter: blur(16px);
    border-right: 1px solid rgba(255,255,255,0.05);
}

/* Remove streamlit padding */
.block-container {
    padding-top: 1.5rem;
}

/* Hero */
.big-title {
    font-size: 64px;
    font-weight: 800;
    letter-spacing: -3px;
    line-height: 1;
    margin-bottom: 10px;
}

.subtitle {
    color: #9f9f9f;
    font-size: 18px;
    margin-bottom: 30px;
}

/* Track Cards */
.track-card {
    background: rgba(255,255,255,0.04);
    border: 1px solid rgba(255,255,255,0.06);
    border-radius: 22px;
    padding: 18px;
    margin-bottom: 18px;
    transition: all 0.25s ease;
    backdrop-filter: blur(14px);
}

.track-card:hover {
    transform: scale(1.01);
    background: rgba(255,255,255,0.07);
}

/* Pills */
.metric-pill {
    display: inline-block;
    background: rgba(255,255,255,0.08);
    padding: 6px 12px;
    border-radius: 999px;
    margin-right: 6px;
    margin-top: 10px;
    font-size: 13px;
}

/* Search Box */
.stTextInput > div > div > input {
    background-color: rgba(255,255,255,0.05);
    border: 1px solid rgba(255,255,255,0.08);
    border-radius: 16px;
    color: white;
    padding: 14px;
    font-size: 16px;
}

/* Buttons */
.stButton button {
    border-radius: 14px;
    border: none;
    background: linear-gradient(135deg, #7c4dff, #00c6ff);
    color: white;
    font-weight: 600;
    padding: 10px 18px;
}

/* Scrollbar */
::-webkit-scrollbar {
    width: 10px;
}

::-webkit-scrollbar-thumb {
    background: #444;
    border-radius: 999px;
}

</style>
""",
    unsafe_allow_html=True,
)

# ============================================================
# LOAD DATA
# ============================================================


@st.cache_data
def load_data() -> pd.DataFrame:

    tracks = []

    for f in Path("metadata").glob("*.json"):
        try:
            track = Features.load(f).to_dict()

            track.setdefault("bpm", 0)
            track.setdefault("danceability", 0)
            track.setdefault("key", "N/A")
            track.setdefault("scale", "")
            track.setdefault("track_name", "Unknown")

            # FIX DANCEABILITY
            dance = track.get("danceability", 0)

            try:
                dance = float(dance)
            except Exception:
                dance = 0

            if dance > 1:
                dance = dance / 100

            dance = min(max(dance, 0), 1)

            track["danceability"] = dance

            tracks.append(track)

        except Exception:
            pass

    return pd.DataFrame(tracks)


df = load_data()

if df.empty:
    st.error("No metadata found. Run the pipeline first.")
    st.stop()

# ============================================================
# HERO
# ============================================================

st.markdown(
    """
<div class="big-title">
AI MUSIC<br>
INTELLIGENCE
</div>
""",
    unsafe_allow_html=True,
)

st.markdown(
    """
<div class="subtitle">
Semantic Discovery • Smart Playlists • Audio Intelligence • CLAP Search
</div>
""",
    unsafe_allow_html=True,
)

# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.markdown("## 🎛 Discovery Engine")

search_query = st.sidebar.text_input(
    "Semantic Search", placeholder="dark warehouse techno..."
)

bpm_min = int(df["bpm"].min())
bpm_max = int(df["bpm"].max())

bpm_range = st.sidebar.slider("BPM Range", bpm_min, bpm_max, (bpm_min, bpm_max))

min_dance = st.sidebar.slider("Minimum Danceability", 0.0, 1.0, 0.2)

sort_mode = st.sidebar.selectbox(
    "Sort Results By", ["Similarity", "Danceability", "BPM", "Track Name"]
)

# ============================================================
# SEARCH ENGINE
# ============================================================

filtered = df.copy()

filtered = filtered[
    (filtered["bpm"] >= bpm_range[0]) & (filtered["bpm"] <= bpm_range[1])
]

filtered = filtered[filtered["danceability"] >= min_dance]

scores = []

for _, row in filtered.iterrows():
    score = 0

    track_name = str(row.get("track_name", "")).lower()

    clap_text = ""

    if isinstance(row.get("clap_matches"), list):
        clap_text = " ".join(
            [x.get("description", "") for x in row["clap_matches"]]
        ).lower()

    if search_query:
        score += SequenceMatcher(None, search_query.lower(), track_name).ratio() * 100

        score += SequenceMatcher(None, search_query.lower(), clap_text).ratio() * 100

    else:
        score += random.randint(20, 80)

    score += float(row.get("danceability", 0)) * 100

    scores.append(score)

filtered["similarity_score"] = scores

# ============================================================
# SORTING
# ============================================================

if sort_mode == "Similarity":
    filtered = filtered.sort_values(by="similarity_score", ascending=False)

elif sort_mode == "Danceability":
    filtered = filtered.sort_values(by="danceability", ascending=False)

elif sort_mode == "BPM":
    filtered = filtered.sort_values(by="bpm", ascending=False)

elif sort_mode == "Track Name":
    filtered = filtered.sort_values(by="track_name")

# ============================================================
# METRICS
# ============================================================

m1, m2, m3, m4 = st.columns(4)

with m1:
    st.metric("Tracks", len(df))

with m2:
    st.metric("Filtered", len(filtered))

with m3:
    st.metric("Avg BPM", round(df["bpm"].mean(), 1))

with m4:
    st.metric("Avg Danceability", round(df["danceability"].mean(), 2))

# ============================================================
# MAIN LAYOUT
# ============================================================

left, right = st.columns([2.4, 1])

# ============================================================
# LEFT COLUMN
# ============================================================

with left:
    st.markdown("## 🔎 Discovery Results")

    top_results = filtered.head(25)

    for _, track in top_results.iterrows():
        similarity = int(track["similarity_score"])

        bpm = track.get("bpm", "N/A")

        key = track.get("key", "N/A")

        scale = track.get("scale", "")

        dance = round(track.get("danceability", 0), 2)

        st.markdown(
            f"""
            <div class="track-card">

                <h3 style="margin-bottom:10px;">
                    {track["track_name"]}
                </h3>

                <span class="metric-pill">
                    🎵 {bpm} BPM
                </span>

                <span class="metric-pill">
                    🎹 {key}{scale}
                </span>

                <span class="metric-pill">
                    💃 Dance {dance}
                </span>

                <span class="metric-pill">
                    🧠 Similarity {similarity}%
                </span>

            </div>
            """,
            unsafe_allow_html=True,
        )

        # AUDIO PLAYER

        audio_file = None

        for ext in [".mp3", ".wav", ".flac", ".m4a"]:
            p = Path("audio_raw") / (f"{track['track_name']}{ext}")

            if p.exists():
                audio_file = p
                break

        if audio_file:
            st.audio(str(audio_file))

        # CLAP TAGS

        if isinstance(track.get("clap_matches"), list):
            tags = sorted(
                track["clap_matches"], key=lambda x: x.get("score", 0), reverse=True
            )[:4]

            cols = st.columns(len(tags))

            for c, t in zip(cols, tags):
                with c:
                    st.caption(f"🎧 {t['description']} ({t['score']:.2f})")

# ============================================================
# RIGHT COLUMN
# ============================================================

with right:
    st.markdown("## 🚀 Smart Playlist")

    if st.button("Generate Intelligent Playlist", use_container_width=True):
        playlist = filtered.sort_values(
            by=["similarity_score", "danceability"], ascending=False
        ).head(20)

        st.session_state["playlist"] = playlist.to_dict("records")

        st.session_state["playlist_index"] = 0

    if "playlist" in st.session_state:
        playlist = st.session_state["playlist"]

        idx = st.session_state["playlist_index"]

        current = playlist[idx]

        st.markdown(
            f"""
            <div class="track-card">

                <h2>▶ NOW PLAYING</h2>

                <h3>
                    {current["track_name"]}
                </h3>

                <span class="metric-pill">
                    {current.get("bpm")} BPM
                </span>

                <span class="metric-pill">
                    {current.get("key")}
                </span>

            </div>
            """,
            unsafe_allow_html=True,
        )

        if st.button("Next Track", use_container_width=True):
            st.session_state["playlist_index"] = (idx + 1) % len(playlist)

            st.rerun()

        st.markdown("### Queue")

        for i, track in enumerate(playlist):
            active = "▶️" if i == idx else "•"

            st.write(f"{active} {track['track_name']} ({track.get('bpm')} BPM)")

# ============================================================
# ANALYTICS
# ============================================================

st.markdown("## 📊 Audio Intelligence")

a1, a2 = st.columns(2)

# ============================================================
# BPM HISTOGRAM
# ============================================================

with a1:
    fig = px.histogram(df, x="bpm", nbins=35, title="BPM Distribution")

    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
    )

    st.plotly_chart(fig, use_container_width=True)

# ============================================================
# DANCEABILITY SCATTER
# ============================================================

with a2:
    fig2 = px.scatter(
        df,
        x="bpm",
        y="danceability",
        hover_name="track_name",
        title="Danceability vs BPM",
    )

    fig2.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
    )

    st.plotly_chart(fig2, use_container_width=True)

# ============================================================
# CLAP TAG CLOUD
# ============================================================

if "clap_matches" in df.columns:
    all_tags = []

    for matches in df["clap_matches"]:
        if isinstance(matches, list):
            all_tags.extend([x.get("description", "") for x in matches])

    counts = Counter(all_tags)

    cloud_df = pd.DataFrame(counts.items(), columns=["Tag", "Count"]).sort_values(
        by="Count", ascending=False
    )

    st.markdown("## ☁ Semantic Mood Space")

    fig3 = px.treemap(cloud_df.head(30), path=["Tag"], values="Count")

    fig3.update_layout(template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)")

    st.plotly_chart(fig3, use_container_width=True)

# ============================================================
# FOOTER
# ============================================================

st.markdown("---")

st.caption("""
Pipeline:
FFmpeg → Essentia → CLAP → FAISS → Semantic Discovery
""")
