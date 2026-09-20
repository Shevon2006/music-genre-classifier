"""
Music Genre Classifier — Streamlit web app
Group project: predicts a track's genre from its audio features (XGBoost).
"""

import os
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st

# ----------------------------------------------------------------------------
# Page setup
# ----------------------------------------------------------------------------
st.set_page_config(
    page_title="Genre Finder",
    page_icon="🎚️",
    layout="wide",
    initial_sidebar_state="expanded",
)

BASE_DIR = Path(__file__).parent
AUDIO_DIR = BASE_DIR / "audio"

# Each genre gets its own accent colour, used for the result card and meters.
GENRE_COLORS = {
    "classical":   "#C9A227",
    "comedy":      "#E4572E",
    "country":     "#B07D3F",
    "edm":         "#00D1B2",
    "heavy-metal": "#8E2DE2",
    "hip-hop":     "#F2B705",
    "jazz":        "#3D7EA6",
    "k-pop":       "#FF4D8D",
    "reggaeton":   "#F25C05",
    "sleep":       "#6C7BD9",
}

GENRE_BLURBS = {
    "classical":   "Acoustic, orchestral, almost no beat to speak of.",
    "comedy":      "Spoken word and laughter — very high speechiness.",
    "country":     "Acoustic guitar, storytelling vocals, steady mid-tempo.",
    "edm":         "Loud, synthetic and built for the dancefloor.",
    "heavy-metal": "Distorted guitars, fast tempo, maximum loudness.",
    "hip-hop":     "Beat-driven with heavy bass and rhythmic vocals.",
    "jazz":        "Acoustic instrumentation, swing feel, live-room warmth.",
    "k-pop":       "Polished pop production with bright synth hooks.",
    "reggaeton":   "Dembow rhythm, deep bass, high danceability.",
    "sleep":       "Quiet ambient textures, no percussion, very low energy.",
}

# ----------------------------------------------------------------------------
# Styling
# ----------------------------------------------------------------------------
st.markdown(
    """
    <link href="https://fonts.googleapis.com/css2?family=Bricolage+Grotesque:opsz,wght@12..96,500;12..96,700&family=Inter:wght@400;500;600&display=swap" rel="stylesheet">
    <style>
      :root {
        --ink:      #12101A;
        --surface:  #1C1928;
        --line:     #2E2A40;
        --text:     #EDEAF4;
        --muted:    #9A93AE;
      }
      .stApp { background: var(--ink); color: var(--text); }
      html, body, [class*="css"] { font-family: 'Inter', system-ui, sans-serif; }

      h1, h2, h3, .display {
        font-family: 'Bricolage Grotesque', 'Inter', sans-serif;
        letter-spacing: -0.02em;
      }

      section[data-testid="stSidebar"] {
        background: var(--surface);
        border-right: 1px solid var(--line);
      }

      .masthead { padding: 0.5rem 0 1.75rem 0; }
      .masthead h1 { font-size: 2.6rem; margin: 0; font-weight: 700; }
      .masthead p  { color: var(--muted); margin: 0.35rem 0 0 0; font-size: 1rem; }

      .result {
        border: 1px solid var(--line);
        border-left: 5px solid var(--accent, #6C7BD9);
        border-radius: 14px;
        background: var(--surface);
        padding: 1.6rem 1.8rem;
      }
      .result .genre {
        font-family: 'Bricolage Grotesque', sans-serif;
        font-size: 3.4rem;
        font-weight: 700;
        line-height: 1.05;
        color: var(--accent, #EDEAF4);
        margin: 0.2rem 0;
      }
      .result .conf  { color: var(--muted); font-size: 0.95rem; margin: 0; }
      .result .blurb { color: var(--text); font-size: 1.02rem; margin: 0.9rem 0 0 0; }

      .meter-row {
        display: flex; align-items: center; gap: 0.9rem; margin: 0.55rem 0;
      }
      .meter-name  { width: 118px; font-size: 0.9rem; color: var(--text); }
      .meter-track {
        flex: 1; height: 9px; border-radius: 99px; background: #262234; overflow: hidden;
      }
      .meter-fill  { height: 100%; border-radius: 99px; }
      .meter-value {
        width: 58px; text-align: right; font-size: 0.88rem;
        color: var(--muted); font-variant-numeric: tabular-nums;
      }

      .fact { border-top: 1px solid var(--line); padding: 0.7rem 0; }
      .fact span { color: var(--muted); font-size: 0.85rem; display: block; }
      .fact strong { font-size: 1.05rem; font-weight: 600; }

      div.stButton > button {
        width: 100%; border-radius: 10px; border: 1px solid var(--line);
        background: #241F35; color: var(--text); padding: 0.6rem 1rem;
        font-weight: 600;
      }
      div.stButton > button:hover { border-color: #5B4FD6; color: #fff; }
      #MainMenu, footer { visibility: hidden; }
    </style>
    """,
    unsafe_allow_html=True,
)


# ----------------------------------------------------------------------------
# Loading
# ----------------------------------------------------------------------------
@st.cache_resource
def load_model_and_artifacts():
    model = joblib.load(BASE_DIR / "best_model.pkl")
    artifacts = joblib.load(BASE_DIR / "preprocessing_artifacts.pkl")
    return model, artifacts


@st.cache_data
def load_songs():
    return pd.read_csv(BASE_DIR / "song_lookup.csv")


try:
    model, ART = load_model_and_artifacts()
    songs = load_songs()
except FileNotFoundError as err:
    st.error(f"A required file is missing: {err.filename}. "
             "Make sure best_model.pkl, preprocessing_artifacts.pkl and "
             "song_lookup.csv sit next to app.py.")
    st.stop()

SCALER = ART["scaler"]
LABELS = ART["label_encoder"]
CLIP_BOUNDS = ART["clip_bounds"]
SCALE_COLS = ART["scale_cols"]
FEATURE_COLUMNS = ART["feature_columns"]
GENRES = list(LABELS.classes_)


# ----------------------------------------------------------------------------
# Prediction
# ----------------------------------------------------------------------------
def build_feature_row(raw: dict) -> pd.DataFrame:
    """Turn raw user input into the exact 30-column row the model expects."""
    row = dict(raw)

    # 1. cap outliers with the bounds learned during preprocessing
    for col, (low, high) in CLIP_BOUNDS.items():
        if col in row:
            row[col] = float(np.clip(row[col], low, high))

    # 2. scale the numeric features with the saved scaler
    numeric = pd.DataFrame([[row[c] for c in SCALE_COLS]], columns=SCALE_COLS)
    scaled = pd.DataFrame(SCALER.transform(numeric), columns=SCALE_COLS)

    # 3. rebuild the one-hot columns
    frame = scaled.copy()
    frame["explicit"] = int(row["explicit"])
    frame["mode"] = int(row["mode"])
    for k in range(12):
        frame[f"key_{k}"] = int(row["key"] == k)
    for ts in [0, 1, 3, 4, 5]:
        frame[f"ts_{ts}"] = int(row["time_signature"] == ts)

    # 4. same column order as training
    return frame.reindex(columns=FEATURE_COLUMNS, fill_value=0)


def predict(frame: pd.DataFrame):
    proba = model.predict_proba(frame)[0]
    order = np.argsort(proba)[::-1]
    return [(GENRES[i], float(proba[i])) for i in order]


# ----------------------------------------------------------------------------
# Result display
# ----------------------------------------------------------------------------
def show_result(ranked, actual_genre=None):
    genre, confidence = ranked[0]
    accent = GENRE_COLORS.get(genre, "#6C7BD9")

    st.markdown(
        f"""
        <div class="result" style="--accent:{accent}">
          <p class="conf">Predicted genre</p>
          <p class="genre">{genre}</p>
          <p class="conf">{confidence:.0%} confidence</p>
          <p class="blurb">{GENRE_BLURBS.get(genre, '')}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if actual_genre is not None:
        if actual_genre == genre:
            st.success(f"Correct — this track is labelled {actual_genre}.")
        else:
            st.warning(f"The dataset labels this track as {actual_genre}.")

    clip = AUDIO_DIR / f"{genre}.mp3"
    if clip.exists():
        st.markdown(f"**Hear what {genre} sounds like**")
        st.audio(str(clip))
    else:
        st.caption(f"No sample clip found at audio/{genre}.mp3")

    st.markdown("#### How the other genres scored")
    for name, p in ranked[:5]:
        colour = GENRE_COLORS.get(name, "#6C7BD9")
        st.markdown(
            f"""
            <div class="meter-row">
              <div class="meter-name">{name}</div>
              <div class="meter-track">
                <div class="meter-fill" style="width:{max(p * 100, 1):.1f}%;background:{colour}"></div>
              </div>
              <div class="meter-value">{p:.1%}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )


# ----------------------------------------------------------------------------
# Header
# ----------------------------------------------------------------------------
st.markdown(
    """
    <div class="masthead">
      <h1>Genre Finder</h1>
      <p>Set a track's audio features and hear which of ten genres it belongs to.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

tab_build, tab_song, tab_about = st.tabs(["Build a track", "Try a real song", "About the model"])

# ----------------------------------------------------------------------------
# Tab 1 — sliders
# ----------------------------------------------------------------------------
with tab_build:
    with st.sidebar:
        st.markdown("### Track controls")

        preset = st.selectbox(
            "Start from a preset",
            ["Custom", "Club banger", "Quiet piano piece", "Garage rock"],
        )
        presets = {
            "Club banger":       dict(dance=0.85, energy=0.92, loud=-4.0, tempo=128.0,
                                      acoustic=0.02, instr=0.6, valence=0.7, speech=0.06),
            "Quiet piano piece": dict(dance=0.25, energy=0.08, loud=-27.0, tempo=76.0,
                                      acoustic=0.97, instr=0.93, valence=0.18, speech=0.04),
            "Garage rock":       dict(dance=0.45, energy=0.88, loud=-5.0, tempo=148.0,
                                      acoustic=0.05, instr=0.05, valence=0.55, speech=0.08),
        }
        p = presets.get(preset, dict(dance=0.55, energy=0.55, loud=-9.0, tempo=120.0,
                                     acoustic=0.35, instr=0.1, valence=0.5, speech=0.07))

        danceability = st.slider("Danceability", 0.0, 1.0, p["dance"], 0.01)
        energy = st.slider("Energy", 0.0, 1.0, p["energy"], 0.01)
        loudness = st.slider("Loudness (dB)", -49.0, 2.0, p["loud"], 0.5)
        speechiness = st.slider("Speechiness", 0.0, 1.0, p["speech"], 0.01)
        acousticness = st.slider("Acousticness", 0.0, 1.0, p["acoustic"], 0.01)
        instrumentalness = st.slider("Instrumentalness", 0.0, 1.0, p["instr"], 0.01)
        liveness = st.slider("Liveness", 0.0, 1.0, 0.15, 0.01)
        valence = st.slider("Positivity (valence)", 0.0, 1.0, p["valence"], 0.01)
        tempo = st.slider("Tempo (BPM)", 40.0, 220.0, p["tempo"], 1.0)
        minutes = st.slider("Duration (minutes)", 0.5, 10.0, 3.3, 0.1)
        popularity = st.slider("Popularity", 0, 100, 40)

        st.markdown("###### Musical details")
        key_names = ["C", "C♯/D♭", "D", "D♯/E♭", "E", "F",
                     "F♯/G♭", "G", "G♯/A♭", "A", "A♯/B♭", "B"]
        key = key_names.index(st.selectbox("Key", key_names, index=0))
        mode = 1 if st.radio("Mode", ["Major", "Minor"], horizontal=True) == "Major" else 0
        time_signature = st.selectbox("Time signature", [3, 4, 5, 1, 0], index=1)
        explicit = st.checkbox("Explicit lyrics", value=False)

    left, right = st.columns([1, 1.15], gap="large")

    with left:
        st.markdown("#### Your track")
        summary = [
            ("Tempo", f"{tempo:.0f} BPM"),
            ("Loudness", f"{loudness:.1f} dB"),
            ("Length", f"{int(minutes)}:{int(round((minutes % 1) * 60)):02d}"),
            ("Key", f"{key_names[key]} {'major' if mode else 'minor'}"),
        ]
        for label, value in summary:
            st.markdown(
                f'<div class="fact"><span>{label}</span><strong>{value}</strong></div>',
                unsafe_allow_html=True,
            )
        st.write("")
        go = st.button("Identify genre", type="primary")

    with right:
        if go:
            raw = dict(
                popularity=popularity,
                duration_ms=minutes * 60_000,
                explicit=explicit,
                danceability=danceability,
                energy=energy,
                loudness=loudness,
                mode=mode,
                speechiness=speechiness,
                acousticness=acousticness,
                instrumentalness=instrumentalness,
                liveness=liveness,
                valence=valence,
                tempo=tempo,
                key=key,
                time_signature=time_signature,
            )
            show_result(predict(build_feature_row(raw)))
        else:
            st.info("Adjust the controls in the sidebar, then select Identify genre.")

# ----------------------------------------------------------------------------
# Tab 2 — pick a real song
# ----------------------------------------------------------------------------
with tab_song:
    st.markdown("#### Pick a track from the dataset")
    st.caption("These are real songs the model was tested against. "
               "Choose one to see whether it gets the genre right.")

    query = st.text_input("Search by song or artist", placeholder="e.g. Bohemian")

    pool = songs
    if query:
        mask = (songs["track_name"].str.contains(query, case=False, na=False) |
                songs["artists"].str.contains(query, case=False, na=False))
        pool = songs[mask]

    if pool.empty:
        st.warning("Nothing matched that search. Try a different word.")
    else:
        pool = pool.head(200)
        options = {
            f"{r.track_name} — {r.artists}": i for i, r in pool.iterrows()
        }
        choice = st.selectbox(f"Matching tracks ({len(pool)} shown)", list(options.keys()))
        if st.button("Identify this track's genre"):
            row = songs.loc[options[choice]]
            frame = row[FEATURE_COLUMNS].to_frame().T.astype(float)
            show_result(predict(frame), actual_genre=row["track_genre"])

# ----------------------------------------------------------------------------
# Tab 3 — about
# ----------------------------------------------------------------------------
with tab_about:
    st.markdown("#### How this works")
    st.write(
        "Six algorithms were trained on 8,250 Spotify tracks across ten genres. "
        "XGBoost scored highest and is the model running here."
    )

    scores = pd.DataFrame({
        "Algorithm": ["XGBoost", "Random Forest", "SVM (RBF)", "KNN",
                      "Decision Tree", "Logistic Regression"],
        "Accuracy": [0.823, 0.810, 0.755, 0.735, 0.709, 0.665],
        "Macro F1": [0.810, 0.794, 0.736, 0.714, 0.688, 0.645],
    })
    st.dataframe(scores, hide_index=True, use_container_width=True)

    st.markdown("#### The ten genres")
    cols = st.columns(2)
    for i, g in enumerate(GENRES):
        with cols[i % 2]:
            st.markdown(
                f'<div class="fact"><span style="color:{GENRE_COLORS[g]}">{g}</span>'
                f'{GENRE_BLURBS.get(g, "")}</div>',
                unsafe_allow_html=True,
            )
