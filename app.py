"""
Genre Finder — Spotify genre prediction (XGBoost)
Streamlit app with an animated result stage and live audio visualiser.
"""

import base64
import html
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st
import streamlit.components.v1 as components

st.set_page_config(page_title="Genre Finder", page_icon="🎧", layout="wide",
                   initial_sidebar_state="collapsed")

BASE_DIR = Path(__file__).parent
AUDIO_DIR = BASE_DIR / "audio"

GENRE_COLORS = {
    "classical":   "#E0B645",
    "comedy":      "#FF7A45",
    "country":     "#D9A066",
    "edm":         "#1FE0C4",
    "heavy-metal": "#B266FF",
    "hip-hop":     "#FFD23F",
    "jazz":        "#5AA9E6",
    "k-pop":       "#FF5FA2",
    "reggaeton":   "#FF8C42",
    "sleep":       "#8C9EFF",
}
GENRE_BLURBS = {
    "classical":   "Acoustic and orchestral, with almost no beat to speak of.",
    "comedy":      "Spoken word and laughter, so speechiness runs very high.",
    "country":     "Acoustic guitar, storytelling vocals and a steady mid-tempo.",
    "edm":         "Loud, synthetic and built for the dancefloor.",
    "heavy-metal": "Distorted guitars, fast tempo and maximum loudness.",
    "hip-hop":     "Beat-driven, with heavy bass and rhythmic vocals.",
    "jazz":        "Acoustic instruments, a swing feel and live-room warmth.",
    "k-pop":       "Polished pop production with bright synth hooks.",
    "reggaeton":   "Dembow rhythm, deep bass and high danceability.",
    "sleep":       "Quiet ambient textures with no percussion and very low energy.",
}

# Slider defaults and presets --------------------------------------------------
DEFAULTS = dict(dance=0.55, energy=0.55, valence=0.50, acoustic=0.35, instr=0.10,
                speech=0.07, live=0.15, tempo=120, loud=-9.0, minutes=3.3, popl=40)
PRESETS = {
    "Festival drop":  dict(dance=0.82, energy=0.95, valence=0.55, acoustic=0.01, instr=0.55,
                           speech=0.06, live=0.25, tempo=128, loud=-3.5),
    "Late-night piano": dict(dance=0.25, energy=0.07, valence=0.20, acoustic=0.98, instr=0.92,
                             speech=0.04, live=0.10, tempo=72, loud=-28.0),
    "Mosh pit":       dict(dance=0.40, energy=0.97, valence=0.35, acoustic=0.01, instr=0.10,
                           speech=0.09, live=0.30, tempo=165, loud=-3.8),
    "Stand-up set":   dict(dance=0.60, energy=0.55, valence=0.60, acoustic=0.75, instr=0.00,
                           speech=0.92, live=0.85, tempo=100, loud=-15.0),
}

for k, v in DEFAULTS.items():
    st.session_state.setdefault(k, v)


def apply_preset(name):
    for k, v in PRESETS[name].items():
        st.session_state[k] = v


# Styling ----------------------------------------------------------------------
st.markdown("""
<link href="https://fonts.googleapis.com/css2?family=Bricolage+Grotesque:opsz,wght@12..96,500;12..96,800&family=Inter:wght@400;500;600&display=swap" rel="stylesheet">
<style>
  :root { --ink:#0F0D17; --panel:#17141F; --line:#2A2638; --text:#EEEAF6; --muted:#9891AD; --brand:#7B6CFF; }
  .stApp { color: var(--text); }
  html, body, [class*="css"], .stMarkdown, label, p { font-family: 'Inter', system-ui, sans-serif; }
  .block-container { padding-top: 2.2rem; max-width: 1280px; }
  header[data-testid="stHeader"] { background: transparent; }
  #MainMenu, footer { visibility: hidden; }
  html, body { background: radial-gradient(1200px 500px at 85% -10%, rgba(123,108,255,.16), transparent 60%), #0F0D17 !important; }
  .stApp, [data-testid="stAppViewContainer"], [data-testid="stMain"] { background: transparent !important; }
  .stApp { position:relative; z-index:1; }
  #ambient-root { position:fixed; inset:0; z-index:0; pointer-events:none; overflow:hidden; }
  #ambient-root img { position:absolute; top:0; left:0; height:auto; opacity:0; will-change:transform, opacity;
      animation: drift var(--d) linear var(--delay) infinite; filter: blur(var(--b)); }
  @keyframes drift {
      0%   { transform: translate(var(--x0), var(--y0)) rotate(var(--r0)); opacity:0; }
      12%  { opacity: var(--o); }
      80%  { opacity: var(--o); }
      100% { transform: translate(var(--x1), var(--y1)) rotate(var(--r1)); opacity:0; } }
  @media (prefers-reduced-motion: reduce) { #ambient-root { display:none; } }
  .hero { position:relative; border-radius:22px; overflow:hidden; border:1px solid var(--line);
          margin-bottom:1.8rem; box-shadow:0 30px 60px -30px rgba(0,0,0,.8); }
  .hero img { display:block; width:100%; height:auto; }
  .hero::after { content:""; position:absolute; inset:auto 0 0 0; height:28%;
          background:linear-gradient(to bottom, transparent, rgba(15,13,23,.55)); pointer-events:none; }
  .brand { display:flex; align-items:flex-end; justify-content:space-between; margin-bottom:1.4rem; }
  .brand h1 { font-family:'Bricolage Grotesque',sans-serif; font-weight:800; font-size:clamp(2.8rem,6vw,4.4rem);
              letter-spacing:-0.045em; margin:0; line-height:1; padding-bottom:.08em; }
  .brand h1 .w { display:inline-block; white-space:nowrap; }
  .brand h1 .l { display:inline-block; opacity:0; transform:translateY(55%) scale(.96); filter:blur(8px);
                 background:linear-gradient(100deg,#EEEAF6 0%,#EEEAF6 30%,#B8AEFF 42%,#FF5FA2 50%,#1FE0C4 58%,#EEEAF6 70%,#EEEAF6 100%);
                 background-size:900% 100%; background-position:100% 0;
                 -webkit-background-clip:text; background-clip:text; color:transparent;
                 animation: rise .8s cubic-bezier(.2,.8,.2,1) forwards, sweep 7s 1.4s ease-in-out infinite; }
  .brand h1 .gap { display:inline-block; width:.28em; }
  @keyframes rise { to { opacity:1; transform:none; filter:none; } }
  @keyframes sweep { 0% { background-position:100% 0; } 55%,100% { background-position:0% 0; } }
  .brand .rule { height:3px; width:0; margin-top:.9rem; border-radius:3px;
                 background:linear-gradient(90deg,#7B6CFF,#FF5FA2,#1FE0C4);
                 animation: draw 1.1s .9s cubic-bezier(.2,.8,.2,1) forwards; }
  @keyframes draw { to { width:120px; } }
  @media (prefers-reduced-motion: reduce){
    .brand h1 .l { animation:none; opacity:1; transform:none; filter:none; }
    .brand .rule { animation:none; width:120px; }
  }
  .brand p { color:var(--muted); margin:.5rem 0 0 0; font-size:1.02rem; max-width:560px; }
  .brand .eq { display:flex; gap:4px; align-items:flex-end; height:34px; }
  .brand .eq i { width:6px; border-radius:3px; background:var(--brand); animation: bob 1.1s ease-in-out infinite; }
  .brand .eq i:nth-child(2){animation-delay:.15s} .brand .eq i:nth-child(3){animation-delay:.3s}
  .brand .eq i:nth-child(4){animation-delay:.45s} .brand .eq i:nth-child(5){animation-delay:.6s}
  @keyframes bob { 0%,100%{height:8px} 50%{height:34px} }
  @media (prefers-reduced-motion: reduce){ .brand .eq i{animation:none;height:18px} }
  /* ---------- panels (bordered containers holding a group title) ---------- */
  div[data-testid="stVerticalBlock"]:has(> div[data-testid="stElementContainer"] .group-title) {
      background: linear-gradient(180deg, #1A1725 0%, #15121E 100%);
      border: 1px solid var(--line) !important; border-radius: 20px;
      box-shadow: inset 0 1px 0 rgba(255,255,255,.04), 0 20px 40px -28px rgba(0,0,0,.9); }
  .group-title { font-family:'Bricolage Grotesque',sans-serif; font-size:1.15rem; font-weight:500; margin:0 0 .15rem 0; }
  .group-sub { color:var(--muted); font-size:.85rem; margin:0 0 .6rem 0; }
  /* ---------- labels ---------- */
  div[data-testid="stWidgetLabel"] p { font-size:.86rem !important; color:#CFC9DD !important; font-weight:500; }
  /* ---------- sliders ---------- */
  div[data-testid="stSlider"] { padding:.1rem .2rem .35rem; border-radius:14px; transition: background .3s ease; }
  div[data-testid="stSlider"]:hover { background:rgba(123,108,255,.045); }
  div[data-testid="stSlider"] div:has(> div > [data-testid="stSliderThumbValue"]) > div:first-child { height:6px !important; border-radius:999px; box-shadow: inset 0 1px 2px rgba(0,0,0,.55);
      transition: filter .3s ease; }
  div[data-testid="stSlider"] div:has(> div > [data-testid="stSliderThumbValue"]) > div:first-child::after { content:""; position:absolute; inset:0; border-radius:inherit; pointer-events:none;
      background:linear-gradient(90deg,#6D5DFC 0%,#A66BFF 45%,#FF5FA2 100%); mix-blend-mode:hue; }
  div[data-testid="stSlider"]:hover div:has(> div > [data-testid="stSliderThumbValue"]) > div:first-child { filter: drop-shadow(0 0 6px rgba(123,108,255,.55)); }
  div[data-testid="stSlider"] div:has(> [data-testid="stSliderThumbValue"]) { width:20px !important; height:20px !important; background:radial-gradient(circle at 35% 30%, #FFFFFF 0%, #E9E5FF 60%, #CFC6FF 100%) !important;
      border:3px solid var(--brand) !important; box-shadow:0 0 0 0 rgba(123,108,255,.0), 0 4px 10px rgba(0,0,0,.55) !important;
      transition: transform .22s cubic-bezier(.2,.8,.2,1), box-shadow .22s ease; cursor:grab; }
  div[data-testid="stSlider"] div:has(> [data-testid="stSliderThumbValue"])[data-hovered] { transform:translate(-50%,-50%) scale(1.12) !important;
      box-shadow:0 0 0 7px rgba(123,108,255,.18), 0 4px 12px rgba(0,0,0,.55) !important; }
  div[data-testid="stSlider"] div:has(> [data-testid="stSliderThumbValue"])[data-dragging] { cursor:grabbing; transform:translate(-50%,-50%) scale(1.2) !important;
      box-shadow:0 0 0 10px rgba(123,108,255,.22), 0 0 24px 4px rgba(123,108,255,.55), 0 4px 12px rgba(0,0,0,.55) !important; }
  div[data-testid="stSlider"] div:has(> [data-testid="stSliderThumbValue"])[data-focus-visible] { box-shadow:0 0 0 3px #0F0D17, 0 0 0 5px #B8AEFF !important; }
  div[data-testid="stSlider"] [data-testid="stSliderThumbValue"] { top:-2.35em !important; padding:.14rem .5rem; border-radius:8px;
      background:#221D33; border:1px solid rgba(123,108,255,.45); color:#F3F0FF !important;
      font-size:.74rem !important; font-weight:600 !important; font-variant-numeric:tabular-nums; letter-spacing:.01em;
      box-shadow:0 6px 16px -6px rgba(0,0,0,.7); transition: transform .22s cubic-bezier(.2,.8,.2,1), background .22s ease; }
  div[data-testid="stSlider"] [data-testid="stSliderThumbValue"]::after { content:""; position:absolute; left:50%; bottom:-4px; width:7px; height:7px;
      background:inherit; border-right:1px solid rgba(123,108,255,.45); border-bottom:1px solid rgba(123,108,255,.45);
      transform:translateX(-50%) rotate(45deg); }
  div[data-testid="stSlider"] div:has(> [data-testid="stSliderThumbValue"])[data-dragging] > [data-testid="stSliderThumbValue"],
  div[data-testid="stSlider"] div:has(> [data-testid="stSliderThumbValue"])[data-hovered] > [data-testid="stSliderThumbValue"] {
      background:linear-gradient(120deg,#6D5DFC,#9B6BFF); border-color:transparent; transform:translateY(-3px); }
  div[data-testid="stSlider"] div:has(> [data-testid="stSliderThumbValue"])[data-dragging] > [data-testid="stSliderThumbValue"]::after,
  div[data-testid="stSlider"] div:has(> [data-testid="stSliderThumbValue"])[data-hovered] > [data-testid="stSliderThumbValue"]::after { background:#8A64FE; border-color:transparent; }
  div[data-testid="stSliderTickBar"] { color:#6F6886 !important; font-size:.72rem; }
  div[data-testid="stSlider"] [data-testid="stWidgetLabel"] { margin-bottom:.9rem; }
  /* ---------- selectboxes and text inputs ---------- */
  div[data-baseweb="select"] > div, div[data-baseweb="input"], div[data-baseweb="base-input"] {
      background:#1A1725 !important; border-radius:12px !important; border-color:var(--line) !important;
      transition: border-color .2s ease, box-shadow .2s ease; }
  div[data-baseweb="select"] > div:hover, div[data-baseweb="input"]:hover { border-color:#4A4166 !important; }
  div[data-baseweb="select"]:focus-within > div, div[data-baseweb="input"]:focus-within {
      border-color:var(--brand) !important; box-shadow:0 0 0 4px rgba(123,108,255,.18); }
  ul[role="listbox"] { background:#1A1725 !important; border:1px solid var(--line); border-radius:12px; }
  ul[role="listbox"] li[aria-selected="true"] { background:rgba(123,108,255,.18) !important; }
  /* ---------- expander ---------- */
  div[data-testid="stExpander"] details { background:#15121E; border:1px solid var(--line) !important;
      border-radius:16px !important; overflow:hidden; }
  div[data-testid="stExpander"] summary { padding:.85rem 1rem; transition: background .2s ease; }
  div[data-testid="stExpander"] summary:hover { background:rgba(123,108,255,.07); }
  div[data-testid="stExpander"] summary p { font-weight:500; }
  /* ---------- secondary buttons (presets) ---------- */
  button[data-testid="stBaseButton-secondary"] {
      border-radius:999px !important; border:1px solid var(--line) !important;
      background:rgba(255,255,255,.025) !important; color:#DAD5E8 !important;
      font-weight:500; font-size:.88rem; padding:.45rem .9rem !important;
      transition: border-color .2s ease, background .2s ease, transform .2s ease, box-shadow .2s ease; }
  button[data-testid="stBaseButton-secondary"]:hover {
      border-color:rgba(123,108,255,.65) !important; background:rgba(123,108,255,.1) !important;
      color:#fff !important; transform: translateY(-1px); box-shadow:0 8px 20px -12px rgba(123,108,255,.9); }
  button[data-testid="stBaseButton-secondary"]:active { transform: translateY(0) scale(.98); }
  /* ---------- primary button ---------- */
  button[data-testid="stBaseButton-primary"] {
      position:relative; overflow:hidden; border:none !important; border-radius:16px !important;
      padding:.95rem 1rem !important; color:#fff !important; font-weight:600; font-size:1.05rem; letter-spacing:.01em;
      background:linear-gradient(120deg,#6D5DFC 0%,#9B6BFF 45%,#E062C9 100%) !important; background-size:180% 100% !important;
      box-shadow:0 14px 34px -14px rgba(123,108,255,.95), inset 0 1px 0 rgba(255,255,255,.25);
      transition: background-position .5s ease, transform .2s ease, box-shadow .2s ease; }
  button[data-testid="stBaseButton-primary"]::after {
      content:""; position:absolute; top:0; left:-60%; width:40%; height:100%;
      background:linear-gradient(100deg, transparent, rgba(255,255,255,.35), transparent);
      transform:skewX(-20deg); transition:left .7s ease; pointer-events:none; }
  button[data-testid="stBaseButton-primary"]:hover {
      background-position:100% 0 !important; transform:translateY(-2px);
      box-shadow:0 20px 40px -14px rgba(160,100,255,1), inset 0 1px 0 rgba(255,255,255,.3); }
  button[data-testid="stBaseButton-primary"]:hover::after { left:120%; }
  button[data-testid="stBaseButton-primary"]:active { transform:translateY(0) scale(.985); }
  button[data-testid="stBaseButton-primary"]:disabled { opacity:.45; box-shadow:none; transform:none; }
  button[data-testid^="stBaseButton"]:focus-visible { outline:none !important;
      box-shadow:0 0 0 3px #0F0D17, 0 0 0 5px #B8AEFF !important; }
  /* ---------- tabs as a segmented pill ---------- */
  div[data-testid="stTabs"] [role="tablist"] { gap:6px !important; background:#15121E; border:1px solid var(--line);
      border-radius:999px; padding:6px; width:fit-content; max-width:100%; margin-bottom:.6rem;
      box-shadow: inset 0 1px 0 rgba(255,255,255,.03); }
  div[data-testid="stTabs"] [role="tablist"]::after { display:none !important; }
  div[data-testid="stTabs"] [data-testid="stTab"] { height:auto !important; padding:.62rem 1.35rem !important;
      border-radius:999px; color:var(--muted) !important; font-weight:500;
      transition: background .25s ease, color .25s ease, box-shadow .25s ease; }
  div[data-testid="stTabs"] [data-testid="stTab"] p { color:inherit !important; font-size:.95rem !important;
      font-weight:500; margin:0; }
  div[data-testid="stTabs"] [data-testid="stTab"][data-hovered] { color:var(--text) !important;
      background:rgba(255,255,255,.04); }
  div[data-testid="stTabs"] [data-testid="stTab"][data-selected] { color:#fff !important;
      background:linear-gradient(120deg,#6D5DFC 0%,#9B6BFF 55%,#C865D6 100%);
      box-shadow:0 8px 22px -10px rgba(123,108,255,.95), inset 0 1px 0 rgba(255,255,255,.22); }
  div[data-testid="stTabs"] [data-testid="stTab"][data-focus-visible] { box-shadow:0 0 0 2px #0F0D17, 0 0 0 4px #B8AEFF !important; }
  div[data-testid="stTabs"] .react-aria-SelectionIndicator { display:none !important; }
  /* ---------- radio + toggle accents ---------- */
  div[data-testid="stRadio"] label p, div[data-testid="stCheckbox"] label p { font-size:.9rem; }
  @media (prefers-reduced-motion: reduce){
    button[data-testid^="stBaseButton"], button[data-testid^="stBaseButton"]::after,
    div[data-testid="stSlider"] div[role="slider"] { transition:none !important; } }
</style>
""", unsafe_allow_html=True)


# Loading ----------------------------------------------------------------------
@st.cache_resource
def load_model_and_artifacts():
    return (joblib.load(BASE_DIR / "best_model.pkl"),
            joblib.load(BASE_DIR / "preprocessing_artifacts.pkl"))


@st.cache_data
def load_songs():
    return pd.read_csv(BASE_DIR / "song_lookup.csv")


@st.cache_data
def audio_data_uri(genre):
    clip = AUDIO_DIR / f"{genre}.mp3"
    if not clip.exists() or clip.stat().st_size == 0:
        return ""
    return "data:audio/mpeg;base64," + base64.b64encode(clip.read_bytes()).decode()


@st.cache_data
def logo_data_uri():
    img = BASE_DIR / "spotify.png"
    if not img.exists():
        return ""
    try:
        from io import BytesIO
        from PIL import Image
        im = Image.open(img).convert("RGBA")
        im.thumbnail((160, 160))
        buf = BytesIO()
        im.save(buf, format="PNG", optimize=True)
        data = buf.getvalue()
    except Exception:
        data = img.read_bytes()
    return "data:image/png;base64," + base64.b64encode(data).decode()


@st.cache_data
def hero_data_uri():
    img = BASE_DIR / "hero.jpg"
    if not img.exists():
        return ""
    return "data:image/jpeg;base64," + base64.b64encode(img.read_bytes()).decode()


try:
    model, ART = load_model_and_artifacts()
    songs = load_songs()
except FileNotFoundError as err:
    st.error(f"Missing file: {err.filename}. Keep best_model.pkl, "
             "preprocessing_artifacts.pkl and song_lookup.csv next to app.py.")
    st.stop()

SCALER, LABELS = ART["scaler"], ART["label_encoder"]
CLIP_BOUNDS, SCALE_COLS, FEATURE_COLUMNS = ART["clip_bounds"], ART["scale_cols"], ART["feature_columns"]
GENRES = list(LABELS.classes_)


# Prediction -------------------------------------------------------------------
def build_feature_row(raw):
    row = dict(raw)
    for col, (low, high) in CLIP_BOUNDS.items():
        if col in row:
            row[col] = float(np.clip(row[col], low, high))
    numeric = pd.DataFrame([[row[c] for c in SCALE_COLS]], columns=SCALE_COLS)
    frame = pd.DataFrame(SCALER.transform(numeric), columns=SCALE_COLS)
    frame["explicit"] = int(row["explicit"])
    frame["mode"] = int(row["mode"])
    for k in range(12):
        frame[f"key_{k}"] = int(row["key"] == k)
    for ts in [0, 1, 3, 4, 5]:
        frame[f"ts_{ts}"] = int(row["time_signature"] == ts)
    return frame.reindex(columns=FEATURE_COLUMNS, fill_value=0)


def predict(frame):
    proba = model.predict_proba(frame)[0]
    order = np.argsort(proba)[::-1]
    return [(GENRES[i], float(proba[i])) for i in order]


# Result stage (custom HTML/JS) ------------------------------------------------
STAGE_HEIGHT = 640

STAGE_TEMPLATE = r"""
<!doctype html><html><head>
<link href="https://fonts.googleapis.com/css2?family=Bricolage+Grotesque:opsz,wght@12..96,500;12..96,800&family=Inter:wght@400;500;600&display=swap" rel="stylesheet">
<style>
  *{box-sizing:border-box} html,body{margin:0;background:transparent;color:#EEEAF6;
    font-family:'Inter',system-ui,sans-serif;overflow:hidden}
  .stage{position:relative;height:__H__px;border-radius:22px;border:1px solid #2A2638;
    background:#17141F;overflow:hidden;padding:30px 34px}
  .glow{position:absolute;inset:-40%;background:radial-gradient(closest-side, var(--c), transparent 70%);
    opacity:0;filter:blur(30px);animation:glow 1.4s .2s ease-out forwards;pointer-events:none}
  @keyframes glow{to{opacity:.22}}

  .idle{height:100%;display:flex;flex-direction:column;align-items:center;justify-content:center;
    text-align:center;color:#9891AD}
  .idle .rings{position:relative;width:170px;height:170px;margin-bottom:26px}
  .idle .rings span{position:absolute;inset:0;border-radius:50%;border:1px solid #3A3450;
    animation:pulse 3s ease-out infinite}
  .idle .rings span:nth-child(2){animation-delay:1s} .idle .rings span:nth-child(3){animation-delay:2s}
  .idle .rings b{position:absolute;inset:58px;border-radius:50%;background:#7B6CFF;opacity:.85}
  @keyframes pulse{from{transform:scale(.45);opacity:1}to{transform:scale(1);opacity:0}}
  .idle h2{font-family:'Bricolage Grotesque',sans-serif;color:#EEEAF6;font-weight:500;font-size:1.6rem;margin:0 0 6px}

  .top{position:relative;display:flex;justify-content:space-between;align-items:flex-start;gap:20px}
  .eyebrow{color:#9891AD;font-size:.9rem;margin:0 0 4px;opacity:0;animation:fade .5s .1s forwards}
  .genre{font-family:'Bricolage Grotesque',sans-serif;font-weight:800;font-size:clamp(3rem,7vw,5.2rem);
    line-height:.95;letter-spacing:-0.04em;margin:0;color:var(--c);white-space:nowrap}
  .genre span{display:inline-block;opacity:0;transform:translateY(60%) rotate(6deg);filter:blur(6px);
    animation:letter .6s cubic-bezier(.2,.8,.2,1) forwards}
  @keyframes letter{to{opacity:1;transform:none;filter:none}}
  .blurb{color:#CFC9DD;margin:14px 0 0;max-width:420px;line-height:1.5;opacity:0;animation:fade .6s 1.1s forwards}
  @keyframes fade{to{opacity:1}}

  .ring{position:relative;width:118px;height:118px;flex:none;opacity:0;animation:fade .4s .5s forwards}
  .ring svg{transform:rotate(-90deg)}
  .ring .bg{stroke:#2A2638} .ring .fg{stroke:var(--c);stroke-linecap:round;
    transition:stroke-dashoffset 1.4s cubic-bezier(.2,.8,.2,1)}
  .ring .num{position:absolute;inset:0;display:flex;flex-direction:column;align-items:center;justify-content:center}
  .ring .num strong{font-family:'Bricolage Grotesque',sans-serif;font-size:1.7rem;font-weight:800}
  .ring .num small{color:#9891AD;font-size:.75rem}

  .player{position:relative;margin-top:26px;display:flex;align-items:center;gap:18px;
    background:#1E1A2B;border:1px solid #2A2638;border-radius:16px;padding:14px 18px;
    opacity:0;animation:fade .6s 1.4s forwards}
  .play{flex:none;width:52px;height:52px;border-radius:50%;border:none;cursor:pointer;
    background:var(--c);display:flex;align-items:center;justify-content:center;
    transition:transform .15s}
  .play:hover{transform:scale(1.06)} .play:focus-visible{outline:3px solid #fff;outline-offset:3px}
  .play svg{fill:#0F0D17}
  .player .meta{flex:none;width:150px}
  .player .meta b{display:block;font-size:.95rem} .player .meta span{color:#9891AD;font-size:.8rem}
  canvas{flex:1;height:52px;width:100%}

  .scores{position:relative;margin-top:24px}
  .scores h3{font-family:'Bricolage Grotesque',sans-serif;font-weight:500;font-size:1.05rem;margin:0 0 10px;
    opacity:0;animation:fade .5s 1.6s forwards}
  .row{display:grid;grid-template-columns:110px 1fr 56px;align-items:center;gap:14px;margin:9px 0;
    opacity:0;animation:fade .4s forwards}
  .row .name{font-size:.9rem} .row .val{text-align:right;color:#9891AD;font-size:.85rem;font-variant-numeric:tabular-nums}
  .track{height:8px;border-radius:99px;background:#262234;overflow:hidden}
  .fill{height:100%;width:0;border-radius:99px;transition:width 1s cubic-bezier(.2,.8,.2,1)}
  .note{color:#9891AD;font-size:.8rem;margin-top:6px}

  @media (prefers-reduced-motion: reduce){
    *{animation-duration:.01ms !important;animation-delay:0s !important;transition:none !important}
  }
</style></head><body>
__BODY__
</body></html>
"""

IDLE_BODY = """
<div class="stage"><div class="idle">
  <div class="rings"><span></span><span></span><span></span><b></b></div>
  <h2>Ready when you are</h2>
  <p>Shape your track on the left, then select Identify genre.</p>
</div></div>
"""


def result_stage_html(ranked, subtitle, run_id):
    genre, conf = ranked[0]
    color = GENRE_COLORS.get(genre, "#7B6CFF")
    letters = "".join(
        f'<span style="animation-delay:{0.25 + i * 0.045:.3f}s">{html.escape(ch) if ch != " " else "&nbsp;"}</span>'
        for i, ch in enumerate(genre)
    )
    rows = "".join(
        f'<div class="row" style="animation-delay:{1.7 + i * 0.08:.2f}s">'
        f'<div class="name">{html.escape(g)}</div>'
        f'<div class="track"><div class="fill" data-w="{max(p * 100, 0.8):.1f}" '
        f'style="background:{GENRE_COLORS.get(g, "#7B6CFF")}"></div></div>'
        f'<div class="val">{p:.1%}</div></div>'
        for i, (g, p) in enumerate(ranked[:5])
    )
    src = audio_data_uri(genre)
    circumference = 2 * np.pi * 50

    body = f"""
<div class="stage" style="--c:{color}" data-run="{run_id}">
  <div class="glow"></div>
  <div class="top">
    <div>
      <p class="eyebrow">{html.escape(subtitle)}</p>
      <h1 class="genre">{letters}</h1>
      <p class="blurb">{html.escape(GENRE_BLURBS.get(genre, ""))}</p>
    </div>
    <div class="ring">
      <svg width="118" height="118" viewBox="0 0 118 118">
        <circle class="bg" cx="59" cy="59" r="50" fill="none" stroke-width="9"/>
        <circle class="fg" id="fg" cx="59" cy="59" r="50" fill="none" stroke-width="9"
          stroke-dasharray="{circumference:.2f}" stroke-dashoffset="{circumference:.2f}"/>
      </svg>
      <div class="num"><strong id="pct">0%</strong><small>confidence</small></div>
    </div>
  </div>

  <div class="player">
    <button class="play" id="play" aria-label="Play sample">
      <svg id="ico" width="20" height="20" viewBox="0 0 20 20"><path d="M5 3l12 7-12 7z"/></svg>
    </button>
    <div class="meta"><b>{html.escape(genre)} sample</b><span id="status">{'Starting…' if src else 'No sample clip found'}</span></div>
    <canvas id="viz"></canvas>
    <audio id="clip" preload="auto" {'src="' + src + '"' if src else ''}></audio>
  </div>

  <div class="scores">
    <h3>How the top genres scored</h3>
    {rows}
  </div>
</div>
<script>
(function(){{
  const conf = {conf:.4f}, C = {circumference:.2f}, color = {json.dumps(color)};
  const reduce = matchMedia('(prefers-reduced-motion: reduce)').matches;

  // confidence ring + count-up
  setTimeout(() => {{
    document.getElementById('fg').style.strokeDashoffset = C * (1 - conf);
    const el = document.getElementById('pct'), target = Math.round(conf * 100), t0 = performance.now();
    const dur = reduce ? 1 : 1400;
    (function tick(t) {{
      const k = Math.min(1, (t - t0) / dur), e = 1 - Math.pow(1 - k, 3);
      el.textContent = Math.round(target * e) + '%';
      if (k < 1) requestAnimationFrame(tick);
    }})(t0);
  }}, 550);

  // probability bars
  setTimeout(() => document.querySelectorAll('.fill').forEach(f => f.style.width = f.dataset.w + '%'), 1800);

  // audio + visualiser
  const audio = document.getElementById('clip'), btn = document.getElementById('play'),
        ico = document.getElementById('ico'), status = document.getElementById('status'),
        cv = document.getElementById('viz'), g = cv.getContext('2d');
  let ctx = null, analyser = null, data = null, wired = false;
  const PLAY = 'M5 3l12 7-12 7z', PAUSE = 'M5 3h4v14H5zM11 3h4v14h-4z';

  function size() {{ const r = cv.getBoundingClientRect(); cv.width = r.width * devicePixelRatio; cv.height = r.height * devicePixelRatio; }}
  size(); addEventListener('resize', size);

  async function wire() {{
    if (wired) return true;
    try {{
      ctx = ctx || new (window.AudioContext || window.webkitAudioContext)();
      if (ctx.state !== 'running') await ctx.resume();
      if (ctx.state !== 'running') return false;       // never route audio into a silent context
      const node = ctx.createMediaElementSource(audio);
      analyser = ctx.createAnalyser(); analyser.fftSize = 128; analyser.smoothingTimeConstant = .8;
      node.connect(analyser); analyser.connect(ctx.destination);
      data = new Uint8Array(analyser.frequencyBinCount); wired = true; return true;
    }} catch (e) {{ return false; }}
  }}

  function draw(t) {{
    const W = cv.width, H = cv.height, n = 40, gap = 3 * devicePixelRatio, bw = (W - gap * (n - 1)) / n;
    g.clearRect(0, 0, W, H); g.fillStyle = color;
    if (wired && !audio.paused) analyser.getByteFrequencyData(data);
    for (let i = 0; i < n; i++) {{
      let v;
      if (audio.paused) v = .08;
      else if (wired) v = data[Math.floor(i * data.length / n * .75)] / 255;
      else v = .25 + .6 * Math.abs(Math.sin(t / 260 + i * .55)) * Math.abs(Math.cos(t / 410 + i * .2));
      const h = Math.max(3 * devicePixelRatio, v * H);
      g.globalAlpha = .45 + .55 * v;
      g.beginPath(); g.roundRect ? g.roundRect(i * (bw + gap), (H - h) / 2, bw, h, bw / 2)
                                 : g.rect(i * (bw + gap), (H - h) / 2, bw, h);
      g.fill();
    }}
    requestAnimationFrame(draw);
  }}
  requestAnimationFrame(draw);

  function setIcon() {{ ico.firstElementChild.setAttribute('d', audio.paused ? PLAY : PAUSE);
                       btn.setAttribute('aria-label', audio.paused ? 'Play sample' : 'Pause sample'); }}
  audio.addEventListener('play', () => {{ setIcon(); status.textContent = 'Playing'; }});
  audio.addEventListener('pause', () => {{ setIcon(); status.textContent = audio.ended ? 'Finished' : 'Paused'; }});

  btn.addEventListener('click', async () => {{
    if (!audio.src) return;
    await wire();
    if (audio.paused) audio.play(); else audio.pause();
  }});

  // try to start automatically once the reveal lands; browsers may block this
  if (audio.src) setTimeout(async () => {{
    await wire();
    audio.play().catch(() => {{ status.textContent = 'Tap play to listen'; }});
  }}, 1500);
}})();
</script>
"""
    return STAGE_TEMPLATE.replace("__H__", str(STAGE_HEIGHT - 4)).replace("__BODY__", body)


def show_stage(body_html):
    # st.iframe replaces components.html in newer Streamlit; fall back on older versions
    if hasattr(st, "iframe"):
        st.iframe(body_html, height=STAGE_HEIGHT)
    else:
        components.html(body_html, height=STAGE_HEIGHT, scrolling=False)


# Ambient background: a few logos drifting slowly behind the page ---------------
AMBIENT = [
    # width px, start x/y, end x/y, start/end rotation, duration, delay, opacity, blur
    (70,  "-10vw", "18vh",  "105vw", "62vh",  "-18deg",  "24deg", "38s",   "0s", .26, "0px"),
    (44,  "108vw", "8vh",   "-12vw", "40vh",   "30deg", "-40deg", "46s", "-12s", .18, "1px"),
    (96,  "20vw",  "110vh", "70vw",  "-15vh",  "-8deg",  "35deg", "52s", "-25s", .12, "3px"),
    (56,  "75vw",  "-12vh", "30vw",  "112vh",  "45deg",  "-5deg", "42s",  "-6s", .22, "0.5px"),
    (38,  "-8vw",  "85vh",  "104vw", "30vh",  "-35deg",  "15deg", "34s", "-19s", .20, "0px"),
]


def ambient_layer():
    src = logo_data_uri()
    if not src:
        return
    specs = json.dumps([
        dict(w=w, style=(f"width:{w}px;--x0:{x0};--y0:{y0};--x1:{x1};--y1:{y1};"
                         f"--r0:{r0};--r1:{r1};--d:{d};--delay:{delay};--o:{o};--b:{b}"))
        for w, x0, y0, x1, y1, r0, r1, d, delay, o, b in AMBIENT
    ])
    script = f"""
<script>
(function () {{
  if (document.getElementById('ambient-root')) return;
  const root = document.createElement('div');
  root.id = 'ambient-root';
  root.setAttribute('aria-hidden', 'true');
  const src = {json.dumps(src)};
  for (const spec of {specs}) {{
    const img = document.createElement('img');
    img.src = src; img.alt = ''; img.setAttribute('style', spec.style);
    root.appendChild(img);
  }}
  document.body.prepend(root);
}})();
</script>"""
    try:
        st.html(script, unsafe_allow_javascript=True)
    except TypeError:
        pass  # older Streamlit without script support: skip the effect


ambient_layer()

# Header -----------------------------------------------------------------------
if hero_data_uri():
    st.markdown(f'<div class="hero"><img src="{hero_data_uri()}" '
                'alt="Audio features flowing into a model that sorts tracks by genre"></div>',
                unsafe_allow_html=True)

def animated_title(text):
    words, i = [], 0
    for word in text.split(" "):
        letters = ""
        for ch in word:
            letters += (f'<span class="l" style="animation-delay:{0.15 + i * 0.05:.2f}s, '
                        f'{1.4 + i * 0.05:.2f}s">{html.escape(ch)}</span>')
            i += 1
        words.append(f'<span class="w">{letters}</span>')
    return '<span class="gap"></span>'.join(words)


st.markdown(f"""
<div class="brand">
  <div>
    <h1 aria-label="Genre Finder">{animated_title("Genre Finder")}</h1>
    <div class="rule"></div>
    <p>Shape a track's sound with the controls, and the model will name its genre and play you a sample.</p>
  </div>
  <div class="eq" aria-hidden="true"><i></i><i></i><i></i><i></i><i></i></div>
</div>
""", unsafe_allow_html=True)

tab_build, tab_song, tab_about = st.tabs(["Build a track", "Try a real song", "About the model"])

# Tab 1 — build a track ----------------------------------------------------------
with tab_build:
    left, right = st.columns([1, 1.15], gap="large")

    with left:
        st.markdown('<p class="group-sub">Start from a preset, or set every control yourself.</p>',
                    unsafe_allow_html=True)
        pcols = st.columns(len(PRESETS))
        for col, name in zip(pcols, PRESETS):
            col.button(name, on_click=apply_preset, args=(name,), width="stretch")

        with st.container(border=True):
            st.markdown('<p class="group-title">Feel</p>'
                        '<p class="group-sub">How it moves you</p>', unsafe_allow_html=True)
            c1, c2 = st.columns(2)
            c1.slider("Danceability", 0.0, 1.0, step=0.01, key="dance")
            c2.slider("Energy", 0.0, 1.0, step=0.01, key="energy")
            c1.slider("Positivity", 0.0, 1.0, step=0.01, key="valence")
            c2.slider("Tempo (BPM)", 40, 220, step=1, key="tempo")

        with st.container(border=True):
            st.markdown('<p class="group-title">Texture</p>'
                        '<p class="group-sub">What it is made of</p>', unsafe_allow_html=True)
            c1, c2 = st.columns(2)
            c1.slider("Acousticness", 0.0, 1.0, step=0.01, key="acoustic")
            c2.slider("Instrumentalness", 0.0, 1.0, step=0.01, key="instr")
            c1.slider("Speechiness", 0.0, 1.0, step=0.01, key="speech")
            c2.slider("Loudness (dB)", -49.0, 2.0, step=0.5, key="loud")

        with st.expander("More details"):
            c1, c2 = st.columns(2)
            c1.slider("Liveness", 0.0, 1.0, step=0.01, key="live")
            c2.slider("Popularity", 0, 100, key="popl")
            c1.slider("Duration (minutes)", 0.5, 10.0, step=0.1, key="minutes")
            key_names = ["C", "C♯", "D", "D♯", "E", "F", "F♯", "G", "G♯", "A", "A♯", "B"]
            key = key_names.index(c2.selectbox("Key", key_names))
            mode = 1 if c1.radio("Mode", ["Major", "Minor"], horizontal=True) == "Major" else 0
            time_signature = c2.selectbox("Beats per bar", [4, 3, 5, 1, 0])
            explicit = c1.toggle("Explicit lyrics")

        identify = st.button("Identify genre", type="primary", width="stretch")

    with right:
        if identify:
            s = st.session_state
            raw = dict(popularity=s.popl, duration_ms=s.minutes * 60_000, explicit=explicit,
                       danceability=s.dance, energy=s.energy, loudness=s.loud, mode=mode,
                       speechiness=s.speech, acousticness=s.acoustic, instrumentalness=s.instr,
                       liveness=s.live, valence=s.valence, tempo=s.tempo, key=key,
                       time_signature=time_signature)
            s.build_result = predict(build_feature_row(raw))
            s.build_run = s.get("build_run", 0) + 1

        if "build_result" in st.session_state:
            show_stage(result_stage_html(st.session_state.build_result,
                                         "Your track sounds like", st.session_state.build_run))
        else:
            show_stage(STAGE_TEMPLATE.replace("__H__", str(STAGE_HEIGHT - 4))
                                     .replace("__BODY__", IDLE_BODY))

# Tab 2 — real songs ---------------------------------------------------------------
with tab_song:
    left, right = st.columns([1, 1.15], gap="large")
    with left:
        with st.container(border=True):
            st.markdown('<p class="group-title">Pick a real track</p>'
                        '<p class="group-sub">Search the songs the model was built from and '
                        'see whether it names the right genre.</p>', unsafe_allow_html=True)
            query = st.text_input("Song or artist", placeholder="Try Mozart, Metallica or BTS")
            pool = songs
            if query:
                pool = songs[songs["track_name"].str.contains(query, case=False, na=False, regex=False) |
                             songs["artists"].str.contains(query, case=False, na=False, regex=False)]
            if pool.empty:
                st.warning("No songs matched that search. Try a shorter word or an artist name.")
                choice = None
            else:
                pool = pool.head(200)
                options = {f"{r.track_name}  ·  {r.artists}": i for i, r in pool.iterrows()}
                choice = st.selectbox(f"{len(pool)} tracks", list(options))
        go_song = st.button("Identify this track", type="primary", width="stretch",
                            disabled=choice is None)

    with right:
        if go_song and choice:
            row = songs.loc[options[choice]]
            ranked = predict(row[FEATURE_COLUMNS].to_frame().T.astype(float))
            st.session_state.song_result = (ranked, row["track_genre"], row["track_name"])
            st.session_state.song_run = st.session_state.get("song_run", 0) + 1

        if "song_result" in st.session_state:
            ranked, actual, name = st.session_state.song_result
            verdict = "Correct" if ranked[0][0] == actual else f"Labelled {actual} in the dataset"
            show_stage(result_stage_html(ranked, f"{name[:40]} — {verdict}",
                                         st.session_state.song_run))
        else:
            show_stage(STAGE_TEMPLATE.replace("__H__", str(STAGE_HEIGHT - 4))
                                     .replace("__BODY__", IDLE_BODY))

# Tab 3 — about ----------------------------------------------------------------------
with tab_about:
    c1, c2 = st.columns([1.2, 1], gap="large")
    with c1:
        with st.container(border=True):
            st.markdown('<p class="group-title">How it works</p>', unsafe_allow_html=True)
            st.write("Six algorithms were trained on 8,250 Spotify tracks across ten genres, "
                     "using the same train/test split and 5-fold cross-validation. "
                     "XGBoost scored highest and powers this app.")
            st.dataframe(pd.DataFrame({
                "Algorithm": ["XGBoost", "Random Forest", "SVM (RBF)", "KNN",
                              "Decision Tree", "Logistic Regression"],
                "Accuracy": [0.823, 0.810, 0.755, 0.735, 0.709, 0.665],
                "Macro F1": [0.810, 0.794, 0.736, 0.714, 0.688, 0.645],
            }), hide_index=True, width="stretch")
    with c2:
        with st.container(border=True):
            st.markdown('<p class="group-title">The ten genres</p>', unsafe_allow_html=True)
            for g in GENRES:
                st.markdown(f'<p style="margin:.35rem 0"><b style="color:{GENRE_COLORS[g]}">{g}</b>'
                            f'<br><span style="color:#9891AD;font-size:.88rem">{GENRE_BLURBS[g]}</span></p>',
                            unsafe_allow_html=True)
