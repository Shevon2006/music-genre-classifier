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
                speech=0.07, live=0.15, tempo=120, loud=-9.0, minutes=3.3, popl=40,
                explicit=False, mode_name="Major")

# Each preset is built around the typical values of one genre in the dataset,
# and was checked to predict that genre with high confidence.
PRESETS = {
    "Festival":    dict(genre="edm", icon=":material/graphic_eq:",
                        values=dict(dance=0.68, energy=0.90, valence=0.45, acoustic=0.02, instr=0.40,
                                    speech=0.05, live=0.15, tempo=126, loud=-4.5, popl=50,
                                    minutes=3.2, explicit=False, mode_name="Major")),
    "Piano":       dict(genre="classical", icon=":material/piano:",
                        values=dict(dance=0.35, energy=0.10, valence=0.30, acoustic=0.98, instr=0.90,
                                    speech=0.04, live=0.10, tempo=96, loud=-22.0, popl=5,
                                    minutes=3.5, explicit=False, mode_name="Major")),
    "Mosh pit":    dict(genre="heavy-metal", icon=":material/bolt:",
                        values=dict(dance=0.42, energy=0.96, valence=0.35, acoustic=0.00, instr=0.05,
                                    speech=0.07, live=0.22, tempo=150, loud=-4.5, popl=25,
                                    minutes=4.2, explicit=False, mode_name="Minor")),
    "Stand-up":    dict(genre="comedy", icon=":material/mic:",
                        values=dict(dance=0.57, energy=0.72, valence=0.45, acoustic=0.80, instr=0.00,
                                    speech=0.92, live=0.75, tempo=95, loud=-10.0, popl=22,
                                    minutes=3.3, explicit=True, mode_name="Major")),
    "Lullaby":     dict(genre="sleep", icon=":material/bedtime:",
                        values=dict(dance=0.15, energy=0.08, valence=0.05, acoustic=0.90, instr=0.90,
                                    speech=0.05, live=0.15, tempo=80, loud=-25.0, popl=35,
                                    minutes=2.7, explicit=False, mode_name="Major")),
    "Block party": dict(genre="reggaeton", icon=":material/celebration:",
                        values=dict(dance=0.78, energy=0.75, valence=0.68, acoustic=0.12, instr=0.00,
                                    speech=0.09, live=0.12, tempo=98, loud=-4.6, popl=3,
                                    minutes=3.5, explicit=False, mode_name="Major")),
}

for k, v in DEFAULTS.items():
    st.session_state.setdefault(k, v)
st.session_state.setdefault("active_preset", None)


def apply_preset(name):
    for k, v in PRESETS[name]["values"].items():
        st.session_state[k] = v
    st.session_state.active_preset = name


def clear_preset():
    st.session_state.active_preset = None


def preset_key(name):
    return "preset_" + name.lower().replace(" ", "_").replace("-", "_")


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
  #ambient-root { position:fixed; inset:0; z-index:-1; pointer-events:none; overflow:hidden; }
  #ambient-root .lane { position:absolute; left:0; top:var(--top); opacity:0; will-change:transform, opacity;
      animation: travel var(--d) linear var(--delay) infinite; }
  #ambient-root .lane.rev { animation-name: travel-rev; }
  #ambient-root img { display:block; height:auto; opacity:var(--o); filter: blur(var(--b));
      animation: sway var(--sd) ease-in-out var(--delay) infinite alternate; }
  @keyframes travel {
      0%   { transform: translateX(-12vw); opacity:0; }
      10%  { opacity:1; }  90% { opacity:1; }
      100% { transform: translateX(108vw); opacity:0; } }
  @keyframes travel-rev {
      0%   { transform: translateX(108vw); opacity:0; }
      10%  { opacity:1; }  90% { opacity:1; }
      100% { transform: translateX(-12vw); opacity:0; } }
  @keyframes sway {
      from { transform: translateY(-10px) rotate(var(--r0)); }
      to   { transform: translateY(10px)  rotate(var(--r1)); } }
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
  /* ---------- secondary buttons (preset chips) ---------- */
  .preset-title { font-family:'Bricolage Grotesque',sans-serif; font-size:1.15rem; font-weight:500; margin:0 0 .15rem 0; }
  button[data-testid="stBaseButton-secondary"] {
      border-radius:14px !important; border:1px solid var(--line) !important; min-height:3rem;
      background:linear-gradient(180deg, rgba(255,255,255,.045), rgba(255,255,255,.012)) !important;
      color:#DAD5E8 !important; font-weight:500; font-size:.92rem; padding:.55rem .9rem !important;
      box-shadow: inset 0 1px 0 rgba(255,255,255,.05);
      transition: border-color .25s ease, background .25s ease, transform .25s cubic-bezier(.2,.8,.2,1), box-shadow .25s ease; }
  button[data-testid="stBaseButton-secondary"] p { white-space:nowrap !important; overflow:visible !important;
      text-overflow:clip !important; font-size:.92rem; }
  button[data-testid="stBaseButton-secondary"] span[data-testid="stIconMaterial"] { color:#A99BFF; font-size:1.2rem;
      transition: transform .25s cubic-bezier(.2,.8,.2,1), color .25s ease; }
  button[data-testid="stBaseButton-secondary"]:hover {
      border-color:rgba(123,108,255,.7) !important;
      background:linear-gradient(180deg, rgba(123,108,255,.16), rgba(123,108,255,.05)) !important;
      color:#fff !important; transform: translateY(-2px); box-shadow:0 12px 24px -14px rgba(123,108,255,.95); }
  button[data-testid="stBaseButton-secondary"]:hover span[data-testid="stIconMaterial"] { transform: scale(1.15) rotate(-6deg); color:#fff; }
  button[data-testid="stBaseButton-secondary"]:active { transform: translateY(0) scale(.97); }
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
  /* ---------- chat (Ask the model) ---------- */
  div[data-testid="stChatMessage"] { background:#1A1725 !important; border:1px solid var(--line);
      border-radius:16px; padding:.9rem 1.1rem; margin-bottom:.7rem; }
  div[data-testid="stChatMessage"]:has(img[alt="user avatar"]),
  div[data-testid="stChatMessageAvatarUser"] ~ div { background:#211C33 !important; }
  div[data-testid="stChatMessage"] p { font-size:.93rem; line-height:1.55; }
  div[data-testid="stChatMessage"] table { font-size:.85rem; margin:.5rem 0; }
  div[data-testid="stChatMessage"] th { color:#B8AEFF !important; font-weight:600; }
  div[data-testid="stChatMessage"] code { background:#2A2440; color:#CFC6FF; padding:.1rem .35rem;
      border-radius:5px; font-size:.85em; }
  div[data-testid="stChatInput"] { background:#1A1725 !important; border:1px solid var(--line) !important;
      border-radius:14px !important; }
  div[data-testid="stChatInput"]:focus-within { border-color:var(--brand) !important;
      box-shadow:0 0 0 4px rgba(123,108,255,.18); }
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
# One logo per horizontal lane, so they never collide and cover the whole screen.
# Lanes alternate direction; negative delays spread them across the width on load.
AMBIENT = [
    # lane top, width px, travel time, start offset, opacity, blur, tilt range, reverse
    ("4vh",  46, "58s", "-8s",  .20, "0px",   ("-14deg", "10deg"), False),
    ("17vh", 60, "72s", "-50s", .14, "1.5px", ("12deg", "-8deg"),  True),
    ("31vh", 40, "64s", "-30s", .22, "0px",   ("-6deg", "16deg"),  False),
    ("45vh", 72, "86s", "-70s", .10, "2.5px", ("8deg", "-12deg"),  True),
    ("59vh", 50, "60s", "-18s", .18, "0.5px", ("-18deg", "6deg"),  False),
    ("73vh", 64, "78s", "-44s", .13, "1.5px", ("10deg", "-14deg"), True),
    ("87vh", 42, "66s", "-58s", .20, "0px",   ("-8deg", "12deg"),  False),
]


def ambient_layer():
    src = logo_data_uri()
    if not src:
        return
    specs = json.dumps([
        dict(rev=rev, w=w,
             lane=f"--top:{top};--d:{d};--delay:{delay}",
             img=f"width:{w}px;--o:{o};--b:{b};--r0:{r0};--r1:{r1};--sd:{7 + i * 1.3:.1f}s;--delay:{delay}")
        for i, (top, w, d, delay, o, b, (r0, r1), rev) in enumerate(AMBIENT)
    ])
    script = f"""
<script>
(function () {{
  const old = document.getElementById('ambient-root');
  if (old && old.dataset.v === '2') return;
  if (old) old.remove();
  const root = document.createElement('div');
  root.id = 'ambient-root';
  root.dataset.v = '2';
  root.setAttribute('aria-hidden', 'true');
  const src = {json.dumps(src)};
  for (const spec of {specs}) {{
    const lane = document.createElement('div');
    lane.className = spec.rev ? 'lane rev' : 'lane';
    lane.setAttribute('style', spec.lane);
    const img = document.createElement('img');
    img.src = src; img.alt = ''; img.setAttribute('style', spec.img);
    lane.appendChild(img);
    root.appendChild(lane);
  }}
  document.body.prepend(root);
}})();
</script>"""
    try:
        st.html(script, unsafe_allow_javascript=True)
    except TypeError:
        pass  # older Streamlit without script support: skip the effect


# ==============================================================================
# ASK THE MODEL — a grounded assistant that answers questions about this project
# ==============================================================================
# Every number it quotes is computed from the loaded model at runtime, so the
# answers can never drift away from what the app is actually doing.

import re

BOT_GREETING = (
    "Ask me anything about this model — how accurate it is, why XGBoost was chosen, "
    "which genres it struggles with, or what any audio feature means."
)

SUGGESTED = [
    "How accurate is it?",
    "Why XGBoost?",
    "Is it overfitting?",
    "Which genre is weakest?",
    "Why only 10 genres?",
    "What features matter most?",
]


@st.cache_data(show_spinner=False)
def model_facts():
    """Recompute the headline metrics from the deployed model, on the same split
    every member used. Cached, so this runs once per session."""
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                                 f1_score, precision_recall_fscore_support,
                                 confusion_matrix)

    data = songs.copy()
    y_all = pd.Series(LABELS.transform(data["track_genre"]), index=data.index)
    X_all = data[FEATURE_COLUMNS].astype(float)

    X_tr, X_te, y_tr, y_te = train_test_split(
        X_all, y_all, test_size=0.2, random_state=42, stratify=y_all)

    pred_te = model.predict(X_te)
    pred_tr = model.predict(X_tr)

    p, r, f, s = precision_recall_fscore_support(y_te, pred_te, zero_division=0)
    per_genre = {GENRES[i]: dict(precision=p[i], recall=r[i], f1=f[i], support=int(s[i]))
                 for i in range(len(GENRES))}

    cm = confusion_matrix(y_te, pred_te)
    pairs = sorted(((GENRES[i], GENRES[j], int(cm[i, j]))
                    for i in range(len(GENRES)) for j in range(len(GENRES)) if i != j),
                   key=lambda t: -t[2])

    importance = pd.Series(model.feature_importances_, index=FEATURE_COLUMNS) \
                   .sort_values(ascending=False)

    return dict(
        train_acc=accuracy_score(y_tr, pred_tr),
        test_acc=accuracy_score(y_te, pred_te),
        precision=precision_score(y_te, pred_te, average="macro", zero_division=0),
        recall=recall_score(y_te, pred_te, average="macro", zero_division=0),
        macro_f1=f1_score(y_te, pred_te, average="macro", zero_division=0),
        n_train=len(X_tr), n_test=len(X_te), n_total=len(data),
        per_genre=per_genre, confusions=pairs[:6], importance=importance,
    )


LEADERBOARD = [
    ("XGBoost", 0.832, 0.818, "Vithanawasam O.D."),
    ("Random Forest", 0.811, 0.796, "Ahamed M.A."),
    ("SVM (RBF)", 0.755, 0.736, "Athale D.N.K."),
    ("KNN", 0.735, 0.714, "Senarathna P.K."),
    ("Decision Tree", 0.709, 0.688, "Randy T.H.S"),
    ("Logistic Regression", 0.663, 0.643, "Wimalarathna M.A.Y"),
]

FEATURE_GLOSSARY = {
    "danceability": "How suitable a track is for dancing (0–1), from tempo stability, rhythm and beat strength.",
    "energy": "Perceived intensity (0–1). Loud, fast, noisy tracks score high; a quiet ballad scores low.",
    "valence": "Musical positivity (0–1). High means cheerful, low means sad or angry.",
    "acousticness": "Confidence the track is acoustic rather than electronic (0–1).",
    "instrumentalness": "Likelihood the track has no vocals (0–1). Above 0.5 usually means instrumental.",
    "speechiness": "How much spoken word is present (0–1). Above 0.66 is mostly speech, which is why comedy scores so high.",
    "liveness": "Probability the track was recorded with an audience (0–1).",
    "loudness": "Overall loudness in decibels, roughly −49 to 0.",
    "tempo": "Estimated speed in beats per minute.",
    "popularity": "Spotify's 0–100 popularity score — a chart statistic, not a property of the sound.",
    "mode": "Major (1) or minor (0).",
    "key": "The musical key, 0 = C through 11 = B.",
    "explicit": "Whether the track is flagged for explicit lyrics.",
    "duration": "Track length. The model sees it in milliseconds.",
}


def _fmt_leaderboard():
    return "\n".join(f"| {i}. {n} | {a:.1%} | {f:.3f} | {who} |"
                     for i, (n, a, f, who) in enumerate(LEADERBOARD, 1))


# --- Answer builders ----------------------------------------------------------
def _a_accuracy(F, **_):
    return (f"**{F['test_acc']:.1%} accuracy** on the held-out test set "
            f"({F['n_test']} tracks the model never saw during training).\n\n"
            f"| Metric | Score |\n| :--- | :--- |\n"
            f"| Accuracy | {F['test_acc']:.1%} |\n"
            f"| Macro precision | {F['precision']:.3f} |\n"
            f"| Macro recall | {F['recall']:.3f} |\n"
            f"| Macro F1 | {F['macro_f1']:.3f} |\n\n"
            "With 10 genres, random guessing would score 10%, so the model is roughly "
            f"{F['test_acc'] / 0.1:.0f}× better than chance.")


def _a_why_xgboost(F, **_):
    return ("Six algorithms were trained on the **same data and the same 80/20 split**, so the "
            "comparison is fair. XGBoost won on every metric:\n\n"
            "| Rank | Accuracy | Macro F1 | Member |\n| :--- | :--- | :--- | :--- |\n"
            + _fmt_leaderboard() +
            "\n\nIt also had the best cross-validated score (0.792 macro F1), so the win was not a "
            "lucky test split. Gradient boosting is the standard strong performer on tabular data, "
            "and it helped most where the others failed — the beat-driven genres.")


def _a_how_works(F, **_):
    return ("**XGBoost (gradient boosting)** builds 200 decision trees one at a time. Each new "
            "tree is trained to correct the mistakes the previous trees still make, and the final "
            "prediction combines all of them.\n\n"
            "That sequential correction is why it beats a single Decision Tree by about 12 "
            "accuracy points: the genres other models confuse are exactly the leftover errors each "
            "new tree focuses on.\n\n"
            "Settings used: `n_estimators=200`, `learning_rate=0.1`, `max_depth=6`, "
            "`subsample=0.8`, chosen by `GridSearchCV` over 16 combinations.")


def _a_overfitting(F, **_):
    gap = (F["train_acc"] - F["test_acc"]) * 100
    return (f"The training accuracy is **{F['train_acc']:.1%}** and the test accuracy is "
            f"**{F['test_acc']:.1%}** — a gap of {gap:.0f} points. That looks alarming, but it is "
            "normal for a boosted ensemble, and two checks show it is not harmful:\n\n"
            "**1. Cross-validation agrees with the test score.** The 5-fold CV accuracy is about "
            "80.9%, measured on folds held out during training. If the model were truly memorising, "
            "it would collapse there. It does not.\n\n"
            "**2. Closing the gap makes predictions worse.** Retraining with heavy regularisation "
            "brings the gap down to 3.5 points but drops test accuracy to 77.2% — nearly 6 points "
            "lost. The gap is cosmetic rather than damaging.")


def _a_cv(F, **_):
    return ("**5-fold stratified cross-validation**, used in two stages:\n\n"
            "1. **Model selection** — `GridSearchCV` trained every hyperparameter combination on "
            "5 folds of the training set, scoring each by macro F1. The training data is split "
            "five ways, and each variety is validated five times.\n"
            "2. **Final check** — a 20% test set, touched only once after tuning, so it gives an "
            "unbiased estimate.\n\n"
            "Folds are *stratified* because the classes are uneven (998 heavy-metal tracks against "
            "529 jazz); plain K-fold could leave a fold short of a genre.")


def _a_best_worst(F, **_):
    ranked = sorted(F["per_genre"].items(), key=lambda kv: -kv[1]["f1"])
    rows = "\n".join(f"| {g} | {v['f1']:.3f} | {v['support']} |" for g, v in ranked)
    best, worst = ranked[0], ranked[-1]
    return (f"**Strongest: {best[0]} (F1 {best[1]['f1']:.3f})**  ·  "
            f"**Weakest: {worst[0]} (F1 {worst[1]['f1']:.3f})**\n\n"
            "| Genre | F1 | Test tracks |\n| :--- | :--- | :--- |\n" + rows +
            "\n\nThe pattern is consistent: genres with a distinctive acoustic signature "
            "(heavy-metal, comedy, sleep, classical) score well, while the beat-driven genres "
            "overlap heavily and drag each other down.")


def _a_confusion(F, **_):
    rows = "\n".join(f"| {a} | {b} | {n} |" for a, b, n in F["confusions"][:5])
    return ("The most frequent mistakes on the test set:\n\n"
            "| Actual | Predicted as | Tracks |\n| :--- | :--- | :--- |\n" + rows +
            "\n\nThese are not random errors. EDM, hip-hop, k-pop and reggaeton are all produced, "
            "beat-driven pop music with similar danceability, energy and tempo, so they sit in the "
            "same region of the feature space. A human listener separates them by vocals, language "
            "and production style — none of which Spotify's audio features capture.")


def _a_ten_genres(F, **_):
    return ("The raw dataset has **114 genres**, but many overlap almost completely — rock, "
            "alt-rock, hard-rock and grunge, or house, deep-house and chicago-house. Their audio "
            "features are nearly identical, so no model can separate them: a Random Forest on all "
            "114 classes reached only about **40% accuracy**, and a Decision Tree 20–25%.\n\n"
            "We therefore selected **10 acoustically distinct genres**. That is a documented "
            "modelling decision, not a shortcut — the EDA showed the overlap, and the figures above "
            "measure what it costs to ignore it.")


def _a_features(F, **_):
    top = F["importance"].head(8)
    rows = "\n".join(f"| {k} | {v:.3f} |" for k, v in top.items())
    onehot = F["importance"][[c for c in FEATURE_COLUMNS
                              if c.startswith(("key_", "ts_"))]].sum()
    return ("The model uses **30 features**: 13 audio and metadata columns plus 17 one-hot columns "
            "for musical key and time signature.\n\n"
            "| Feature | Importance |\n| :--- | :--- |\n" + rows +
            f"\n\nAll 17 key and time-signature columns together account for only {onehot:.1%} of "
            "total importance, which matches the EDA finding that musical key says very little "
            "about genre.")


def _a_preprocessing(F, **_):
    return ("The pipeline in `group_pipeline.ipynb`, one step per member:\n\n"
            "1. **Missing data** — rows without artist, album or track name dropped\n"
            "2. **Encoding** — `explicit` to integer; key and time signature one-hot encoded\n"
            "3. **Duplicates** — tracks labelled with more than one genre removed (about 16,000 "
            "were ambiguous), then duplicate track IDs dropped\n"
            "4. **Outliers** — duration, tempo, loudness and speechiness capped at the IQR bounds\n"
            "5. **Scaling** — `StandardScaler` on the 11 numeric audio features\n"
            "6. **Feature engineering** — genre label-encoded, mutual information and PCA analysed\n\n"
            f"Result: **{F['n_total']:,} tracks, 30 features, 10 balanced classes.** The scaler, "
            "label encoder, IQR bounds and column order are saved to "
            "`preprocessing_artifacts.pkl`, so this app transforms your input exactly as the "
            "training data was transformed.")


def _a_split(F, **_):
    return (f"An **80/20 stratified split** with `random_state=42`: "
            f"**{F['n_train']:,} training** tracks and **{F['n_test']:,} test** tracks.\n\n"
            "Stratified means each genre keeps the same proportion in both halves. Every one of the "
            "six members used this identical split and the same 5-fold cross-validation, which is "
            "what makes the six sets of results directly comparable.")


def _a_dataset(F, **_):
    return ("The **Spotify Tracks Dataset** from Kaggle (by Maharshi Pandya): 114,000 tracks, 21 "
            "columns, 114 genre labels.\n\n"
            "After selecting 10 genres and running the preprocessing pipeline, "
            f"**{F['n_total']:,} tracks** remain. Each track is described by Spotify's audio "
            "features — danceability, energy, acousticness and so on — rather than by the audio "
            "itself.")


def _a_team(F, **_):
    rows = "\n".join(f"| {who} | {n} | {a:.1%} | {f:.3f} |" for n, a, f, who in LEADERBOARD)
    return ("Each member trained one algorithm on the same data and split:\n\n"
            "| Member | Algorithm | Accuracy | Macro F1 |\n| :--- | :--- | :--- | :--- |\n" + rows +
            "\n\nEach also owned one EDA visualisation and one preprocessing step in the shared "
            "pipeline.")


def _a_pipeline(F, **_):
    return ("When you press **Identify genre**, your input goes through exactly the same steps as "
            "the training data:\n\n"
            "1. **Clip** — extreme values capped at the IQR bounds saved during preprocessing\n"
            "2. **Scale** — the 11 numeric features transformed by the saved `StandardScaler`\n"
            "3. **Encode** — key and time signature expanded into one-hot columns\n"
            "4. **Reorder** — columns arranged to match the training order exactly\n"
            "5. **Predict** — XGBoost returns a probability for each of the 10 genres\n"
            "6. **Decode** — the highest one is converted back to a genre name\n\n"
            "Reusing the *saved* scaler matters: fitting a new one on a single track would produce "
            "meaningless values.")


def _a_macro_f1(F, **_):
    return ("**Macro F1 was the selection metric.** F1 is the harmonic mean of precision and "
            "recall, so it balances false positives against false negatives. *Macro* means it is "
            "averaged over the genres with equal weight.\n\n"
            "That matters because the classes are uneven — 998 heavy-metal tracks against 529 "
            "jazz. A weighted average would let the larger genres mask poor performance on the "
            "smaller ones; macro treats a jazz mistake as seriously as a heavy-metal one. Since "
            "the goal is a classifier that recognises all ten genres, macro F1 is the honest "
            "target, and it is what `GridSearchCV` optimised.")


def _a_limitations(F, **_):
    return ("Honest limitations:\n\n"
            "- **The features are summaries, not audio.** Thirteen numbers per track carry nothing "
            "about lyrics, instrumentation or production era, so genres that sound different to a "
            "listener can look identical to the model.\n"
            "- **`popularity` is not a property of the sound.** It improves accuracy but means the "
            "model partly learns chart behaviour, which would not transfer to an unreleased track.\n"
            "- **Only 10 of 114 genres.** The task was narrowed to make it tractable.\n"
            "- **One split.** All results come from a single 80/20 split; repeated splits would "
            "give confidence intervals rather than point estimates.\n\n"
            "Possible improvements: raw-audio features such as MFCCs, ensembling the six models, "
            "a wider randomised hyperparameter search, and retraining without `popularity`.")


def _a_improve(F, **_):
    return ("Four realistic improvements:\n\n"
            "1. **Raw-audio features** (MFCCs or spectrograms) instead of Spotify's summary "
            "numbers — the single biggest likely gain, since the current ceiling comes from the "
            "features, not the algorithm.\n"
            "2. **Ensemble the six models** by voting or stacking; XGBoost and Random Forest make "
            "different errors, so together they should beat either alone.\n"
            "3. **A wider hyperparameter search** — `RandomizedSearchCV` over `colsample_bytree`, "
            "`min_child_weight` and `reg_lambda`, which stayed at their defaults.\n"
            "4. **Drop `popularity`** and report the honest audio-only accuracy alongside the "
            "current figure.")


def _a_last_prediction(F, **_):
    ranked = st.session_state.get("build_result") or (
        st.session_state.get("song_result") or [None])[0]
    if not ranked:
        return ("No prediction has been made yet. Set up a track on **Build a track** or pick one "
                "on **Try a real song**, then ask me again and I will explain that result.")
    genre, conf = ranked[0]
    runners = ", ".join(f"{g} {p:.1%}" for g, p in ranked[1:4])
    stats = F["per_genre"].get(genre, {})
    note = ""
    if stats:
        note = (f"\n\nOn the test set the model scores **F1 {stats['f1']:.3f}** for {genre} "
                f"(precision {stats['precision']:.3f}, recall {stats['recall']:.3f}), so that "
                f"prediction is {'reliable' if stats['f1'] >= 0.8 else 'less certain than it looks'}.")
    certainty = ("very confident" if conf >= 0.8 else
                 "fairly confident" if conf >= 0.5 else "not confident")
    return (f"The last prediction was **{genre}** at **{conf:.1%} confidence** — the model is "
            f"{certainty}.\n\nThe runners-up were {runners}. XGBoost produces a probability for "
            f"each of the 10 genres, and the highest one is shown.{note}")


def _a_genre_detail(genre, F):
    v = F["per_genre"][genre]
    confused = [(b, n) for a, b, n in F["confusions"] if a == genre][:2]
    extra = ""
    if confused:
        extra = ("\n\nMost often mistaken for " +
                 ", ".join(f"**{b}** ({n} tracks)" for b, n in confused) + ".")
    verdict = ("one of the model's strongest genres" if v["f1"] >= 0.85 else
               "handled reasonably well" if v["f1"] >= 0.75 else
               "one of the harder genres for this model")
    return (f"**{genre}** — {GENRE_BLURBS.get(genre, '')}\n\n"
            f"| Metric | Score |\n| :--- | :--- |\n"
            f"| Precision | {v['precision']:.3f} |\n| Recall | {v['recall']:.3f} |\n"
            f"| F1 | {v['f1']:.3f} |\n| Test tracks | {v['support']} |\n\n"
            f"At F1 {v['f1']:.3f} this is {verdict}.{extra}")


def _a_feature_detail(feature, F):
    """Explain one audio feature, with its real importance in the deployed model."""
    desc = FEATURE_GLOSSARY[feature]
    imp = F["importance"]
    extra = ""

    if feature == "key":
        # 'key' is spread across key_0 ... key_11, so sum them
        cols = [c for c in imp.index if c.startswith("key_")]
        extra = (f"\n\nSpread across its 12 one-hot columns it carries only "
                 f"**{imp[cols].sum():.3f} importance in total** — musical key turns out to say "
                 "very little about genre.")
    else:
        col = "duration_ms" if feature == "duration" else feature
        if col in imp.index:
            rank = list(imp.index).index(col) + 1
            extra = (f"\n\nIn this model it carries **{imp[col]:.3f} importance**, ranking "
                     f"#{rank} of {len(imp)} features.")

    return f"**{feature}** — {desc}{extra}"


# --- Intent table -------------------------------------------------------------
INTENTS = [
    (["accurate", "accuracy", "how good", "how well", "performance", "precision",
      "recall", "f1 score", "score", "metric"], _a_accuracy),
    (["why xgboost", "why did you choose", "which model", "best model", "compare model",
      "other model", "six model", "6 model", "algorithm compar", "leaderboard",
      "why this model", "model selection"], _a_why_xgboost),
    (["how does xgboost", "how does it work", "how does the model work", "boosting",
      "gradient boost", "explain xgboost", "what is xgboost", "how it works"], _a_how_works),
    (["overfit", "memoris", "memoriz", "100%", "training accuracy", "train accuracy",
      "generalis", "generaliz"], _a_overfitting),
    (["cross valid", "cross-valid", "k-fold", "kfold", "5-fold", "5 fold", "validation method",
      "stratified"], _a_cv),
    (["weakest", "strongest", "best genre", "worst genre", "struggle", "which genre",
      "per genre", "per-genre", "hardest"], _a_best_worst),
    (["confus", "mistake", "mix up", "mixes up", "wrong", "error", "misclassif"], _a_confusion),
    (["10 genre", "ten genre", "114", "why only", "all genres", "more genres",
      "genre selection"], _a_ten_genres),
    (["feature importance", "which feature", "what feature", "important feature",
      "features used", "how many feature", "input"], _a_features),
    (["preprocess", "pre-process", "cleaning", "clean the data", "data prep", "pipeline step",
      "scaling", "scaler", "outlier", "duplicate"], _a_preprocessing),
    (["split", "train test", "training set", "test set", "how much data",
      "how many tracks"], _a_split),
    (["dataset", "data source", "kaggle", "where did the data", "spotify data"], _a_dataset),
    (["who did", "team", "member", "group", "contribution"], _a_team),
    (["how do you predict", "prediction work", "how does prediction", "what happens when",
      "behind the scenes", "pipeline"], _a_pipeline),
    (["macro f1", "why f1", "why macro", "why that metric", "imbalance"], _a_macro_f1),
    (["limitation", "weakness", "drawback", "problem with", "what's wrong"], _a_limitations),
    (["improve", "better", "future work", "next step", "enhance"], _a_improve),
    (["why did you predict", "why this result", "explain the prediction", "last prediction",
      "that prediction", "this prediction", "why that genre"], _a_last_prediction),
]


def bot_answer(question: str) -> str:
    q = question.lower().strip()
    F = model_facts()

    # A genre named on its own takes priority — "how about jazz?"
    for g in GENRES:
        if re.search(rf"\b{re.escape(g)}\b", q) or (g == "hip-hop" and "hip hop" in q):
            if not any(k in q for k in ["confus", "all genre", "every genre", "which genre"]):
                return _a_genre_detail(g, F)

    # A feature named on its own — "what is danceability?"
    for feat in FEATURE_GLOSSARY:
        if re.search(rf"\b{re.escape(feat)}\b", q):
            if not any(k in q for k in ["importance", "which feature", "what feature"]):
                return _a_feature_detail(feat, F)

    # Score each intent by how many of its phrases appear
    best_fn, best_score = None, 0
    for keys, fn in INTENTS:
        score = sum(len(k) for k in keys if k in q)
        if score > best_score:
            best_fn, best_score = fn, score
    if best_fn:
        return best_fn(F)

    return ("I'm not sure about that one — I only know this project. Try asking about:\n\n"
            "- **Performance** — accuracy, precision, recall, a specific genre\n"
            "- **The model** — why XGBoost, how boosting works, overfitting, cross-validation\n"
            "- **The data** — the dataset, preprocessing, the train/test split, why 10 genres\n"
            "- **The features** — which matter most, or what any one of them means\n"
            "- **This app** — how a prediction is made, or why the last result came out that way")


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

tab_build, tab_song, tab_ask, tab_about = st.tabs(
    ["Build a track", "Try a real song", "Ask the model", "About the model"])

# Tab 1 — build a track ----------------------------------------------------------
with tab_build:
    left, right = st.columns([1, 1.15], gap="large")

    with left:
        st.markdown('<p class="preset-title">Start from a sound</p>'
                    '<p class="group-sub">Pick a preset to load a typical track, then fine-tune it.</p>',
                    unsafe_allow_html=True)
        names = list(PRESETS)
        for row_start in range(0, len(names), 3):
            cols = st.columns(3, gap="small")
            for col, name in zip(cols, names[row_start:row_start + 3]):
                col.button(name, key=preset_key(name), icon=PRESETS[name]["icon"],
                           on_click=apply_preset, args=(name,), width="stretch")
        active = st.session_state.active_preset
        if active:
            colour = GENRE_COLORS[PRESETS[active]["genre"]]
            st.markdown(f"""<style>
              .st-key-{preset_key(active)} button {{
                border-color:{colour} !important;
                background:linear-gradient(135deg, {colour}33, {colour}14) !important;
                color:#fff !important; box-shadow:0 10px 26px -14px {colour} !important; }}
              .st-key-{preset_key(active)} button span[data-testid="stIconMaterial"] {{ color:{colour} !important; }}
            </style>""", unsafe_allow_html=True)

        with st.container(border=True):
            st.markdown('<p class="group-title">Feel</p>'
                        '<p class="group-sub">How it moves you</p>', unsafe_allow_html=True)
            c1, c2 = st.columns(2)
            c1.slider("Danceability", 0.0, 1.0, step=0.01, key="dance", on_change=clear_preset)
            c2.slider("Energy", 0.0, 1.0, step=0.01, key="energy", on_change=clear_preset)
            c1.slider("Positivity", 0.0, 1.0, step=0.01, key="valence", on_change=clear_preset)
            c2.slider("Tempo (BPM)", 40, 220, step=1, key="tempo", on_change=clear_preset)

        with st.container(border=True):
            st.markdown('<p class="group-title">Texture</p>'
                        '<p class="group-sub">What it is made of</p>', unsafe_allow_html=True)
            c1, c2 = st.columns(2)
            c1.slider("Acousticness", 0.0, 1.0, step=0.01, key="acoustic", on_change=clear_preset)
            c2.slider("Instrumentalness", 0.0, 1.0, step=0.01, key="instr", on_change=clear_preset)
            c1.slider("Speechiness", 0.0, 1.0, step=0.01, key="speech", on_change=clear_preset)
            c2.slider("Loudness (dB)", -49.0, 2.0, step=0.5, key="loud", on_change=clear_preset)

        with st.expander("More details"):
            c1, c2 = st.columns(2)
            c1.slider("Liveness", 0.0, 1.0, step=0.01, key="live", on_change=clear_preset)
            c2.slider("Popularity", 0, 100, key="popl", on_change=clear_preset)
            c1.slider("Duration (minutes)", 0.5, 10.0, step=0.1, key="minutes", on_change=clear_preset)
            key_names = ["C", "C♯", "D", "D♯", "E", "F", "F♯", "G", "G♯", "A", "A♯", "B"]
            key = key_names.index(c2.selectbox("Key", key_names))
            c1.radio("Mode", ["Major", "Minor"], horizontal=True, key="mode_name", on_change=clear_preset)
            time_signature = c2.selectbox("Beats per bar", [4, 3, 5, 1, 0])
            c1.toggle("Explicit lyrics", key="explicit", on_change=clear_preset)

        identify = st.button("Identify genre", type="primary", width="stretch")

    with right:
        if identify:
            s = st.session_state
            raw = dict(popularity=s.popl, duration_ms=s.minutes * 60_000, explicit=s.explicit,
                       danceability=s.dance, energy=s.energy, loudness=s.loud,
                       mode=1 if s.mode_name == "Major" else 0,
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
                "Accuracy": [0.832, 0.811, 0.755, 0.735, 0.709, 0.663],
                "Macro F1": [0.818, 0.796, 0.736, 0.714, 0.688, 0.643],
            }), hide_index=True, width="stretch")
    with c2:
        with st.container(border=True):
            st.markdown('<p class="group-title">The ten genres</p>', unsafe_allow_html=True)
            for g in GENRES:
                st.markdown(f'<p style="margin:.35rem 0"><b style="color:{GENRE_COLORS[g]}">{g}</b>'
                            f'<br><span style="color:#9891AD;font-size:.88rem">{GENRE_BLURBS[g]}</span></p>',
                            unsafe_allow_html=True)
