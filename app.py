import json
import re
import sqlite3
from pathlib import Path
from typing import Any, Dict, List, Tuple

import pandas as pd
import streamlit as st
from deep_translator import GoogleTranslator
import time
from deep_translator import exceptions

st.set_page_config(page_title="JSON Translator Pro", layout="wide")

# ------------------------------
# Config / Paths
# ------------------------------
APP_DIR = Path(".")
DB_PATH = APP_DIR / "translations.db"
GLOSSARY_PATH = APP_DIR / "glossary.json"

# ------------------------------
# SQLite Cache
# ------------------------------
def init_db():
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute(
        '''
        CREATE TABLE IF NOT EXISTS translations (
            source_text TEXT NOT NULL,
            target_lang TEXT NOT NULL,
            translated_text TEXT NOT NULL,
            PRIMARY KEY (source_text, target_lang)
        )
        '''
    )
    conn.commit()
    return conn

def cache_get(conn, source_text: str, target_lang: str):
    cur = conn.cursor()
    cur.execute(
        "SELECT translated_text FROM translations WHERE source_text=? AND target_lang=?",
        (source_text, target_lang),
    )
    row = cur.fetchone()
    return row[0] if row else None

def cache_put(conn, source_text: str, target_lang: str, translated_text: str):
    cur = conn.cursor()
    cur.execute(
        "INSERT OR REPLACE INTO translations (source_text, target_lang, translated_text) VALUES (?, ?, ?)",
        (source_text, target_lang, translated_text),
    )
    conn.commit()

# ------------------------------
# Glossary
# ------------------------------
def load_glossary() -> Dict[str, str]:
    if GLOSSARY_PATH.exists():
        try:
            with open(GLOSSARY_PATH, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return {}
    return {}

def save_glossary(glossary: Dict[str, str]):
    with open(GLOSSARY_PATH, "w", encoding="utf-8") as f:
        json.dump(glossary, f, ensure_ascii=False, indent=2)

# ------------------------------
# Translator
# ------------------------------
_translators: Dict[str, GoogleTranslator] = {}

def get_translator(target: str) -> GoogleTranslator:
    if target not in _translators:
        _translators[target] = GoogleTranslator(source="en", target=target)
    return _translators[target]

'''def mask_terms(text: str, terms: Tuple[str, ...]):
    """
    Mask preserve-terms before translation (case-insensitive exact word match).
    Returns masked text + placeholder map.
    """
    placeholders: Dict[str, str] = {}
    masked = text
    for i, term in enumerate(terms):
        if not term:
            continue
        placeholder = f"__TERM_{i}__"
        masked = re.sub(rf"\b{re.escape(term)}\b", placeholder, masked, flags=re.IGNORECASE)
        placeholders[placeholder] = term
    return masked, placeholders

def restore_placeholders(text: str, placeholders: Dict[str, str]) -> str:
    for p, term in placeholders.items():
        text = text.replace(p, term)
    return text
'''
import base64

def _b64_encode(s: str) -> str:
    return base64.b64encode(s.encode("utf-8")).decode("ascii")

def _b64_decode(s: str) -> str:
    return base64.b64decode(s.encode("ascii")).decode("utf-8")

def mask_terms(text: str, terms: Tuple[str, ...]) -> Tuple[str, Dict[str,str]]:
    """
    Mask preserve-terms before translation using base64-wrapped placeholders.
    Returns masked text and a map {placeholder: original_term}.
    """
    placeholders: Dict[str, str] = {}
    masked = text
    for i, term in enumerate(terms):
        term = term.strip()
        if not term:
            continue
        # build safe placeholder using base64 so translator won't corrupt it
        b64 = _b64_encode(term)
        placeholder = f"@@{b64}@@"
        # replace whole-word matches ignoring case
        masked = re.sub(rf"\b{re.escape(term)}\b", placeholder, masked, flags=re.IGNORECASE)
        placeholders[placeholder] = term
    return masked, placeholders

def restore_placeholders(text: str, placeholders: Dict[str,str]) -> str:
    """
    Restore placeholders of form @@<base64>@@ back to original terms.
    Works even if translator keeps case or minor spacing around.
    """
    restored = text

    # First, replace exact placeholders if present
    for placeholder, original in placeholders.items():
        if placeholder in restored:
            restored = restored.replace(placeholder, original)

    # Then catch any base64 tokens preserved but maybe with slight spacing or punctuation.
    # Find any @@...@@ pattern and try to decode; if decoded value exists in placeholders values, replace.
    matches = re.findall(r"@@([A-Za-z0-9+/=]+)@@", restored)
    for b64 in matches:
        try:
            dec = _b64_decode(b64)
        except Exception:
            continue
        # Replace all occurrences of @@b64@@ (exact) with decoded term
        restored = restored.replace(f"@@{b64}@@", dec)

    # As a last defense: if translator inserted spaces inside the placeholder like "@@ b64 @@" or changed separators,
    # try to catch patterns like "@@\s*b64\s*@@"
    def replace_loose(m):
        b64 = re.sub(r"\s+", "", m.group(1))
        try:
            dec = _b64_decode(b64)
            return dec
        except Exception:
            return m.group(0)  # leave as-is

    restored = re.sub(r"@@\s*([A-Za-z0-9+/=\s]+)\s*@@", replace_loose, restored)

    return restored

def translate_string(s: str, target: str, terms_to_preserve: Tuple[str, ...], conn: sqlite3.Connection) -> str:
    if not s:
        return s
    cached = cache_get(conn, s, target)
    if cached:
        return cached
    masked, placeholders = mask_terms(s, terms_to_preserve)
    try:
        translated_masked = get_translator(target).translate(masked)
    except Exception as e:
        st.warning(f"Translation error: {e}")
        translated_masked = s
    translated = restore_placeholders(translated_masked, placeholders)
    cache_put(conn, s, target, translated)
    return translated
    


def translate_with_retry(text, target, terms_to_preserve, conn, retries=5, delay=2):
    for attempt in range(retries):
        try:
            return translate_string(text, target, terms_to_preserve, conn)
        except Exception as e:
            if attempt < retries - 1:
                time.sleep(delay)
            else:
                st.warning(f"Translation failed for snippet: {e}")
                return text  # fallback to original
# ------------------------------
# JSON flatten / unflatten
# ------------------------------
def flatten_json(obj: Any, parent_key: str = "", sep: str = ".") -> Dict[str, Any]:
    items: Dict[str, Any] = {}
    if isinstance(obj, dict):
        for k, v in obj.items():
            new_key = f"{parent_key}{sep}{k}" if parent_key else k
            items.update(flatten_json(v, new_key, sep=sep))
    elif isinstance(obj, list):
        for idx, v in enumerate(obj):
            new_key = f"{parent_key}[{idx}]"
            items.update(flatten_json(v, new_key, sep=sep))
    else:
        items[parent_key] = obj
    return items

def parse_path(path: str) -> List[Any]:
    parts: List[Any] = []
    buf = ""
    i = 0
    while i < len(path):
        ch = path[i]
        if ch == '.':
            if buf:
                parts.append(buf)
                buf = ""
            i += 1
        elif ch == '[':
            if buf:
                parts.append(buf)
                buf = ""
            j = path.find(']', i)
            idx = int(path[i+1:j])
            parts.append(idx)
            i = j + 1
        else:
            buf += ch
            i += 1
    if buf:
        parts.append(buf)
    return parts

def ensure_list_size(lst: List[Any], size: int):
    while len(lst) <= size:
        lst.append(None)

def unflatten_json(flat: Dict[str, Any]) -> Any:
    root: Any = {}
    for path, value in flat.items():
        tokens = parse_path(path)
        cur = root
        for i, token in enumerate(tokens):
            last = i == len(tokens) - 1
            if isinstance(token, str):
                if last:
                    if not isinstance(cur, dict):
                        # convert to dict if needed
                        # If it's a list, this is a structural conflict; in our input we avoid such conflicts.
                        pass
                    if not isinstance(cur, dict):
                        cur = {}
                    cur[token] = value
                else:
                    nxt = tokens[i+1]
                    if isinstance(nxt, int):
                        # next is list
                        if token not in cur or not isinstance(cur[token], list):
                            cur[token] = []
                        cur = cur[token]
                    else:
                        if token not in cur or not isinstance(cur[token], dict):
                            cur[token] = {}
                        cur = cur[token]
            else:  # list index
                if not isinstance(cur, list):
                    # create list if missing
                    # If cur is dict, it means previous step created a list holder in a key
                    if isinstance(cur, dict):
                        # This path only occurs when previous token ensured list at cur = cur[token]
                        pass
                    else:
                        cur = []
                ensure_list_size(cur, token)
                if last:
                    cur[token] = value
                else:
                    if cur[token] is None:
                        # decide next container by peeking
                        nxt = tokens[i+1]
                        cur[token] = [] if isinstance(nxt, int) else {}
                    cur = cur[token]
    # If root contains a single "_list" key pattern we won't use it here; our builder stores directly.
    return root

# ------------------------------
# Translation pipeline
# ------------------------------
def translate_flat_map(flat_map: Dict[str, Any], target: str, terms_to_preserve: Tuple[str, ...], conn: sqlite3.Connection, glossary: Dict[str, str]) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for path, val in flat_map.items():
        if isinstance(val, str):
            key_name = path.split(".")[-1]
            if key_name in glossary:
                out[path] = glossary[key_name]
            else:
                out[path] = translate_with_retry(val, target, terms_to_preserve, conn)

        else:
            out[path] = val
    return out

# ------------------------------
# UI
# ------------------------------
#st.title("💡 AI-Powered JSON Translator — Multi-Language + Glossary Management")
st.markdown("""
    <div style='background-color: #1E1E1E; padding: 20px; border-radius: 8px; text-align: center;'>
        <h2 style='color: #FFD700;'> AI-Powered JSON Translator</h2>
        <h4 style='color: #FFFFFF;'>Multilingual + Glossary Management</h4>
    </div>
""", unsafe_allow_html=True)
#st.image("logo.png", width=100)


with st.sidebar:
    st.header("Settings")
    languages_map = {
        "Arabic (ar)": "ar",
        "Dutch (nl)": "nl",
        "French (fr)": "fr",
        "Farsi (fa)": "fa",
        "German (de)": "de",
        "Greek (el)": "el",
        "Hindi (hi)": "hi",
        "Indonesian (id)": "id",
        "Italian (it)": "it",
        "Japanese (ja)": "ja",
        "Korean (ko)": "ko",
        "Malay (ms)": "ms",
        "Mandarin Chinese (zh-CN)": "zh-CN",
        "Mandarin (Singapore)": "zh-CN",
        "Polish (pl)": "pl",
        "Portuguese (pt)": "pt",
        "Romanian (ro)": "ro",
        "Russian (ru)": "ru",
        "Spanish (es)": "es", 
        "Turkish (tr)": "tr",
        "Ukrainian (uk)": "uk",
        "Urdu (ur)": "ur"
        
            
    }
    Lang_Fallback={
         "zh-SG": "zh-CN",
         "pt-BR":"pt"
    }

    target_lang_labels = list(languages_map.keys())
    chosen_labels = st.multiselect("Target languages", target_lang_labels, default=["Malay (ms)"])
    chosen_targets = [languages_map[l] for l in chosen_labels]

    preserve_terms_input = st.text_input("Preserve terms (comma-separated)", "AML,KYC,OTP,MyKad")
    terms_to_preserve = tuple([t.strip() for t in preserve_terms_input.split(",") if t.strip()])

    conn = init_db()
    st.markdown("---")
    st.caption(f"Cache DB: {DB_PATH}")

st.header("Upload JSON File")
uploaded = st.file_uploader("Upload English JSON File To Translate!", type=["json"])


st.header("Glossary Management")
glossary = load_glossary()
gloss_df = pd.DataFrame([{"key": k, "translation": v} for k, v in glossary.items()]) if glossary else pd.DataFrame(columns=["key", "translation"])
edited_gloss = st.data_editor(gloss_df, num_rows="dynamic", use_container_width=True, key="glossary_editor")
if st.button("Save Glossary"):
    new_gloss = {}
    for _, row in edited_gloss.iterrows():
        k = str(row.get("key", "")).strip()
        v = str(row.get("translation", "")).strip()
        if k:
            new_gloss[k] = v
    save_glossary(new_gloss)
    st.success("Glossary saved.")
    glossary = new_gloss

st.header(" Click 👇 ")
colA, colB = st.columns([1,1])
with colA:
    run_btn = st.button("Run Translation")
with colB:
    review_lang = st.selectbox("Language to review", options=chosen_targets if chosen_targets else ["ms"])

original_flat = None
translated_by_lang: Dict[str, Dict[str, Any]] = {}

if uploaded:
    try:
        raw = uploaded.read().decode("utf-8")
        original = json.loads(raw)
        st.subheader("Uploaded JSON (full)")
        st.markdown(f"""
        <div style='
        max-height: 300px;        /* limit height (adjust to 250-400px as needed) */
        overflow-y: auto;         /* add scrollbar if content is long */
        border: 1px solid #00ff00;
        border-radius: 8px;
        background-color: #000000; /* dark background */
        color: #00ff00;           /* green text like your screenshot */
        padding: 10px;
        font-family: monospace;
        white-space: pre-wrap;
        '>
        {raw}
        </div>
        """,
        unsafe_allow_html=True
    )

        original_flat = flatten_json(original)
    except Exception as e:
        st.error(f"Could not parse JSON: {e}")

if uploaded and run_btn and original_flat and chosen_targets:
    with st.spinner("Translating..."):
        for tgt in chosen_targets:
            translated_flat = translate_flat_map(original_flat, tgt, terms_to_preserve, conn, glossary)
            translated_by_lang[tgt] = translated_flat
        st.success("Translation completed.")

    # 4) Review table for selected language
    if review_lang in translated_by_lang:
        ai_flat = translated_by_lang[review_lang]
        rows = []
        for path, val in original_flat.items():
            if isinstance(val, str):
                rows.append({
                    "path": path,
                    "key": path.split(".")[-1],
                    "original": val,
                    "translated": ai_flat.get(path, val),
                    "final": ai_flat.get(path, val),
                })
        review_df = pd.DataFrame(rows)
        st.header(f"Review & Adjust ({review_lang})")
        st.caption("Edit the 'final' column, then click 'Save Final & Download'")
        edited = st.data_editor(
            review_df,
            num_rows="fixed",
            use_container_width=True,
            column_config={
                "path": st.column_config.Column(disabled=True),
                "key": st.column_config.Column(disabled=True),
                "original": st.column_config.Column(disabled=True),
                "translated": st.column_config.Column(disabled=True),
            },
            key="review_editor"
        )

        final_flat = {}
        for _, row in edited.iterrows():
            final_flat[row["path"]] = row["final"]
        for path, val in original_flat.items():
            if not isinstance(val, str):
                final_flat[path] = val

        try:
            final_json_obj = unflatten_json(final_flat)
            final_pretty = json.dumps(final_json_obj, ensure_ascii=False, indent=2)
        except Exception as e:
            st.error(f"Rebuild error: {e}")
            final_pretty = None

        col1, col2 = st.columns([1,1])
        with col1:
            if final_pretty:
                st.subheader(f"Final JSON Preview ({review_lang})")
                st.markdown(f"""
                <div style='
                max-height: 400px;
                overflow-y: auto;
                border: 1px solid #00ff00;
                border-radius: 8px;
                padding: 10px;
                background-color: #000000;  /* dark background */
                color: #00ff00;             /* light text for visibility */
                font-family: monospace;
                white-space: pre-wrap;
            '>
            {final_pretty}
            </div>
            """,
            unsafe_allow_html=True
            )

        with col2:
            if final_pretty:
                st.download_button(
                    label=f"Download reviewed_{review_lang}.json",
                    data=final_pretty,
                    file_name=f"reviewed_{review_lang}.json",
                    mime="application/json",
                )

    st.header("Bulk Download")
    for tgt in chosen_targets:
        flat = translated_by_lang[tgt]
        try:
            obj = unflatten_json(flat)
            pretty = json.dumps(obj, ensure_ascii=False, indent=2)
            st.download_button(
                label=f"Download translated_{tgt}.json",
                data=pretty,
                file_name=f"translated_{tgt}.json",
                mime="application/json",
                key=f"download_button_{tgt}"
            )
        except Exception as e:
            st.warning(f"Could not build JSON for {tgt}: {e}")

st.markdown("---")
st.caption("Tip: For very large files, rely on the download buttons even if the preview feels heavy. Keys remain unchanged; only string values are translated.")
