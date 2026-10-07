import os
import re

import streamlit as st
import streamlit.components.v1 as components
from google import genai
from google.genai import types

st.set_page_config(page_title="AI Code Studio", page_icon="💻", layout="wide")

DEFAULT_MODEL = "gemini-3.5-flash-lite"

LANGS = [
    "Auto-detect", "Python", "JavaScript", "TypeScript", "HTML/CSS", "Java",
    "C", "C++", "C#", "Go", "Rust", "PHP", "Ruby", "Swift", "Kotlin", "Dart",
    "SQL", "Bash", "PowerShell", "R", "MATLAB", "Lua", "Perl", "Scala",
    "Haskell", "Solidity", "VBA",
]
EXT = {
    "python": "py", "javascript": "js", "js": "js", "typescript": "ts",
    "html": "html", "css": "css", "java": "java", "c": "c", "cpp": "cpp",
    "c++": "cpp", "csharp": "cs", "c#": "cs", "go": "go", "rust": "rs",
    "php": "php", "ruby": "rb", "swift": "swift", "kotlin": "kt",
    "dart": "dart", "sql": "sql", "bash": "sh", "sh": "sh",
    "powershell": "ps1", "r": "r", "matlab": "m", "lua": "lua",
    "perl": "pl", "scala": "scala", "haskell": "hs", "solidity": "sol",
    "vba": "bas",
}

BASE_SYSTEM = (
    "You are an expert senior software engineer. Write correct, complete, "
    "runnable, well-structured code with sensible error handling. Put every "
    "file in its own fenced code block with the language tag. Keep "
    "explanations short: how to run it, plus key notes. If the request is "
    "ambiguous, make a reasonable assumption and state it. When the user asks "
    "for changes, return the full updated code, not fragments."
)


def system_prompt(lang: str, mode: str) -> str:
    s = BASE_SYSTEM
    if lang != "Auto-detect":
        s += f" Use {lang} unless the user explicitly asks for another language."
    if mode == "Vibe coding":
        s += (
            " The user is vibe coding: they describe ideas loosely and iterate. "
            "Build on the previous code, keep what works, and apply their latest "
            "request."
        )
    return s


def to_contents(messages):
    """Convert chat history to Gemini format (assistant -> 'model')."""
    return [
        types.Content(
            role="model" if m["role"] == "assistant" else "user",
            parts=[types.Part(text=m["content"])],
        )
        for m in messages
    ]


def last_code(text: str):
    blocks = re.findall(r"```(\w*)\n(.*?)```", text, re.S)
    return blocks[-1] if blocks else None


# ---------- Sidebar ----------
with st.sidebar:
    st.title("💻 AI Code Studio")
    api_key = st.text_input(
        "Gemini API key",
        type="password",
        value=os.getenv("GEMINI_API_KEY", ""),
        help="Get a key at aistudio.google.com/apikey",
    )
    model = st.text_input("Model", value=DEFAULT_MODEL)
    lang = st.selectbox("Language", LANGS)
    mode = st.radio(
        "Mode",
        ["Vibe coding", "Prompt"],
        help="Vibe coding keeps the whole conversation so you can iterate. "
        "Prompt treats each request as a fresh task.",
    )
    max_tokens = st.slider("Max output tokens", 1000, 32000, 8000, 500)
    if st.button("🗑️ Clear chat", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

if "messages" not in st.session_state:
    st.session_state.messages = []

st.header("Describe it. Get working code.")
st.caption("Any language, from a plain prompt or a loose idea you refine by chatting.")

# ---------- History ----------
for m in st.session_state.messages:
    with st.chat_message(m["role"]):
        st.markdown(m["content"])

# ---------- Input ----------
user_input = st.chat_input("e.g. Build a snake game in Python, then make it faster...")

if user_input:
    if not api_key:
        st.error("Add your Gemini API key in the sidebar.")
        st.stop()

    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    history = (
        st.session_state.messages
        if mode == "Vibe coding"
        else [{"role": "user", "content": user_input}]
    )

    client = genai.Client(api_key=api_key)
    with st.chat_message("assistant"):
        box = st.empty()
        full = ""
        try:
            stream = client.models.generate_content_stream(
                model=model.strip() or DEFAULT_MODEL,
                contents=to_contents(history),
                config=types.GenerateContentConfig(
                    system_instruction=system_prompt(lang, mode),
                    max_output_tokens=max_tokens,
                ),
            )
            for chunk in stream:
                if chunk.text:
                    full += chunk.text
                    box.markdown(full + "▌")
            box.markdown(full)
        except Exception as e:
            full = ""
            box.error(f"API error: {e}")

    if full:
        st.session_state.messages.append({"role": "assistant", "content": full})
        st.rerun()

# ---------- Actions on latest answer ----------
if st.session_state.messages and st.session_state.messages[-1]["role"] == "assistant":
    found = last_code(st.session_state.messages[-1]["content"])
    if found:
        tag, code = found
        ext = EXT.get(tag.lower(), "txt")
        c1, c2 = st.columns(2)
        c1.download_button(
            "⬇️ Download code", code, file_name=f"code.{ext}", use_container_width=True
        )
        if tag.lower() == "html":
            with st.expander("▶️ Live preview", expanded=True):
                components.html(code, height=500, scrolling=True)
