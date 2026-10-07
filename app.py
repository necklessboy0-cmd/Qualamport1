import os

from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from google import genai
from google.genai import types
from pydantic import BaseModel

app = FastAPI()  # Vercel looks for this top-level `app`

DEFAULT_MODEL = os.getenv("GEMINI_MODEL", "gemini-3.5-flash-lite")

BASE_SYSTEM = (
    "You are an expert senior software engineer. Write correct, complete, "
    "runnable, well-structured code with sensible error handling. Put every "
    "file in its own fenced code block with the language tag. Keep "
    "explanations short: how to run it, plus key notes. If the request is "
    "ambiguous, make a reasonable assumption and state it. When the user asks "
    "for changes, return the full updated code, not fragments."
)


class Msg(BaseModel):
    role: str
    content: str


class ChatReq(BaseModel):
    messages: list[Msg]
    lang: str = "Auto-detect"
    mode: str = "vibe"


@app.post("/api/chat")
def chat(req: ChatReq):
    key = os.getenv("GEMINI_API_KEY")
    if not key:
        raise HTTPException(500, "GEMINI_API_KEY is not set on the server.")

    system = BASE_SYSTEM
    if req.lang != "Auto-detect":
        system += f" Use {req.lang} unless the user explicitly asks otherwise."
    msgs = req.messages if req.mode == "vibe" else req.messages[-1:]
    if req.mode == "vibe":
        system += (
            " The user is vibe coding: they describe ideas loosely and iterate. "
            "Build on the previous code and apply their latest request."
        )

    contents = [
        types.Content(
            role="model" if m.role == "assistant" else "user",
            parts=[types.Part(text=m.content)],
        )
        for m in msgs
    ]
    try:
        client = genai.Client(api_key=key)
        res = client.models.generate_content(
            model=DEFAULT_MODEL,
            contents=contents,
            config=types.GenerateContentConfig(
                system_instruction=system, max_output_tokens=8000
            ),
        )
        return {"text": res.text or ""}
    except Exception as e:
        raise HTTPException(500, str(e))


@app.get("/", response_class=HTMLResponse)
def home():
    return HTML


HTML = r"""<!doctype html>
<html lang="en"><head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>AI Code Studio</title>
<style>
:root{--bg:#0f1115;--card:#171a21;--bd:#262b36;--tx:#e6e8ee;--mu:#8b93a7;--ac:#6ea8fe}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--tx);font:15px/1.5 system-ui,sans-serif;display:flex;flex-direction:column;height:100vh}
header{padding:10px 14px;border-bottom:1px solid var(--bd);display:flex;gap:8px;flex-wrap:wrap;align-items:center}
header h1{font-size:16px;margin:0 auto 0 0}
select,button,textarea{background:var(--card);color:var(--tx);border:1px solid var(--bd);border-radius:8px;padding:7px 10px;font:inherit}
button{cursor:pointer}button:hover{border-color:var(--ac)}
#chat{flex:1;overflow:auto;padding:14px;display:flex;flex-direction:column;gap:12px}
.m{max-width:900px;width:100%;margin:0 auto}
.u{background:var(--card);border:1px solid var(--bd);border-radius:10px;padding:10px 12px;white-space:pre-wrap}
.a .t{white-space:pre-wrap;padding:4px 2px}
.code{border:1px solid var(--bd);border-radius:10px;margin:8px 0;overflow:hidden}
.code .bar{display:flex;gap:6px;align-items:center;background:var(--card);padding:5px 8px;color:var(--mu);font-size:12px}
.code .bar span{margin-right:auto}
.code .bar button{padding:3px 8px;font-size:12px}
pre{margin:0;padding:10px;overflow:auto;background:#0b0d11;font:13px/1.45 ui-monospace,Menlo,monospace}
iframe{width:100%;height:360px;border:0;background:#fff}
footer{padding:10px 14px;border-top:1px solid var(--bd);display:flex;gap:8px}
textarea{flex:1;resize:none;height:48px}
#send{background:var(--ac);color:#0b0d11;border:0;font-weight:600}
.err{color:#ff8a8a}
</style></head><body>
<header>
  <h1>💻 AI Code Studio</h1>
  <select id="lang"></select>
  <select id="mode"><option value="vibe">Vibe coding</option><option value="prompt">Prompt</option></select>
  <button id="clear">Clear</button>
</header>
<div id="chat"><div class="m a"><div class="t">Describe what you want to build in any language. In Vibe coding mode, keep chatting to refine it.</div></div></div>
<footer>
  <textarea id="inp" placeholder="e.g. Build a to-do app in HTML, then add dark mode..."></textarea>
  <button id="send">Send</button>
</footer>
<script>
const LANGS=["Auto-detect","Python","JavaScript","TypeScript","HTML/CSS","Java","C","C++","C#","Go","Rust","PHP","Ruby","Swift","Kotlin","Dart","SQL","Bash","PowerShell","R","MATLAB","Lua","Perl","Scala","Haskell","Solidity","VBA"];
const EXT={python:"py",javascript:"js",js:"js",typescript:"ts",html:"html",css:"css",java:"java",c:"c",cpp:"cpp","c++":"cpp",csharp:"cs","c#":"cs",go:"go",rust:"rs",php:"php",ruby:"rb",swift:"swift",kotlin:"kt",dart:"dart",sql:"sql",bash:"sh",sh:"sh",powershell:"ps1",r:"r",matlab:"m",lua:"lua",perl:"pl",scala:"scala",haskell:"hs",solidity:"sol",vba:"bas"};
const $=id=>document.getElementById(id);
const chat=$("chat"),inp=$("inp"),sendBtn=$("send");
let messages=[];
LANGS.forEach(l=>{const o=document.createElement("option");o.textContent=l;$("lang").appendChild(o)});

function el(tag,cls,txt){const e=document.createElement(tag);if(cls)e.className=cls;if(txt!=null)e.textContent=txt;return e}

function render(text){
  const box=el("div","m a");
  const re=/```(\w*)\n([\s\S]*?)```/g;
  let last=0,m;
  while((m=re.exec(text))){
    if(m.index>last)box.appendChild(el("div","t",text.slice(last,m.index)));
    const lang=m[1]||"text",code=m[2];
    const wrap=el("div","code"),bar=el("div","bar"),pre=el("pre");
    bar.appendChild(el("span",null,lang));
    const copy=el("button",null,"Copy");
    copy.onclick=()=>{navigator.clipboard.writeText(code);copy.textContent="Copied";setTimeout(()=>copy.textContent="Copy",1200)};
    const dl=el("button",null,"Download");
    dl.onclick=()=>{const a=document.createElement("a");a.href=URL.createObjectURL(new Blob([code]));a.download="code."+(EXT[lang.toLowerCase()]||"txt");a.click()};
    bar.append(copy,dl);
    if(lang.toLowerCase()==="html"){
      const pv=el("button",null,"Preview");
      pv.onclick=()=>{
        const old=wrap.querySelector("iframe");
        if(old){old.remove();return}
        const f=document.createElement("iframe");
        f.setAttribute("sandbox","allow-scripts");f.srcdoc=code;wrap.appendChild(f);
      };
      bar.appendChild(pv);
    }
    pre.textContent=code;
    wrap.append(bar,pre);box.appendChild(wrap);
    last=re.lastIndex;
  }
  if(last<text.length)box.appendChild(el("div","t",text.slice(last)));
  chat.appendChild(box);chat.scrollTop=chat.scrollHeight;
}

async function send(){
  const text=inp.value.trim();
  if(!text||sendBtn.disabled)return;
  inp.value="";
  messages.push({role:"user",content:text});
  chat.appendChild(el("div","m u",text));
  const wait=el("div","m a");wait.appendChild(el("div","t","Thinking..."));
  chat.appendChild(wait);chat.scrollTop=chat.scrollHeight;
  sendBtn.disabled=true;
  try{
    const r=await fetch("/api/chat",{method:"POST",headers:{"Content-Type":"application/json"},
      body:JSON.stringify({messages,lang:$("lang").value,mode:$("mode").value})});
    const d=await r.json();
    wait.remove();
    if(!r.ok)throw new Error(d.detail||"Request failed");
    messages.push({role:"assistant",content:d.text});
    render(d.text);
  }catch(e){
    wait.remove();
    const b=el("div","m a");b.appendChild(el("div","t err","Error: "+e.message));chat.appendChild(b);
    messages.pop();
  }
  sendBtn.disabled=false;
}
sendBtn.onclick=send;
inp.addEventListener("keydown",e=>{if(e.key==="Enter"&&!e.shiftKey){e.preventDefault();send()}});
$("clear").onclick=()=>{messages=[];chat.innerHTML=""};
</script></body></html>
"""
