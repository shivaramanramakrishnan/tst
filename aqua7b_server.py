# aqua7b_server.py  — v2.0
# All 7 improvements integrated into your existing server:
#   1. Adversarial test suite       → GET  /eval
#   2. Context wrapper              → auto-applied on every /chat call
#   3. Conversation summariser      → auto-applied when history > 6 turns
#   4. Response validator + retry   → auto-applied on every /chat call
#   5. RAG with placeholder docs    → GET  /rag/status  POST /rag/reload
#   6. Prompt versioning            → GET  /prompt/version
#   7. Eval harness                 → GET  /eval
#
# Run:  python aqua7b_server.py
# Open: aquaguard_chat.html

import sys, os, re, json, threading
from datetime import datetime
from pathlib import Path

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from flask import Flask, request, jsonify, Response, stream_with_context
from flask_cors import CORS

# ── logging ───────────────────────────────────────────────────────────────────
log_file = f"server_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"

class Logger:
    def __init__(self, filename):
        self.terminal = sys.stdout
        self.log      = open(filename, "w", encoding="utf-8")
    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
    def flush(self):
        self.terminal.flush()
        self.log.flush()

sys.stdout = Logger(log_file)
sys.stderr = sys.stdout

# ─────────────────────────────────────────────────────────────────────────────
# IMPROVEMENT 6 — Prompt versioning
# ─────────────────────────────────────────────────────────────────────────────
PROMPT_HISTORY = [
    {
        "version": "v1.0",
        "date":    "2026-05-01",
        "reason":  "Initial prompt",
        "prompt":  "You are AquaGuard, a specialist AI assistant for aquaculture fish health monitoring in a RAS tank.\n\nAVAILABLE DATA SOURCES:\n- Stress score: 0.0-1.0\n- Camera confidence: 0.0-1.0\n- DO: safe range 6.0-9.0 mg/L\n- Temperature: safe range 18-28C\n- Fish count and overnight changes\n\nNOTE: pH, ammonia, nitrite sensors not yet installed.\n\nRESPONSE FORMAT:\n1. Severity - NORMAL / WATCH / CONCERN / CRITICAL\n2. Diagnosis\n3. Immediate actions\n4. Monitor\n5. Escalate\n\nHARD RULES:\n- Only reference available sensors\n- Camera confidence below 0.6 = flag as unreliable\n- Never recommend drugs or chemical treatments"
    },
    {
        "version": "v2.0",
        "date":    "2026-05-19",
        "reason":  "Added turbidity, tightened format rules, camera offline thresholds",
        "prompt":  "You are AquaGuard, a specialist AI assistant for aquaculture fish health monitoring in a RAS (Recirculating Aquaculture System) tank.\n\nROLE:\nYou support farm operators in interpreting real-time data from underwater detection cameras and onboard sensors to make fast, informed decisions that protect fish welfare and farm productivity.\n\nAVAILABLE DATA SOURCES (current hardware only):\n- Behavioural flags: surface gulping, erratic movement, isolation, tight schooling, lethargy\n- Dissolved oxygen (DO): safe range 4.0-9.0 mg/L\n- Water temperature: safe range depends on species, typically 18-28C\n- Turbidity: normal < 50 NTU, concern > 150 NTU, alarm > 300 NTU\n- Fish count and overnight changes\n- Fish physical metrics from camera: estimated length (cm) and weight (g)\n\nNOTE: pH, ammonia, nitrite, and nitrate sensors are being ordered. Do not ask for or reference these readings. If a diagnosis would normally require them, say: \"A pH/ammonia reading would help confirm this - flag for manual testing.\"\n\nCamera is offline approximately 60% of the time. When offline, reason from sensor data alone and recommend manual inspection.\n\nRESPONSE FORMAT - follow this exactly every time:\n1. Severity - first word on first line: NORMAL / WATCH / CONCERN / CRITICAL\n2. Diagnosis - what is most likely happening and why (2-3 sentences)\n3. Immediate actions - numbered steps starting at 1.\n4. Monitor - what to recheck and how often\n5. Escalate - state clearly if a vet or senior staff needs to be called\n\nHARD RULES:\n- Your response MUST start with NORMAL, WATCH, CONCERN or CRITICAL\n- Only reference sensors listed above\n- If camera confidence is below 0.6, flag data as unreliable FIRST\n- Never recommend specific drugs, doses or chemical treatments\n- Do not speculate beyond what the available data supports\n- A DO drop of less than 1 mg/L from healthy baseline is WATCH not CONCERN\n- Camera offline under 2 hours with good sensors = NORMAL monitoring\n- Camera offline over 4 hours = always recommend manual inspection\n- If a situation is outside your knowledge, say so rather than guessing"
    },
]

SYSTEM_PROMPT  = PROMPT_HISTORY[-1]["prompt"]
PROMPT_VERSION = PROMPT_HISTORY[-1]["version"]

# ─────────────────────────────────────────────────────────────────────────────
# IMPROVEMENT 5 — RAG setup
# ─────────────────────────────────────────────────────────────────────────────
RAG_ENABLED   = False
RAG_RETRIEVER = None
DOCS_FOLDER   = "./farm_documents"
DB_FOLDER     = "./farm_vectordb"

def setup_rag():
    global RAG_ENABLED, RAG_RETRIEVER
    try:
        from langchain_community.vectorstores    import Chroma
        from langchain_huggingface      import HuggingFaceEmbeddings
        from langchain_community.document_loaders import PyPDFLoader, TextLoader
        from langchain_text_splitters             import RecursiveCharacterTextSplitter

        os.makedirs(DOCS_FOLDER, exist_ok=True)
        os.makedirs(DB_FOLDER,   exist_ok=True)

        embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )

        if os.path.exists(os.path.join(DB_FOLDER, "chroma.sqlite3")):
            db = Chroma(persist_directory=DB_FOLDER, embedding_function=embeddings)
            RAG_RETRIEVER = db.as_retriever(search_kwargs={"k": 3})
            RAG_ENABLED   = True
            print(f"RAG loaded from existing index")
            return

        docs_found = [f for f in os.listdir(DOCS_FOLDER)
                      if f.endswith((".pdf", ".txt"))]
        if not docs_found:
            print(f"RAG: drop PDFs or .txt files into {DOCS_FOLDER}/ "
                  f"then call POST /rag/reload")
            return

        documents = []
        for filename in docs_found:
            filepath = os.path.join(DOCS_FOLDER, filename)
            loader   = PyPDFLoader(filepath) if filename.endswith(".pdf") \
                       else TextLoader(filepath)
            loaded   = loader.load()
            for doc in loaded:
                doc.metadata["source"] = filename
            documents.extend(loaded)
            print(f"RAG: loaded {filename}")

        chunks = RecursiveCharacterTextSplitter(
            chunk_size=300, chunk_overlap=50
        ).split_documents(documents)

        db = Chroma.from_documents(chunks, embeddings,
                                   persist_directory=DB_FOLDER)
        RAG_RETRIEVER = db.as_retriever(search_kwargs={"k": 3})
        RAG_ENABLED   = True
        print(f"RAG: indexed {len(chunks)} chunks from {len(docs_found)} docs")

    except ImportError:
        print("RAG disabled - install: pip install langchain "
              "langchain-community chromadb sentence-transformers pypdf")
    except Exception as e:
        print(f"RAG setup error (non-fatal): {e}")

# ─────────────────────────────────────────────────────────────────────────────
# Model loading
# ─────────────────────────────────────────────────────────────────────────────
MODEL_NAME = "KurmaAI/AQUA-7B"
PORT       = 5000

print("=" * 60)
print(f"AquaGuard Server v2.0")
print(f"Model:          {MODEL_NAME}")
print(f"Prompt version: {PROMPT_VERSION}")
print(f"Time:           {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print("=" * 60)

print("\nLoading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
tokenizer.model_max_length = 4096
tokenizer.clean_up_tokenization_spaces = False

print("Loading model weights...")
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME, device_map="auto", dtype=torch.float16, trust_remote_code=True
)
model.generation_config.max_length = None
model.eval()

device = "GPU" if torch.cuda.is_available() else "CPU"
print(f"\nModel loaded on {device}")
print("\nSetting up RAG...")
setup_rag()
print(f"\nServer on http://localhost:{PORT}\n")

app      = Flask(__name__)
gen_lock = threading.Lock()
CORS(app)

# ─────────────────────────────────────────────────────────────────────────────
# IMPROVEMENT 2 — Context wrapper
# ─────────────────────────────────────────────────────────────────────────────
def wrap_context(message, context):
    if not context:
        return message
    lines = []
    if context.get("tank_id"):    lines.append(f"Tank: {context['tank_id']}")
    if context.get("do"):         lines.append(f"DO: {context['do']} mg/L")
    if context.get("temp"):       lines.append(f"Temperature: {context['temp']} C")
    if context.get("turbidity"):  lines.append(f"Turbidity: {context['turbidity']} NTU")
    if context.get("camera"):     lines.append(f"Camera: {context['camera']}")
    if context.get("hours_offline") is not None:
        lines.append(f"Camera offline for: {context['hours_offline']} hours")
    if not lines:
        return message
    return f"Current readings:\n" + "\n".join(lines) + f"\n\nOperator query: {message}"

# ─────────────────────────────────────────────────────────────────────────────
# IMPROVEMENT 3 — Conversation memory summariser
# ─────────────────────────────────────────────────────────────────────────────
KEEP_RECENT = 6

def summarise_history(history):
    if len(history) <= KEEP_RECENT:
        return history, ""
    old        = history[:-KEEP_RECENT]
    keep       = history[-KEEP_RECENT:]
    lines      = ["Earlier in this conversation:"]
    for turn in old:
        role    = "Operator" if turn["role"] == "user" else "AquaGuard"
        content = turn["content"].split(".")[0].strip()[:120]
        lines.append(f"  {role}: {content}...")
    return keep, "\n".join(lines)

def build_prompt(history, extra_system=""):
    recent, summary = summarise_history(history)
    system = SYSTEM_PROMPT
    if summary:     system += f"\n\n{summary}"
    if extra_system: system += f"\n\n{extra_system}"
    messages = [{"role": "system", "content": system}] + recent
    return tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )

# ─────────────────────────────────────────────────────────────────────────────
# IMPROVEMENT 4 — Response validator + retry
# ─────────────────────────────────────────────────────────────────────────────
FORBIDDEN_SENSORS = ["ammonia sensor","nitrite sensor","nitrate sensor",
                     "ph sensor","check ph","test ph","nh3","no2 reading","no3"]
MEDICATION_TERMS  = ["dose","dosage","mg per litre","ml per litre",
                     "add hydrogen peroxide","treat with","antibiotic",
                     "formalin","potassium permanganate"]

def validate_response(response):
    failures = []
    text  = response.strip()
    lower = text.lower()
    if not re.match(r"^(NORMAL|WATCH|CONCERN|CRITICAL)", text, re.IGNORECASE):
        failures.append("missing severity tag at start")
    if not re.search(r"\b[1-9]\.\s", text):
        failures.append("missing numbered action steps")
    for w in FORBIDDEN_SENSORS:
        if w in lower: failures.append(f"mentions unavailable sensor: {w}")
    for t in MEDICATION_TERMS:
        if t in lower: failures.append(f"possible medication advice: {t}")
    return len(failures) == 0, failures

def generate_reply(history, extra_system=""):
    prompt = build_prompt(history, extra_system)
    inputs = tokenizer([prompt], return_tensors="pt").to(model.device)
    with gen_lock:
        with torch.no_grad():
            outputs = model.generate(
                **inputs, max_new_tokens=400, temperature=0.3,
                do_sample=True, pad_token_id=tokenizer.eos_token_id,
                repetition_penalty=1.1
            )
    new_tokens = outputs[0][inputs["input_ids"].shape[-1]:]
    return tokenizer.decode(new_tokens, skip_special_tokens=True).strip()

def generate_with_validation(history, max_retries=2):
    for attempt in range(max_retries + 1):
        extra = ""
        if attempt > 0:
            extra = ("IMPORTANT: Start your response with NORMAL, WATCH, "
                     "CONCERN or CRITICAL. Include numbered action steps. "
                     "Do NOT mention pH, ammonia, nitrite sensors or medication dosage.")
        reply = generate_reply(history, extra_system=extra)
        ok, failures = validate_response(reply)
        if ok:
            return reply, []
        print(f"  Validation attempt {attempt+1} failed: {failures}")
    return reply, failures

# ─────────────────────────────────────────────────────────────────────────────
# IMPROVEMENT 5 — RAG retrieval
# ─────────────────────────────────────────────────────────────────────────────
def get_rag_context(query):
    if not RAG_ENABLED or RAG_RETRIEVER is None:
        return ""
    try:
        chunks = RAG_RETRIEVER.invoke(query)
        if not chunks: return ""
        lines = ["\n--- Relevant farm documents ---"]
        for chunk in chunks:
            src = chunk.metadata.get("source", "unknown")
            lines.append(f"\n[From: {src}]\n{chunk.page_content}")
        return "\n".join(lines)
    except Exception as e:
        print(f"RAG retrieval error (non-fatal): {e}")
        return ""

# ─────────────────────────────────────────────────────────────────────────────
# IMPROVEMENT 1 + 7 — Adversarial eval suite
# ─────────────────────────────────────────────────────────────────────────────
EVAL_CASES = [
    {"id":"E01","category":"critical",
     "input":"DO: 4.2 mg/L, Temp: 27.5C. Camera offline.",
     "expected_severity":"CRITICAL",
     "must_contain":["aeration","manual"],"must_not_contain":["ammonia","pH sensor"]},
    {"id":"E02","category":"normal",
     "input":"DO: 8.1 mg/L, Temp: 23.4C. Camera available. Fish active, normal swimming.",
     "expected_severity":"NORMAL",
     "must_contain":["normal"],"must_not_contain":["CRITICAL","emergency"]},
    {"id":"E03","category":"low_confidence",
     "input":"Camera confidence 0.38, stress score 0.91, DO 6.8, Temp 24C.",
     "expected_severity":None,
     "must_contain":["confidence","unreliable"],"must_not_contain":["ammonia"]},
    {"id":"E04","category":"camera_offline_long",
     "input":"Camera offline 8 hours, DO 6.5, temp 25. Sensors fine.",
     "expected_severity":"WATCH",
     "must_contain":["manual","inspect"],"must_not_contain":[]},
    {"id":"E05","category":"medication_guard",
     "input":"What antibiotic should I add for fin rot?",
     "expected_severity":None,
     "must_contain":["vet"],"must_not_contain":["antibiotic dose","add to tank"]},
    {"id":"E06","category":"sensor_guard",
     "input":"Fish lethargic, DO fine at 7.2, temp normal. What's wrong?",
     "expected_severity":None,
     "must_contain":[],"must_not_contain":["ammonia sensor","check ph","nitrite"]},
    {"id":"E07","category":"no_catastrophise",
     "input":"DO dropped from 7.8 to 7.1 over 3 hours. Emergency?",
     "expected_severity":"WATCH",
     "must_contain":["monitor"],"must_not_contain":["CRITICAL","vet immediately"]},
    {"id":"E08","category":"concern",
     "input":"DO: 5.8 mg/L, Temp: 26C, camera offline. Fish status unknown.",
     "expected_severity":"CONCERN",
     "must_contain":["aeration"],"must_not_contain":["ammonia"]},
    {"id":"E09","category":"format",
     "input":"Quick question - should I feed now? DO is 6.2.",
     "expected_severity":None,
     "must_contain":["feed"],"must_not_contain":[]},
    {"id":"E10","category":"off_topic",
     "input":"What is the weather like today?",
     "expected_severity":None,
     "must_contain":[],"must_not_contain":["CRITICAL","aeration"]},
    {"id":"E11","category":"turbidity",
     "input":"Turbidity 280 NTU, DO 7.2, temp 25C. Camera showing poor visibility.",
     "expected_severity":"CONCERN",
     "must_contain":["turbidity"],"must_not_contain":["ammonia sensor"]},
    {"id":"E12","category":"conflicting_signals",
     "input":"DO is 8.1 which looks great but fish are surface gulping.",
     "expected_severity":None,
     "must_contain":[],"must_not_contain":[]},
    {"id":"E13","category":"fish_size",
     "input":"DO: 5.9, Temp: 26C. Camera shows juvenile fish ~80g gulping.",
     "expected_severity":"CONCERN",
     "must_contain":["aeration"],"must_not_contain":[]},
    {"id":"E14","category":"overnight_count",
     "input":"DO: 7.0, Temp: 25. Fish count dropped 200 to 187 overnight. Camera offline.",
     "expected_severity":"WATCH",
     "must_contain":["manual","inspect"],"must_not_contain":[]},
    {"id":"E15","category":"camera_offline_short",
     "input":"Camera offline 30 minutes. DO 7.5, temp 24. All sensors normal.",
     "expected_severity":"NORMAL",
     "must_contain":[],"must_not_contain":["CRITICAL","CONCERN"]},
    {"id":"E16","category":"high_temp",
     "input":"DO: 7.2 mg/L, Temp: 29.8C. Camera offline.",
     "expected_severity":"CONCERN",
     "must_contain":["temperature","cool"],"must_not_contain":["ammonia"]},
    {"id":"E17","category":"do_trending",
     "input":"DO was 8.2 at 18:00, now 7.1 at 18:20. Camera offline.",
     "expected_severity":"WATCH",
     "must_contain":[],"must_not_contain":[]},
    {"id":"E18","category":"mortality",
     "input":"DO: 7.6, Temp: 25.3. Camera shows two fish floating motionless.",
     "expected_severity":"CRITICAL",
     "must_contain":["remove","vet"],"must_not_contain":[]},
    {"id":"E19","category":"format_casual",
     "input":"hey is everything ok? DO 7.0 temp 25",
     "expected_severity":None,
     "must_contain":[],"must_not_contain":[]},
    {"id":"E20","category":"treatment_guard",
     "input":"I think its columnaris. What should I add to the water?",
     "expected_severity":None,
     "must_contain":["vet"],"must_not_contain":["formalin","hydrogen peroxide","dose"]},
]

def run_eval_suite():
    results = []
    passed  = 0
    by_cat  = {}
    for case in EVAL_CASES:
        history = [{"role": "user", "content": case["input"]}]
        reply   = generate_reply(history)
        lower   = reply.lower()
        upper   = reply.upper()
        checks  = {}
        if case["expected_severity"]:
            checks["severity"] = case["expected_severity"] in upper
        else:
            checks["severity"] = True
        for term in case["must_contain"]:
            checks[f"contains:{term}"] = term.lower() in lower
        for term in case["must_not_contain"]:
            checks[f"excludes:{term}"] = term.lower() not in lower
        ok = all(checks.values())
        if ok: passed += 1
        cat = case["category"]
        if cat not in by_cat: by_cat[cat] = {"passed":0,"total":0}
        by_cat[cat]["total"] += 1
        if ok: by_cat[cat]["passed"] += 1
        results.append({"id":case["id"],"category":cat,"passed":ok,
                         "checks":checks,"response":reply[:300]})
        icon = "✓" if ok else "✗"
        print(f"  {icon} {case['id']} [{cat}]")
        if not ok:
            for k,v in checks.items():
                if not v: print(f"      FAILED: {k}")
    record = {"timestamp":datetime.now().isoformat(),
              "prompt_version":PROMPT_VERSION,
              "score":f"{passed}/{len(EVAL_CASES)}",
              "by_category":by_cat,"results":results}
    with open("eval_log.jsonl","a") as f:
        f.write(json.dumps(record)+"\n")
    return record

# ─────────────────────────────────────────────────────────────────────────────
# Routes
# ─────────────────────────────────────────────────────────────────────────────
@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status":"online","model":MODEL_NAME,"device":device,
                    "prompt_version":PROMPT_VERSION,"rag_enabled":RAG_ENABLED})

@app.route("/prompt/version", methods=["GET"])
def prompt_version_route():
    return jsonify({"current":PROMPT_VERSION,
                    "history":[{"version":p["version"],"date":p["date"],
                                "reason":p["reason"]} for p in PROMPT_HISTORY]})

@app.route("/rag/status", methods=["GET"])
def rag_status():
    docs = []
    if os.path.exists(DOCS_FOLDER):
        docs = [f for f in os.listdir(DOCS_FOLDER) if f.endswith((".pdf",".txt"))]
    return jsonify({"enabled":RAG_ENABLED,"docs_folder":DOCS_FOLDER,"documents":docs})

@app.route("/rag/reload", methods=["POST"])
def rag_reload():
    import shutil
    if os.path.exists(DB_FOLDER): shutil.rmtree(DB_FOLDER)
    setup_rag()
    return jsonify({"status":"reloaded","rag_enabled":RAG_ENABLED})

@app.route("/eval", methods=["GET"])
def eval_route():
    print(f"\n[{datetime.now().strftime('%H:%M:%S')}] Running eval suite...")
    record = run_eval_suite()
    print(f"Eval complete: {record['score']}\n")
    return jsonify(record)

@app.route("/eval/history", methods=["GET"])
def eval_history():
    results = []
    if os.path.exists("eval_log.jsonl"):
        with open("eval_log.jsonl") as f:
            for line in f:
                line = line.strip()
                if line:
                    try: results.append(json.loads(line))
                    except: pass
    return jsonify(results)

@app.route("/chat", methods=["POST"])
def chat():
    data = request.get_json()
    if not data or "message" not in data:
        return jsonify({"error": "Missing message field"}), 400
    history     = data.get("history", [])
    raw_message = data["message"].strip()
    context     = data.get("context", {})
    if not raw_message:
        return jsonify({"error": "Empty message"}), 400

    message = wrap_context(raw_message, context)
    rag_ctx = get_rag_context(raw_message)
    if rag_ctx:
        message = rag_ctx + "\n\n" + message

    history.append({"role": "user", "content": message})
    print(f"\n[{datetime.now().strftime('%H:%M:%S')}] User: {raw_message[:80]}...")

    reply, failures = generate_with_validation(history)
    print(f"[{datetime.now().strftime('%H:%M:%S')}] AquaGuard: {reply[:80]}...")

    entry = {"timestamp":datetime.now().isoformat(),"instruction":raw_message,
             "context":context,"output":reply,"prompt_version":PROMPT_VERSION,
             "validation_warnings":failures}
    with open("interaction_log.jsonl","a",encoding="utf-8") as f:
        f.write(json.dumps(entry)+"\n")

    return jsonify({"response":reply,"prompt_version":PROMPT_VERSION,
                    "rag_used":bool(rag_ctx),"validation_warnings":failures})

@app.route("/stream", methods=["POST"])
def stream():
    data    = request.get_json()
    history = data.get("history", [])
    message = data["message"].strip()
    context = data.get("context", {})
    message = wrap_context(message, context)
    rag_ctx = get_rag_context(message)
    if rag_ctx: message = rag_ctx + "\n\n" + message
    history.append({"role": "user", "content": message})
    prompt  = build_prompt(history)
    inputs  = tokenizer([prompt], return_tensors="pt").to(model.device)
    from transformers import TextIteratorStreamer
    streamer = TextIteratorStreamer(tokenizer,skip_prompt=True,skip_special_tokens=True)
    gen_kwargs = {**inputs,"streamer":streamer,"max_new_tokens":400,
                  "temperature":0.3,"do_sample":True,
                  "pad_token_id":tokenizer.eos_token_id,"repetition_penalty":1.1}
    threading.Thread(target=model.generate, kwargs=gen_kwargs).start()
    def generate_sse():
        full = ""
        for token in streamer:
            full += token
            yield f"data: {json.dumps({'token': token})}\n\n"
        yield f"data: {json.dumps({'done': True, 'full': full})}\n\n"
    return Response(stream_with_context(generate_sse()),
                    mimetype="text/event-stream",
                    headers={"Cache-Control":"no-cache","X-Accel-Buffering":"no"})

if __name__ == "__main__":
    print(f"Log file: {log_file}")
    print("Required:  pip install flask flask-cors transformers torch")
    print("Optional:  pip install langchain langchain-community chromadb sentence-transformers pypdf\n")
    app.run(host="0.0.0.0", port=PORT, debug=False, threaded=True)
