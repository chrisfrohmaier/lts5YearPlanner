import os
import io
import json
import base64
import datetime
from pathlib import Path
from flask import Flask, request, jsonify, send_file, render_template_string
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

# Optional MongoDB support
try:
    from pymongo import MongoClient
except Exception:
    MongoClient = None

app = Flask(__name__)
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
DATA_DIR.mkdir(exist_ok=True)

MONGO_URI = os.environ.get("MONGO_URI")  # e.g. "mongodb://user:pass@host:port/db"
MONGO_DB = os.environ.get("MONGO_DB", "sky_app")
MONGO_COLLECTION = os.environ.get("MONGO_COLLECTION", "json_inputs")
mongo_client = None
if MONGO_URI and MongoClient:
    try:
        mongo_client = MongoClient(MONGO_URI, serverSelectionTimeoutMS=3000)
        # quick check
        mongo_client.server_info()
    except Exception:
        mongo_client = None


INDEX_HTML = """
<!doctype html>
<html>
<head>
  <meta charset="utf-8">
  <title>Sky JSON Editor & Plotter</title>
  <style>
    body { font-family: Arial, sans-serif; margin: 20px; }
    #editor { width: 100%; height: 400px; border: 1px solid #ccc; }
    #controls { margin-top: 10px; }
    button { margin-right: 8px; padding: 8px 12px; }
    img { max-width: 100%; margin-top: 12px; border: 1px solid #ddd; }
    #message { margin-top: 8px; color: green; }
  </style>
  <link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/jsoneditor@9.7.3/dist/jsoneditor.min.css">
</head>
<body>
  <h2>Sky JSON Editor & Plotter</h2>
  <div id="editor"></div>
  <div id="controls">
    <button id="saveRepo">Save to Repository</button>
    <button id="saveRemote">Save to Remote DB</button>
    <button id="generatePlot">Generate Sky Plot</button>
  </div>
  <div id="message"></div>
  <div id="plotContainer"><img id="plotImg" src="" alt="Sky plot will appear here"></div>

  <script src="https://cdn.jsdelivr.net/npm/jsoneditor@9.7.3/dist/jsoneditor.min.js"></script>
  <script>
    const container = document.getElementById("editor");
    const options = {
      mode: 'code',
      mainMenuBar: false,
      navigationBar: true
    };
    const editor = new JSONEditor(container, options, {
      "targets": [
        {"ra": 10.684, "dec": 41.269, "label": "M31"},
        {"ra": 83.633, "dec": 22.0145, "label": "Crab"}
      ]
    });

    function showMessage(text, isError=false) {
      const el = document.getElementById("message");
      el.style.color = isError ? "red" : "green";
      el.textContent = text;
      setTimeout(() => { el.textContent = ""; }, 6000);
    }

    async function postJSON(action) {
      let data;
      try {
        data = editor.get();
      } catch (err) {
        showMessage("Invalid JSON: " + err.message, true);
        return null;
      }
      const payload = { action: action, json: data };
      const resp = await fetch("/save", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload)
      });
      return resp.json();
    }

    async function saveRepo() {
      const res = await postJSON("file");
      if (res && res.ok) showMessage("Saved to repository: " + res.path);
      else showMessage("Save failed", true);
    }

    async function saveRemote() {
      const res = await postJSON("db");
      if (res && res.ok) showMessage("Saved to remote DB (id: " + (res.id || res.error) + ")");
      else showMessage("Save failed", true);
    }

    async function generatePlot() {
      let data;
      try { data = editor.get(); } catch (err) { showMessage("Invalid JSON: " + err.message, true); return; }
      const resp = await fetch("/plot", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ json: data })
      });
      const blob = await resp.blob();
      if (blob.size === 0) { showMessage("Plot generation failed", true); return; }
      const url = URL.createObjectURL(blob);
      document.getElementById("plotImg").src = url;
      showMessage("Plot generated");
    }

    document.getElementById("saveRepo").addEventListener("click", saveRepo);
    document.getElementById("saveRemote").addEventListener("click", saveRemote);
    document.getElementById("generatePlot").addEventListener("click", generatePlot);
  </script>
</body>
</html>
"""


@app.route("/", methods=["GET"])
def index():
    return render_template_string(INDEX_HTML)


def save_to_file(json_obj):
    timestamp = datetime.datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
    filename = DATA_DIR / f"input_{timestamp}.json"
    with open(filename, "w", encoding="utf-8") as f:
        json.dump(json_obj, f, indent=2, ensure_ascii=False)
    return str(filename.relative_to(BASE_DIR))


def save_to_mongo(json_obj):
    if not mongo_client:
        raise RuntimeError("MongoDB client not configured")
    db = mongo_client[MONGO_DB]
    coll = db[MONGO_COLLECTION]
    res = coll.insert_one({"data": json_obj, "created_at": datetime.datetime.utcnow()})
    return str(res.inserted_id)


@app.route("/save", methods=["POST"])
def save():
    payload = request.get_json(force=True)
    action = payload.get("action", "file")
    data = payload.get("json")
    if not isinstance(data, (dict, list)):
        return jsonify({"ok": False, "error": "JSON payload must be an object or array"}), 400
    try:
        if action == "db":
            if not mongo_client:
                return jsonify({"ok": False, "error": "MongoDB not configured"}), 500
            inserted_id = save_to_mongo(data)
            return jsonify({"ok": True, "id": inserted_id})
        else:
            path = save_to_file(data)
            return jsonify({"ok": True, "path": path})
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)}), 500


def make_sky_plot(json_obj):
    """
    Expects json_obj to be a list of objects or a dict with a top-level list.
    Each entry should have 'ra' (degrees), 'dec' (degrees), optional 'label'.
    Produces a PNG bytes object.
    """
    # Normalize input into a list
    if isinstance(json_obj, dict):
        # If it's a dict with key 'targets' or 'sources', try these
        if "targets" in json_obj and isinstance(json_obj["targets"], list):
            items = json_obj["targets"]
        else:
            # attempt to treat dict as a single object with ra/dec
            items = [json_obj]
    elif isinstance(json_obj, list):
        items = json_obj
    else:
        raise ValueError("Unsupported JSON structure for plotting")

    ras = []
    decs = []
    labels = []
    for it in items:
        try:
            ra = float(it.get("ra"))
            dec = float(it.get("dec"))
            ras.append(ra)
            decs.append(dec)
            labels.append(str(it.get("label", "")))
        except Exception:
            # skip invalid entries
            continue

    if len(ras) == 0:
        raise ValueError("No valid RA/Dec found in JSON")

    # Convert RA (0-360) to radians for Aitoff (-pi to +pi)
    ra_deg = np.array(ras)
    dec_deg = np.array(decs)
    # shift RA from [0,360) to [-180,180)
    ra_shifted = ((ra_deg + 180) % 360) - 180
    ra_rad = np.radians(ra_shifted)
    dec_rad = np.radians(dec_deg)

    fig = plt.figure(figsize=(8, 4.5))
    ax = fig.add_subplot(111, projection="aitoff")
    ax.grid(True)
    sc = ax.scatter(-ra_rad, dec_rad, s=30, c='red', alpha=0.7)  # negative RA to match sky convention
    # add labels if present
    for x, y, label in zip(-ra_rad, dec_rad, labels):
        if label:
            ax.text(x, y, " " + label, fontsize=8, ha='left', va='center')

    ax.set_title("Sky plot (Aitoff projection)")

    buf = io.BytesIO()
    plt.tight_layout()
    fig.savefig(buf, format="png", dpi=150)
    plt.close(fig)
    buf.seek(0)
    return buf


@app.route("/plot", methods=["POST"])
def plot():
    payload = request.get_json(force=True)
    data = payload.get("json")
    if data is None:
        return jsonify({"ok": False, "error": "Missing JSON payload"}), 400
    try:
        buf = make_sky_plot(data)
        return send_file(buf, mimetype="image/png", as_attachment=False, download_name="sky.png")
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)}), 400


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)), debug=False)