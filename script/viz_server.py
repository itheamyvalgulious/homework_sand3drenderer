"""Flask front-end for the interactive SAND SDF ray-march visualizer.

Serves a single HTML page with a control panel (light position, color,
intensity, camera elevation/azimuth, resolution, ambient, model selector) and
an image box. ``POST /render`` calls :class:`script.viz_render.FieldCache` to
ray-march the selected field, shade it with a point light, and return a PNG.

Usage::

    .venv/bin/python script/viz_server.py

then open http://127.0.0.1:8080/ in a browser.
"""
from __future__ import annotations

import base64
import io
import os
import sys

# Bootstrap the project root so ``from script...`` / ``from src...`` work when
# this file is run directly via ``.venv/bin/python script/viz_server.py``.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from flask import Flask, Response, jsonify, request

app = Flask(__name__)

# Lazily-built singleton cache. Constructing it loads torch checkpoints / builds
# octree fields, which can take a few seconds and may even fail if CUDA is busy
# or viz_render's checkpoints are not on disk. We therefore defer both the
# ``FieldCache`` *import* and its construction to first use so importing this
# module never crashes the server — important because ``viz_render`` may not
# exist yet if this file is being imported during a build race with the other
# agent's file. ``get_cache()`` then fails only on actual use.
_CACHE = None  # type: ignore[var-annotated]
_CACHE_DEVICE: str | None = None


def get_cache():
    """Return the process-wide :class:`FieldCache`, building it on first use.

    Imports :class:`script.viz_render.FieldCache` lazily so that a missing
    ``viz_render`` module only errors when the cache is actually requested, not
    at server import time.
    """
    global _CACHE, _CACHE_DEVICE
    if _CACHE is not None:
        return _CACHE
    from script.viz_render import FieldCache  # deferred import (race-safe)

    device = "cuda"
    try:
        import torch

        if not torch.cuda.is_available():
            device = "cpu"
    except Exception:
        device = "cpu"
    try:
        _CACHE = FieldCache(device=device)
    except Exception as exc:  # pragma: no cover - best-effort fallback
        # If CUDA construction failed (e.g. OOM / busy), retry on CPU so the
        # server still serves /health and /models instead of crashing.
        if device != "cpu":
            _CACHE = FieldCache(device="cpu")
        else:
            raise RuntimeError(f"FieldCache construction failed: {exc}") from exc
    _CACHE_DEVICE = _CACHE._device if hasattr(_CACHE, "_device") else device  # type: ignore[attr-defined]
    return _CACHE


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.route("/health")
def health():
    """Liveness + model count probe (never builds the cache eagerly)."""
    n = 0
    try:
        n = len(get_cache().list_models())
    except Exception:
        n = 0
    return jsonify({"ok": True, "models": n})


@app.route("/models")
def models():
    """JSON list of available models for the <select> dropdown."""
    try:
        return jsonify(get_cache().list_models())
    except Exception as exc:
        return jsonify({"error": str(exc)}), 500


def _f(val, default: float) -> float:
    """Parse a JSON value to float, falling back to ``default``."""
    try:
        return float(val)
    except (TypeError, ValueError):
        return default


@app.route("/render", methods=["POST"])
def render():
    """Ray-march the selected model and return a base64 PNG + stats."""
    try:
        body = request.get_json(silent=True) or {}
        tag = body.get("tag")
        if not tag:
            return jsonify({"error": "missing 'tag'"}), 400

        lp = body.get("light_pos", [2.0, 2.0, 2.0])
        if not isinstance(lp, (list, tuple)) or len(lp) != 3:
            lp = [2.0, 2.0, 2.0]
        light_pos = (_f(lp[0], 2.0), _f(lp[1], 2.0), _f(lp[2], 2.0))

        lc = body.get("light_color", [1.0, 1.0, 1.0])
        if not isinstance(lc, (list, tuple)) or len(lc) != 3:
            lc = [1.0, 1.0, 1.0]
        light_color = (
            min(max(_f(lc[0], 1.0), 0.0), 1.0),
            min(max(_f(lc[1], 1.0), 0.0), 1.0),
            min(max(_f(lc[2], 1.0), 0.0), 1.0),
        )

        intensity = _f(body.get("intensity", 1.0), 1.0)
        elev = _f(body.get("elev", 25.0), 25.0)
        azim = _f(body.get("azim", 45.0), 45.0)
        ambient = _f(body.get("ambient", 0.15), 0.15)
        try:
            img_size = int(body.get("img_size", 384))
        except (TypeError, ValueError):
            img_size = 384
        if img_size not in (256, 384, 512):
            img_size = 384

        result = get_cache().render(
            tag=tag,
            light_pos=light_pos,
            light_color=light_color,
            intensity=intensity,
            elev=elev,
            azim=azim,
            img_size=img_size,
            ambient=ambient,
        )
        png_b64 = base64.b64encode(result["png"]).decode("ascii")
        return jsonify({"png": png_b64, "stats": result["stats"]})
    except KeyError as exc:
        return jsonify({"error": f"unknown model tag: {exc}"}), 400
    except Exception as exc:
        return jsonify({"error": str(exc)}), 400


# ---------------------------------------------------------------------------
# HTML page (embedded, no external dependencies, Chinese labels)
# ---------------------------------------------------------------------------

HTML_PAGE = """<!doctype html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>SAND SDF 交互渲染器</title>
<style>
  * { box-sizing: border-box; }
  body { margin:0; font-family: -apple-system, "PingFang SC", "Microsoft YaHei", system-ui, sans-serif;
         background:#1b1f24; color:#e6e6e6; }
  .layout { display:flex; gap:24px; padding:24px; min-height:100vh; }
  .panel { width:320px; flex:0 0 320px; background:#242a31; border:1px solid #2f3742;
           border-radius:10px; padding:20px; overflow-y:auto; }
  .stage { flex:1; display:flex; flex-direction:column; align-items:center;
           justify-content:flex-start; padding:20px; }
  h1 { font-size:18px; margin:0 0 18px 0; color:#7fd1ff; }
  .group { margin-bottom:16px; }
  .group > label { display:block; font-size:13px; color:#9fb0c2; margin-bottom:6px; }
  .row { display:flex; gap:8px; align-items:center; }
  input[type=number], input[type=text], select {
    width:100%; padding:6px 8px; background:#1b2128; color:#e6e6e6;
    border:1px solid #2f3742; border-radius:5px; font-size:13px; }
  input[type=range] { flex:1; }
  .small { width:64px; }
  input[type=color] { width:42px; height:30px; padding:0; border:1px solid #2f3742;
                     border-radius:5px; background:none; cursor:pointer; }
  button.render { width:100%; padding:11px; background:#2f6fd4; color:#fff; border:none;
                  border-radius:6px; font-size:15px; cursor:pointer; margin-top:6px; }
  button.render:disabled { background:#3a4452; cursor:not-allowed; }
  .imgbox { min-width:420px; min-height:420px; background:#0e1116;
            border:1px solid #2f3742; border-radius:10px; display:flex;
            align-items:center; justify-content:center; overflow:hidden; }
  .imgbox img { max-width:100%; max-height:75vh; display:block; }
  .placeholder { color:#5b6573; font-size:14px; }
  .stats { margin-top:12px; font-size:13px; color:#9fb0c2; min-height:20px; }
  .err { color:#ff8b8b; }
  .hint { font-size:11px; color:#5b6573; margin-top:4px; }
  .chk { display:flex; align-items:center; gap:8px; font-size:13px; color:#9fb0c2; }
  .chk input { width:auto; }
</style>
</head>
<body>
<div class="layout">
  <div class="panel">
    <h1>SAND SDF 交互渲染器</h1>

    <div class="group">
      <label>模型</label>
      <select id="model"></select>
    </div>

    <div class="group">
      <label>光源位置</label>
      <div class="row">
        <input type="text" id="lx" class="small" value="2.0">
        <input type="text" id="ly" class="small" value="2.0">
        <input type="text" id="lz" class="small" value="2.0">
      </div>
      <div class="hint">X / Y / Z</div>
    </div>

    <div class="group">
      <label>光源颜色</label>
      <div class="row">
        <input type="color" id="cpick" value="#ffffff">
        <input type="text" id="cr" class="small" value="1.0">
        <input type="text" id="cg" class="small" value="1.0">
        <input type="text" id="cb" class="small" value="1.0">
      </div>
      <div class="hint">取色器 / R / G / B (0..1)</div>
    </div>

    <div class="group">
      <label>光照强度</label>
      <input type="text" id="intensity" value="1.0">
    </div>

    <div class="group">
      <label>环境光</label>
      <input type="text" id="ambient" value="0.15">
    </div>

    <div class="group">
      <label>相机仰角 (度)</label>
      <div class="row">
        <input type="range" id="elev_slider" min="-89" max="89" step="1" value="25">
        <input type="text" id="elev" class="small" value="25">
      </div>
    </div>

    <div class="group">
      <label>相机方位角 (度)</label>
      <div class="row">
        <input type="range" id="azim_slider" min="0" max="360" step="1" value="45">
        <input type="text" id="azim" class="small" value="45">
      </div>
    </div>

    <div class="group">
      <label>渲染分辨率</label>
      <select id="img_size">
        <option value="256">256</option>
        <option value="384" selected>384</option>
        <option value="512">512</option>
      </select>
    </div>

    <div class="group chk">
      <input type="checkbox" id="realtime">
      <label for="realtime">实时 (滑动自动渲染)</label>
    </div>

    <button class="render" id="renderbtn" onclick="doRender()">渲染</button>
  </div>

  <div class="stage">
    <div class="imgbox" id="imgbox">
      <span class="placeholder">点击「渲染」生成图像</span>
    </div>
    <div class="stats" id="stats"></div>
  </div>
</div>

<script>
const $ = id => document.getElementById(id);
let rtTimer = null;

// --- color picker <-> RGB text sync ---
function hexToRgb01(h) {
  const n = parseInt(h.slice(1), 16);
  return [((n>>16)&255)/255, ((n>>8)&255)/255, (n&255)/255];
}
function rgb01ToHex(r,g,b) {
  const f = v => Math.max(0,Math.min(255, Math.round(Math.max(0,Math.min(1,v))*255)));
  return '#' + [f(r),f(g),f(b)].map(x=>x.toString(16).padStart(2,'0')).join('');
}
$('cpick').addEventListener('input', () => {
  const [r,g,b] = hexToRgb01($('cpick').value);
  $('cr').value = r.toFixed(3); $('cg').value = g.toFixed(3); $('cb').value = b.toFixed(3);
});
['cr','cg','cb'].forEach(id => $(id).addEventListener('input', () => {
  $('cpick').value = rgb01ToHex(parseFloat($('cr').value)||0,
                               parseFloat($('cg').value)||0,
                               parseFloat($('cb').value)||0);
}));

// --- elev/azim slider <-> text sync ---
function syncSliderText(sliderId, textId) {
  $(sliderId).addEventListener('input', () => { $(textId).value = $(sliderId).value; maybeRT(); });
  $(textId).addEventListener('input', () => { $(sliderId).value = $(textId).value; });
}
syncSliderText('elev_slider','elev');
syncSliderText('azim_slider','azim');

function maybeRT() {
  if (!$('realtime').checked) return;
  clearTimeout(rtTimer);
  rtTimer = setTimeout(doRender, 300);
}

function buildBody() {
  // num() returns the parsed value when finite (incl. 0 and negatives);
  // only falls back to the default on NaN/empty. Using `||` here is wrong
  // because parseFloat("0")===0 is falsy and would be replaced by default.
  const num = (id, d) => { const n = parseFloat($(id).value); return Number.isFinite(n) ? n : d; };
  const num3 = (a,b,c, d) => [num(a,d), num(b,d), num(c,d)];
  return {
    tag: $('model').value,
    light_pos: num3('lx','ly','lz', 2.0),
    light_color: num3('cr','cg','cb', 1.0),
    intensity: num('intensity', 1.0),
    ambient: num('ambient', 0.15),
    elev: num('elev', 25.0),
    azim: num('azim', 45.0),
    img_size: parseInt($('img_size').value, 10) || 384,
  };
}

async function doRender() {
  const btn = $('renderbtn');
  const stats = $('stats');
  btn.disabled = true; btn.textContent = '渲染中…'; stats.className = 'stats';
  try {
    const resp = await fetch('/render', {
      method: 'POST',
      headers: {'Content-Type':'application/json'},
      body: JSON.stringify(buildBody()),
    });
    const data = await resp.json();
    if (!resp.ok || data.error) {
      throw new Error(data.error || ('HTTP ' + resp.status));
    }
    const box = $('imgbox');
    box.innerHTML = '<img alt="rendered" src="data:image/png;base64,' + data.png + '">';
    const s = data.stats || {};
    stats.textContent = `耗时 ${Number(s.total_s).toFixed(2)}s · 命中 ${s.n_hit} / ${s.n_rays} (${(Number(s.hit_frac)*100).toFixed(1)}%)`;
  } catch (e) {
    stats.className = 'stats err';
    stats.textContent = '错误: ' + e.message;
  } finally {
    btn.disabled = false; btn.textContent = '渲染';
  }
}

// --- load model list on startup ---
fetch('/models').then(r=>r.json()).then(list => {
  const sel = $('model');
  (Array.isArray(list) ? list : []).forEach(m => {
    const o = document.createElement('option');
    o.value = m.tag; o.textContent = m.label;
    sel.appendChild(o);
  });
  // select the first model so a first "渲染" click works without manual picking
  if (sel.options.length) sel.selectedIndex = 0;
}).catch(()=>{});
</script>
</body>
</html>
"""


@app.route("/")
def index():
    """Serve the visualizer HTML page (no-cache so the latest HTML ships)."""
    resp = Response(HTML_PAGE, mimetype="text/html")
    resp.headers["Cache-Control"] = "no-store, no-cache, must-revalidate"
    return resp


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=16200, debug=False)
