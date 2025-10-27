# tess_embed_3d.py
# single-file pipeline: process TOI (TESS) -> write JSON -> write viewer.html -> open in browser
# (Update: adds KPI blocks for Total, Candidates, Confirmed at top-left)

import os, json, math, argparse, webbrowser
import numpy as np
import pandas as pd
from pathlib import Path

# ---------------------------
# CLI
# ---------------------------
p = argparse.ArgumentParser()
p.add_argument("--data", default="data/TESS Project Candidates.csv", help="path to TESS CSV/XLSX")
p.add_argument("--projector", choices=["tsne","umap","pca"], default="tsne", help="3D projector")
p.add_argument("--out_dir", default=".", help="output folder for json/html")
p.add_argument("--seed", type=int, default=42)
args = p.parse_args()

DATA_PATH = args.data
OUT_DIR   = Path(args.out_dir)
OUT_DIR.mkdir(parents=True, exist_ok=True)

# === Config (Tabular TESS features) ==========================================
data_path   = "data/TESS Project Candidates.csv"   # <-- change to your file
target_col  = "tfopwg_disp"                        # CP / PC

# Core transit/orbital + host-star features
feature_cols_base = [
    "pl_orbper","pl_trandurh","pl_trandep",
    "pl_rade","pl_eqt","pl_insol",
    "st_teff","st_logg","st_rad","st_tmag","st_dist"
]

# HIGH-LEVERAGE ADDITIONS: (no TESS binary FP flags provided in this table)
fp_flag_cols = []  # kept for API-compat with KOI script

# Final tabular feature set for this run
feature_cols = feature_cols_base + fp_flag_cols

# Keep IDs for traceability (never used as features)
f_ids_tess = ["toi", "tid"]

# Label policy:
#   "confirmed_vs_candidate" → binary (CP=1, PC=0)
label_policy = "confirmed_vs_candidate"
positive_label = "CP"
negative_label = "PC"

# ---------------------------
# Column configuration (TESS)
# ---------------------------
TARGET_COL    = target_col
FEATURE_COLS  = feature_cols
ID_COLS       = f_ids_tess

LABEL_POLICY  = label_policy
POS_LABEL     = positive_label
NEG_LABEL     = negative_label

OUT_JSON = OUT_DIR / "tess_tsne3d_points.json"
OUT_HTML = OUT_DIR / "viewer.html"
RANDOM_STATE = args.seed

# ---------------------------
# Load
# ---------------------------
ext = os.path.splitext(DATA_PATH)[1].lower()
if ext in (".xls", ".xlsx"):
    df_raw = pd.read_excel(DATA_PATH)
else:
    df_raw = pd.read_csv(DATA_PATH)

df_raw.columns = df_raw.columns.astype(str).str.strip()
required = set([TARGET_COL] + FEATURE_COLS + ID_COLS)
missing = [c for c in required if c not in df_raw.columns]
if missing:
    raise KeyError(f"Missing required columns: {missing}")

df = df_raw[ID_COLS + [TARGET_COL] + FEATURE_COLS].copy()

if LABEL_POLICY == "confirmed_vs_candidate":
    keep = {POS_LABEL, NEG_LABEL}
    df = df[df[TARGET_COL].isin(keep)].reset_index(drop=True)
    y = df[TARGET_COL].map({POS_LABEL: 1, NEG_LABEL: 0}).astype(int)
else:
    y = df[TARGET_COL].astype(str)

for c in FEATURE_COLS:
    df[c] = pd.to_numeric(df[c], errors="coerce")
df = df.dropna(subset=FEATURE_COLS).reset_index(drop=True)
y = y.loc[df.index]

X = df[FEATURE_COLS].values

# ---------------------------
# Scale + project to 3D
# ---------------------------
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

proj = args.projector.lower()
if proj == "tsne":
    try:
        from sklearn.manifold import TSNE
        tsne = TSNE(
            n_components=3, learning_rate="auto", init="pca",
            perplexity=min(30, max(5, len(df)//50)), n_iter=1000, random_state=RANDOM_STATE
        )
        coords = tsne.fit_transform(X_scaled)
        projector_used = "t-SNE"
    except Exception:
        from sklearn.decomposition import PCA
        coords = PCA(n_components=3, random_state=RANDOM_STATE).fit_transform(X_scaled)
        projector_used = "PCA (fallback)"
elif proj == "umap":
    try:
        import umap
        coords = umap.UMAP(n_components=3, random_state=RANDOM_STATE).fit_transform(X_scaled)
        projector_used = "UMAP"
    except Exception:
        from sklearn.decomposition import PCA
        coords = PCA(n_components=3, random_state=RANDOM_STATE).fit_transform(X_scaled)
        projector_used = "PCA (fallback)"
else:  # pca
    from sklearn.decomposition import PCA
    coords = PCA(n_components=3, random_state=RANDOM_STATE).fit_transform(X_scaled)
    projector_used = "PCA"

# ---------------------------
# Cluster (HDBSCAN -> KMeans)
# ---------------------------
try:
    import hdbscan
    min_sz = max(10, len(df)//50)
    clusterer = hdbscan.HDBSCAN(min_cluster_size=min_sz)
    clabels = clusterer.fit_predict(coords)
    clustering_used = f"HDBSCAN(min_cluster_size={min_sz})"
except Exception:
    from sklearn.cluster import KMeans
    k = min(8, max(2, int(round(math.sqrt(len(df)) / 4))))
    clabels = KMeans(n_clusters=k, n_init=10, random_state=RANDOM_STATE).fit_predict(coords)
    clustering_used = f"KMeans(k={k})"

# ---------------------------
# Package points
# ---------------------------
actual_names = {0: NEG_LABEL, 1: POS_LABEL}
points = []
for i, (x, y_, z) in enumerate(coords):
    row = df.iloc[i]
    lab = int(y.iloc[i]) if str(y.iloc[i]).isdigit() else 0
    points.append({
        "x": float(x), "y": float(y_), "z": float(z),
        "tid": str(row.get("tid", "")),
        "toi": str(row.get("toi", "")),
        "actual_label": lab,
        "actual_label_name": actual_names.get(lab, str(lab)),
        "cluster": int(clabels[i]),
        # handy attributes for tooltip/encodings
        "pl_rade": float(row.get("pl_rade", np.nan)),
        "pl_eqt": float(row.get("pl_eqt", np.nan)),
    })

OUT_JSON.write_text(json.dumps(points), encoding="utf-8")

# ---------------------------
# Write viewer.html (safe string) — now with KPI blocks
# ---------------------------
VIEWER_HTML = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8"/>
<meta name="viewport" content="width=device-width, initial-scale=1"/>
<title>TESS 3D Space</title>
<script src="https://cdnjs.cloudflare.com/ajax/libs/three.js/r128/three.min.js"></script>
<script src="https://cdn.jsdelivr.net/npm/three@0.128.0/examples/js/controls/OrbitControls.js"></script>
<style>
  body{margin:0; font-family:Helvetica, Arial, sans-serif; background:#fff}
  #wrap{position:relative; width:100%; height:min(95vh,100svh); min-height:640px; overflow:hidden;}
  #title{position:absolute; top:16px; left:16px; z-index:10; color:#333; font-weight:700}
  #sub{font-size:12px; color:#666; font-weight:400}

  /* KPI strip */
  #kpis{position:absolute; top:56px; left:16px; z-index:9; display:flex; gap:12px}
  .kpi{background:rgba(255,255,255,0.96); padding:10px 12px; border-radius:12px; box-shadow:0 2px 10px rgba(0,0,0,.08); min-width:120px}
  .kpi .v{font-size:20px; font-weight:700; color:#222; line-height:1}
  .kpi .l{font-size:11px; color:#666; margin-top:4px; text-transform:uppercase; letter-spacing:.04em}

  #panel{position:absolute; top:12px; right:12px; z-index:10; background:rgba(255,255,255,0.96);
         padding:10px; border-radius:12px; box-shadow:0 2px 10px rgba(0,0,0,.08); min-width:240px}
  #panel label{display:block; font-size:12px; color:#555; margin:8px 0 6px}
  #panel select, #panel input[type="text"], #panel input[type="range"]{width:100%; padding:6px 8px; margin-bottom:8px; border:1px solid #e0e0e0; border-radius:8px}
  #legend{margin-top:6px; font-size:12px; max-height:160px; overflow:auto}
  .legend-item{display:flex; align-items:center; margin:4px 0}
  .legend-dot{width:12px; height:12px; border-radius:50%; margin-right:6px; border:2px solid transparent}
  #tip{position:absolute; z-index:1000; background:rgba(0,0,0,.9); color:#fff; padding:8px 12px; border-radius:6px; font-size:13px; pointer-events:none; display:none}
  #footer{position:absolute; bottom:10px; left:16px; z-index:5; font-size:12px; color:#777; background:rgba(255,255,255,.85); padding:6px 8px; border-radius:8px}
  .pill{display:inline-block; padding:2px 8px; border-radius:999px; background:#f3f3f3; margin-right:6px}
</style>
</head>
<body>
<div id="wrap">
  <div id="title">TESS 3D Space<div id="sub">features: 11 transit/stellar features · unsupervised embedding</div></div>

  <!-- KPI blocks -->
  <div id="kpis">
    <div class="kpi"><div id="kpi-total" class="v">—</div><div class="l">Total points</div></div>
    <div class="kpi"><div id="kpi-cand"  class="v">—</div><div class="l">Candidates</div></div>
    <div class="kpi"><div id="kpi-conf"  class="v">—</div><div class="l">Confirmed</div></div>
  </div>

  <div id="panel">
    <label>Color by</label>
    <select id="colorMode">
      <option value="actual">Actual label (CP/PC)</option>
      <option value="cluster">Cluster (unsupervised)</option>
    </select>

    <label>Search by ID/name</label>
    <input id="q" type="text" placeholder="e.g., 8260536 or TOI-5398"/>

    <label>Spread</label>
    <input id="spread" type="range" min="0" max="4" step="0.1" value="3"/>

    <label>Point size</label>
    <input id="psize" type="range" min="0.4" max="1.6" step="0.1" value="1.2"/>

    <div id="legend"></div>
  </div>
  <div id="tip"></div>
  <div id="footer"><span class="pill">drag = orbit</span><span class="pill">wheel = zoom</span><span class="pill">shift+drag = pan</span></div>
</div>

<script>
const actualColor = {"0":"#B22222","1":"#228B22"}; // PC (red) / CP (green)
const clusterPalette = ["#4C78A8","#F58518","#E45756","#72B7B2","#54A24B","#EECA3B","#B279A2","#FF9DA6","#9D755D","#BAB0AC"];
const clusterColor = c => clusterPalette[Math.abs(c) % clusterPalette.length];

async function loadData(){
  const res = await fetch("tess_tsne3d_points.json", {cache:"no-cache"});
  if(!res.ok) throw new Error("failed to load data: "+res.status);
  return await res.json();
}
function normalize(d){
  const xs=d.map(v=>v.x), ys=d.map(v=>v.y), zs=d.map(v=>v.z);
  const minx=Math.min(...xs), maxx=Math.max(...xs), miny=Math.min(...ys), maxy=Math.max(...ys), minz=Math.min(...zs), maxz=Math.max(...zs);
  const scale = 90 / Math.max(maxx-minx, maxy-miny, maxz-minz);
  d.forEach(v=>{ v.X=(v.x - minx - (maxx-minx)/2)*scale; v.Y=(v.y - miny - (maxy-miny)/2)*scale; v.Z=(v.z - minz - (maxz-minz)/2)*scale; });
  d.forEach(v=>{ v.X0=v.X; v.Y0=v.Y; v.Z0=v.Z; });
  return d;
}
function axisLabel(text,pos){
  const c=document.createElement('canvas'), ctx=c.getContext('2d'); c.width=64; c.height=64;
  ctx.font='bold 44px Arial'; ctx.fillStyle='black'; ctx.textAlign='center'; ctx.textBaseline='middle'; ctx.fillText(text,32,32);
  const s=new THREE.Sprite(new THREE.SpriteMaterial({map:new THREE.CanvasTexture(c)})); s.position.copy(pos); s.scale.set(4,4,1); return s;
}
function buildLegend(el, mode, data, cloudActual, cloudCluster){
  el.innerHTML='';
  if (mode==='actual'){
    [['#B22222','PC (candidate)'],['#228B22','CP (confirmed)']].forEach(([c,n])=>{
      const div=document.createElement('div'); div.className='legend-item';
      div.innerHTML=`<span class="legend-dot" style="background:${c}"></span><span>${n}</span>`;
      el.appendChild(div);
    });
    cloudActual.visible=true; cloudCluster.visible=false;
  } else {
    const uniq=[...new Set(data.map(d=>d.cluster))].sort((a,b)=>a-b);
    uniq.forEach(c=>{
      const col=clusterColor(c);
      const div=document.createElement('div'); div.className='legend-item';
      div.innerHTML=`<span class="legend-dot" style="background:${col}"></span><span>cluster ${c}</span>`;
      el.appendChild(div);
    });
    cloudActual.visible=false; cloudCluster.visible=true;
  }
}

loadData().then(DATA=>{
  normalize(DATA);

  // --- KPIs ---
  const total = DATA.length;
  const confirmed = DATA.filter(d=>d.actual_label===1).length;
  const candidates = total - confirmed;
  document.getElementById('kpi-total').textContent = total.toLocaleString();
  document.getElementById('kpi-cand').textContent  = candidates.toLocaleString();
  document.getElementById('kpi-conf').textContent  = confirmed.toLocaleString();

  const wrap = document.getElementById('wrap');
  const renderer = new THREE.WebGLRenderer({antialias:true}); wrap.appendChild(renderer.domElement);
  renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
  const scene = new THREE.Scene(); scene.background = new THREE.Color(0xffffff);
  const camera = new THREE.PerspectiveCamera(75, 1, 0.1, 5000); camera.position.set(80,80,80);
  function fit(){ const w=wrap.clientWidth, h=wrap.clientHeight; renderer.setSize(w,h,false); camera.aspect=w/h; camera.updateProjectionMatrix(); }
  fit(); new ResizeObserver(fit).observe(wrap);

  const controls = new THREE.OrbitControls(camera, renderer.domElement);
  controls.enableDamping=true; controls.dampingFactor=.06; controls.minDistance=20; controls.maxDistance=400;
  scene.add(new THREE.AmbientLight(0xffffff,.8));
  const dl=new THREE.DirectionalLight(0xffffff,.35); dl.position.set(50,100,30); scene.add(dl);
  scene.add(new THREE.GridHelper(200,20,0xeaeaea,0xf5f5f5));

  const axes=new THREE.Group();
  axes.add(new THREE.Line(new THREE.BufferGeometry().setFromPoints([new THREE.Vector3(0,0,0), new THREE.Vector3(100,0,0)]), new THREE.LineBasicMaterial({color:0x000000})));
  axes.add(new THREE.Line(new THREE.BufferGeometry().setFromPoints([new THREE.Vector3(0,0,0), new THREE.Vector3(0,100,0)]), new THREE.LineBasicMaterial({color:0x000000})));
  axes.add(new THREE.Line(new THREE.BufferGeometry().setFromPoints([new THREE.Vector3(0,0,0), new THREE.Vector3(0,0,100)]), new THREE.LineBasicMaterial({color:0x000000})));
  scene.add(axes);
  scene.add(axisLabel('X', new THREE.Vector3(110,0,0)));
  scene.add(axisLabel('Y', new THREE.Vector3(0,110,0)));
  scene.add(axisLabel('Z', new THREE.Vector3(0,0,110)));

  // geometry & materials
  const sphere = new THREE.SphereGeometry(1,16,16);
  const matPC = new THREE.MeshBasicMaterial({ color: 0xB22222 }); // candidate
  const matCP = new THREE.MeshBasicMaterial({ color: 0x228B22 }); // confirmed
  const clusterMaterials = clusterPalette.map(col => new THREE.MeshBasicMaterial({ color: col }));

  const actualGroup = new THREE.Group();
  const clusterGroup = new THREE.Group();
  scene.add(actualGroup, clusterGroup);

  const actualMeshes = [];
  const clusterMeshes = [];
  
  DATA.forEach((d, i) => {
    const mat = d.actual_label === 1 ? matCP : matPC;
    const mesh = new THREE.Mesh(sphere, mat);
    mesh.position.set(d.X, d.Y, d.Z);
    mesh.userData = { index: i, data: d };
    actualGroup.add(mesh);
    actualMeshes.push(mesh);
    
    const clusterIdx = Math.abs(d.cluster) % clusterPalette.length;
    const clusterMesh = new THREE.Mesh(sphere, clusterMaterials[clusterIdx]);
    clusterMesh.position.set(d.X, d.Y, d.Z);
    clusterMesh.userData = { index: i, data: d };
    clusterGroup.add(clusterMesh);
    clusterMeshes.push(clusterMesh);
  });

  // interactive spread + size
  const spreadEl = document.getElementById('spread');
  const psizeEl  = document.getElementById('psize');
  function rebuildPositions(){
    const spread = parseFloat(spreadEl.value);
    const psize  = parseFloat(psizeEl.value);
    DATA.forEach((d, i) => {
      const x = d.X0 * spread, y = d.Y0 * spread, z = d.Z0 * spread;
      actualMeshes[i].position.set(x, y, z);
      actualMeshes[i].scale.set(psize, psize, psize);
      clusterMeshes[i].position.set(x, y, z);
      clusterMeshes[i].scale.set(psize, psize, psize);
    });
  }
  spreadEl.addEventListener('input', rebuildPositions);
  psizeEl.addEventListener('input', rebuildPositions);
  rebuildPositions();

  // legend + toggle
  const legend = document.getElementById('legend');
  const modeSel = document.getElementById('colorMode');
  function refreshLegend(){ buildLegend(legend, modeSel.value, DATA, actualGroup, clusterGroup); }
  modeSel.addEventListener('change', refreshLegend); 
  refreshLegend();

  // search
  document.getElementById('q').addEventListener('keydown', e=>{
    if(e.key!=='Enter') return;
    const needle = e.target.value.trim().toLowerCase(); if(!needle) return;
    const i = DATA.findIndex(d =>
      (d.tid && String(d.tid).toLowerCase().includes(needle)) ||
      (d.toi && String(d.toi).toLowerCase().includes(needle))
    );
    if (i>=0){
      const spread = parseFloat(spreadEl.value);
      const d = DATA[i]; 
      const target = new THREE.Vector3(d.X0*spread, d.Y0*spread, d.Z0*spread);
      controls.target.copy(target);
      camera.position.lerp(new THREE.Vector3(target.x+30,target.y+30,target.z+30), 0.6);
      const ring=new THREE.Mesh(new THREE.RingGeometry(2.0,2.4,32), new THREE.MeshBasicMaterial({color:0x111111, side:THREE.DoubleSide, transparent:true, opacity:.85}));
      ring.position.copy(target); ring.rotation.x=-Math.PI/2; scene.add(ring); setTimeout(()=>scene.remove(ring), 1400);
    }
  });

  // tooltip
  const tip = document.getElementById('tip');
  const ray = new THREE.Raycaster(); const mouse = new THREE.Vector2();
  function onMove(e){
    const rect = renderer.domElement.getBoundingClientRect();
    mouse.x = ((e.clientX-rect.left)/rect.width)*2 - 1;
    mouse.y = -((e.clientY-rect.top)/rect.height)*2 + 1;
    ray.setFromCamera(mouse, camera);
    const target = (modeSel.value==='actual') ? actualGroup : clusterGroup;
    const hits = ray.intersectObjects(target.children);
    if (hits.length){
      const d = hits[0].object.userData.data;
      tip.innerHTML = `<strong>${d.toi || '—'}</strong><br>
                       tid: ${d.tid}<br>
                       label: ${d.actual_label_name}<br>
                       cluster: ${d.cluster}<br>
                       R⊕: ${Number.isFinite(d.pl_rade)?d.pl_rade.toFixed(2):'—'}, T_eq: ${Number.isFinite(d.pl_eqt)?d.pl_eqt.toFixed(0):'—'}`;
      tip.style.display='block'; 
      tip.style.left=(e.clientX+10)+'px'; 
      tip.style.top=(e.clientY-28)+'px';
      document.body.style.cursor='pointer';
    } else { 
      tip.style.display='none'; 
      document.body.style.cursor='default'; 
    }
  }
  renderer.domElement.addEventListener('mousemove', onMove);

  // animate
  function animate(){ controls.update(); renderer.render(scene, camera); requestAnimationFrame(animate); } 
  animate();
}).catch(err=>{
  document.body.innerHTML = '<pre style="padding:20px; white-space:pre-wrap">'+String(err)+'</pre>';
});
</script>
</body>
</html>"""

OUT_HTML.write_text(VIEWER_HTML, encoding="utf-8")

print(f"✅ wrote {OUT_JSON.name} ({len(points)} points)")
print(f"✅ wrote {OUT_HTML.name}")
print(f"   projector: {projector_used} | clustering: {clustering_used}")

# ---------------------------
# Serve viewer.html on localhost (foreground)
# ---------------------------
import os, time, webbrowser
from http.server import ThreadingHTTPServer, SimpleHTTPRequestHandler

PORT = 8008
os.chdir(args.out_dir)  # make sure viewer.html and tess_tsne3d_points.json are here

server = ThreadingHTTPServer(("127.0.0.1", PORT), SimpleHTTPRequestHandler)
print(f"🌐 Serving on http://127.0.0.1:{PORT}/viewer.html  (Ctrl+C to stop)")

# open the browser, then block in serve_forever()
webbrowser.open(f"http://127.0.0.1:{PORT}/viewer.html")
try:
    server.serve_forever()
except KeyboardInterrupt:
    pass
finally:
    server.server_close()
    print("👋 Server stopped")
