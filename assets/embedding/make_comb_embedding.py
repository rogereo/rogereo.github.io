# tnsa_embed_3d.py
# single-file pipeline: combine KOI (Kepler) + TESS -> embed -> viewer.html -> serve
# (Update: adds KPI blocks for Total, Candidates, Confirmed at top-left)

import os, json, math, argparse, webbrowser
import numpy as np
import pandas as pd
from pathlib import Path

# ---------------------------
# CLI
# ---------------------------
p = argparse.ArgumentParser()
p.add_argument("--koi",  default="data/Kepler Object of Interest.csv", help="path to KOI CSV/XLSX")
p.add_argument("--tess", default="data/TESS Project Candidates.csv",   help="path to TESS CSV/XLSX")
p.add_argument("--projector", choices=["tsne","umap","pca"], default="tsne", help="3D projector")
p.add_argument("--out_dir", default=".", help="output folder for json/html")
p.add_argument("--seed", type=int, default=42)
args = p.parse_args()

KOI_PATH  = Path(args.koi)
TESS_PATH = Path(args.tess)
OUT_DIR   = Path(args.out_dir)
OUT_DIR.mkdir(parents=True, exist_ok=True)

OUT_JSON = OUT_DIR / "tnsa_tsne3d_points.json"
OUT_HTML = OUT_DIR / "viewer.html"
RANDOM_STATE = args.seed

# ---------------------------
# Unified schema config
# ---------------------------
koi_target_col = "koi_disposition"
koi_keep_map = {
    "koi_period":   "period",
    "koi_duration": "duration_hours",
    "koi_depth":    "depth_ppm",
    "koi_prad":     "prad_re",
    "koi_teq":      "teq_k",
    "koi_steff":    "st_teff",
    "koi_slogg":    "st_logg",
    "koi_srad":     "st_rad",
}

tess_target_col = "tfopwg_disp"  # CP/PC or CF/CN
tess_keep_map = {
    "pl_orbper":   "period",
    "pl_trandurh": "duration_hours",
    "pl_trandep":  "depth_ppm",
    "pl_rade":     "prad_re",
    "pl_eqt":      "teq_k",
    "st_teff":     "st_teff",
    "st_logg":     "st_logg",
    "st_rad":      "st_rad",
}

def normalize_label(val):
    mapping = {
        "CONFIRMED": "CONFIRMED",
        "CANDIDATE": "CANDIDATE",
        "CP": "CONFIRMED", "CF": "CONFIRMED",
        "PC": "CANDIDATE", "CN": "CANDIDATE",
    }
    if pd.isna(val):
        return None
    return mapping.get(str(val).strip().upper(), None)

# ---------------------------
# Load helpers
# ---------------------------
def load_any(path: Path) -> pd.DataFrame:
    ext = path.suffix.lower()
    if ext in (".xls", ".xlsx"):
        return pd.read_excel(path)
    return pd.read_csv(path)

# ---------------------------
# Load & harmonize KOI
# ---------------------------
df_koi_raw = load_any(KOI_PATH)
df_koi_raw.columns = df_koi_raw.columns.astype(str).str.strip()

need_koi = list(koi_keep_map.keys())
missing_koi = [c for c in need_koi + [koi_target_col] if c not in df_koi_raw.columns]
if missing_koi:
    raise KeyError(f"KOI missing required columns: {missing_koi}")

df_koi = df_koi_raw[need_koi + [koi_target_col]].copy()
df_koi.rename(columns=koi_keep_map, inplace=True)
df_koi["disposition"] = df_koi[koi_target_col].apply(normalize_label)
df_koi.drop(columns=[koi_target_col], inplace=True)
for c in ["kepid", "kepler_name", "kepoi_name"]:
    if c not in df_koi_raw.columns:
        df_koi[c] = np.nan
    else:
        df_koi[c] = df_koi_raw[c]
df_koi["source"] = "KOI"
df_koi = df_koi[df_koi["disposition"].isin(["CONFIRMED","CANDIDATE"])]

# ---------------------------
# Load & harmonize TESS
# ---------------------------
df_tess_raw = load_any(TESS_PATH)
df_tess_raw.columns = df_tess_raw.columns.astype(str).str.strip()

need_tess = list(tess_keep_map.keys())
missing_tess = [c for c in need_tess + [tess_target_col] if c not in df_tess_raw.columns]
if missing_tess:
    raise KeyError(f"TESS missing required columns: {missing_tess}")

df_tess = df_tess_raw[need_tess + [tess_target_col]].copy()
df_tess.rename(columns=tess_keep_map, inplace=True)
df_tess["disposition"] = df_tess[tess_target_col].apply(normalize_label)
df_tess.drop(columns=[tess_target_col], inplace=True)
for c in ["toi", "tid", "tic", "toi_name"]:
    if c not in df_tess_raw.columns:
        df_tess[c] = np.nan
    else:
        df_tess[c] = df_tess_raw[c]
df_tess["source"] = "TESS"
df_tess = df_tess[df_tess["disposition"].isin(["CONFIRMED","CANDIDATE"])]

# ---------------------------
# Union on shared feature set
# ---------------------------
shared_feature_cols = sorted(
    set(df_koi.columns).intersection(df_tess.columns)
    - {"disposition","source","kepid","kepler_name","kepoi_name","toi","tid","tic","toi_name"}
)
if not shared_feature_cols:
    raise RuntimeError("No shared features between KOI and TESS after mapping.")

for c in shared_feature_cols:
    df_koi[c]  = pd.to_numeric(df_koi[c],  errors="coerce")
    df_tess[c] = pd.to_numeric(df_tess[c], errors="coerce")

df_koi_small  = df_koi[shared_feature_cols + ["disposition","source","kepid","kepler_name","kepoi_name"]]
df_tess_small = df_tess[shared_feature_cols + ["disposition","source","toi","tid","tic","toi_name"]]

combo_df = pd.concat([df_koi_small, df_tess_small], ignore_index=True)
before = len(combo_df)
combo_df = combo_df.dropna(subset=shared_feature_cols + ["disposition","source"]).reset_index(drop=True)
after = len(combo_df)
print(f"Shared features: {shared_feature_cols}")
print(f"Dropped {before - after} rows with missing values (kept {after})")
print(combo_df["disposition"].value_counts())

# ---------------------------
# Labels & matrix
# ---------------------------
y = combo_df["disposition"].map({"CONFIRMED":1, "CANDIDATE":0}).astype(int).values
X = combo_df[shared_feature_cols].values

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
            perplexity=min(30, max(5, len(combo_df)//50)), n_iter=1000, random_state=RANDOM_STATE
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
else:
    from sklearn.decomposition import PCA
    coords = PCA(n_components=3, random_state=RANDOM_STATE).fit_transform(X_scaled)
    projector_used = "PCA"

# ---------------------------
# Cluster (HDBSCAN -> KMeans)
# ---------------------------
try:
    import hdbscan
    min_sz = max(10, len(combo_df)//50)
    clusterer = hdbscan.HDBSCAN(min_cluster_size=min_sz)
    clabels = clusterer.fit_predict(coords)
    clustering_used = f"HDBSCAN(min_cluster_size={min_sz})"
except Exception:
    from sklearn.cluster import KMeans
    k = min(10, max(2, int(round(math.sqrt(len(combo_df)) / 4))))
    clabels = KMeans(n_clusters=k, n_init=10, random_state=RANDOM_STATE).fit_predict(coords)
    clustering_used = f"KMeans(k={k})"

# ---------------------------
# Package points
# ---------------------------
points = []
for i, (x, y_, z) in enumerate(coords):
    row = combo_df.iloc[i]
    points.append({
        "x": float(x), "y": float(y_), "z": float(z),
        "source": row["source"],
        "actual_label": int(row["disposition"] == "CONFIRMED"),
        "actual_label_name": row["disposition"],
        "cluster": int(clabels[i]),
        "kepid":       str(row.get("kepid", "")) if pd.notna(row.get("kepid", np.nan)) else "",
        "kepler_name": str(row.get("kepler_name","")) if pd.notna(row.get("kepler_name", np.nan)) else "",
        "kepoi_name":  str(row.get("kepoi_name","")) if pd.notna(row.get("kepoi_name", np.nan)) else "",
        "toi":         str(row.get("toi","")) if pd.notna(row.get("toi", np.nan)) else "",
        "tid":         str(row.get("tid","")) if pd.notna(row.get("tid", np.nan)) else "",
        "tic":         str(row.get("tic","")) if pd.notna(row.get("tic", np.nan)) else "",
        "toi_name":    str(row.get("toi_name","")) if pd.notna(row.get("toi_name", np.nan)) else "",
        "prad_re": float(row.get("prad_re", np.nan)),
        "teq_k":   float(row.get("teq_k", np.nan)),
    })

OUT_JSON.write_text(json.dumps(points), encoding="utf-8")

# ---------------------------
# Write viewer.html (with KPI blocks)
# ---------------------------
VIEWER_HTML = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8"/>
<meta name="viewport" content="width=device-width, initial-scale=1"/>
<title>TNSA 3D Space (KOI + TESS)</title>
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
         padding:10px; border-radius:12px; box-shadow:0 2px 10px rgba(0,0,0,.08); min-width:260px}
  #panel label{display:block; font-size:12px; color:#555; margin:8px 0 6px}
  #panel select, #panel input[type="text"], #panel input[type="range"]{width:100%; padding:6px 8px; margin-bottom:8px; border:1px solid #e0e0e0; border-radius:8px}
  #legend{margin-top:6px; font-size:12px; max-height:180px; overflow:auto}
  .legend-item{display:flex; align-items:center; margin:4px 0}
  .legend-dot{width:12px; height:12px; border-radius:50%; margin-right:6px; border:2px solid transparent}
  #tip{position:absolute; z-index:1000; background:rgba(0,0,0,.9); color:#fff; padding:8px 12px; border-radius:6px; font-size:13px; pointer-events:none; display:none}
  #footer{position:absolute; bottom:10px; left:16px; z-index:5; font-size:12px; color:#777; background:rgba(255,255,255,.85); padding:6px 8px; border-radius:8px}
  .pill{display:inline-block; padding:2px 8px; border-radius:999px; background:#f3f3f3; margin-right:6px}
</style>
</head>
<body>
<div id="wrap">
  <div id="title">TNSA 3D Space<div id="sub">KOI + TESS · shared features · unsupervised embedding</div></div>

  <!-- KPI blocks -->
  <div id="kpis">
    <div class="kpi"><div id="kpi-total" class="v">—</div><div class="l">Total points</div></div>
    <div class="kpi"><div id="kpi-cand"  class="v">—</div><div class="l">Candidates</div></div>
    <div class="kpi"><div id="kpi-conf"  class="v">—</div><div class="l">Confirmed</div></div>
  </div>

  <div id="panel">
    <label>Color by</label>
    <select id="colorMode">
      <option value="actual">Actual label (CONFIRMED/CANDIDATE)</option>
      <option value="source">Source (KOI/TESS)</option>
      <option value="cluster">Cluster (unsupervised)</option>
    </select>

    <label>Search (ID or name)</label>
    <input id="q" type="text" placeholder="e.g., 11446443, Kepler-22, TOI-5398, 8260536"/>

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
const labelColor  = {"0":"#B22222","1":"#228B22"};
const sourceColor = {"KOI":"#6C5CE7","TESS":"#00B894"};
const clusterPalette = ["#4C78A8","#F58518","#E45756","#72B7B2","#54A24B","#EECA3B","#B279A2","#FF9DA6","#9D755D","#BAB0AC"];
const clusterColor = c => clusterPalette[Math.abs(c) % clusterPalette.length];

async function loadData(){
  const res = await fetch("tnsa_tsne3d_points.json", {cache:"no-cache"});
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
function buildLegend(el, mode, data, gLabel, gSource, gCluster){
  el.innerHTML='';
  if (mode==='actual'){
    [['#B22222','CANDIDATE'],['#228B22','CONFIRMED']].forEach(([c,n])=>{
      const div=document.createElement('div'); div.className='legend-item';
      div.innerHTML=`<span class="legend-dot" style="background:${c}"></span><span>${n}</span>`;
      el.appendChild(div);
    });
    gLabel.visible=true; gSource.visible=false; gCluster.visible=false;
  } else if (mode==='source'){
    Object.entries(sourceColor).forEach(([k,c])=>{
      const div=document.createElement('div'); div.className='legend-item';
      div.innerHTML=`<span class="legend-dot" style="background:${c}"></span><span>${k}</span>`;
      el.appendChild(div);
    });
    gLabel.visible=false; gSource.visible=true; gCluster.visible=false;
  } else {
    const uniq=[...new Set(data.map(d=>d.cluster))].sort((a,b)=>a-b);
    uniq.forEach(c=>{
      const col=clusterColor(c);
      const div=document.createElement('div'); div.className='legend-item';
      div.innerHTML=`<span class="legend-dot" style="background:${col}"></span><span>cluster ${c}</span>`;
      el.appendChild(div);
    });
    gLabel.visible=false; gSource.visible=false; gCluster.visible=true;
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

  // label materials
  const matCand = new THREE.MeshBasicMaterial({ color: 0xB22222 });
  const matConf = new THREE.MeshBasicMaterial({ color: 0x228B22 });

  // source materials
  const matKOI  = new THREE.MeshBasicMaterial({ color: sourceColor["KOI"] });
  const matTESS = new THREE.MeshBasicMaterial({ color: sourceColor["TESS"] });

  // cluster materials
  const clusterMaterials = clusterPalette.map(col => new THREE.MeshBasicMaterial({ color: col }));

  const gLabel   = new THREE.Group();
  const gSource  = new THREE.Group();
  const gCluster = new THREE.Group();
  scene.add(gLabel, gSource, gCluster);

  const meshesLabel = [], meshesSource = [], meshesCluster = [];

  DATA.forEach((d, i) => {
    // label view
    const mLab = d.actual_label === 1 ? matConf : matCand;
    const a = new THREE.Mesh(sphere, mLab); a.position.set(d.X, d.Y, d.Z); a.userData = { i, d };
    gLabel.add(a); meshesLabel.push(a);

    // source view
    const mSrc = (d.source === "KOI") ? matKOI : matTESS;
    const b = new THREE.Mesh(sphere, mSrc); b.position.set(d.X, d.Y, d.Z); b.userData = { i, d };
    gSource.add(b); meshesSource.push(b);

    // cluster view
    const idx = Math.abs(d.cluster) % clusterPalette.length;
    const c = new THREE.Mesh(sphere, clusterMaterials[idx]); c.position.set(d.X, d.Y, d.Z); c.userData = { i, d };
    gCluster.add(c); meshesCluster.push(c);
  });

  // interactive spread + size
  const spreadEl = document.getElementById('spread');
  const psizeEl  = document.getElementById('psize');

  function rebuildPositions(){
    const spread = parseFloat(spreadEl.value);
    const psize  = parseFloat(psizeEl.value);
    [meshesLabel, meshesSource, meshesCluster].forEach(arr=>{
      arr.forEach((m)=>{
        const d = m.userData.d;
        m.position.set(d.X0*spread, d.Y0*spread, d.Z0*spread);
        m.scale.set(psize, psize, psize);
      });
    });
  }
  spreadEl.addEventListener('input', rebuildPositions);
  psizeEl.addEventListener('input', rebuildPositions);
  rebuildPositions();

  // legend + toggle
  const legend = document.getElementById('legend');
  const modeSel = document.getElementById('colorMode');
  function refreshLegend(){ buildLegend(legend, modeSel.value, DATA, gLabel, gSource, gCluster); }
  modeSel.addEventListener('change', refreshLegend);
  refreshLegend();

  // search across KOI & TESS identifiers/names
  document.getElementById('q').addEventListener('keydown', e=>{
    if(e.key!=='Enter') return;
    const needle = e.target.value.trim().toLowerCase(); if(!needle) return;
    const idx = DATA.findIndex(d =>
      (d.kepid && String(d.kepid).toLowerCase().includes(needle)) ||
      (d.kepler_name && d.kepler_name.toLowerCase().includes(needle)) ||
      (d.kepoi_name && d.kepoi_name.toLowerCase().includes(needle)) ||
      (d.tid && String(d.tid).toLowerCase().includes(needle)) ||
      (d.tic && String(d.tic).toLowerCase().includes(needle)) ||
      (d.toi && String(d.toi).toLowerCase().includes(needle)) ||
      (d.toi_name && d.toi_name.toLowerCase().includes(needle))
    );
    if (idx>=0){
      const spread = parseFloat(spreadEl.value);
      const d = DATA[idx];
      const target = new THREE.Vector3(d.X0*spread, d.Y0*spread, d.Z0*spread);
      controls.target.copy(target);
      camera.position.lerp(new THREE.Vector3(target.x+30,target.y+30,target.z+30), 0.6);
      const ring=new THREE.Mesh(new THREE.RingGeometry(2.0,2.6,40), new THREE.MeshBasicMaterial({color:0x111111, side:THREE.DoubleSide, transparent:true, opacity:.85}));
      ring.position.copy(target); ring.rotation.x=-Math.PI/2; scene.add(ring); setTimeout(()=>scene.remove(ring), 1500);
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

    const mode = modeSel.value;
    const target = (mode==='actual') ? gLabel : (mode==='source' ? gSource : gCluster);
    const hits = ray.intersectObjects(target.children);

    if (hits.length){
      const d = hits[0].object.userData.d;
      const name = d.kepler_name || d.kepoi_name || d.toi_name || d.toi || d.kepid || d.tid || '—';
      const idtxt = [
        d.source,
        d.kepler_name ? `kepler: ${d.kepler_name}` : "",
        d.kepoi_name  ? `kepoi: ${d.kepoi_name}`   : "",
        d.kepid       ? `kepid: ${d.kepid}`        : "",
        d.toi_name    ? `toi_name: ${d.toi_name}`  : "",
        d.toi         ? `toi: ${d.toi}`            : "",
        d.tid         ? `tid: ${d.tid}`            : "",
        d.tic         ? `tic: ${d.tic}`            : "",
      ].filter(Boolean).join(" · ");

      tip.innerHTML = `<strong>${name}</strong><br>
                       ${idtxt}<br>
                       label: ${d.actual_label_name} · cluster: ${d.cluster}<br>
                       R⊕: ${Number.isFinite(d.prad_re)?d.prad_re.toFixed(2):'—'}, T_eq: ${Number.isFinite(d.teq_k)?d.teq_k.toFixed(0):'—'}`;
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
os.chdir(args.out_dir)  # make sure viewer.html and tnsa_tsne3d_points.json are here

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
