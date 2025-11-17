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
p.add_argument("--dataset", choices=["koi","tess","both"], default="both", help="which dataset/features to use")
p.add_argument("--projector", choices=["tsne","umap","pca"], default="tsne", help="3D projector")
p.add_argument("--out_dir", default=".", help="output folder for json/html")
p.add_argument("--seed", type=int, default=42)
args = p.parse_args()

KOI_PATH  = Path(args.koi)
TESS_PATH = Path(args.tess)
OUT_DIR   = Path(args.out_dir)
OUT_DIR.mkdir(parents=True, exist_ok=True)

OUT_JSON = OUT_DIR / "tnsa_tsne3d_points.json"
OUT_HTML = OUT_DIR / "viewer_comb.html"
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

dataset = args.dataset.lower()

# ---------------------------
# Load dataframes and select features by dataset mode
# ---------------------------
if dataset == "koi":
    # KOI: use user-provided training features (base + FP flags)
    df_koi_raw = load_any(KOI_PATH)
    df_koi_raw.columns = df_koi_raw.columns.astype(str).str.strip()

    koi_features_base = [
        "koi_period", "koi_time0bk", "koi_duration", "koi_depth",
        "koi_prad", "koi_teq", "koi_steff", "koi_slogg", "koi_srad", "koi_kepmag",
    ]
    koi_features_fp = ["koi_fpflag_nt","koi_fpflag_ss","koi_fpflag_co","koi_fpflag_ec"]
    koi_required = koi_features_base + koi_features_fp + [koi_target_col]
    missing = [c for c in koi_required if c not in df_koi_raw.columns]
    if missing:
        raise KeyError(f"KOI missing required columns for selected features: {missing}")

    df = df_koi_raw[koi_required].copy()
    df["disposition"] = df[koi_target_col].apply(normalize_label)
    df.drop(columns=[koi_target_col], inplace=True)
    for c in ["kepid", "kepler_name", "kepoi_name"]:
        if c not in df_koi_raw.columns:
            df[c] = np.nan
        else:
            df[c] = df_koi_raw[c]
    df["source"] = "KOI"
    feature_cols = koi_features_base + koi_features_fp

elif dataset == "tess":
    # TESS: use user-provided training features (base set)
    df_tess_raw = load_any(TESS_PATH)
    df_tess_raw.columns = df_tess_raw.columns.astype(str).str.strip()

    tess_features_base = [
        "pl_orbper","pl_trandurh","pl_trandep",
        "pl_rade","pl_eqt","pl_insol",
        "st_teff","st_logg","st_rad","st_tmag","st_dist",
    ]
    tess_required = tess_features_base + [tess_target_col]
    missing = [c for c in tess_required if c not in df_tess_raw.columns]
    if missing:
        raise KeyError(f"TESS missing required columns for selected features: {missing}")

    df = df_tess_raw[tess_required].copy()
    df["disposition"] = df[tess_target_col].apply(normalize_label)
    df.drop(columns=[tess_target_col], inplace=True)
    for c in ["toi", "tid", "tic", "toi_name"]:
        if c not in df_tess_raw.columns:
            df[c] = np.nan
        else:
            df[c] = df_tess_raw[c]
    df["source"] = "TESS"
    feature_cols = tess_features_base

else:
    # BOTH: harmonize to unified shared schema (mapped features)
    # Load & harmonize KOI
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

    # Load & harmonize TESS
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

    # Union on shared feature set from the unified schema
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

    # Labels & matrix
    y = combo_df["disposition"].map({"CONFIRMED":1, "CANDIDATE":0}).astype(int).values
    X = combo_df[shared_feature_cols].values

# For single-dataset modes, finish building X/y and combo_df
if dataset in ("koi","tess"):
    # Filter to valid labels and numeric features
    df = df[df["disposition"].isin(["CONFIRMED","CANDIDATE"])]
    for c in feature_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    before = len(df)
    df = df.dropna(subset=feature_cols + ["disposition"]).reset_index(drop=True)
    after = len(df)
    print(f"Features ({dataset}): {feature_cols}")
    print(f"Dropped {before - after} rows with missing values (kept {after})")
    print(df["disposition"].value_counts())

    y = df["disposition"].map({"CONFIRMED":1, "CANDIDATE":0}).astype(int).values
    X = df[feature_cols].values
    combo_df = df

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
        # include commonly-inspected values, with fallbacks for dataset-specific columns
        "prad_re": float(
            row.get("prad_re", np.nan)
            if pd.notna(row.get("prad_re", np.nan)) else
            row.get("koi_prad", row.get("pl_rade", np.nan))
        ),
        "teq_k":   float(
            row.get("teq_k", np.nan)
            if pd.notna(row.get("teq_k", np.nan)) else
            row.get("koi_teq", row.get("pl_eqt", np.nan))
        ),
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
<title>KOI + TESS 3D Space Visualization</title>
<script src="https://cdnjs.cloudflare.com/ajax/libs/three.js/r128/three.min.js"></script>
<script src="https://cdn.jsdelivr.net/npm/three@0.128.0/examples/js/controls/OrbitControls.js"></script>
<style>
  body{margin:0; font-family:'Segoe UI', Helvetica, Arial, sans-serif; background:#f8f9fa}
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

VIEWER_HTML = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8"/>
<meta name="viewport" content="width=device-width, initial-scale=1"/>
<title>KOI + TESS 3D Space Explorer</title>
<script src="https://cdnjs.cloudflare.com/ajax/libs/three.js/r128/three.min.js"></script>
<script src="https://cdn.jsdelivr.net/npm/three@0.128.0/examples/js/controls/OrbitControls.js"></script>
<style>
  body{margin:0; font-family:'Segoe UI', Helvetica, Arial, sans-serif; background:#f8f9fa}
  #wrap{position:relative; width:100%; height:min(95vh,100svh); min-height:640px; overflow:hidden;}
  #title{position:absolute; top:16px; left:16px; z-index:10; color:#333; font-weight:700; font-size:20px}
  #sub{font-size:13px; color:#666; font-weight:400; margin-top:4px}

  #left-sliders{position:absolute; left:16px; top:174px; z-index:9; display:flex; flex-direction:column; gap:12px}
  #left-sliders .sliderline{background:#fff; padding:10px 12px; border-radius:12px; box-shadow:0 2px 12px rgba(0,0,0,.08); width:min(320px, 70vw)}
  #left-sliders label{display:block; font-size:12px; color:#555; margin-bottom:6px; font-weight:500}
  #left-sliders input[type="range"]{width:100%; height:6px; appearance:none; background:#e9ecef; border-radius:999px; outline:none}
  #left-sliders input[type="range"]::-webkit-slider-thumb{appearance:none; width:16px; height:16px; border-radius:50%; background:#1971ff; border:none; box-shadow:0 1px 3px rgba(0,0,0,.2)}

  #kpis{position:absolute; top:84px; left:16px; z-index:9; display:flex; gap:12px}
  .kpi{background:rgba(255,255,255,0.98); padding:12px 16px; border-radius:12px; box-shadow:0 2px 12px rgba(0,0,0,.1); min-width:120px; transition:all 0.2s; position:relative; overflow:hidden}
  .kpi:hover{transform:translateY(-2px); box-shadow:0 4px 16px rgba(0,0,0,.12)}
  .kpi .v{font-size:24px; font-weight:700; color:#222; line-height:1}
  .kpi .l{font-size:11px; color:#666; margin-top:4px; text-transform:uppercase; letter-spacing:.05em}
  .kpi.candidate{border-top:3px solid #DC143C}
  .kpi.confirmed{border-top:3px solid #32CD32}

  #color-legend{position:absolute; top:125px; left:16px; z-index:11; display:flex; gap:20px; background:rgba(255,255,255,0.95); padding:8px 12px; border-radius:8px; font-size:12px; box-shadow:0 2px 8px rgba(0,0,0,.05)}
  .color-dot{width:12px; height:12px; border-radius:50%; border:2px solid #fff; box-shadow:0 1px 3px rgba(0,0,0,.2)}

  #panel{position:absolute; top:12px; right:12px; z-index:10; background:rgba(255,255,255,0.98);
         padding:14px; border-radius:12px; box-shadow:0 2px 12px rgba(0,0,0,.1); min-width:260px}
  #panel label{display:block; font-size:12px; color:#555; margin:10px 0 6px; font-weight:500}
  #panel select, #panel input[type="text"], #panel input[type="range"]{width:100%; padding:6px 10px; margin-bottom:8px; border:1px solid #ddd; border-radius:8px; background:#fff}
  #panel select:focus, #panel input[type="text"]:focus{border-color:#4C78A8; outline:none; box-shadow:0 0 0 2px rgba(76,120,168,.2)}

  #tip{position:absolute; z-index:1000; background:rgba(20,20,20,.95); color:#fff; padding:10px 14px; border-radius:8px; font-size:13px; pointer-events:none; display:none; backdrop-filter:blur(10px); box-shadow:0 4px 12px rgba(0,0,0,.3)}
  #tip strong{color:#4C78A8; font-size:14px}
  #tip .row{margin:4px 0}

  #footer{position:absolute; bottom:12px; left:16px; z-index:5; font-size:12px; color:#666; background:rgba(255,255,255,.95); padding:8px 12px; border-radius:10px; box-shadow:0 2px 8px rgba(0,0,0,.08)}
  .pill{display:inline-block; padding:3px 10px; border-radius:999px; background:#e8eaed; margin-right:8px; color:#555; font-weight:500}

  #loader{position:absolute; top:50%; left:50%; transform:translate(-50%,-50%); z-index:100}
  .spinner{border:3px solid #f3f3f3; border-top:3px solid #4C78A8; border-radius:50%; width:50px; height:50px; animation:spin 1s linear infinite}
  @keyframes spin{0%{transform:rotate(0deg)} 100%{transform:rotate(360deg)}}
  /* Hide overlapping legend */
  #color-legend{display:none !important}
</style>
</head>
<body>
<div id="wrap">
  <div id="loader"><div class="spinner"></div></div>
  <div id="title">KOI + TESS 3D Space Explorer<div id="sub">Combined dataset • shared features • t-SNE/UMAP/PCA</div></div>

  <div id="kpis">
    <div class="kpi candidate"><div id="kpi-cand" class="v">—</div><div class="l">Candidates</div></div>
    <div class="kpi confirmed"><div id="kpi-conf" class="v">—</div><div class="l">Confirmed</div></div>
  </div>

  <div id="color-legend">
    <div class="color-item"><span class="color-dot" style="background:#DC143C"></span>Candidate</div>
    <div class="color-item"><span class="color-dot" style="background:#32CD32"></span>Confirmed</div>
  </div>

  <div id="left-sliders">
    <div class="sliderline"><label for="spread">Point spread</label><input id="spread" type="range" min="0.5" max="4" step="0.1" value="2.5"/></div>
    <div class="sliderline"><label for="psize">Point size</label><input id="psize" type="range" min="0.3" max="2.0" step="0.1" value="1.0"/></div>
  </div>
  <div id="tip"></div>

  <div id="footer">
    <span class="pill">drag = orbit</span>
    <span class="pill">wheel = zoom</span>
    <span class="pill">shift+drag = pan</span>
  </div>
</div>

<script>
const clusterPalette = ["#4C78A8","#F58518","#E45756","#72B7B2","#54A24B","#EECA3B","#B279A2","#FF9DA6","#9D755D","#BAB0AC"];
const clusterColor = c => clusterPalette[Math.abs(c) % clusterPalette.length];

async function loadData(){
  const res = await fetch("tnsa_tsne3d_points.json", {cache:"no-cache"});
  if(!res.ok) throw new Error("Failed to load data: "+res.status);
  return await res.json();
}

function normalize(d){
  const xs=d.map(v=>v.x), ys=d.map(v=>v.y), zs=d.map(v=>v.z);
  const minx=Math.min(...xs), maxx=Math.max(...xs);
  const miny=Math.min(...ys), maxy=Math.max(...ys);
  const minz=Math.min(...zs), maxz=Math.max(...zs);
  const scale = 100 / Math.max(maxx-minx, maxy-miny, maxz-minz);
  d.forEach(v=>{ v.X=(v.x-minx-(maxx-minx)/2)*scale; v.Y=(v.y-miny-(maxy-miny)/2)*scale; v.Z=(v.z-minz-(maxz-minz)/2)*scale; v.X0=v.X; v.Y0=v.Y; v.Z0=v.Z;});
  return d;
}

function createAxisLabel(text, pos, color='#333'){
  const c=document.createElement('canvas'), ctx=c.getContext('2d'); c.width=128; c.height=128;
  ctx.font='bold 48px Arial'; ctx.fillStyle=color; ctx.textAlign='center'; ctx.textBaseline='middle'; ctx.fillText(text, 64, 64);
  const s=new THREE.Sprite(new THREE.SpriteMaterial({map:new THREE.CanvasTexture(c), transparent:true})); s.position.copy(pos); s.scale.set(5,5,1); return s;
}

function updateColorMode(mode, data, actualGroup, clusterGroup){
  const colorLegend = document.getElementById('color-legend');
  if (mode==='actual'){
    actualGroup.visible=true; clusterGroup.visible=false;
    colorLegend.innerHTML = `
      <div class="color-item"><span class="color-dot" style="background:#DC143C"></span>Candidate</div>
      <div class="color-item"><span class="color-dot" style="background:#32CD32"></span>Confirmed</div>
    `;
  } else {
    actualGroup.visible=false; clusterGroup.visible=true;
    const uniq = [...new Set(data.map(d=>d.cluster))].sort((a,b)=>a-b);
    colorLegend.innerHTML = uniq.slice(0,4).map(c=>`<div class="color-item"><span class="color-dot" style="background:${clusterColor(c)}"></span>Cluster ${c}</div>`).join('');
    if(uniq.length>4){ colorLegend.innerHTML += `<div class="color-item" style="color:#999">+${uniq.length-4} more</div>`; }
  }
}

loadData().then(DATA=>{
  normalize(DATA);
  document.getElementById('loader').style.display='none';

  const total = DATA.length;
  const confirmed = DATA.filter(d=>String(d.actual_label_name||'').toUpperCase()==='CONFIRMED').length;
  const candidates = total - confirmed;
  document.getElementById('kpi-cand').textContent = candidates.toLocaleString();
  document.getElementById('kpi-conf').textContent = confirmed.toLocaleString();

  const wrap = document.getElementById('wrap');
  const renderer = new THREE.WebGLRenderer({antialias:true, alpha:false}); wrap.appendChild(renderer.domElement);
  renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2)); renderer.shadowMap.enabled=true; renderer.shadowMap.type=THREE.PCFSoftShadowMap;

  const scene=new THREE.Scene(); scene.background=new THREE.Color(0xf8f9fa); scene.fog=new THREE.Fog(0xf8f9fa,200,500);
  const camera=new THREE.PerspectiveCamera(60,1,0.1,1000); camera.position.set(120,80,120);
  function fit(){ const w=wrap.clientWidth,h=wrap.clientHeight; renderer.setSize(w,h,false); camera.aspect=w/h; camera.updateProjectionMatrix(); }
  fit(); new ResizeObserver(fit).observe(wrap);

  const controls=new THREE.OrbitControls(camera, renderer.domElement); controls.enableDamping=true; controls.dampingFactor=.05; controls.minDistance=30; controls.maxDistance=400; controls.autoRotate=false; controls.autoRotateSpeed=.5;

  scene.add(new THREE.AmbientLight(0xffffff,.6)); const dl=new THREE.DirectionalLight(0xffffff,.4); dl.position.set(50,100,50); dl.castShadow=true; scene.add(dl); const dl2=new THREE.DirectionalLight(0xffffff,.2); dl2.position.set(-50,50,-50); scene.add(dl2);

  scene.add(new THREE.GridHelper(200,20,0xaaaaaa,0xdddddd));
  const axes=new THREE.Group(); const L=100;
  axes.add(new THREE.Line(new THREE.BufferGeometry().setFromPoints([new THREE.Vector3(0,0,0), new THREE.Vector3(L,0,0)]), new THREE.LineBasicMaterial({color:0x000000})));
  axes.add(new THREE.Line(new THREE.BufferGeometry().setFromPoints([new THREE.Vector3(0,0,0), new THREE.Vector3(0,L,0)]), new THREE.LineBasicMaterial({color:0x000000})));
  axes.add(new THREE.Line(new THREE.BufferGeometry().setFromPoints([new THREE.Vector3(0,0,0), new THREE.Vector3(0,0,L)]), new THREE.LineBasicMaterial({color:0x000000})));
  scene.add(axes);
  scene.add(createAxisLabel('X', new THREE.Vector3(110,0,0), '#000000'));
  scene.add(createAxisLabel('Y', new THREE.Vector3(0,110,0), '#000000'));
  scene.add(createAxisLabel('Z', new THREE.Vector3(0,0,110), '#000000'));

  const sphere=new THREE.SphereGeometry(0.5,12,12);
  const matCand=new THREE.MeshPhongMaterial({color:0xDC143C, emissive:0xDC143C, emissiveIntensity:.2, shininess:100});
  const matConf=new THREE.MeshPhongMaterial({color:0x32CD32, emissive:0x32CD32, emissiveIntensity:.2, shininess:100});
  const clusterMaterials=clusterPalette.map(col=>new THREE.MeshPhongMaterial({color:col, emissive:col, emissiveIntensity:.1, shininess:80}));

  const actualGroup=new THREE.Group(); const clusterGroup=new THREE.Group(); scene.add(actualGroup, clusterGroup);
  const actualMeshes=[]; const clusterMeshes=[];
  DATA.forEach((d,i)=>{
    const isConf = String(d.actual_label_name||'').toUpperCase()==='CONFIRMED';
    const mesh=new THREE.Mesh(sphere, isConf?matConf:matCand); mesh.position.set(d.X,d.Y,d.Z); mesh.userData={index:i,data:d}; mesh.castShadow=mesh.receiveShadow=true; actualGroup.add(mesh); actualMeshes.push(mesh);
    const ci=Math.abs(d.cluster)%clusterPalette.length; const cmesh=new THREE.Mesh(sphere, clusterMaterials[ci]); cmesh.position.set(d.X,d.Y,d.Z); cmesh.userData={index:i,data:d}; cmesh.castShadow=cmesh.receiveShadow=true; clusterGroup.add(cmesh); clusterMeshes.push(cmesh);
  });

  const spreadEl=document.getElementById('spread'); const psizeEl=document.getElementById('psize');
  function updatePositions(){ const spread=parseFloat(spreadEl.value), psize=parseFloat(psizeEl.value); DATA.forEach((d,i)=>{ const x=d.X0*spread,y=d.Y0*spread,z=d.Z0*spread; actualMeshes[i].position.set(x,y,z); actualMeshes[i].scale.set(psize,psize,psize); clusterMeshes[i].position.set(x,y,z); clusterMeshes[i].scale.set(psize,psize,psize); }); }
  spreadEl.addEventListener('input', updatePositions); psizeEl.addEventListener('input', updatePositions); updatePositions();

  const modeSel=document.getElementById('colorMode'); const hasMode=!!modeSel; function refreshMode(){ const mode=hasMode?modeSel.value:'actual'; updateColorMode(mode, DATA, actualGroup, clusterGroup);} if(hasMode){ modeSel.addEventListener('change', refreshMode);} refreshMode();

  const qEl=document.getElementById('q'); if(qEl){ qEl.addEventListener('keydown', e=>{ if(e.key!=='Enter') return; const needle=e.target.value.trim().toLowerCase(); if(!needle) return; const i=DATA.findIndex(d=> (d.kepid && String(d.kepid).toLowerCase().includes(needle)) || (d.kepler_name && String(d.kepler_name).toLowerCase().includes(needle)) || (d.toi && String(d.toi).toLowerCase().includes(needle)) ); if(i>=0){ const s=parseFloat(spreadEl.value); const d=DATA[i]; const target=new THREE.Vector3(d.X0*s,d.Y0*s,d.Z0*s); controls.target.lerp(target,.5); camera.position.lerp(new THREE.Vector3(target.x+40,target.y+40,target.z+40),.5); const ring=new THREE.Mesh(new THREE.TorusGeometry(3,.3,8,20), new THREE.MeshBasicMaterial({color:0xFFD700, transparent:true, opacity:.9})); ring.position.copy(target); scene.add(ring); let sc=1; const anim=()=>{ sc+=.02; ring.scale.set(sc,sc,sc); ring.material.opacity=Math.max(0,.9-(sc-1)*2); if(ring.material.opacity>0){ requestAnimationFrame(anim);} else { scene.remove(ring);} }; anim(); setTimeout(()=>e.target.value='',1500);} else { e.target.style.borderColor='#ff4444'; setTimeout(()=>e.target.style.borderColor='',500);} }); }

  const tip=document.getElementById('tip'); const ray=new THREE.Raycaster(); const mouse=new THREE.Vector2();
  function onMove(e){ const rect=renderer.domElement.getBoundingClientRect(); mouse.x=((e.clientX-rect.left)/rect.width)*2-1; mouse.y=-((e.clientY-rect.top)/rect.height)*2+1; ray.setFromCamera(mouse,camera); const target=(hasMode && modeSel.value==='actual')?actualGroup:clusterGroup; const hits=ray.intersectObjects(target.children); if(hits.length){ const d=hits[0].object.userData.data; const name = d.kepler_name && d.kepler_name!=='nan' ? d.kepler_name : (d.toi?`TOI ${d.toi}`:'Unnamed'); tip.innerHTML = `
        <strong>${name}</strong>
        <div class="row">Source: ${d.source}</div>
        <div class="row">KepID: ${d.kepid||'—'}; TOI: ${d.toi||'—'}</div>
        <div class="row">Status: <span style=\"color:${String(d.actual_label_name||'').toUpperCase()==='CONFIRMED'?'#32CD32':'#DC143C'}\">${d.actual_label_name}</span></div>
        <div class="row">Cluster: ${d.cluster}</div>
        <div class="row">Radius: ${Number.isFinite(d.prad_re)?d.prad_re.toFixed(2)+' R⊕':'—'}</div>
        <div class="row">Temp: ${Number.isFinite(d.teq_k)?d.teq_k.toFixed(0)+' K':'—'}</div>`; tip.style.display='block'; tip.style.left=(e.clientX+15)+'px'; tip.style.top=(e.clientY-20)+'px'; document.body.style.cursor='pointer'; hits[0].object.scale.setScalar(parseFloat(psizeEl.value)*1.5);
    } else { tip.style.display='none'; document.body.style.cursor='default'; const psize=parseFloat(psizeEl.value); actualMeshes.forEach(m=>m.scale.setScalar(psize)); clusterMeshes.forEach(m=>m.scale.setScalar(psize)); } }
  renderer.domElement.addEventListener('mousemove', onMove); renderer.domElement.addEventListener('mouseleave', ()=>{ tip.style.display='none'; document.body.style.cursor='default'; });

  function animate(){ controls.update(); renderer.render(scene,camera); requestAnimationFrame(animate);} animate();
}).catch(err=>{
  document.getElementById('loader').style.display='none';
  document.body.innerHTML = '<div style="padding:40px; text-align:center; color:#d32f2f"><h2>Error Loading Data</h2><pre style="white-space:pre-wrap">'+String(err)+'</pre><p>Make sure tnsa_tsne3d_points.json is in the same directory</p></div>';
});
</script>
</body>
</html>
"""

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
os.chdir(args.out_dir)  # make sure viewer_comb.html and tnsa_tsne3d_points.json are here

server = ThreadingHTTPServer(("127.0.0.1", PORT), SimpleHTTPRequestHandler)
print(f"🌐 Serving on http://127.0.0.1:{PORT}/viewer.html  (Ctrl+C to stop)")

# open the browser, then block in serve_forever()
print(f"Open: http://127.0.0.1:{PORT}/viewer_comb.html")
webbrowser.open(f"http://127.0.0.1:{PORT}/viewer_comb.html")
try:
    server.serve_forever()
except KeyboardInterrupt:
    pass
finally:
    server.server_close()
    print("👋 Server stopped")
