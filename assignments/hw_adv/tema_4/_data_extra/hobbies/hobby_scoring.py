# pip install numpy matplotlib scikit-learn requests

import os, json, requests, numpy as np
import matplotlib
# Try interactive; fall back to headless
try:
    matplotlib.use("TkAgg")
except Exception:
    matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.patheffects as pe
from matplotlib.patches import Ellipse
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
from collections import defaultdict

# -------------------- CONFIG --------------------
CHAT_PORT = 8080
EMBED_PORT = 8082
EMBED_MODEL = "nomic-embed-text-v1.5"
N_CLUSTERS = 10
DOMINANT_TO_CIRCLE = 5
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
def P(x): return os.path.join(SCRIPT_DIR, x)

# VS Code dark-like theme
plt.rcParams.update({
    "axes.facecolor": "#1e1e1e", "figure.facecolor": "#1e1e1e",
    "axes.edgecolor": "#DDDDDD", "axes.labelcolor": "#DDDDDD",
    "xtick.color": "#CCCCCC", "ytick.color": "#CCCCCC",
    "text.color": "#DDDDDD", "grid.color": "#3a3a3a",
    "legend.edgecolor": "#1e1e1e", "legend.facecolor": "#2d2d30"
})

# -------------------- DATA --------------------
with open(P("hobbies.csv"), "r", encoding="utf-8") as f:
    hobbies = [ln.strip() for ln in f if ln.strip()]

# -------------------- EMBEDDINGS --------------------
def embed_texts(texts):
    r = requests.post(
        f"http://localhost:{EMBED_PORT}/v1/embeddings",
        json={"model": EMBED_MODEL, "input": texts},
        timeout=120,
    )
    r.raise_for_status()
    return np.array([d["embedding"] for d in r.json()["data"]], dtype=np.float32)

embeddings = embed_texts(hobbies)
print(f"[DEBUG] Got embeddings: shape={embeddings.shape}, dtype={embeddings.dtype}")
print(f"[DEBUG] First 5 dims of first vector: {embeddings[0][:5]}")

# cosine similarity
norm = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
sim_matrix = norm @ norm.T
print(f"[DEBUG] Cosine similarity matrix: shape={sim_matrix.shape}")
print(f"[DEBUG] min={sim_matrix.min():.4f}, max={sim_matrix.max():.4f}")
np.savetxt(P("similarity.csv"), sim_matrix, delimiter=",")

def show_closest(hobby, top_n=5):
    if hobby not in hobbies:
        print(f"Hobby '{hobby}' not found."); return
    i = hobbies.index(hobby); sims = sim_matrix[i]
    order = np.argsort(-sims)
    print(f"\nClosest hobbies to '{hobby}':")
    for j in [k for k in order if k != i][:top_n]:
        print(f"  {hobbies[j]:<22} {sims[j]:.3f}")

show_closest("Cycling", 5)
show_closest("Chess", 5)

# heatmap without labels
plt.figure(figsize=(16, 14))
im = plt.imshow(sim_matrix, cmap="plasma")
plt.colorbar(im, fraction=0.046, pad=0.04)
plt.xticks([]); plt.yticks([])
plt.title("Hobby Similarity (Cosine) — no labels")
plt.tight_layout(); plt.savefig(P("heatmap_nolabels.png"), dpi=300); plt.close()

# -------------------- t-SNE + KMEANS --------------------
perplexity = max(5, min(40, len(hobbies)//6))
reduced = TSNE(n_components=2, random_state=42, perplexity=perplexity, init="pca").fit_transform(embeddings)
labels = KMeans(n_clusters=N_CLUSTERS, random_state=42, n_init=10).fit_predict(embeddings)

# -------------------- CLUSTER NAMES (chat first, simple indexed fallback) --------------------
def get_chat_model():
    try:
        r = requests.get(f"http://localhost:{CHAT_PORT}/v1/models", timeout=5)
        r.raise_for_status()
        data = r.json().get("data", [])
        return data[0]["id"] if data else None
    except Exception:
        return None
CHAT_MODEL = get_chat_model() or "auto"

def name_cluster_via_chat(hobby_list):
    prompt = {
        "instruction": "Return a SHORT category (1–3 words). Output ONLY JSON: {\"label\":\"<short>\"}.",
        "hobbies": hobby_list[:50]
    }
    try:
        r = requests.post(
            f"http://localhost:{CHAT_PORT}/v1/chat/completions",
            json={"model": CHAT_MODEL, "temperature": 0.2, "max_tokens": 20,
                  "messages": [
                      {"role":"system","content":"You return concise JSON only."},
                      {"role":"user","content":json.dumps(prompt)}
                  ]},
            timeout=20,
        )
        r.raise_for_status()
        out = r.json()["choices"][0]["message"]["content"].strip()
        label = str(json.loads(out).get("label","")).strip().strip('"').strip("'")
        if label and label.lower() not in {"misc","other","general","hobbies","cluster"}:
            return label[:50]
    except Exception:
        pass
    return None

clusters = defaultdict(list)
for h, lab in zip(hobbies, labels):
    clusters[lab].append(h)

cluster_names = {}
for cid in range(N_CLUSTERS):
    name = name_cluster_via_chat(sorted(clusters[cid])) or f"Cluster {cid}"
    cluster_names[cid] = name

# -------------------- PLOTTING HELPERS --------------------
def lighten(color, amt=0.80):
    r,g,b,a = mcolors.to_rgba(color)
    return (1 - (1 - r)*amt, 1 - (1 - g)*amt, 1 - (1 - b)*amt, 1)

def add_ellipse(ax, pts, color, nsig=2.2, lw=2.2, alpha_fill=0.0, alpha_edge=0.9, z=1):
    if len(pts) < 3: return
    mean = pts.mean(axis=0); cov = np.cov(pts.T)
    vals, vecs = np.linalg.eigh(cov); idx = vals.argsort()[::-1]
    vals, vecs = vals[idx], vecs[:, idx]
    ang = np.degrees(np.arctan2(*vecs[:,0][::-1]))
    w,h = 2*nsig*np.sqrt(np.maximum(vals, 1e-9))
    ax.add_patch(Ellipse(mean, w, h, angle=ang,
                         facecolor=lighten(color, 0.88), edgecolor=color,
                         lw=lw, alpha=alpha_fill, zorder=z))
    ax.add_patch(Ellipse(mean, w, h, angle=ang,
                         facecolor=(0,0,0,0), edgecolor=color,
                         lw=lw, alpha=alpha_edge, zorder=z+0.1))

def draw_tsne(filename, with_ellipses=True, show=False):
    fig, ax = plt.subplots(figsize=(14, 12))
    cmap = plt.get_cmap("tab20", N_CLUSTERS)
    colors = [cmap(i) for i in range(N_CLUSTERS)]

    # points
    ax.scatter(reduced[:,0], reduced[:,1],
               c=[colors[l] for l in labels],
               s=24, alpha=0.9, edgecolors="none", zorder=2)

    # ellipses
    if with_ellipses:
        sizes = {cid: len(clusters[cid]) for cid in range(N_CLUSTERS)}
        top = set(sorted(sizes, key=sizes.get, reverse=True)[:DOMINANT_TO_CIRCLE])
        for cid in range(N_CLUSTERS):
            pts = reduced[labels==cid]
            add_ellipse(ax, pts, colors[cid],
                        lw=3.0 if cid in top else 2.0,
                        alpha_fill=0.0 if cid in top else 0.18,
                        alpha_edge=0.95 if cid in top else 0.8, z=1)

    # center names (match dot color)
    centers = np.vstack([reduced[labels==i].mean(axis=0) for i in range(N_CLUSTERS)])
    for cid, (x,y) in enumerate(centers):
        ax.text(x, y, cluster_names[cid],
                color=colors[cid], fontsize=11, weight="bold", ha="center", va="center",
                zorder=3, path_effects=[pe.withStroke(linewidth=3.5, foreground="black", alpha=0.6)])

    # no legend; clean layout
    ax.set_title("Hobby Clusters (t-SNE + KMeans)")
    fig.tight_layout()
    fig.savefig(P(filename), dpi=300)
    if show:
        try: plt.show()
        except Exception: pass
    plt.close(fig)

# save both variants
draw_tsne("tsne_clusters.png",             with_ellipses=True,  show=True)
draw_tsne("tsne_clusters_no_ellipses.png", with_ellipses=False, show=False)

# -------------------- COLORED MARKDOWN --------------------
cmap = plt.get_cmap("tab20", N_CLUSTERS)
hex_color = {cid: mcolors.to_hex(cmap(cid)) for cid in range(N_CLUSTERS)}
sizes = {cid: len(clusters[cid]) for cid in range(N_CLUSTERS)}
order = sorted(sizes, key=sizes.get, reverse=True)

md = []
md.append("# Hobby Clusters (t-SNE + KMeans)\n")
md.append('<p><img src="tsne_clusters.png" width="900"/></p>\n')
md.append('<p><img src="tsne_clusters_no_ellipses.png" width="900"/></p>\n')
md.append(f"- Total hobbies: **{len(hobbies)}**\n- Clusters: **{N_CLUSTERS}**\n")

for cid in order:
    col = hex_color[cid]
    md.append(f'\n<h2 style="color:{col}">{cluster_names[cid]} ({sizes[cid]})</h2>')
    md.append("<ul>")
    for h in sorted(clusters[cid]):
        md.append(f'<li><span style="color:{col}">{h}</span></li>')
    md.append("</ul>")

with open(P("tsne_clusters_colored.md"), "w", encoding="utf-8") as f:
    f.write("\n".join(md))

print("Saved: tsne_clusters.png, tsne_clusters_no_ellipses.png, "
      "tsne_clusters_colored.md, heatmap_nolabels.png, similarity.csv")
