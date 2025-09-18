# plot_tsne_clusters.py
import os, numpy as np, matplotlib.pyplot as plt, matplotlib.colors as mcolors, matplotlib.patheffects as pe
from matplotlib.patches import Ellipse
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
from collections import defaultdict
from data_loader import HobbyData
from plot_style import apply_dark_theme

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
def P(x): return os.path.join(SCRIPT_DIR, x)

N_CLUSTERS = 10
DOMINANT_TO_CIRCLE = 5

apply_dark_theme()

data = HobbyData()
embeddings = data.embeddings
hobbies = data.hobbies

perplexity = max(5, min(40, len(hobbies)//6))
reduced = TSNE(n_components=2, random_state=42, perplexity=perplexity, init="pca").fit_transform(embeddings)
labels = KMeans(n_clusters=N_CLUSTERS, random_state=42, n_init=10).fit_predict(embeddings)

clusters = defaultdict(list)
for h, lab in zip(hobbies, labels):
    clusters[lab].append(h)

cluster_names = {cid: f"Cluster {cid}" for cid in range(N_CLUSTERS)}

def lighten(color, amt=0.80):
    r,g,b,a = mcolors.to_rgba(color)
    return (1 - (1 - r)*amt, 1 - (1 - g)*amt, 1 - (1 - b)*amt, 1)

def add_ellipse(ax, pts, color, nsig=2.2, lw=2.2):
    if len(pts) < 3: return
    mean = pts.mean(axis=0); cov = np.cov(pts.T)
    vals, vecs = np.linalg.eigh(cov); idx = vals.argsort()[::-1]
    vals, vecs = vals[idx], vecs[:, idx]
    ang = np.degrees(np.arctan2(*vecs[:,0][::-1]))
    w,h = 2*nsig*np.sqrt(np.maximum(vals, 1e-9))
    ax.add_patch(Ellipse(mean, w, h, angle=ang,
                         facecolor=lighten(color, 0.88), edgecolor=color,
                         lw=lw, alpha=0.25, zorder=1))

def draw_tsne(filename, with_ellipses=True):
    fig, ax = plt.subplots(figsize=(14, 12))
    cmap = plt.get_cmap("tab20", N_CLUSTERS)
    colors = [cmap(i) for i in range(N_CLUSTERS)]

    ax.scatter(reduced[:,0], reduced[:,1],
               c=[colors[l] for l in labels], s=24, alpha=0.9, edgecolors="none")

    if with_ellipses:
        sizes = {cid: len(clusters[cid]) for cid in range(N_CLUSTERS)}
        top = set(sorted(sizes, key=sizes.get, reverse=True)[:DOMINANT_TO_CIRCLE])
        for cid in top:
            add_ellipse(ax, reduced[labels==cid], colors[cid])

    centers = np.vstack([reduced[labels==i].mean(axis=0) for i in range(N_CLUSTERS)])
    for cid, (x,y) in enumerate(centers):
        ax.text(x, y, cluster_names[cid], color=colors[cid],
                fontsize=11, weight="bold", ha="center", va="center",
                path_effects=[pe.withStroke(linewidth=3.5, foreground="black", alpha=0.6)])

    ax.set_title("Hobby Clusters (t-SNE + KMeans)")
    fig.tight_layout()
    fig.savefig(P(filename), dpi=300)
    plt.close(fig)

draw_tsne("tsne_clusters.png", with_ellipses=True)
draw_tsne("tsne_clusters_no_ellipses.png", with_ellipses=False)

# Markdown
cmap = plt.get_cmap("tab20", N_CLUSTERS)
hex_color = {cid: mcolors.to_hex(cmap(cid)) for cid in range(N_CLUSTERS)}
sizes = {cid: len(clusters[cid]) for cid in range(N_CLUSTERS)}
order = sorted(sizes, key=sizes.get, reverse=True)

md = ["# Hobby Clusters (t-SNE + KMeans)\n"]
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

print("Saved: tsne_clusters.png, tsne_clusters_no_ellipses.png, tsne_clusters_colored.md")
