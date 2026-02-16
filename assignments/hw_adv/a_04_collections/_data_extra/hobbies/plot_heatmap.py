# plot_heatmap.py
import os, numpy as np, matplotlib.pyplot as plt
from data_loader import HobbyData
from plot_style import apply_dark_theme

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
def P(x): return os.path.join(SCRIPT_DIR, x)

apply_dark_theme()

data = HobbyData()

# limit to 20 hobbies
N = 20
sim_matrix = data.similarity[:N, :N]
labels = data.hobbies[:N]

# moderate figure size
plt.figure(figsize=(10, 8))  # 10*150 = 1500px wide, 8*150 = 1200px tall
im = plt.imshow(sim_matrix, cmap="plasma")
plt.colorbar(im, fraction=0.046, pad=0.04)

plt.xticks(range(len(labels)), labels, rotation=90, fontsize=12, fontweight="bold")
plt.yticks(range(len(labels)), labels, fontsize=12, fontweight="bold")

plt.title("Hobby Similarity (Cosine)", fontsize=18, fontweight="bold")

# no tight_layout, let matplotlib keep spacing
plt.savefig(P("heatmap_20.png"), dpi=150, bbox_inches="tight")

plt.show()

np.savetxt(P("similarity_20.csv"), sim_matrix, delimiter=",")
print("Saved: heatmap_20.png, similarity_20.csv")
