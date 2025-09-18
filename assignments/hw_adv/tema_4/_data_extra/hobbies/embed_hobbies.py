# embed_hobbies.py
import os, json, requests, numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
def P(x): return os.path.join(SCRIPT_DIR, x)

EMBED_PORT = 8082
EMBED_MODEL = "nomic-embed-text-v1.5"

# with open(P("hobbies.csv"), "r", encoding="utf-8") as f:
#     hobbies = [ln.strip() for ln in f if ln.strip()]

# hobbies = ["golf", "gaming", "flying", "chess", "reading",
#            "cooking", "swimming", "running", "cycling", "hiking",
#            "painting", "music", "coding", "yoga", "dancing",
#            "fishing", "climbing", "travel", "movies", "photography"]

with open(P("hobbies.json"), "r", encoding="utf-8") as f:
    hobbies = json.load(f)

def embed_texts(texts):
    r = requests.post(
        f"http://localhost:{EMBED_PORT}/v1/embeddings",
        json={"model": EMBED_MODEL, "input": texts},
        timeout=120,
    )
    r.raise_for_status()
    return np.array([d["embedding"] for d in r.json()["data"]], dtype=np.float32)

embeddings = embed_texts(hobbies)

# cosine similarity
norm = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
sim_matrix = norm @ norm.T

np.save(P("embeddings.npy"), embeddings)
np.save(P("similarity.npy"), sim_matrix)
with open(P("hobbies.json"), "w", encoding="utf-8") as f:
    json.dump(hobbies, f, ensure_ascii=False, indent=2)

print(f"Saved: embeddings.npy, similarity.npy, hobbies.json")
