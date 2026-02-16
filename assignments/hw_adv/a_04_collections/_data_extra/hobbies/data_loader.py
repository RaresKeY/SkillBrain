# data_loader.py
import os, json, numpy as np
from sklearn.cluster import KMeans
from collections import defaultdict

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
def P(x): return os.path.join(SCRIPT_DIR, x)

class HobbyData:
    def __init__(self, n_clusters=10, random_state=42):
        self.embeddings = np.load(P("embeddings.npy"))
        self.similarity = np.load(P("similarity.npy"))
        with open(P("hobbies.json"), "r", encoding="utf-8") as f:
            self.hobbies = json.load(f)

        # default clustering
        self.n_clusters = n_clusters
        self.cluster_labels = KMeans(
            n_clusters=n_clusters, random_state=random_state, n_init=10
        ).fit_predict(self.embeddings)

    def closest(self, hobby, top_n=5):
        """Find closest hobbies to a single hobby."""
        if hobby not in self.hobbies:
            return []
        i = self.hobbies.index(hobby)
        sims = self.similarity[i]
        order = np.argsort(-sims)
        return [(self.hobbies[j], float(sims[j]))
                for j in order if j != i][:top_n]

    def get_clusters(self):
        """Return dict: cluster_id -> list of hobbies."""
        clusters = defaultdict(list)
        for h, lab in zip(self.hobbies, self.cluster_labels):
            clusters[lab].append(h)
        return dict(clusters)

    def group_closest(self, hobby_list, top_n=5):
        """
        Find closest hobbies to a group of hobbies.
        Uses the average embedding of the group.
        """
        idxs = [self.hobbies.index(h) for h in hobby_list if h in self.hobbies]
        if not idxs:
            return []

        group_vec = self.embeddings[idxs].mean(axis=0)
        group_vec = group_vec / np.linalg.norm(group_vec)

        norms = self.embeddings / np.linalg.norm(self.embeddings, axis=1, keepdims=True)
        sims = norms @ group_vec

        exclude = set(idxs)
        order = np.argsort(-sims)
        return [(self.hobbies[j], float(sims[j]))
                for j in order if j not in exclude][:top_n]
    

if __name__ == "__main__":
    data = HobbyData()

    print("Closest to 'Cycling':", data.closest("Cycling", 5))

    clusters = data.get_clusters()
    print("Cluster 0:", clusters[0])

    print("Group closest (['Cycling','Running']):", data.group_closest(["Cycling", "Running"], 5))
    print("Group closest (['Cycling']):", data.group_closest(["Cycling"], 5))

    

