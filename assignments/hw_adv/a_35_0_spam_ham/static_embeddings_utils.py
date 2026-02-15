import json
import re
import struct
import hashlib
import pickle
import zipfile
from pathlib import Path
from typing import Iterable, List, Tuple

import numpy as np
import pandas as pd
from gensim.models import KeyedVectors, Word2Vec
from huggingface_hub import snapshot_download

DEFAULT_REPO_ID = "LogicLark-QuantumQuill/static-embeddings-en-50m-v2"
TOKEN_RE = re.compile(r"[a-z0-9']+")


class HashedStaticEmbeddings:
    """Wrapper for safetensors matrices that do not ship a token vocabulary.

    We map tokens to rows with a deterministic hash, so the static matrix can
    still be used as a lightweight embedding table.
    """

    def __init__(self, matrix: np.ndarray):
        self.matrix = matrix
        self.vector_size = int(matrix.shape[1])
        # Keep API-compatible attribute for existing logging.
        self.key_to_index = {f"__hashed_rows__": int(matrix.shape[0])}

    def __contains__(self, token: str) -> bool:
        return bool(token)

    def __getitem__(self, token: str) -> np.ndarray:
        digest = hashlib.blake2b(token.encode("utf-8"), digest_size=8).digest()
        idx = int.from_bytes(digest, byteorder="little", signed=False) % self.matrix.shape[0]
        return self.matrix[idx]


class VocabStaticEmbeddings:
    """Static embedding wrapper backed by an explicit token->row vocab mapping."""

    def __init__(self, matrix: np.ndarray, vocab: dict):
        self.matrix = matrix
        self.vocab = vocab
        self.vector_size = int(matrix.shape[1])
        self.key_to_index = vocab

    def __contains__(self, token: str) -> bool:
        return token in self.vocab

    def __getitem__(self, token: str) -> np.ndarray:
        return self.matrix[self.vocab[token]]


def clean_text(text: str) -> str:
    return str(text).lower().strip()


def tokenize(text: str) -> List[str]:
    return TOKEN_RE.findall(clean_text(text))


def load_dataset(dataset_path: Path) -> pd.DataFrame:
    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset not found: {dataset_path}")

    df = pd.read_csv(dataset_path, encoding="latin-1")
    if {"label", "text"}.issubset(df.columns):
        out = df[["label", "text"]].copy()
    elif {"v1", "v2"}.issubset(df.columns):
        out = df[["v1", "v2"]].copy()
        out.columns = ["label", "text"]
    else:
        raise ValueError("Unsupported CSV format. Expected label/text or v1/v2")

    out["label"] = out["label"].astype(str).str.lower().str.strip()
    out = out[out["label"].isin(["ham", "spam"])].copy()
    out["label_num"] = out["label"].map({"ham": 0, "spam": 1})
    out["text"] = out["text"].astype(str).map(clean_text)
    out = out[out["text"].str.len() > 0].reset_index(drop=True)
    if out.empty:
        raise ValueError("Dataset has no valid rows after preprocessing.")
    return out


def _iter_candidate_files(model_dir: Path) -> Iterable[Path]:
    files = [p for p in model_dir.rglob("*") if p.is_file()]

    priority = [
        "vectors.kv",
        "model.kv",
        "word2vec.kv",
        "model.bin",
        "vectors.bin",
        "word2vec.bin",
        "model.txt",
        "vectors.txt",
        "word2vec.txt",
        "model.vec",
        "vectors.vec",
        "word2vec.vec",
        "static_embeddings_en_50m_pruned_fp16_v2.safetensors",
    ]

    name_to_file = {p.name.lower(): p for p in files}
    seen = set()

    for name in priority:
        p = name_to_file.get(name)
        if p and p not in seen:
            seen.add(p)
            yield p

    for p in files:
        lower = p.name.lower()
        if lower.endswith((".kv", ".bin", ".txt", ".vec", ".model", ".safetensors")) and p not in seen:
            seen.add(p)
            yield p


def _load_first_tensor_from_safetensors(path: Path) -> np.ndarray:
    with path.open("rb") as f:
        header_len = struct.unpack("<Q", f.read(8))[0]
        header = json.loads(f.read(header_len))

        tensor_keys = [k for k in header.keys() if k != "__metadata__"]
        if not tensor_keys:
            raise ValueError(f"No tensors found in safetensors file: {path}")

        first_key = tensor_keys[0]
        info = header[first_key]
        if info.get("dtype") != "F16":
            raise ValueError(f"Unsupported dtype in {path}: {info.get('dtype')}")

        shape = info.get("shape")
        offsets = info.get("data_offsets")
        if not shape or not offsets:
            raise ValueError(f"Invalid tensor metadata in {path}")

        start, end = int(offsets[0]), int(offsets[1])
        f.seek(8 + header_len + start)
        raw = f.read(end - start)

    arr = np.frombuffer(raw, dtype=np.float16)
    arr = arr.reshape(shape).astype(np.float32)
    return arr


def _load_vocab_from_pth(path: Path) -> dict:
    with zipfile.ZipFile(path, "r") as zf:
        data_name = [n for n in zf.namelist() if n.endswith("data.pkl")]
        if not data_name:
            raise ValueError(f"No data.pkl in vocab checkpoint: {path}")
        payload = pickle.loads(zf.read(data_name[0]))

    if not isinstance(payload, dict):
        raise ValueError(f"Unexpected vocab payload type: {type(payload)}")
    vocab = payload.get("vocab")
    if not isinstance(vocab, dict):
        raise ValueError("Vocab checkpoint missing dict key 'vocab'")
    return vocab


def load_keyed_vectors(path: Path):
    suffix = path.suffix.lower()
    if suffix == ".kv":
        return KeyedVectors.load(str(path), mmap="r")
    if suffix == ".model":
        return Word2Vec.load(str(path)).wv
    if suffix in {".bin", ".txt", ".vec"}:
        return KeyedVectors.load_word2vec_format(str(path), binary=(suffix == ".bin"))
    if suffix == ".safetensors":
        matrix = _load_first_tensor_from_safetensors(path)
        vocab_path = path.with_suffix(".vocab.pth")
        if vocab_path.exists():
            vocab = _load_vocab_from_pth(vocab_path)
            return VocabStaticEmbeddings(matrix, vocab)
        return HashedStaticEmbeddings(matrix)
    raise ValueError(f"Unsupported embedding file: {path}")


def download_and_load_embeddings(
    repo_id: str = DEFAULT_REPO_ID,
    cache_dir: Path = None,
) -> Tuple[KeyedVectors, Path]:
    cache_base = cache_dir or (Path(__file__).resolve().parent / "models")
    cache_base.mkdir(parents=True, exist_ok=True)

    model_dir = Path(
        snapshot_download(
            repo_id=repo_id,
            cache_dir=str(cache_base),
            local_dir=str(cache_base / repo_id.replace("/", "__")),
            local_dir_use_symlinks=False,
        )
    )

    last_error = None
    for candidate in _iter_candidate_files(model_dir):
        try:
            kv = load_keyed_vectors(candidate)
            return kv, candidate
        except Exception as exc:
            last_error = exc
            continue

    raise RuntimeError(
        f"Could not load embeddings from {repo_id} in {model_dir}. "
        f"Last error: {last_error}"
    )


def texts_to_avg_vectors(texts: Iterable[str], kv: KeyedVectors) -> Tuple[np.ndarray, float]:
    dim = int(kv.vector_size)
    rows = []
    total_tokens = 0
    covered_tokens = 0

    for text in texts:
        tokens = tokenize(text)
        vecs = []
        for tok in tokens:
            total_tokens += 1
            if tok in kv:
                covered_tokens += 1
                vecs.append(kv[tok])

        if vecs:
            rows.append(np.mean(np.vstack(vecs), axis=0))
        else:
            rows.append(np.zeros(dim, dtype=np.float32))

    coverage = (covered_tokens / total_tokens) if total_tokens else 0.0
    return np.vstack(rows).astype(np.float32), coverage


def write_metadata(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
