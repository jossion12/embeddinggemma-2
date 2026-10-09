# -*- coding: utf-8 -*-
"""多模态文件库核心：把一个目录内的文档/图片/音频/视频编码进统一向量索引。

设计要点：
- 每个被收录的目录是一个“集合”（Collection），索引数据存放在项目内
  collections/<slug>/（sqlite 元数据 + faiss 向量），不污染源目录
- 增量扫描：mtime_ns + size 判断变更；新增→编码，变更→重编码，删除→移除；
  未变更的文件直接跳过（“已经 embedding 过的就不用再 embedding”）
- 向量统一 256 维（MRL 截断），四类模态在同一空间，支持跨模态检索

CLI:
    python library.py add "D:/某个目录"     # 收录目录并扫描
    python library.py list                  # 列出所有集合
    python library.py scan [slug|all]       # 手动增量扫描
    python library.py remove <slug>         # 移除集合（只删索引，不动源文件）
    python library.py search <slug> "查询"  # 命令行检索
"""
import argparse
import hashlib
import json
import re
import shutil
import sqlite3
import threading
import time
from pathlib import Path

import faiss
import numpy as np
from PIL import Image

from avcodec import TARGET_SR, detect_media_type, read_audio_bytes, read_video_frames
from doccodec import DOC_EXTS, extract_text, make_snippet

ROOT = Path(__file__).resolve().parent
COLLECTIONS_DIR = ROOT / "collections"
MODEL_PATH = r"D:\model\modelscope\google\embeddinggemma-2"
DIMS = 256

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".gif"}
MODALITY_BY_EXT = {
    **{e: "image" for e in IMAGE_EXTS},
    **{e: "audio" for e in (".wav", ".mp3", ".flac", ".ogg", ".m4a", ".aac")},
    **{e: "video" for e in (".mp4", ".webm", ".mov", ".mkv", ".avi", ".m4v")},
    **{e: "document" for e in DOC_EXTS},
}
MAX_AUDIO_SECONDS = 320  # 8K 上下文按 25 tokens/秒约 327 秒，超限跳过避免截断失真

CREATE_SQL = """
CREATE TABLE IF NOT EXISTS files (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    path TEXT NOT NULL UNIQUE,        -- 相对集合根目录的 posix 路径
    mtime_ns INTEGER NOT NULL,        -- 变更检测：st_mtime_ns
    size INTEGER NOT NULL,            -- 变更检测：st_size
    modality TEXT NOT NULL,           -- image / audio / video / document
    embedding BLOB NOT NULL,          -- float32[DIMS]
    snippet TEXT,                     -- 文档摘要片段（其他模态为 NULL）
    embedded_at REAL NOT NULL
);
CREATE TABLE IF NOT EXISTS meta (key TEXT PRIMARY KEY, value TEXT);
CREATE TABLE IF NOT EXISTS labels (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT NOT NULL,               -- 用户命名的概念标签
    anchor_id INTEGER NOT NULL,       -- 锚点图片 files.id
    threshold REAL NOT NULL,          -- 传播时的相似度阈值
    created_at REAL NOT NULL
);
CREATE TABLE IF NOT EXISTS label_docs (
    label_id INTEGER NOT NULL REFERENCES labels(id) ON DELETE CASCADE,
    doc_id INTEGER NOT NULL,          -- 关联到的 files.id
    sim REAL NOT NULL,                -- 与锚点的相似度
    PRIMARY KEY (label_id, doc_id)
);
CREATE INDEX IF NOT EXISTS idx_label_docs_doc ON label_docs(doc_id);
"""

# 全进程只允许一个扫描/写入（模型推理 + 索引落盘都是重资源）
SCAN_LOCK = threading.Lock()


def _like_escape(s: str) -> str:
    return s.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")


def modality_of(name: str) -> str | None:
    return MODALITY_BY_EXT.get(Path(name).suffix.lower())


def make_slug(root: Path) -> str:
    digest = hashlib.sha1(str(root.resolve()).encode("utf-8")).hexdigest()[:8]
    name = re.sub(r"[^\w一-鿿-]+", "_", root.name) or "collection"
    return f"{name}-{digest}"


class Embedder:
    """惰性单例：封装模型的四种模态编码（统一 256 维 + 归一化）。"""

    _instance = None
    _lock = threading.Lock()

    @classmethod
    def get(cls) -> "Embedder":
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    import torch
                    from sentence_transformers import SentenceTransformer

                    device = "cuda" if torch.cuda.is_available() else "cpu"
                    model = SentenceTransformer(
                        MODEL_PATH,
                        model_kwargs={"torch_dtype": torch.float32},  # fp16 会出 NaN
                        device=device,
                    )
                    model[0].processor.image_processor.max_soft_tokens = 280
                    cls._instance = cls(model)
        return cls._instance

    def __init__(self, model):
        self.model = model

    @staticmethod
    def _t(vecs) -> np.ndarray:
        return np.ascontiguousarray(vecs, dtype=np.float32)

    def encode_images(self, imgs) -> np.ndarray:
        return self._t(self.model.encode(
            imgs, normalize_embeddings=True, truncate_dim=DIMS, show_progress_bar=False))

    def encode_audios(self, waves: list[np.ndarray]) -> np.ndarray:
        batch = [{"audio": {"array": w, "sampling_rate": TARGET_SR}} for w in waves]
        return self._t(self.model.encode(
            batch, normalize_embeddings=True, truncate_dim=DIMS, show_progress_bar=False))

    def encode_videos(self, frames_list: list) -> np.ndarray:
        batch = [{"video": frames} for frames in frames_list]
        return self._t(self.model.encode(
            batch, normalize_embeddings=True, truncate_dim=DIMS, show_progress_bar=False))

    def encode_documents(self, texts: list[str]) -> np.ndarray:
        # 文档已按 "title: x | text: y" 组装，不加任务前缀（模型卡规范）
        return self._t(self.model.encode(
            texts, normalize_embeddings=True, truncate_dim=DIMS, show_progress_bar=False))

    def encode_query(self, text: str, dims: int = DIMS) -> np.ndarray:
        return self._t(self.model.encode(
            [text], prompt_name="SearchQuery",
            normalize_embeddings=True, truncate_dim=dims, show_progress_bar=False))[0]


class Collection:
    """一个被收录的目录及其索引。"""

    def __init__(self, cdir: Path):
        self.dir = cdir
        self.slug = cdir.name
        self.meta = json.loads((cdir / "meta.json").read_text(encoding="utf-8"))
        self.root = Path(self.meta["root"])
        self.db_path = cdir / "index.db"
        self.faiss_path = cdir / "vectors.faiss"
        self.scanning = False
        self._index: faiss.Index | None = None
        self._index_version = -1

    # ---------- 生命周期 ----------

    @classmethod
    def create(cls, root: Path) -> "Collection":
        root = root.resolve()
        if not root.is_dir():
            raise ValueError(f"目录不存在: {root}")
        cdir = COLLECTIONS_DIR / make_slug(root)
        if cdir.exists():
            raise ValueError(f"该目录已被收录（{cdir.name}）")
        cdir.mkdir(parents=True)
        (cdir / "meta.json").write_text(json.dumps(
            {"name": root.name, "root": str(root), "created_at": time.time()},
            ensure_ascii=False, indent=2), encoding="utf-8")
        col = cls(cdir)
        conn = col._connect()
        conn.close()
        return col

    @classmethod
    def load(cls, slug: str) -> "Collection | None":
        cdir = COLLECTIONS_DIR / slug
        if not (cdir / "meta.json").exists():
            return None
        return cls(cdir)

    @property
    def name(self) -> str:
        return self.meta.get("name") or self.slug.rsplit("-", 1)[0]

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path, timeout=30)
        conn.execute("PRAGMA journal_mode=WAL")
        conn.executescript(CREATE_SQL)
        return conn

    @staticmethod
    def _meta_get(conn: sqlite3.Connection, key: str, default: str = "") -> str:
        row = conn.execute("SELECT value FROM meta WHERE key=?", (key,)).fetchone()
        return row[0] if row else default

    @staticmethod
    def _meta_set(conn: sqlite3.Connection, key: str, value: str) -> None:
        conn.execute(
            "INSERT INTO meta(key,value) VALUES(?,?) "
            "ON CONFLICT(key) DO UPDATE SET value=excluded.value", (key, value))

    def info(self) -> dict:
        conn = self._connect()
        counts = dict(conn.execute(
            "SELECT modality, count(*) FROM files GROUP BY modality").fetchall())
        version = int(self._meta_get(conn, "version", "0"))
        last_scan = float(self._meta_get(conn, "last_scan", "0"))
        conn.close()
        return {
            "slug": self.slug, "name": self.name, "root": str(self.root),
            "total": sum(counts.values()), "counts": counts, "version": version,
            "last_scan": last_scan, "scanning": self.scanning,
        }

    # ---------- 索引（内存缓存 + 版本号失效） ----------

    def _load_index(self) -> faiss.Index:
        conn = self._connect()
        version = int(self._meta_get(conn, "version", "0"))
        conn.close()
        if self._index is not None and version == self._index_version:
            return self._index
        if self.faiss_path.exists():
            index = faiss.read_index(str(self.faiss_path))
        else:
            index = faiss.IndexIDMap(faiss.IndexFlatIP(DIMS))
        self._index = index
        self._index_version = version
        return index

    def _save_index(self, conn: sqlite3.Connection) -> None:
        faiss.write_index(self._index, str(self.faiss_path))
        self._meta_set(conn, "version", str(int(self._meta_get(conn, "version", "0")) + 1))
        self._index_version = int(self._meta_get(conn, "version", "0"))

    # ---------- 增量扫描 ----------

    def _load_payload(self, rel: str):
        """按模态加载模型输入。失败抛异常（由调用方记为错误并跳过）。"""
        p = self.root / rel
        mod = modality_of(rel)
        if mod == "image":
            return Image.open(p).convert("RGB")
        if mod == "audio":
            wave = read_audio_bytes(p)
            if len(wave) / TARGET_SR > MAX_AUDIO_SECONDS:
                raise ValueError(f"音频超过 {MAX_AUDIO_SECONDS} 秒")
            return wave
        if mod == "video":
            return read_video_frames(p)
        if mod == "document":
            text = extract_text(p)
            if len(text.strip()) < 4:
                raise ValueError("文档无有效文本（可能是扫描件 PDF）")
            return text
        raise ValueError(f"未知模态: {rel}")

    @staticmethod
    def _doc_embedding_text(rel: str, text: str) -> tuple[str, str]:
        """组装文档编码文本（title: 文件名 | text: 内容）并生成摘要。"""
        title = Path(rel).stem
        content = " ".join(text.split())
        return f"title: {title} | text: {content}", make_snippet(text)

    def scan(self, embedder: Embedder | None = None) -> dict:
        """增量扫描：新增编码 / 变更重编码 / 删除移除 / 未变跳过。"""
        if self.scanning:
            return {"status": "already_scanning"}
        self.scanning = True
        stats = {"added": 0, "updated": 0, "removed": 0, "unchanged": 0, "errors": []}
        try:
            embedder = embedder or Embedder.get()
            conn = self._connect()
            known = {r[0]: (r[1], r[2], r[3]) for r in conn.execute(
                "SELECT path, mtime_ns, size, id FROM files")}

            todo_new: list[tuple[str, object]] = []
            todo_update: list[tuple[str, int, object]] = []
            seen: set[str] = set()
            for p in sorted(self.root.rglob("*")):
                if not p.is_file() or not modality_of(p.name):
                    continue
                rel = p.relative_to(self.root).as_posix()
                try:
                    st = p.stat()
                except OSError:
                    continue
                seen.add(rel)
                if rel in known:
                    mtime, size, fid = known[rel]
                    if mtime == st.st_mtime_ns and size == st.st_size:
                        stats["unchanged"] += 1
                        continue
                    todo_update.append((rel, fid, st))
                else:
                    todo_new.append((rel, st))

            stale_ids = [fid for rel, (_, _, fid) in known.items() if rel not in seen]

            # ---- 编码（按模态分批）----
            jobs = []  # (rel, fid, st, modality, payload)
            for rel, fid, st in todo_update:
                try:
                    jobs.append((rel, fid, st, modality_of(rel), self._load_payload(rel)))
                except Exception as e:
                    stats["errors"].append(f"{rel}: {e}")
            for rel, st in todo_new:
                try:
                    jobs.append((rel, None, st, modality_of(rel), self._load_payload(rel)))
                except Exception as e:
                    stats["errors"].append(f"{rel}: {e}")

            by_mod: dict[str, list] = {}
            for job in jobs:
                by_mod.setdefault(job[3], []).append(job)

            def encode_group(mod: str, group: list) -> np.ndarray:
                if mod == "image":
                    return embedder.encode_images([j[4] for j in group])
                if mod == "audio":
                    return embedder.encode_audios([j[4] for j in group])
                if mod == "video":
                    return embedder.encode_videos([j[4] for j in group])
                texts, snippets = [], []
                for j in group:
                    emb_text, snip = self._doc_embedding_text(j[0], j[4])
                    texts.append(emb_text)
                    snippets.append(snip)
                for j, snip in zip(group, snippets):
                    j_payload[j[0]] = snip  # type: ignore[name-defined]
                return embedder.encode_documents(texts)

            j_payload: dict[str, str] = {}
            results: dict[str, np.ndarray] = {}
            for mod, group in by_mod.items():
                vecs = encode_group(mod, group)
                if vecs.shape[0] != len(group):
                    raise RuntimeError(f"{mod} 批次返回 {vecs.shape[0]} 条，输入 {len(group)} 条")
                for job, vec in zip(group, vecs):
                    results[job[0]] = vec

            # ---- 落库 + 更新索引 ----
            index = self._load_index()
            with conn:
                if stale_ids or todo_update:
                    remove_ids = np.array(stale_ids + [fid for _, fid, _ in todo_update],
                                          dtype=np.int64)
                    index.remove_ids(faiss.IDSelectorBatch(remove_ids.size, faiss.swig_ptr(remove_ids)))
                    stats["removed"] += len(stale_ids)
                    for fid in stale_ids:
                        conn.execute("DELETE FROM files WHERE id=?", (fid,))
                    # 级联清理标签关联（锚点被删则整个标签删除）
                    conn.executemany("DELETE FROM label_docs WHERE doc_id=?",
                                     [(fid,) for fid in stale_ids])
                    conn.executemany("DELETE FROM labels WHERE anchor_id=?",
                                     [(fid,) for fid in stale_ids])

                add_vecs, add_ids = [], []
                for rel, fid, st, mod, _ in jobs:
                    vec = results[rel]
                    snippet = j_payload.get(rel)
                    blob = np.ascontiguousarray(vec, dtype=np.float32).tobytes()
                    if fid is None:  # 新增
                        cur = conn.execute(
                            "INSERT INTO files(path,mtime_ns,size,modality,embedding,snippet,embedded_at)"
                            " VALUES(?,?,?,?,?,?,?)",
                            (rel, st.st_mtime_ns, st.st_size, mod, blob, snippet, time.time()))
                        fid = cur.lastrowid
                        stats["added"] += 1
                    else:  # 变更（id 不变，向量替换）
                        conn.execute(
                            "UPDATE files SET mtime_ns=?, size=?, embedding=?, snippet=?, embedded_at=?"
                            " WHERE id=?",
                            (st.st_mtime_ns, st.st_size, blob, snippet, time.time(), fid))
                        stats["updated"] += 1
                    add_vecs.append(vec)
                    add_ids.append(fid)

                if add_vecs:
                    index.add_with_ids(
                        np.vstack(add_vecs).astype(np.float32),
                        np.array(add_ids, dtype=np.int64))

                self._save_index(conn)
                self._meta_set(conn, "last_scan", str(time.time()))
            conn.close()
            stats["status"] = "ok"
            return stats
        finally:
            self.scanning = False

    # ---------- 检索 ----------

    def search(self, qvec: np.ndarray, k: int = 12) -> list[dict]:
        index = self._load_index()
        if index.ntotal == 0:
            return []
        k = max(1, min(k, index.ntotal))
        v = np.asarray(qvec, dtype=np.float32).reshape(1, -1)
        faiss.normalize_L2(v)
        sims, ids = index.search(v, k)
        conn = self._connect()
        out = []
        for s, i in zip(sims[0], ids[0]):
            row = conn.execute(
                "SELECT path, modality, snippet FROM files WHERE id=?", (int(i),)).fetchone()
            if not row:
                continue
            rel, mod, snippet = row
            out.append({
                "path": rel,
                "abs": str(self.root / rel),
                "url": f"/files/{self.slug}/{rel}",
                "modality": mod,
                "label": Path(rel).parent.name or self.name,
                "snippet": snippet or "",
                "score": round(float(s), 4),
                "collection": self.slug,
                "collection_name": self.name,
            })
        conn.close()
        return out

    # ---------- 概念标签：命名一张图，关联一批相似图 ----------

    def create_label(self, anchor_rel: str, name: str, threshold: float = 0.5,
                     top_k: int = 200) -> dict:
        """以 anchor_rel 图片为锚点，把名称传播给集合内所有相似图片。

        纯向量计算（锚点向量已在索引中），不调用模型。
        返回标签信息与关联到的图片数量。
        """
        name = name.strip()
        if not name:
            raise ValueError("名称不能为空")
        if not 0 < threshold < 1:
            raise ValueError("threshold 必须在 (0, 1) 区间")
        conn = self._connect()
        try:
            row = conn.execute("SELECT id FROM files WHERE path=?", (anchor_rel,)).fetchone()
            if not row:
                raise ValueError(f"文件不在索引中（请先扫描收录）: {anchor_rel}")
            anchor_id = int(row[0])
            index = self._load_index()
            if index.ntotal == 0:
                raise ValueError("集合索引为空")
            vec = np.frombuffer(
                conn.execute("SELECT embedding FROM files WHERE id=?", (anchor_id,)).fetchone()[0],
                dtype=np.float32).reshape(1, -1)
            k = min(index.ntotal, max(top_k, int(top_k * 1.5)))
            sims, ids = index.search(np.ascontiguousarray(vec), k)
            picked = [(int(i), float(s)) for s, i in zip(sims[0], ids[0])
                      if s >= threshold][:top_k]
            if not any(i == anchor_id for i, _ in picked):
                picked.insert(0, (anchor_id, 1.0))
            with conn:
                cur = conn.execute(
                    "INSERT INTO labels(name,anchor_id,threshold,created_at) VALUES(?,?,?,?)",
                    (name, anchor_id, threshold, time.time()))
                lid = int(cur.lastrowid)
                conn.executemany(
                    "INSERT OR REPLACE INTO label_docs(label_id,doc_id,sim) VALUES(?,?,?)",
                    [(lid, i, s) for i, s in picked])
            return {"id": lid, "name": name, "anchor": anchor_rel,
                    "threshold": threshold, "count": len(picked)}
        finally:
            conn.close()

    def list_labels(self) -> list[dict]:
        conn = self._connect()
        rows = conn.execute(
            "SELECT l.id, l.name, f.path, l.threshold, l.created_at,"
            " (SELECT count(*) FROM label_docs ld WHERE ld.label_id = l.id)"
            " FROM labels l LEFT JOIN files f ON f.id = l.anchor_id"
            " ORDER BY l.created_at DESC").fetchall()
        conn.close()
        return [{"id": r[0], "name": r[1], "anchor": r[2], "threshold": r[3],
                 "created_at": r[4], "count": r[5]} for r in rows]

    def delete_label(self, label_id: int) -> bool:
        conn = self._connect()
        try:
            with conn:
                n = conn.execute("DELETE FROM labels WHERE id=?", (label_id,)).rowcount
                conn.execute("DELETE FROM label_docs WHERE label_id=?", (label_id,))
            return n > 0
        finally:
            conn.close()

    def label_search(self, name: str) -> list[dict]:
        """按名称命中概念标签，返回关联图片（精确匹配优先，再按相似度排序）。"""
        conn = self._connect()
        rows = conn.execute(
            "SELECT DISTINCT l.id, l.name, ld.doc_id, ld.sim FROM labels l"
            " JOIN label_docs ld ON ld.label_id = l.id"
            " JOIN files f ON f.id = ld.doc_id"
            " WHERE l.name = ? COLLATE NOCASE"
            "    OR l.name LIKE ? ESCAPE '\\'"
            "    OR (length(l.name) >= 2 AND ? LIKE '%' || l.name || '%' ESCAPE '\\')"
            " ORDER BY l.name = ? COLLATE NOCASE DESC, ld.sim DESC",
            (name, f"%{_like_escape(name)}%", name, name)).fetchall()
        out = []
        for lid, lname, doc_id, sim in rows:
            row = conn.execute(
                "SELECT path, modality, snippet FROM files WHERE id=?", (doc_id,)).fetchone()
            if not row:
                continue
            rel, mod, snippet = row
            out.append({
                "path": rel,
                "abs": str(self.root / rel),
                "url": f"/files/{self.slug}/{rel}",
                "modality": mod,
                "label": Path(rel).parent.name or self.name,
                "snippet": snippet or "",
                "score": round(float(sim), 4),
                "collection": self.slug,
                "collection_name": self.name,
                "matched_label": lname,
                "label_id": lid,
            })
        conn.close()
        return out

    def delete(self) -> None:
        """删除索引数据（不动源目录）。"""
        if self.dir.exists():
            shutil.rmtree(self.dir, ignore_errors=True)
        self._index = None
        self._index_version = -1


# ---------- 注册表：集合就是 collections/ 下的子目录 ----------

def list_collections() -> list[Collection]:
    COLLECTIONS_DIR.mkdir(exist_ok=True)
    out = []
    for cdir in sorted(COLLECTIONS_DIR.iterdir()):
        if (cdir / "meta.json").exists():
            out.append(Collection(cdir))
    return out


def get_collection(slug: str) -> Collection | None:
    return Collection.load(slug)


def search_all(qvec: np.ndarray, k: int = 12, slug: str | None = None) -> list[dict]:
    """跨集合检索（或指定单个集合），按分数合并取 top-k。"""
    cols = [get_collection(slug)] if slug else list_collections()
    merged: list[dict] = []
    for col in cols:
        if col is None:
            continue
        merged.extend(col.search(qvec, k=k))
    merged.sort(key=lambda r: -r["score"])
    return merged[:k]


# ---------- CLI ----------

def main():
    ap = argparse.ArgumentParser(description="多模态文件库管理")
    sub = ap.add_subparsers(dest="cmd", required=True)

    p_add = sub.add_parser("add", help="收录目录并扫描")
    p_add.add_argument("path", type=Path)

    sub.add_parser("list", help="列出所有集合")

    p_scan = sub.add_parser("scan", help="手动增量扫描")
    p_scan.add_argument("slug", nargs="?", default="all")

    p_rm = sub.add_parser("remove", help="移除集合（只删索引，不动源文件）")
    p_rm.add_argument("slug")

    p_search = sub.add_parser("search", help="命令行检索")
    p_search.add_argument("slug")
    p_search.add_argument("query")
    p_search.add_argument("-k", type=int, default=8)

    args = ap.parse_args()

    if args.cmd == "add":
        col = Collection.create(args.path)
        print(f"已收录 {col.root} -> {col.slug}，开始扫描…")
        stats = col.scan()
        print(json.dumps(stats, ensure_ascii=False, indent=2))
    elif args.cmd == "list":
        cols = list_collections()
        if not cols:
            print("（还没有收录任何目录）")
        for c in cols:
            i = c.info()
            print(f"{i['slug']}  {i['root']}  共 {i['total']} 条 {i['counts']}")
    elif args.cmd == "scan":
        cols = list_collections() if args.slug == "all" else [Collection.load(args.slug)]
        for c in cols:
            if c is None:
                print(f"集合不存在: {args.slug}")
                break
            print(f"扫描 {c.slug} ({c.root})…")
            t0 = time.time()
            stats = c.scan()
            print(f"  耗时 {time.time() - t0:.0f}s: " + json.dumps(stats, ensure_ascii=False))
    elif args.cmd == "remove":
        col = Collection.load(args.slug)
        if col is None:
            print(f"集合不存在: {args.slug}")
            return
        col.delete()
        print(f"已移除 {args.slug}（源目录未动）")
    elif args.cmd == "search":
        col = Collection.load(args.slug)
        if col is None:
            print(f"集合不存在: {args.slug}")
            return
        qvec = Embedder.get().encode_query(args.query)
        for r in col.search(qvec, k=args.k):
            print(f"  {r['score']:.4f} [{r['modality']:8}] {r['path']}")


if __name__ == "__main__":
    main()
