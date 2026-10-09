# -*- coding: utf-8 -*-
"""把 index/embeddings.npz 中的存量索引迁移到 collections/（不重新编码）。

img/ 与 av/ 各成为一个集合；向量直接从 npz 导入，文件 mtime/size 取磁盘实况，
因此迁移后增量扫描能正确识别后续变更。

用法: python migrate_legacy.py
"""
import time
from pathlib import Path

import faiss
import numpy as np

from library import COLLECTIONS_DIR, DIMS, Collection, make_slug, modality_of

ROOT = Path(__file__).resolve().parent
LEGACY_NPZ = ROOT / "index" / "embeddings.npz"


def import_subset(col: Collection, paths: list[str], embs: np.ndarray, prefix: str) -> int:
    """导入 path 以 prefix 开头的条目，返回成功条数。"""
    conn = col._connect()
    index = faiss.IndexIDMap(faiss.IndexFlatIP(DIMS))
    n, skipped = 0, 0
    add_vecs, add_ids = [], []
    for rel_full, vec in zip(paths, embs):
        if not rel_full.startswith(prefix):
            continue
        rel = rel_full[len(prefix):]
        p = col.root / rel
        if not p.exists():
            skipped += 1
            continue
        st = p.stat()
        blob = np.ascontiguousarray(vec, dtype=np.float32).tobytes()
        cur = conn.execute(
            "INSERT INTO files(path,mtime_ns,size,modality,embedding,snippet,embedded_at)"
            " VALUES(?,?,?,?,?,NULL,?)",
            (rel, st.st_mtime_ns, st.st_size, modality_of(rel), blob, time.time()))
        add_vecs.append(np.asarray(vec, dtype=np.float32))
        add_ids.append(cur.lastrowid)
        n += 1
    if add_vecs:
        index.add_with_ids(np.vstack(add_vecs), np.array(add_ids, dtype=np.int64))
    faiss.write_index(index, str(col.faiss_path))
    with conn:
        col._meta_set(conn, "version", "1")
        col._meta_set(conn, "last_scan", str(time.time()))
    conn.close()
    print(f"  {col.slug}: 导入 {n} 条（跳过缺失文件 {skipped}）")
    return n


def main():
    data = np.load(LEGACY_NPZ, allow_pickle=False)
    paths = [str(p) for p in data["paths"]]
    embs = data["embs"].astype(np.float32)
    data.close()
    print(f"读取存量索引: {len(paths)} 条, 维度 {embs.shape[1]}")

    total = 0
    for subdir in ("img", "av"):
        root = ROOT / subdir
        slug = make_slug(root)
        cdir = COLLECTIONS_DIR / slug
        if cdir.exists():
            print(f"  {slug} 已存在，跳过（如需重建请先 python library.py remove {slug}）")
            continue
        col = Collection.create(root)
        total += import_subset(col, paths, embs, subdir + "/")

    print(f"迁移完成，共导入 {total} 条。")


if __name__ == "__main__":
    main()
