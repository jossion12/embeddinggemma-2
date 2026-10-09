# -*- coding: utf-8 -*-
"""统一多模态语义搜索服务：收录任意目录，文档/图片/音频/视频自动建索引并持续同步。

用法:
    python server.py [--host 127.0.0.1] [--port 7100] [--scan-interval 300]
    scan-interval 为后台自动增量扫描周期（秒），0 表示关闭。

主要接口:
    GET  /api/collections                 集合列表（各类数量、扫描状态）
    POST /api/collections                 收录目录 {"path": "D:/..."}，后台扫描
    DELETE /api/collections/{slug}        移除集合（只删索引缓存，不动源文件）
    POST /api/collections/{slug}/scan     手动触发增量扫描
    POST /api/collections/{slug}/labels   命名图片并关联相似图 {"path","name","threshold"?}
    GET  /api/collections/{slug}/labels   标签列表
    DELETE /api/collections/{slug}/labels/{id}  删除标签
    GET  /api/search?q=&collection=&k=    文字搜索（可限定集合）
    GET  /api/stats                       汇总状态
    POST /v1/search                       跨模态搜索（文本 / data URL / 本地路径）
    POST /v1/embeddings                   OpenAI 兼容嵌入接口
    GET  /files/{slug}/{path}             源文件预览/下载（防目录穿越）
"""
import argparse
import base64
import io
import json
import mimetypes
import threading
import time
from pathlib import Path

import numpy as np
from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import FileResponse, JSONResponse
from PIL import Image
from pydantic import BaseModel

from avcodec import TARGET_SR, detect_media_type, read_audio_bytes, read_video_frames
from library import DIMS, Collection, Embedder, get_collection, list_collections, search_all

ROOT = Path(__file__).resolve().parent
STATIC_DIR = ROOT / "static"

app = FastAPI(title="多模态语义搜索")
_encode_lock = threading.Lock()

# 集合实例缓存（slug → Collection），避免每次请求重读 meta.json
_col_cache: dict[str, Collection] = {}
_col_cache_lock = threading.Lock()


def _get_collection(slug: str) -> Collection:
    with _col_cache_lock:
        col = _col_cache.get(slug)
        if col is None:
            col = get_collection(slug)
            if col is None:
                raise HTTPException(404, f"集合不存在: {slug}")
            _col_cache[slug] = col
        return col


def _drop_collection(slug: str) -> None:
    with _col_cache_lock:
        _col_cache.pop(slug, None)


def _scan_async(col: Collection) -> None:
    def run():
        try:
            stats = col.scan()
            print(f"[scan] {col.slug}: {json.dumps(stats, ensure_ascii=False)}", flush=True)
        except Exception as e:
            print(f"[scan] {col.slug} 失败: {e}", flush=True)

    threading.Thread(target=run, daemon=True).start()


# ---------------- 集合管理 ----------------

class AddCollectionRequest(BaseModel):
    path: str


@app.get("/api/collections")
def api_collections():
    return {"collections": [c.info() for c in list_collections()]}


@app.post("/api/collections")
def api_add_collection(req: AddCollectionRequest):
    root = Path(req.path).expanduser()
    if not root.is_dir():
        raise HTTPException(400, f"目录不存在: {root}")
    try:
        col = Collection.create(root)
    except ValueError as e:
        raise HTTPException(409, str(e))
    _scan_async(col)
    return {"slug": col.slug, "name": col.name, "root": str(col.root), "scanning": True}


@app.delete("/api/collections/{slug}")
def api_remove_collection(slug: str):
    col = _get_collection(slug)
    col.delete()
    _drop_collection(slug)
    return {"removed": slug, "note": "仅删除索引缓存，源目录未动"}


@app.post("/api/collections/{slug}/scan")
def api_scan_collection(slug: str):
    col = _get_collection(slug)
    if col.scanning:
        return {"slug": slug, "scanning": True}
    _scan_async(col)
    return {"slug": slug, "scanning": True}


# ---------------- 概念标签：命名一张图，检索一批相关图 ----------------

class CreateLabelRequest(BaseModel):
    path: str         # 锚点图片的相对路径（files.path）
    name: str         # 用户起的名称
    threshold: float = 0.5  # 相似度传播阈值


@app.post("/api/collections/{slug}/labels")
def api_create_label(slug: str, req: CreateLabelRequest):
    col = _get_collection(slug)
    try:
        return col.create_label(req.path, req.name, threshold=req.threshold)
    except ValueError as e:
        raise HTTPException(400, str(e))


@app.get("/api/collections/{slug}/labels")
def api_list_labels(slug: str):
    col = _get_collection(slug)
    return {"labels": col.list_labels()}


@app.delete("/api/collections/{slug}/labels/{label_id}")
def api_delete_label(slug: str, label_id: int):
    col = _get_collection(slug)
    if not col.delete_label(label_id):
        raise HTTPException(404, f"标签不存在: {label_id}")
    return {"deleted": label_id}


def _merge_label_hits(query: str, slug: str | None, vector_results: list[dict],
                      k: int) -> tuple[list[dict], int]:
    """文字查询：命中概念标签的图片置顶，再补充向量检索结果（按 集合+路径 去重）。"""
    if not query.strip():
        return vector_results[:k], 0
    cols = [get_collection(slug)] if slug else list_collections()
    hits: list[dict] = []
    for c in cols:
        if c is not None:
            hits.extend(c.label_search(query))
    if not hits:
        return vector_results[:k], 0
    seen = {(r["collection"], r["path"]) for r in hits}
    merged = hits + [r for r in vector_results
                     if (r["collection"], r["path"]) not in seen]
    limit = min(max(k, len(hits)), 4 * k)
    return merged[:limit], len(hits)


@app.get("/api/stats")
def api_stats():
    cols = list_collections()
    infos = [c.info() for c in cols]
    model_ready = Embedder._instance is not None
    return {
        "ready": model_ready and any(i["total"] for i in infos),
        "model_ready": model_ready,
        "collections": len(infos),
        "indexed": sum(i["total"] for i in infos),
        "counts": {m: sum(i["counts"].get(m, 0) for i in infos)
                   for m in ("image", "audio", "video", "document")},
    }


# ---------------- 文件访问（防目录穿越） ----------------

@app.get("/files/{slug}/{relpath:path}")
def serve_file(slug: str, relpath: str):
    col = _get_collection(slug)
    root = col.root.resolve()
    target = (root / relpath).resolve()
    if not target.is_relative_to(root) or not target.is_file():
        raise HTTPException(404, "文件不存在")
    media_type = mimetypes.guess_type(str(target))[0] or "application/octet-stream"
    return FileResponse(target, media_type=media_type)


# ---------------- 搜索 ----------------

@app.get("/api/search")
def api_search(
    q: str = Query(..., min_length=1),
    collection: str | None = None,
    k: int = Query(24, ge=1, le=100),
):
    with _encode_lock:
        qvec = Embedder.get().encode_query(q)
    results = search_all(qvec, k=k, slug=collection)
    results, label_hits = _merge_label_hits(q, collection, results, k)
    return {"query": q, "count": len(results), "label_hits": label_hits,
            "results": results}


def _err(message, status=400, err_type="invalid_request_error"):
    return JSONResponse(status_code=status,
                        content={"error": {"message": message, "type": err_type, "code": None}})


def _embed_items(items: list[str], dims: int, prompt_name: str | None):
    """把输入项（文本 / data:image / data:audio / data:video / 本地路径）编码为归一化向量。

    返回 (vectors, token_count, error)。error 非 None 时应直接返回给调用方。
    """
    texts, text_pos = [], []
    images, image_pos = [], []
    audios, audio_pos = [], []
    videos, video_pos = [], []
    for i, s in enumerate(items):
        try:
            if s.startswith("data:image/"):
                payload = base64.b64decode(s.split(",", 1)[1])
                images.append(Image.open(io.BytesIO(payload)).convert("RGB"))
                image_pos.append(i)
            elif s.startswith("data:audio/"):
                payload = base64.b64decode(s.split(",", 1)[1])
                audios.append(read_audio_bytes(payload))
                audio_pos.append(i)
            elif s.startswith("data:video/"):
                payload = base64.b64decode(s.split(",", 1)[1])
                videos.append(read_video_frames(payload))
                video_pos.append(i)
            elif detect_media_type(s) and Path(s).exists():
                # 服务端本地音视频文件路径（音视频输入不加文本前缀）
                if detect_media_type(s) == "audio":
                    audios.append(read_audio_bytes(s))
                else:
                    videos.append(read_video_frames(s))
                (audio_pos if detect_media_type(s) == "audio" else video_pos).append(i)
            else:
                texts.append(s)
                text_pos.append(i)
        except Exception as e:
            return None, 0, _err(f"input[{i}]: 无法解码输入 ({e})")

    model = Embedder.get().model
    token_count = 0
    vecs_out: list[np.ndarray | None] = [None] * len(items)
    kwargs = dict(normalize_embeddings=True, truncate_dim=dims, show_progress_bar=False)
    with _encode_lock:
        if texts:
            kw = dict(kwargs)
            if prompt_name:
                kw["prompt_name"] = prompt_name
            vecs = model.encode(texts, **kw)
            try:
                tok = model.tokenize(texts)
                token_count += int(sum(len(ids) for ids in tok["input_ids"]))
            except Exception:
                token_count += sum(max(1, len(t) // 4) for t in texts)
            for pos, v in zip(text_pos, vecs):
                vecs_out[pos] = np.asarray(v, dtype=np.float32)
        if images:
            vecs = model.encode(images, **kwargs)
            token_count += 280 * len(images)
            for pos, v in zip(image_pos, vecs):
                vecs_out[pos] = np.asarray(v, dtype=np.float32)
        if audios or videos:
            # dict 是“单条输入”（列表值会被池化成一条向量），批量必须传 dict 列表
            mm_inputs = (
                [{"audio": {"array": w, "sampling_rate": TARGET_SR}} for w in audios]
                + [{"video": v} for v in videos]
            )
            vecs = model.encode(mm_inputs, **kwargs)
            token_count += int(sum(len(w) for w in audios) / TARGET_SR * 25)  # 25 tokens/秒
            token_count += sum(len(f) * 140 for f in videos)  # 140 tokens/帧
            for pos, v in zip(audio_pos + video_pos, vecs):
                vecs_out[pos] = np.asarray(v, dtype=np.float32)
    return vecs_out, token_count, None


class SearchRequest(BaseModel):
    """语义搜索：query 为文本、data:image/audio/video base64 或本地音视频路径。"""
    query: str
    k: int = 24
    collection: str | None = None
    prompt_name: str | None = "SearchQuery"


def _query_type(query: str) -> str:
    for prefix, qt in (("data:image/", "image"), ("data:audio/", "audio"), ("data:video/", "video")):
        if query.startswith(prefix):
            return qt
    return detect_media_type(query) or "text"


@app.post("/v1/search")
def openai_search(req: SearchRequest):
    if not req.query.strip():
        return _err("query must not be empty")
    if req.collection:
        _get_collection(req.collection)  # 404 校验

    t0 = time.time()
    vecs_out, _, error = _embed_items([req.query], DIMS, req.prompt_name)
    if error is not None:
        return error
    qvec = vecs_out[0]
    assert qvec is not None
    encode_ms = (time.time() - t0) * 1000

    results = search_all(qvec, k=req.k, slug=req.collection)
    results, label_hits = _merge_label_hits(req.query, req.collection, results, req.k)
    return {
        "object": "list",
        "query_type": _query_type(req.query),
        "count": len(results),
        "label_hits": label_hits,
        "encode_ms": round(encode_ms),
        "data": [{"object": "search_result", **r} for r in results],
        "model": "google/embeddinggemma-2",
    }


# ---------------- OpenAI 兼容嵌入 ----------------

MODEL_ID = "google/embeddinggemma-2"
MRL_DIMS = (768, 512, 256, 128)


class EmbeddingsRequest(BaseModel):
    model: str = MODEL_ID
    input: str | list[str]
    encoding_format: str = "float"
    dimensions: int | None = None
    prompt_name: str | None = "SearchQuery"


@app.get("/v1/models")
def openai_models():
    return {"object": "list", "data": [{
        "id": MODEL_ID, "object": "model", "created": 1760000000, "owned_by": "google"}]}


@app.post("/v1/embeddings")
def openai_embeddings(req: EmbeddingsRequest):
    items = [req.input] if isinstance(req.input, str) else list(req.input)
    if not items:
        return _err("input must not be empty")
    if any(not isinstance(x, str) for x in items):
        return _err("each input item must be a string")
    dims = req.dimensions or 768
    if dims not in MRL_DIMS:
        return _err(f"dimensions must be one of {list(MRL_DIMS)} (Matryoshka truncation)")
    if req.encoding_format not in ("float", "base64"):
        return _err("encoding_format must be 'float' or 'base64'")

    vecs_out, token_count, error = _embed_items(items, dims, req.prompt_name)
    if error is not None:
        return error

    import struct

    data = []
    for i, v in enumerate(vecs_out):
        assert v is not None
        if req.encoding_format == "base64":
            emb: object = base64.b64encode(struct.pack(f"<{len(v)}f", *map(float, v))).decode()
        else:
            emb = [round(float(x), 6) for x in v]
        data.append({"object": "embedding", "embedding": emb, "index": i})

    return {
        "object": "list",
        "data": data,
        "model": req.model or MODEL_ID,
        "usage": {"prompt_tokens": token_count, "total_tokens": token_count},
    }


# ---------------- 前端与启动 ----------------

@app.get("/")
def index():
    return FileResponse(STATIC_DIR / "index.html")


def _autoscan_loop(interval: int) -> None:
    while True:
        time.sleep(interval)
        try:
            for col in list_collections():
                if not col.scanning:
                    stats = col.scan()
                    changed = stats["added"] or stats["updated"] or stats["removed"]
                    if changed or stats["errors"]:
                        print(f"[autoscan] {col.slug}: {json.dumps(stats, ensure_ascii=False)}",
                              flush=True)
        except Exception as e:
            print(f"[autoscan] 失败: {e}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=7100)
    ap.add_argument("--scan-interval", type=int, default=300,
                    help="后台自动增量扫描周期（秒），0 关闭")
    args = ap.parse_args()

    if args.scan_interval > 0:
        threading.Thread(target=_autoscan_loop, args=(args.scan_interval,),
                         daemon=True).start()
        print(f"[autoscan] 每 {args.scan_interval} 秒自动增量扫描")

    import uvicorn

    uvicorn.run(app, host=args.host, port=args.port, log_level="warning")


if __name__ == "__main__":
    main()
