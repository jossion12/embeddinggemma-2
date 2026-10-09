# 多模态语义搜索 · EmbeddingGemma 2

基于 [Google EmbeddingGemma 2](https://huggingface.co/google/embeddinggemma-2) 的本地多模态语义搜索：
收录任意目录，把其中的**文档、图片、音频、视频**统一编码进同一个 256 维向量索引（Matryoshka 截断），
支持文字 / 图片 / 音频 / 视频互相搜索，目录内容变更后索引自动增量更新。

## 特性

- **统一索引**：txt / md / pdf / docx、jpg / png / webp / bmp / gif、wav / mp3 / flac / ogg / m4a、mp4 / webm / mov / mkv 等
- **增量同步**：按 `mtime + size` 检测变更——新增自动编码、修改自动重编码、删除自动移除、未变直接跳过（不重复 embedding）
- **自动扫描**：后台定时增量扫描（周期可配），也可手动/API 触发
- **多集合**：可同时收录多个目录，跨集合或按目录搜索；索引数据存于项目内 `collections/`，不污染源目录
- **OpenAI 兼容接口**：`/v1/embeddings`、`/v1/search` 可直接用 OpenAI SDK 调用
- **跨模态**：四种模态在同一向量空间，文字可以搜音频，音频可以搜视频
- **概念标签**：给一张图片命名，自动关联集合内所有相似图片；之后用该名称检索即可召回整组相关图片

## 环境

- Python ≥ 3.12，CUDA 显卡（CPU 也可运行，较慢）
- 模型：EmbeddingGemma 2，默认路径 `D:\model\modelscope\google\embeddinggemma-2`
  （在 `library.py` 的 `MODEL_PATH` 修改）

```bash
# 安装依赖（uv 会按 pyproject.toml 里的 pytorch-cu130 索引装 CUDA 版 torch）
uv pip install -r pyproject.toml --python .venv/Scripts/python.exe

# 纯 pip 用户需自行指定 CUDA 版 torch 索引：
# pip install torch torchvision --index-url https://download.pytorch.org/whl/cu130
# pip install sentence-transformers "transformers>=5" fastapi "uvicorn[standard]" pillow numpy faiss-cpu soundfile av pypdf python-docx
```

## 快速开始

### 1. 命令行

```bash
python library.py add "D:\我的资料"        # 收录目录并建立索引（后续为增量）
python library.py list                     # 查看所有集合
python library.py scan                     # 手动增量扫描全部集合
python library.py search <slug> "红烧肉"   # 命令行检索
python library.py remove <slug>            # 移除集合（只删索引，不动源文件）
```

### 2. 启动服务

```bash
python server.py                       # 默认 127.0.0.1:7100，每 300 秒自动扫描
python server.py --scan-interval 0     # 关闭自动扫描
```

打开 `http://127.0.0.1:7100`：搜索框输入文字，或拖拽图片/音频/视频跨模态搜索；
点「＋ 收录目录」把任意目录纳入搜索范围。

### 3. 概念标签（命名一张图，检索一批图）

在图片结果卡片上点「🏷 命名」输入名称（如「展会物料」），系统会以该图为锚点，
把名称自动关联到集合内所有相似图片（默认相似度 ≥ 0.5，上限 200 张，纯向量计算不调模型）。
之后搜索这个名称——或任何包含它的句子——关联图片会置顶返回并带「标签」徽标。
搜索框下方的标签栏可点击检索、✕ 删除标签；标签随索引存储，源目录不受影响，
锚点文件被删除时标签会自动清理。

```bash
curl -X POST http://127.0.0.1:7100/api/collections/<slug>/labels \
     -H "Content-Type: application/json" \
     -d '{"path": "相对/路径.jpg", "name": "展会物料", "threshold": 0.5}'
```

## API

| 接口 | 说明 |
|---|---|
| `GET /api/collections` | 集合列表（数量、扫描状态） |
| `POST /api/collections` | 收录目录 `{"path": "D:/..."}`，后台扫描 |
| `DELETE /api/collections/{slug}` | 移除集合（仅删索引缓存） |
| `POST /api/collections/{slug}/scan` | 手动触发增量扫描 |
| `POST /api/collections/{slug}/labels` | 命名图片并关联相似图 `{"path","name","threshold"?}` |
| `GET /api/collections/{slug}/labels` | 标签列表（名称、锚点、关联数量） |
| `DELETE /api/collections/{slug}/labels/{id}` | 删除标签（只解除关联，不影响图片） |
| `GET /api/search?q=&collection=&k=` | 文字搜索（可限定集合；命中概念标签的结果置顶，`label_hits` 为命中数） |
| `GET /api/stats` | 汇总状态 |
| `POST /v1/search` | 跨模态搜索：文本 / `data:image|audio:video;base64,...` / 本地音视频路径（文字查询命中标签时置顶返回） |
| `POST /v1/embeddings` | OpenAI 兼容嵌入（支持 `dimensions` 768/512/256/128 MRL 截断） |
| `GET /files/{slug}/{path}` | 源文件预览/下载（防目录穿越） |

`/v1/search` 请求示例：

```json
{"query": "一份关于量子计算的笔记", "k": 10}
{"query": "data:audio/wav;base64,...", "collection": "mydocs-1a2b3c4d"}
```

## 工作原理

```
目录文件 ──扫描(mtime+size)──► SQLite 元数据 + Faiss IndexFlatIP(256d, IndexIDMap)
                                  │
新增 → 按模态分批编码 ──────────────┤
修改 → 重编码替换向量（id 不变）     │
删除 → 索引与库同步移除 ◄──────────┘
```

- 文档按 EmbeddingGemma 2 规范编码：`title: 文件名 | text: 内容`，查询用 `SearchQuery` 前缀
- 音频按 16kHz 单声道送入（25 tokens/秒，单文件上限约 320 秒）；视频默认 1 fps 抽帧（140 tokens/帧，上限 58 帧）
- 变更检测用 `mtime + size`；若文件内容变了但 mtime 被保留（如某些同步工具），不会触发更新

## 实测性能（RTX 2080 Ti）

- 音频编码约 40 条/秒（5 秒/条），图片批量约 30+ 张/秒
- 3.8 万条 256 维向量检索 < 50ms
- 音频→音频同类检索 top-1：50/50（ESC-50 环境声数据集）

## 注意事项

- **不要用 float16 推理**（模型卡明确禁止，会静默产生 NaN/退化向量）；本项目的 fp32 在 2080 Ti 上足够快
- Windows 下用 Git Bash 的 curl 测试中文查询时，注意命令行参数会按 GBK 编码发送，中文可能乱码——用 Python/HTTP 客户端或文件方式传 JSON
- `index/` 目录是重构前的旧版索引，已无任何代码引用，确认 `collections/` 正常后可删除
- 收录很大的目录时首次扫描耗时与文件数成正比（增量扫描只处理变更部分）

## 项目结构

```
├── library.py        # 核心：集合管理 / 增量扫描 / 统一检索（SQLite + Faiss）
├── server.py         # FastAPI 服务：集合 API、自动扫描、文件服务、OpenAI 兼容接口
├── doccodec.py       # 文档文本提取（txt/md/pdf/docx）
├── avcodec.py        # 音视频解码（16kHz 单声道 / 抽帧）
├── migrate_legacy.py # 旧版索引迁移工具（已运行过一次）
├── static/           # 前端页面
├── collections/      # 各集合的索引数据（自动生成，可用 library.py remove 清理）
├── img/ av/          # 示例数据集（动物图片 + 环境声音频）
└── pyproject.toml
```
