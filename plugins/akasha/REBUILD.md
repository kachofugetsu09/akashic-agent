# Akasha 记忆库重建指南

> 场景：以前聊过一些内容（都在 `sessions.db` 里），想从这些对话**重新建出 akasha 的图**
> （比如改了图算法/参数、清了脏图、或换了机器）。

## 心智模型

- **`sessions.db`** = 唯一真相源（原始 user/assistant 消息，不动）。
- **`akasha.db`** = 在真相之上的**索引 + 图**，由重放 `sessions.db` 算出来：
  - `akasha_nodes` / `akasha_edges` / `akasha_salience_state` —— 图本体（重建会清掉重算）
  - `akasha_query_log` / `akasha_activation_events` —— 诊断（重建重算）
  - `akasha_embedding_cache` —— **每条消息的 embedding 缓存**（重建**默认保留**）
  - `fts_token_idf` —— FTS 稀有词的 IDF 表（FTS 种子用，单独一步建）

重建 = 拿 `sessions.db` 的消息，按时间顺序重放过 akasha 的激活/建边逻辑，把图算出来落回 `akasha.db`。
**embedding 不参与重算**——重放复用缓存里的向量。所以有没有缓存，决定走哪条路。

> 重建是**确定性**的（已修 `core.py` 的 `PYTHONHASHSEED` 漂移）：同一份 `sessions.db` + 同一份缓存，
> 每次重建出的图逐位一致、可复现。

---

## 路径 A — 有缓存（常态，~3 分钟）

平时聊天时线上 engine 已经把每条消息 embed 并写进 `akasha_embedding_cache`（见 `engine.py` `_on_turn_committed`）。
所以**只要你是在用过的库上重建**，缓存就是热的，直接跑：

```bash
python scripts/build_akasha_db.py \
  --config config.toml \
  --sessions-db ~/.akashic/workspace/sessions.db \
  --db-path    ~/.akashic/workspace/memory/akasha.db
```

（不带参数时用默认路径：`~/.akashic/workspace/{sessions.db, memory/akasha.db}`、`config.toml`。）

做了什么：
1. 自动备份旧 `akasha.db`（`*.bak-时间戳`）。
2. `reset_schema()` 清掉图表与诊断，**保留 `embedding_cache`**。
3. 用内存 `MemoryStore` 全库快速重放（graph_fast + fast_dense），末尾一次性批量落库。
4. 写迁移记录，打印 `nodes/edges/query_logs/activation_events` 计数。

输出里看 `cache_hits / cache_misses`：**miss=0 说明缓存全命中**，图完整。

> 这是"以前聊了内容、想重建图"的标准做法——embedding 没变，只是把图算法重新跑一遍。

---

## 路径 B — 没缓存（冷启动 / 删过缓存 / 换了 embedding 模型）

`build_akasha_db.py` **只读缓存、不生成 embedding**。缓存里查不到的消息会被**直接跳过**
（`_load_embeddings_from_cache` 里 `cache_misses += 1`），结果就是**图残缺甚至为空**。

会落到这条路的情况：
- 全新 / 空的 `akasha.db`（从没在上面聊过）；
- 手动删过 `akasha_embedding_cache`；
- **换了 embedding 模型**——缓存命中键是 `(message_id, model, content_hash)`，model 一变全部失配。

解决：**先预热缓存，再走路径 A。** 目前没有现成脚本，预热的最小范式（与线上同一个 `Embedder`、
**同一个 config 里的 model**）：

```python
import asyncio, sqlite3
from agent.config_models import Config
from memory2.embedder import Embedder
from plugins.akasha.core import SourceMessage
from plugins.akasha.store import AkashaStore

cfg = Config.load("config.toml")
emb_cfg = cfg.memory.embedding          # base_url / api_key / model
embedder = Embedder(emb_cfg.base_url, emb_cfg.api_key, model=emb_cfg.model)
store = AkashaStore("~/.akashic/workspace/memory/akasha.db")

src = sqlite3.connect("~/.akashic/workspace/sessions.db"); src.row_factory = sqlite3.Row
rows = src.execute(
    "SELECT id, session_key, seq, role, content, ts FROM messages "
    "WHERE role IN ('user','assistant')"
).fetchall()

async def warm():
    msgs = [SourceMessage(r["id"], r["session_key"], r["seq"], r["role"], r["content"], r["ts"]) for r in rows]
    vecs = await embedder.embed_batch([m.content for m in msgs])
    for m, v in zip(msgs, vecs):
        store.upsert_cached_embedding(message=m, model=emb_cfg.model, embedding=v)

asyncio.run(warm())
store.close()
```

> ⚠️ model 必须和 `config.toml` 的 `memory.embedding.model` 一致，否则重建时仍然失配跳过。
> embed 会调用外部 API（DashScope 风格），耗时和条数成正比；预热一次后缓存就持久了。

预热完，再跑**路径 A** 的 `build_akasha_db.py` 即可。

---

## 从零重建还要补：FTS / IDF 索引

`build_akasha_db.py` **不建** `fts_token_idf`（FTS 稀有词种子用的 IDF 表）。全新库或想刷新 IDF：

```bash
python scripts/build_fts_idf.py
```

它扫 `sessions.db` 全部消息、jieba 切词算 IDF、写回 `akasha.db:fts_token_idf`。

---

## 速查

| 情况 | 步骤 |
|---|---|
| 用过的库，改了图逻辑/参数想重建 | 直接 `build_akasha_db.py`（路径 A） |
| 全新空库 / 删过缓存 | 先预热缓存（路径 B）→ `build_akasha_db.py` → `build_fts_idf.py` |
| 换了 embedding 模型 | 同"全新库"：必须重新预热（旧缓存全失配） |
| 想确认图完整 | 看输出 `cache_misses` 是否为 0 |

> 实现细节：快速重放后端在 `plugins/akasha/fast/`（内存 `MemoryStore` + graph_fast/fast_dense + `dump.dump_to_db`），
> 对拍/确定性回归见 `tests/test_fast_rebuild_parity.py`。
