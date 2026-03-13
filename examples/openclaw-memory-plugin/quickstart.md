# Quickstart: 本地 Embedding + VikingDB（Qwen3-Embedding-0.6B）

本指南使用 Docker Compose 在本地启动 **Qwen3-Embedding-0.6B** embedding 服务和 **VikingDB** 向量数据库，无需火山引擎 API Key，适合本地开发和离线场景。

---

## 前置条件

- Docker >= 24
- Docker Compose >= 2.20
- 磁盘空间 ≥ 5GB（镜像 + 模型权重）

---

## 快速启动

```bash
git clone https://github.com/winterfell2021/OpenViking-with-Qwen.git
cd OpenViking-with-Qwen

# 构建并启动（首次 build 会从 ModelScope 下载模型，约 1.1GB）
docker compose -f docker-compose.embedding.yml up -d --build

# 查看构建/启动日志
docker compose -f docker-compose.embedding.yml logs -f
```

启动后两个服务：

| 服务 | 地址 | 说明 |
|------|------|------|
| embedding | http://localhost:8000 | Qwen3-Embedding-0.6B，POST /embed |
| openviking (VikingDB) | http://localhost:1933 | 向量存储与检索 API |

验证：

```bash
curl http://localhost:8000/health
# {"status":"ok","model":"Qwen/Qwen3-Embedding-0.6B","dim":1024}

curl http://localhost:1933/health
# {"status":"healthy","active_requests":0}
```

---

## 运行全流程测试

```bash
pip install requests
python tests/test_vikingdb_embedding.py
```

预期输出：

```
  Step 1: Embed sample texts       → Embedded 5 texts, dim=1024
  Step 2: Create collection        → code=0
  Step 3: Create index             → code=0
  Step 4: Upsert data              → code=0
  Step 5: Search by vector         → code=0
  Step 6: Verify results           → 返回语义最相关结果
  All steps passed!
```

---

## ov.conf 配置

将以下配置写入 `~/.openviking/ov.conf`，使 OpenViking 主服务使用本地 VikingDB：

```json
{
  "server": {
    "host": "127.0.0.1",
    "port": 1933
  },
  "storage": {
    "vectordb": {
      "backend": "vikingdb",
      "vikingdb": {
        "host": "http://localhost:1933"
      }
    }
  },
  "embedding": {
    "dense": {
      "provider": "openai",
      "api_key": "local",
      "model": "Qwen3-Embedding-0.6B",
      "api_base": "http://localhost:8000",
      "dimension": 1024
    }
  }
}
```

> **注意**：embedding service 暴露的是 `/embed` 接口，不是 OpenAI 兼容格式。
> 如需与 OpenClaw 插件的 `local` 模式配合，请使用标准 `ov.conf`（volcengine 或 openai provider），
> 本地 embedding 服务主要用于直接调用 VikingDB API 的场景（如 `test_vikingdb_embedding.py`）。

---

## OpenClaw 插件配置（连接本地 VikingDB）

如果你已有运行中的 OpenViking 主服务，可以让 OpenClaw 插件指向本地：

```bash
openclaw config set plugins.enabled true --json
openclaw config set plugins.slots.memory memory-openviking
openclaw config set plugins.entries.memory-openviking.config.mode remote
openclaw config set plugins.entries.memory-openviking.config.baseUrl "http://localhost:1933"
```

---

## 停止服务

```bash
docker compose -f docker-compose.embedding.yml down
```

---

**另见：** [INSTALL-ZH.md](./INSTALL-ZH.md) · [INSTALL.md](./INSTALL.md)
