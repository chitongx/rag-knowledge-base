#!/usr/bin/env bash
# 一键部署 + 自检：构建镜像 -> 启动 -> 健康检查 -> 输出访问地址
# 用法：./deploy.sh   （需已安装 Docker）
set -euo pipefail
cd "$(dirname "$0")"

echo "[1/4] 构建镜像（首次约 5-10 分钟，含依赖与模型下载）"
docker compose build

echo "[2/4] 启动服务"
docker compose up -d

echo "[3/4] 等待服务就绪（首次启动需下载向量模型，约 1-3 分钟）"
READY=0
for i in $(seq 1 40); do
  if curl -s -m 3 http://localhost:8000/health 2>/dev/null | grep -q ok; then
    READY=1
    break
  fi
  sleep 3
done

echo "[4/4] 验收"
if [ "$READY" = "1" ]; then
  echo "✅ 服务已上线："
  echo "   演示页（推荐）: http://localhost:8000/"
  echo "   API 文档       : http://localhost:8000/docs"
  echo "   健康检查       : http://localhost:8000/health"
  echo ""
  echo "   # 快速验证检索接口："
  echo "   curl -X POST http://localhost:8000/retrieve -H 'Content-Type: application/json' -d '{\"query\":\"SPD是什么模式？\",\"mode\":\"hybrid_rerank\",\"top_k\":2}'"
else
  echo "❌ 服务未在预期时间内就绪，请查看日志：docker compose logs -f"
  exit 1
fi
