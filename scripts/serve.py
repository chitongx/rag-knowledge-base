"""启动 FastAPI 演示服务。

用法：python -m scripts.serve
文档：http://localhost:8000/docs
"""
import sys
import uvicorn
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import SERVE_HOST, SERVE_PORT

if __name__ == "__main__":
    uvicorn.run("src.api:app", host=SERVE_HOST, port=SERVE_PORT, reload=False)
