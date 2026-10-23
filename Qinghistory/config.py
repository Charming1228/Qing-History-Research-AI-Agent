"""清鉴全局配置。

所有密钥与可调参数一律通过环境变量读取，避免把明文密钥提交到版本库。

使用方法：
  1. 复制项目根目录的 `.env.example` 为 `.env` 并填入你自己的密钥；
  2. 启动前 `export $(cat .env | xargs)`（Linux/macOS）或通过系统环境变量设置；
  3. 或在代码里直接 `os.environ["DEEPSEEK_API_KEY"] = "..."`。

环境变量一览：
  DEEPSEEK_API_KEY      必填，DeepSeek API 密钥
  DEEPSEEK_BASE_URL     可选，默认 https://api.deepseek.com/v1
  DEEPSEEK_MODEL        可选，默认 deepseek-chat
  MOARK_API_KEY         Moark 中转 API 密钥（微调数据构建脚本使用）
  MOARK_BASE_URL        可选，默认 https://api.moark.com/v1
  MOARK_MODEL           可选，默认 DeepSeek-V4-Pro
"""
from __future__ import annotations

import os

from openai import OpenAI


def _require_env(name: str) -> str:
    value = os.environ.get(name, "").strip()
    if not value:
        raise RuntimeError(
            f"缺少环境变量 {name}，请先设置后重试（参考项目根目录 .env.example）。"
        )
    return value


def get_llm_client() -> OpenAI:
    """构建 DeepSeek 客户端（agent.py / agent2.py / agent4.py 共用）。"""
    api_key = _require_env("DEEPSEEK_API_KEY")
    base_url = os.environ.get("DEEPSEEK_BASE_URL", "https://api.deepseek.com/v1")
    return OpenAI(api_key=api_key, base_url=base_url)


def get_llm_model(default: str = "deepseek-chat") -> str:
    """读取 DeepSeek 模型名，未设置时使用传入的默认值。"""
    return os.environ.get("DEEPSEEK_MODEL", default)
