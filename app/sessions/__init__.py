"""
会话管理模块（对齐 Claude Code 的 session 设计）

研究报告类任务是长任务（10-30 分钟），同步请求-响应模型不可用。
本模块提供：
  - 会话提交：立即返回 session_id，后台执行
  - 状态查询：轮询进度
  - 结果获取：最终结果 + trace + 消息历史（transcript）

设计参考（Claude Code / Agent SDK sessions）：
  - session = 一次 agent 执行的完整记录（对话历史 transcript 落盘）
  - session_id 是恢复/追溯的标识
  - 消息历史（messages）随会话落盘——支持中断后恢复执行（resume）

设计原则（接口抽象 + 单一实现）：
  SessionStore 是抽象接口，JsonSessionStore 是当前实现（A 方案）。
  上层（SessionManager / API）只依赖 SessionStore 接口，
  将来换 SQLite 等存储只需新增实现类，零架构级改动。
"""

from .base import SessionRecord, SessionStatus, SessionStore
from .json_store import JsonSessionStore
from .manager import SessionManager

__all__ = ["SessionRecord", "SessionStatus", "SessionStore", "JsonSessionStore", "SessionManager"]
