"""
会话存储——JSON 文件实现

每个会话一个文件：data/sessions/{session_id}.json
人类可读、零依赖；每会话独立文件，无并发写冲突。

未来计划：新增 SqliteSessionStore(SessionStore)，
替换本类的注册即可，上层代码零改动。
"""

import json
import os
import shutil
import time
from dataclasses import asdict

from .base import SessionRecord, SessionStatus, SessionStore


class JsonSessionStore(SessionStore):
    """
    基于文件系统的 SessionStore 实现。

    目录结构：
      data/sessions/{session_id}.json
    """

    def __init__(self, sessions_dir: str):
        """
        Args:
            sessions_dir: 会话存储目录（通常为 data/sessions）
        """
        self.sessions_dir = sessions_dir
        os.makedirs(sessions_dir, exist_ok=True)

    def _path(self, session_id: str) -> str:
        return os.path.join(self.sessions_dir, f"{session_id}.json")

    async def create(self, record: SessionRecord) -> None:
        data = asdict(record)
        # 显式存枚举的 value（"running"/"done"/"failed"）：
        # str-Enum 直接 json 序列化会存成 name 形式（如 "SessionStatus.DONE"），
        # 读回来无法还原为枚举
        data["status"] = record.status.value
        with open(self._path(record.session_id), "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    async def update(self, session_id: str, patch: dict) -> None:
        record = await self.get(session_id)
        if record is None:
            return
        for key, value in patch.items():
            if hasattr(record, key):
                setattr(record, key, value)
        record.updated_at = time.time()
        await self.create(record)

    async def get(self, session_id: str) -> SessionRecord | None:
        path = self._path(session_id)
        if not os.path.exists(path):
            return None
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        # 字符串 → 枚举（否则 record.status 是纯字符串，.value 会报错）
        data["status"] = SessionStatus(data["status"])
        return SessionRecord(**data)

    async def list(self) -> list[SessionRecord]:
        if not os.path.isdir(self.sessions_dir):
            return []
        records = []
        for fn in os.listdir(self.sessions_dir):
            if not fn.endswith(".json"):
                continue
            record = await self.get(fn[:-5])
            if record:
                records.append(record)
        # 按创建时间倒序（最新的在前）
        records.sort(key=lambda r: r.created_at, reverse=True)
        return records

    async def delete(self, session_id: str) -> None:
        """删除会话记录文件及其工作区目录（笔记等）"""
        path = self._path(session_id)
        if os.path.exists(path):
            os.remove(path)
        # 会话工作区目录（笔记）：data/sessions/{session_id}/
        ws_dir = os.path.join(self.sessions_dir, session_id)
        if os.path.isdir(ws_dir):
            shutil.rmtree(ws_dir, ignore_errors=True)

    async def cleanup(self, max_sessions: int) -> int:
        """
        保留最近 max_sessions 个会话，删除更旧的。

        按创建时间（created_at）而非文件 mtime 判定新旧——
        避免 Claude Code 的已知坑：文件 mtime 被备份/同步重置
        导致新会话被误删（Issue #62250）。

        Returns:
            删除的会话数量
        """
        records = await self.list()  # 已按创建时间倒序
        if len(records) <= max_sessions:
            return 0
        removed = 0
        for record in records[max_sessions:]:
            await self.delete(record.session_id)
            removed += 1
        return removed
