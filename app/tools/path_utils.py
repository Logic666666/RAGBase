"""
代码库路径解析公共工具

read_file / read_pdf 等工具共用——消除重复实现。
（曾因两处复制同一实现导致同一 bug 出现两次）
"""

import os


def resolve_codebase_path(codebase_dir: str, path: str) -> str | None:
    """
    解析代码库内文件路径（兼容两种格式），并校验存在且在代码库内。

    格式 1（推荐）：相对路径，如 src/main.py
    格式 2（容错）：完整路径，如 ./data/kb/xxx/source/src/main.py
                    （search_kb 返回的 metadata.source 格式）

    关键：拼接后必须 os.path.exists 才返回——
    否则拼接出的"重复路径"（codebase/data/kb/...）会被误判为有效，
    导致完整路径永远无法解析（曾因此 bug 导致 read_pdf 全部失败）。

    Returns:
        解析后的绝对路径；不存在或不在代码库内返回 None
    """
    codebase_root = os.path.realpath(codebase_dir)

    # 尝试 1：作为相对路径拼接（必须存在才有效）
    full = os.path.realpath(os.path.join(codebase_root, path))
    if full.startswith(codebase_root) and os.path.exists(full):
        return full

    # 尝试 2：作为完整路径（已在代码库内）
    full = os.path.realpath(path)
    if full.startswith(codebase_root) and os.path.exists(full):
        return full

    return None
