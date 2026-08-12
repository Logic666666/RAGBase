"""
PDF 阅读工具

解决情境：研究报告需要阅读 PDF 文档（论文、技术报告等），
read_file 只支持文本，PDF 是二进制格式。

方案：用 pypdf 提取 PDF 文本内容返回给 LLM。
  - 适用于文本型 PDF（如论文、报告）
  - 扫描版 PDF（图片）无法提取文本，需要 OCR——不在本工具范围

安全：路径校验限定在代码库/知识库目录内（与 read_file 一致）。
"""

import os

from ..base import BaseTool, ToolSpec
from ..path_utils import resolve_codebase_path


class ReadPdfTool(BaseTool):
    """读取 PDF 文件内容（提取文本）"""

    def __init__(self, codebase_dir: str):
        self.codebase_dir = codebase_dir

    @property
    def spec(self) -> ToolSpec:
        return ToolSpec(
            name="read_pdf",
            description=(
                "读取 PDF 文件的内容（提取文本）。"
                "当知识库中包含 PDF 文档（论文、技术报告等）时调用，"
                "用于获取文档内容进行分析。"
                "路径使用相对代码库根目录的路径（如 docs/paper.pdf），"
                "不要带 ./data/kb/ 等前缀。"
                "调用示例：{\"tool\": \"read_pdf\", \"arguments\": {\"path\": \"docs/paper.pdf\"}}"
            ),
            parameters={
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "相对代码库根目录的 PDF 文件路径",
                    },
                    "max_chars": {
                        "type": "integer",
                        "description": "最多返回的字符数（默认 4000，防止长文档刷屏）",
                        "default": 4000,
                    },
                },
                "required": ["path"],
                "additionalProperties": False,
            },
        )

    async def run(self, path: str, max_chars: int = 4000) -> str:
        # 路径校验（与 read_file 共用公共函数，防路径穿越）
        full_path = resolve_codebase_path(self.codebase_dir, path)
        if full_path is None:
            return f"路径无效: {path}（只能读取代码库内已存在的文件）"
        if not os.path.isfile(full_path):
            return f"文件不存在: {path}"
        if not full_path.lower().endswith(".pdf"):
            return f"不是 PDF 文件: {path}"

        # pypdf 提取文本
        from pypdf import PdfReader

        try:
            reader = PdfReader(full_path)
            parts = []
            for page in reader.pages:
                text = page.extract_text() or ""
                if text.strip():
                    parts.append(text.strip())
                if sum(len(p) for p in parts) >= max_chars:
                    break
            content = "\n\n".join(parts)
        except Exception as e:
            return f"PDF 读取失败: {type(e).__name__}: {str(e)}"

        # 清理非法字符：pypdf 可能提取出孤立代理字符（surrogate），
        # 无法 UTF-8 编码会导致后续请求崩溃
        content = content.encode("utf-8", errors="ignore").decode("utf-8")

        if not content.strip():
            return "该 PDF 无法提取文本（可能是扫描版，需要 OCR）。"

        if len(content) > max_chars:
            content = content[:max_chars] + "\n...(内容过长已截断)"
        return content
