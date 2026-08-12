"""
PDF 阅读工具单元测试

验证：路径校验、非 PDF 拒绝、pypdf 提取、路径穿越防护。
用 mock pypdf 避免依赖真实 PDF 文件。
"""

import os
from unittest.mock import patch

import pytest

from app.tools.builtin.read_pdf import ReadPdfTool


class FakePage:
    def __init__(self, text):
        self._text = text

    def extract_text(self):
        return self._text


class FakeReader:
    def __init__(self, pages):
        self.pages = pages


@pytest.fixture
def tool(tmp_path):
    return ReadPdfTool(str(tmp_path))


@pytest.mark.asyncio
async def test_extracts_pdf_text(tool, tmp_path):
    """PDF 提取文本（mock pypdf）"""
    pdf_path = tmp_path / "paper.pdf"
    pdf_path.write_bytes(b"fake-pdf-bytes")

    fake_reader = FakeReader([FakePage("这是论文摘要内容"), FakePage("这是正文内容")])
    with patch("pypdf.PdfReader", return_value=fake_reader):
        result = await tool.run("paper.pdf")
    assert "这是论文摘要内容" in result
    assert "这是正文内容" in result


@pytest.mark.asyncio
async def test_truncates_long_pdf(tool, tmp_path):
    """超长 PDF 内容截断"""
    pdf_path = tmp_path / "long.pdf"
    pdf_path.write_bytes(b"fake")

    long_text = "x" * 5000
    fake_reader = FakeReader([FakePage(long_text)])
    with patch("pypdf.PdfReader", return_value=fake_reader):
        result = await tool.run("long.pdf", max_chars=100)
    assert "已截断" in result


@pytest.mark.asyncio
async def test_rejects_non_pdf(tool, tmp_path):
    """非 PDF 文件拒绝"""
    (tmp_path / "note.txt").write_text("hello", encoding="utf-8")
    result = await tool.run("note.txt")
    assert "不是 PDF" in result


@pytest.mark.asyncio
async def test_path_traversal_blocked(tool):
    """路径穿越防护"""
    result = await tool.run("../../secret.pdf")
    assert "路径无效" in result


@pytest.mark.asyncio
async def test_nonexistent_file(tool):
    result = await tool.run("no_such.pdf")
    assert "路径无效" in result


@pytest.mark.asyncio
async def test_surrogate_chars_cleaned(tool, tmp_path):
    """
    pypdf 可能提取出孤立代理字符（surrogate），无法 UTF-8 编码，
    会导致后续 embedding 请求崩溃——提取后必须清理。
    """
    pdf_path = tmp_path / "surrogate.pdf"
    pdf_path.write_bytes(b"fake")

    bad_text = "正常内容\uD835数学符号"  # \ud835 是孤立高代理
    fake_reader = FakeReader([FakePage(bad_text)])
    with patch("pypdf.PdfReader", return_value=fake_reader):
        result = await tool.run("surrogate.pdf")

    # 不崩溃，正常内容保留
    assert "正常内容" in result
    # 清理后可以安全 UTF-8 编码
    result.encode("utf-8")


@pytest.mark.asyncio
async def test_pdf_ingestion_extracts_text(settings, tmp_path):
    """
    上传入库流程：_collect_docs 对 PDF 提取文本并分块，
    确保 PDF 内容进入向量库（可被 search_kb 检索）。
    """
    from unittest.mock import patch as mock_patch
    from app.services.kb import KnowledgeBaseService

    # 构造一个假 PDF 文件在知识库 source 目录
    kb = KnowledgeBaseService(settings)
    kb.create_kb("pdf_kb")
    pdf_path = kb.kb_source_dir("pdf_kb") + "/paper.pdf"
    with open(pdf_path, "wb") as f:
        f.write(b"fake-pdf")

    # mock pypdf 返回两页文本
    fake_reader = FakeReader([FakePage("PDF 第一页内容：深度学习模型介绍"), FakePage("PDF 第二页内容：性能优化方法")])
    with mock_patch("pypdf.PdfReader", return_value=fake_reader):
        docs = kb._collect_docs([pdf_path])

    # 提取的文本被分块成 docs（含 source 元数据）
    assert len(docs) > 0
    texts = " ".join(t for t, _ in docs)
    assert "深度学习模型介绍" in texts
    assert "性能优化方法" in texts
    assert all("paper.pdf" in m["source"] for _, m in docs)
