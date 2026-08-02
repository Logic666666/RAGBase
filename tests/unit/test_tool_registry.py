"""
工具注册中心 + jsonschema 参数校验单元测试

验证：
  - 工具注册 / 获取
  - schemas() 生成 LLM 可读的工具列表
  - execute() 的参数校验（缺少必填、类型错误、未知参数）
  - execute() 的异常兜底（工具内部错误转文本）
"""

import pytest

from app.tools.base import BaseTool, ToolSpec
from app.tools.registry import ToolRegistry


class FakeTool(BaseTool):
    """带参数校验的 mock 工具"""

    @property
    def spec(self) -> ToolSpec:
        return ToolSpec(
            name="fake_tool",
            description="测试工具",
            parameters={
                "type": "object",
                "properties": {
                    "query": {"type": "string"},
                    "top_k": {"type": "integer", "default": 4},
                },
                "required": ["query"],
                "additionalProperties": False,
            },
        )

    async def run(self, query: str, top_k: int = 4) -> str:
        if query == "boom":
            raise RuntimeError("内部错误")
        return f"OK: {query} x{top_k}"


@pytest.fixture
def registry():
    r = ToolRegistry()
    r.register(FakeTool())
    return r


# ──────────────────────────────────────────────
# 注册与发现
# ──────────────────────────────────────────────

def test_register_and_get(registry):
    tool = registry.get("fake_tool")
    assert tool is not None
    assert registry.get("nonexistent") is None


def test_schemas_contains_tool_info(registry):
    schemas = registry.schemas()
    assert len(schemas) == 1
    assert schemas[0]["name"] == "fake_tool"
    assert "description" in schemas[0]
    assert schemas[0]["parameters"]["type"] == "object"
    assert "query" in schemas[0]["parameters"]["properties"]


# ──────────────────────────────────────────────
# 参数校验
# ──────────────────────────────────────────────

@pytest.mark.asyncio
async def test_execute_valid_params(registry):
    result = await registry.execute("fake_tool", {"query": "hello", "top_k": 3})
    assert result == "OK: hello x3"


@pytest.mark.asyncio
async def test_execute_missing_required_param(registry):
    result = await registry.execute("fake_tool", {"top_k": 3})
    assert "工具参数错误" in result
    assert "query" in result


@pytest.mark.asyncio
async def test_execute_wrong_type(registry):
    result = await registry.execute("fake_tool", {"query": "hello", "top_k": "abc"})
    assert "工具参数错误" in result


@pytest.mark.asyncio
async def test_execute_unknown_param(registry):
    """LLM 拼错参数名（qeruy）应被 additionalProperties 拦截"""
    result = await registry.execute("fake_tool", {"qeruy": "hello"})
    assert "工具参数错误" in result


@pytest.mark.asyncio
async def test_execute_unknown_tool(registry):
    """工具不存在时返回错误文本而非抛异常"""
    result = await registry.execute("nonexistent", {})
    assert "不存在" in result


@pytest.mark.asyncio
async def test_execute_catches_runtime_error(registry):
    """工具内部异常 → 转为错误文本（不崩溃）"""
    result = await registry.execute("fake_tool", {"query": "boom"})
    assert "工具执行失败" in result
    assert "RuntimeError" in result
