"""
文本分块工具模块

将长文本分割成适合向量存储和检索的小块。
核心策略：递归优先分割（RecursiveCharacterTextSplitter 的手搓实现）：
  - 优先在自然边界（段落 → 行 → 句子 → 单词 → 字符）切分
  - 相邻块之间保留 overlap，避免在语义中途切断
  - 太小的块合并回相邻块，提高检索效率

使用示例：
    chunks = split_text("很长很长的文档...", chunk_size=500, chunk_overlap=50)
    # → ["第一段...", "第二段...", ...]
"""

from typing import Optional


# 默认分隔符优先级列表
"""
优先级从高到低
切分时会先尝试用高优先级（段落级）切，切不动了才降级到句子级
""(空字符串) 是最后的兜底——按字符数硬切
"""
DEFAULT_SEPARATORS = [
    "\n\n",  # 段落（双换行 → 语义最完整的边界）
    "\n",    # 行（单换行 → 代码行/列表项的自然边界）
    "。",    # 中文句号
    "！",    # 中文感叹号
    "？",    # 中文问号
    ".",     # 英文句号
    "!",     # 英文感叹号
    "?",     # 英文问号
    " ",     # 空格（单词级）
    "",      # 字符级（兜底——按 chunk_size 硬切）
]

# 公开接口
def split_text(
    text: str,
    separators: Optional[list[str]] = None,
    chunk_size: int = 1200,
    chunk_overlap: int = 150,
) -> list[str]:
    """
    将文本分割成适合向量检索的块。

    两阶段处理：
    1. 递归切分（_split_by_separators）
       → 从高优先级分隔符开始切，切不动了降级
       → 直到所有块 ≤ chunk_size
    2. 合并+重叠（_merge_and_overlap）
       → 把太小的块合并回去
       → 相邻块之间加上 overlap 字符

    Args:
        text:          要分割的文本
        separators:    分隔符优先级列表（None 则使用默认值）
        chunk_size:    每块的最大字符数
        chunk_overlap: 相邻块之间的重叠字符数

    Returns:
        分割后的文本块列表
    """
    if not text:
        return []

    sep_list = separators if separators is not None else DEFAULT_SEPARATORS

    # 阶段 1：递归切分到每块 ≤ chunk_size
    pieces = _split_by_separators(text, sep_list, chunk_size)

    # 阶段 2：合并小块 + 添加 overlap
    return _merge_and_overlap(pieces, chunk_size, chunk_overlap)

# 阶段 1：递归切分
def _split_by_separators(
    text: str,
    separators: list[str],
    chunk_size: int,
) -> list[str]:
    """
    递归切分核心。

    算法逻辑：
    1. 如果 text <= chunk_size：
        直接返回 [text]
    2. 如果 separators 已经用完（再也没有分隔符能试了）：
        按 chunk_size 硬切
    3. 取出 separators[0] 作为当前分隔符 sep：
       a. 如果 sep 是 ""（字符级兜底）：
           按 chunk_size 硬切
       b. 如果 sep 不在 text 中（当前分隔符切不动）：
           用 separators[1:] 递归  ← 降级尝试下一级分隔符
       c. 用 sep 切分 text：
           对每个切出来的小块：
             - 如果小块 <= chunk_size：直接保留
             - 如果小块 > chunk_size：用 separators[1:] 递归切

    递归时跳过当前分隔符（separators[1:]），因为一个段落用 "\n\n" 切完之后，剩下的块还太大，
    说明 "\n\n" 这个级别已经处理完了，需要用更细的粒度（如 "\n" 或 "。"）继续切。
    """
    # 边界条件 1：空文本
    if not text:
        return []

    # 边界条件 2：已经够小了，不需要再切
    if len(text) <= chunk_size:
        return [text]

    # 边界条件 3：没有分隔符可用了，硬切
    if not separators:
        return _hard_split(text, chunk_size)

    sep = separators[0]

    # 兜底：空字符串分隔符 = 按字符数硬切
    if sep == "":
        return _hard_split(text, chunk_size)

    # 如果当前分隔符不在文本中 → 降级尝试下一级
    if sep not in text:
        return _split_by_separators(text, separators[1:], chunk_size)

    # 核心切分逻辑
    # split(sep) 会去掉分隔符，但分隔符本身携带语义信息（如段落间距），
    # 所以我们把它加回到每个小块（最后一块除外）的末尾。
    raw_pieces = text.split(sep)
    pieces_with_sep = [
        piece + sep if i < len(raw_pieces) - 1 else piece
        for i, piece in enumerate(raw_pieces)
    ]

    # 对每块递归切分
    result: list[str] = []
    for piece in pieces_with_sep:
        if len(piece) <= chunk_size:
            result.append(piece)
        else:
            # ！递归降级：用剩下的分隔符继续切
            result.extend(
                _split_by_separators(piece, separators[1:], chunk_size)
            )

    return result


def _hard_split(text: str, chunk_size: int) -> list[str]:
    """最后的兜底方案：按 chunk_size 硬切，不关心语义边界"""
    return [text[i:i + chunk_size] for i in range(0, len(text), chunk_size)]


# 阶段 2：合并 + 重叠
def _merge_and_overlap(
    pieces: list[str],
    chunk_size: int,
    chunk_overlap: int,
) -> list[str]:
    """
    合并小块 + 添加重叠。

    递归切分后可能有大量极小块的（如按句号切后每句十几个字），
    需要做两件事：

    1. 合并：把小段拼回大段，直到快接近 chunk_size
       → 检索粒度过细反而降低检索质量
       → 合并后每块包含完整的 1~N 个句子，语义更完整

    2. 重叠：相邻块之间共享部分文本
       → 避免一个语义单元（如一个段落）恰好在块边界被切断
       → 切断的那部分信息，通过 overlap"挂载"到下一块开头
       → 这样检索时无论命中哪一块，都能找到完整的上下文

    Args:
        pieces:        递归切分后的小块列表
        chunk_size:    目标块大小
        chunk_overlap: 重叠字符数

    Returns:
        合并+重叠后的块列表
    """
    if not pieces:
        return []

    # 合并
    merged = _merge_pieces(pieces, chunk_size)

    # 添加重叠
    if chunk_overlap > 0 and len(merged) > 1:
        merged = _add_overlap(merged, chunk_overlap)

    return merged


def _merge_pieces(pieces: list[str], chunk_size: int) -> list[str]:
    """
    将太小的块合并到相邻块中。

    策略：顺序扫描，如果当前块 + 下一块 ≤ chunk_size，就合并。
    最简单的合并策略——贪心从前往后合并。
    """
    merged: list[str] = []
    current = pieces[0]

    for piece in pieces[1:]:
        if len(current) + len(piece) <= chunk_size:
            current += piece
        else:
            merged.append(current)
            current = piece

    if current:
        merged.append(current)

    return merged


def _add_overlap(chunks: list[str], overlap_size: int) -> list[str]:
    """
    在相邻块之间添加重叠文本。

    "从上一块末尾借 overlap_size 个字符，拼到下一块开头"。
    这样即使一个知识点恰好在块边界被切断了，它的前半段也会作为 overlap
    出现在下一块的开头，被检索到。

    overlap 是"附加上去的"，叠加后下一块可能超过 chunk_size。
    这是有意为之——overlap 不计入块大小预算。
    """
    result: list[str] = []
    result.append(chunks[0])

    for i in range(1, len(chunks)):
        prev = chunks[i - 1]
        curr = chunks[i]

        # 从上一块末尾取 overlap_text
        if len(prev) >= overlap_size:
            overlap_text = prev[-overlap_size:]
        else:
            overlap_text = prev  # 上一块本身比 overlap 还小，全部拿来

        result.append(overlap_text + curr)

    return result
