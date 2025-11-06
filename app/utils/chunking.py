from langchain_text_splitters import RecursiveCharacterTextSplitter


def chunk_texts(text: str) -> list[str]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1200,
        chunk_overlap=150,
        separators=["\n\n", "\n", ".", " ", ""],
    )
    return splitter.split_text(text)


