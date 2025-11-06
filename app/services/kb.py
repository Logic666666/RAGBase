import os
import shutil
from typing import List, Iterable

from fastapi import UploadFile

from ..settings import Settings
from .vectorstore import VectorStore
from ..utils.chunking import chunk_texts


SUPPORTED_EXTS = {
    ".txt", ".md", ".py", ".java", ".sql", ".json", ".csv"
}


class KnowledgeBaseService:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings

    # base directories
    def kb_root(self, name: str) -> str:
        return os.path.join(self.settings.data_dir, "kb", name)

    def kb_source_dir(self, name: str) -> str:
        return os.path.join(self.kb_root(name), "source")

    def kb_vector_dir(self, name: str) -> str:
        return os.path.join(self.settings.data_dir, "vectorstore", name)

    def create_kb(self, name: str) -> None:
        os.makedirs(self.kb_source_dir(name), exist_ok=True)
        os.makedirs(self.kb_vector_dir(name), exist_ok=True)

    def list_kb(self) -> List[str]:
        base = os.path.join(self.settings.data_dir, "kb")
        if not os.path.exists(base):
            return []
        return sorted([d for d in os.listdir(base) if os.path.isdir(os.path.join(base, d))])

    def delete_kb(self, name: str) -> None:
        shutil.rmtree(self.kb_root(name), ignore_errors=True)
        shutil.rmtree(self.kb_vector_dir(name), ignore_errors=True)

    async def save_and_ingest_files(self, name: str, files: List[UploadFile]) -> int:
        self.create_kb(name)
        saved_paths: List[str] = []
        for f in files:
            ext = os.path.splitext(f.filename or "")[1].lower()
            if ext not in SUPPORTED_EXTS:
                # skip unsupported silently
                continue
            target = os.path.join(self.kb_source_dir(name), f.filename)
            os.makedirs(os.path.dirname(target), exist_ok=True)
            with open(target, "wb") as out:
                out.write(await f.read())
            saved_paths.append(target)

        # Build docs and index
        docs = self._collect_docs(saved_paths)
        VectorStore(self.settings).add_documents(self.kb_vector_dir(name), docs)
        return len(docs)

    def ingest_git_repo(self, name: str, repo_url: str, branch: str | None, username: str | None, token: str | None) -> int:
        from git import Repo
        self.create_kb(name)
        tmp_dir = os.path.join(self.kb_root(name), "git_tmp")
        if os.path.exists(tmp_dir):
            shutil.rmtree(tmp_dir, ignore_errors=True)
        os.makedirs(tmp_dir, exist_ok=True)

        # Build URL with credentials if provided
        url = repo_url
        if username and token and repo_url.startswith("https://") and "@" not in repo_url:
            url = repo_url.replace("https://", f"https://{username}:{token}@")

        Repo.clone_from(url, tmp_dir, branch=branch or None)

        # Copy supported files into source
        saved_paths: List[str] = []
        for root, _, files in os.walk(tmp_dir):
            for fn in files:
                ext = os.path.splitext(fn)[1].lower()
                if ext not in SUPPORTED_EXTS:
                    continue
                src = os.path.join(root, fn)
                rel = os.path.relpath(src, tmp_dir)
                dst = os.path.join(self.kb_source_dir(name), rel)
                os.makedirs(os.path.dirname(dst), exist_ok=True)
                shutil.copy2(src, dst)
                saved_paths.append(dst)

        shutil.rmtree(tmp_dir, ignore_errors=True)

        docs = self._collect_docs(saved_paths)
        VectorStore(self.settings).add_documents(self.kb_vector_dir(name), docs)
        return len(docs)

    def _collect_docs(self, paths: List[str]):
        texts = []
        metadatas = []
        for p in paths:
            try:
                with open(p, "r", encoding="utf-8", errors="ignore") as f:
                    content = f.read()
                for chunk in chunk_texts(content):
                    texts.append(chunk)
                    metadatas.append({"source": p})
            except Exception:
                continue
        return list(zip(texts, metadatas))


