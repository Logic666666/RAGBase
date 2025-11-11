import os
import shutil
from typing import List, Iterable

from fastapi import UploadFile

from ..settings import Settings
from .vectorstore import VectorStore
from ..utils.chunking import chunk_texts


# 支持的文件扩展名集合，用于验证上传文件类型
SUPPORTED_EXTS = {
    ".txt", ".md", ".py", ".java", ".sql", ".json", ".csv"
}


class KnowledgeBaseService:
    """
    知识库服务类，负责管理知识库的创建、删除、文件存储和向量索引构建
    
    该服务提供了通过文件上传和Git仓库导入两种方式构建知识库的能力，
    并将处理后的文档内容存储到向量数据库中，以便后续检索。
    """
    
    def __init__(self, settings: Settings) -> None:
        """
        初始化知识库服务
        
        Args:
            settings: 应用程序设置对象，包含数据存储路径等配置信息
        """
        self.settings = settings

    # ------------------------------
    # 路径管理方法
    # ------------------------------
    
    def kb_root(self, name: str) -> str:
        """
        获取知识库的根目录路径
        
        Args:
            name: 知识库名称
            
        Returns:
            知识库根目录的绝对路径
        """
        return os.path.join(self.settings.data_dir, "kb", name)

    def kb_source_dir(self, name: str) -> str:
        """
        获取知识库的源文件存储目录路径
        
        Args:
            name: 知识库名称
            
        Returns:
            源文件存储目录的绝对路径
        """
        return os.path.join(self.kb_root(name), "source")

    def kb_vector_dir(self, name: str) -> str:
        """
        获取知识库的向量存储目录路径
        
        Args:
            name: 知识库名称
            
        Returns:
            向量存储目录的绝对路径
        """
        # 修复路径计算，确保与向量存储服务使用相同的路径
        vector_path = os.path.join(self.settings.data_dir, "vectorstore", name)
        # 添加调试日志
        import logging
        logging.debug(f"知识库 '{name}' 向量存储路径: {vector_path}")
        return vector_path

    # ------------------------------
    # 知识库管理方法
    # ------------------------------
    
    def create_kb(self, name: str) -> None:
        """
        创建新的知识库目录结构
        
        该方法会创建知识库的根目录、源文件目录和向量存储目录，
        如果目录已存在则不会抛出异常。
        
        Args:
            name: 知识库名称
        """
        os.makedirs(self.kb_source_dir(name), exist_ok=True)
        os.makedirs(self.kb_vector_dir(name), exist_ok=True)

    def list_kb(self) -> List[str]:
        """
        列出所有可用的知识库名称
        
        Returns:
            按字母排序的知识库名称列表，如果没有知识库则返回空列表
        """
        base = os.path.join(self.settings.data_dir, "kb")
        if not os.path.exists(base):
            return []
        return sorted([d for d in os.listdir(base) if os.path.isdir(os.path.join(base, d))])

    def delete_kb(self, name: str) -> bool:
        """
        删除指定的知识库及其所有内容
        
        该方法会删除知识库的源文件目录和向量存储目录，
        并返回删除操作的实际结果。
        
        Args:
            name: 要删除的知识库名称
            
        Returns:
            bool: 如果所有目录都被成功删除则返回True，否则返回False
        """
        # 记录初始状态
        initial_root_exists = os.path.exists(self.kb_root(name))
        initial_vector_exists = os.path.exists(self.kb_vector_dir(name))
        
        # 如果两个目录都不存在，直接返回True
        if not initial_root_exists and not initial_vector_exists:
            return True
            
        # 尝试删除目录，处理Windows权限问题
        try:
            if initial_root_exists:
                # 对于Windows系统，先尝试修改文件权限
                if os.name == 'nt':
                    self._modify_windows_permissions(self.kb_root(name))
                shutil.rmtree(self.kb_root(name))
                
            if initial_vector_exists:
                if os.name == 'nt':
                    self._modify_windows_permissions(self.kb_vector_dir(name))
                shutil.rmtree(self.kb_vector_dir(name))
                
        except Exception as e:
            # 第一次删除失败，尝试使用更激进的方法
            try:
                if initial_root_exists and os.path.exists(self.kb_root(name)):
                    if os.name == 'nt':
                        # 使用Windows命令行强制删除
                        import subprocess
                        # 使用PowerShell命令强制删除（最高权限）
                        subprocess.run(
                            ['powershell', '-Command', 'Remove-Item', '-Path', f'"{self.kb_root(name)}"', '-Recurse', '-Force', '-ErrorAction', 'Stop'],
                            check=True,
                            capture_output=True,
                            text=True
                        )
                    else:
                        # Linux/macOS系统使用chmod + rm -rf
                        subprocess.run(
                            ['chmod', '-R', '777', self.kb_root(name)],
                            check=True
                        )
                        shutil.rmtree(self.kb_root(name))
                        
                if initial_vector_exists and os.path.exists(self.kb_vector_dir(name)):
                    if os.name == 'nt':
                        import subprocess
                        subprocess.run(
                            ['cmd', '/c', 'rmdir', '/s', '/q', self.kb_vector_dir(name)],
                            check=True,
                            capture_output=True,
                            text=True
                        )
                    else:
                        subprocess.run(
                            ['chmod', '-R', '777', self.kb_vector_dir(name)],
                            check=True
                        )
                        shutil.rmtree(self.kb_vector_dir(name))
            except Exception as e:
                # 记录删除失败的异常信息
                import logging
                logging.error(f"删除知识库 '{name}' 失败: {str(e)}")
                return False
                
        # 验证删除结果
        root_deleted = not os.path.exists(self.kb_root(name))
        vector_deleted = not os.path.exists(self.kb_vector_dir(name))
        
        return root_deleted and vector_deleted
        
    def _modify_windows_permissions(self, path: str) -> None:
        """修改Windows系统文件权限以允许删除"""
        import ctypes
        from ctypes import wintypes
        
        # 设置文件权限常量
        FILE_READ_DATA = 0x0001
        FILE_WRITE_DATA = 0x0002
        FILE_DELETE = 0x00010000
        
        # 获取当前进程令牌
        hToken = wintypes.HANDLE()
        if not ctypes.windll.advapi32.OpenProcessToken(
            ctypes.windll.kernel32.GetCurrentProcess(),
            0x0020 | 0x0008,  # TOKEN_ADJUST_PRIVILEGES | TOKEN_QUERY
            ctypes.byref(hToken)
        ):
            return
            
        # 启用删除权限
        luid = wintypes.LUID()
        if ctypes.windll.advapi32.LookupPrivilegeValueW(
            None, "SeDeleteFilePrivilege", ctypes.byref(luid)
        ):
            tp = ctypes.create_string_buffer(1024)
            ctypes.cast(tp, ctypes.POINTER(wintypes.TOKEN_PRIVILEGES)).contents.PrivilegeCount = 1
            ctypes.cast(tp, ctypes.POINTER(wintypes.TOKEN_PRIVILEGES)).contents.Privileges[0].Luid = luid
            ctypes.cast(tp, ctypes.POINTER(wintypes.TOKEN_PRIVILEGES)).contents.Privileges[0].Attributes = 0x00000002  # SE_PRIVILEGE_ENABLED
            
            ctypes.windll.advapi32.AdjustTokenPrivileges(
                hToken, False, ctypes.cast(tp, ctypes.POINTER(wintypes.TOKEN_PRIVILEGES)),
                0, None, None
            )
        
        # 递归修改目录权限
        for root, dirs, files in os.walk(path):
            for name in dirs + files:
                item_path = os.path.join(root, name)
                try:
                    # 获取文件句柄
                    hFile = ctypes.windll.kernel32.CreateFileW(
                        item_path,
                        FILE_READ_DATA | FILE_WRITE_DATA | FILE_DELETE,
                        0x0007,  # FILE_SHARE_READ | FILE_SHARE_WRITE | FILE_SHARE_DELETE
                        None,
                        3,  # OPEN_EXISTING
                        0x02000000,  # FILE_ATTRIBUTE_NORMAL
                        None
                    )
                    
                    if hFile != wintypes.HANDLE(-1).value:
                        # 获取当前安全描述符
                        pSD = wintypes.LPVOID()
                        if ctypes.windll.kernel32.GetSecurityInfo(
                            hFile,
                            1,  # SE_FILE_OBJECT
                            0, None, None, None, None,
                            ctypes.byref(pSD)
                        ) == 0:
                            # 设置所有者为当前用户
                            ctypes.windll.kernel32.SetSecurityInfo(
                                hFile,
                                1,  # SE_FILE_OBJECT
                                0x00000001,  # OWNER_SECURITY_INFORMATION
                                None, None, None, pSD
                            )
                        ctypes.windll.kernel32.CloseHandle(hFile)
                except Exception:
                    continue
        
        # 如果两个目录都不存在，直接返回True
        if not initial_root_exists and not initial_vector_exists:
            return True
            
        # 尝试删除目录，处理Windows权限问题
        try:
            if initial_root_exists:
                # 对于Windows系统，先尝试修改文件权限
                if os.name == 'nt':
                    self._modify_windows_permissions(self.kb_root(name))
                shutil.rmtree(self.kb_root(name))
                
            if initial_vector_exists:
                if os.name == 'nt':
                    self._modify_windows_permissions(self.kb_vector_dir(name))
                shutil.rmtree(self.kb_vector_dir(name))
                
        except Exception as e:
            # 第一次删除失败，尝试使用更激进的方法
            try:
                if initial_root_exists and os.path.exists(self.kb_root(name)):
                    if os.name == 'nt':
                        # 使用Windows命令行强制删除
                        import subprocess
                        subprocess.run(
                            ['cmd', '/c', 'rmdir', '/s', '/q', self.kb_root(name)],
                            check=True,
                            capture_output=True,
                            text=True
                        )
                    else:
                        # Linux/macOS系统使用chmod + rm -rf
                        subprocess.run(
                            ['chmod', '-R', '777', self.kb_root(name)],
                            check=True
                        )
                        shutil.rmtree(self.kb_root(name))
                        
                if initial_vector_exists and os.path.exists(self.kb_vector_dir(name)):
                    if os.name == 'nt':
                        import subprocess
                        subprocess.run(
                            ['cmd', '/c', 'rmdir', '/s', '/q', self.kb_vector_dir(name)],
                            check=True,
                            capture_output=True,
                            text=True
                        )
                    else:
                        subprocess.run(
                            ['chmod', '-R', '777', self.kb_vector_dir(name)],
                            check=True
                        )
                        shutil.rmtree(self.kb_vector_dir(name))
            except Exception as e:
                # 记录删除失败的异常信息
                import logging
                logging.error(f"删除知识库 '{name}' 失败: {str(e)}")
                return False
        except Exception as e:
            # 记录删除失败的异常信息
            import logging
            logging.error(f"删除知识库 '{name}' 失败: {str(e)}")
            return False
            
        # 验证删除结果
        root_deleted = not os.path.exists(self.kb_root(name))
        vector_deleted = not os.path.exists(self.kb_vector_dir(name))
        
        return root_deleted and vector_deleted

    # ------------------------------
    # 文档导入方法
    # ------------------------------
    
    async def save_and_ingest_files(self, name: str, files: List[UploadFile]) -> int:
        """
        保存上传的文件并将其内容导入知识库
        
        处理流程:
        1. 创建知识库目录结构
        2. 验证并保存上传的文件
        3. 读取文件内容并分块处理
        4. 将分块后的文本添加到向量存储
        
        Args:
            name: 目标知识库名称
            files: 从FastAPI接收到的上传文件列表
            
        Returns:
            成功导入向量存储的文档块数量
        """
        self.create_kb(name)
        saved_paths: List[str] = []
        
        # 保存上传的文件
        for f in files:
            # 获取文件扩展名并转换为小写
            ext = os.path.splitext(f.filename or "")[1].lower()
            if ext not in SUPPORTED_EXTS:
                # 跳过不支持的文件类型，不抛出错误
                continue
                
            # 构建目标文件路径
            target = os.path.join(self.kb_source_dir(name), f.filename)
            os.makedirs(os.path.dirname(target), exist_ok=True)
            
            # 保存文件内容
            with open(target, "wb") as out:
                out.write(await f.read())
            saved_paths.append(target)

        # 处理文档并添加到向量存储
        docs = self._collect_docs(saved_paths)
        VectorStore(self.settings).add_documents(self.kb_vector_dir(name), docs)
        return len(docs)

    def ingest_git_repo(self, name: str, repo_url: str, branch: str | None, username: str | None, token: str | None) -> int:
        """
        从Git仓库导入代码文件并构建知识库

        处理流程:
        1. 创建知识库目录结构
        2. 克隆Git仓库到临时目录（支持重试和Git加速）
        3. 筛选并复制支持的文件类型到知识库
        4. 读取文件内容并分块处理
        5. 将分块后的文本添加到向量存储

        Args:
            name: 目标知识库名称
            repo_url: Git仓库URL
            branch: 要克隆的分支名称，None表示使用默认分支
            username: Git认证用户名，None表示不需要认证
            token: Git认证令牌，None表示不需要认证

        Returns:
            成功导入向量存储的文档块数量

        Raises:
            HTTPException: 如果Git克隆失败，包含详细的错误信息
        """
        from git import Repo, GitCommandError
        import logging
        import time
        import subprocess
        
        self.create_kb(name)
        
        # 准备临时目录 - 使用唯一名称避免冲突
        import uuid
        tmp_dir = os.path.join(self.kb_root(name), f"git_tmp_{uuid.uuid4().hex[:8]}")
        
        # 确保目录不存在
        if os.path.exists(tmp_dir):
            # 如果目录已存在，使用更稳健的删除方法
            max_retries = 5
            for attempt in range(max_retries):
                try:
                    # 先尝试修改文件权限
                    for root, dirs, files in os.walk(tmp_dir):
                        for file in files:
                            file_path = os.path.join(root, file)
                            try:
                                os.chmod(file_path, 0o777)  # 设置完全权限
                            except:
                                pass
                    shutil.rmtree(tmp_dir, ignore_errors=False)
                    break
                except (PermissionError, OSError) as e:
                    if attempt == max_retries - 1:
                        # 如果还是无法删除，使用ignore_errors=True继续
                        shutil.rmtree(tmp_dir, ignore_errors=True)
                        break
                    time.sleep(0.5)  # 等待更长时间后重试
        
        # 创建新的空目录
        os.makedirs(tmp_dir, exist_ok=True)

        # 构建带有认证信息的URL（如果提供）
        url = repo_url
        if username and token and repo_url.startswith("https://") and "@" not in repo_url:
            url = repo_url.replace("https://", f"https://{username}:{token}@")

        # 克隆Git仓库（支持重试、超时和错误处理）
        max_clone_retries = 3
        retry_delay = 2  # 秒
        
        for attempt in range(max_clone_retries):
            try:
                # 尝试使用Git加速服务（如果可用）
                accelerated_url = self._get_accelerated_git_url(url)
                
                logging.info(f"克隆Git仓库 (尝试 {attempt + 1}/{max_clone_retries}): {accelerated_url}")
                
                # 使用GitPython克隆仓库，设置超时
                import signal
                from git import Repo
                
                # 设置克隆选项，包括超时
                clone_options = {
                    'branch': branch or None,
                    'timeout': self.settings.git_timeout
                }
                
                # 克隆仓库
                Repo.clone_from(accelerated_url, tmp_dir, **clone_options)
                break  # 成功则跳出循环
                
            except GitCommandError as e:
                error_msg = str(e)
                logging.error(f"Git克隆失败 (尝试 {attempt + 1}/{max_clone_retries}): {error_msg}")
                
                # 检查是否是JSON解析错误
                if "Unexpected token" in error_msg and "is not valid JSON" in error_msg:
                    logging.warning("检测到JSON解析错误，可能是Git服务端返回了HTML错误页面")
                    
                if attempt == max_clone_retries - 1:
                    # 最后一次尝试也失败，清理临时目录并抛出异常
                    shutil.rmtree(tmp_dir, ignore_errors=True)
                    # 最后一次尝试也失败，尝试使用Git命令行工具
                    logging.warning("GitPython克隆失败，尝试使用Git命令行工具...")
                    if self._fallback_to_git_cli(url, tmp_dir, branch):
                        break  # 命令行克隆成功
                    else:
                        # 清理临时目录并抛出异常
                        shutil.rmtree(tmp_dir, ignore_errors=True)
                        from fastapi import HTTPException
                        raise HTTPException(
                            status_code=500,
                            detail=f"无法克隆Git仓库: {error_msg}. 请检查仓库URL、认证信息和网络连接。"
                        )
                
                # 使用优化的重试策略等待
                wait_time = self._optimize_retry_strategy(attempt, max_clone_retries)
                logging.info(f"等待 {wait_time} 秒后重试...")
                time.sleep(wait_time)
                
            except Exception as e:
                error_msg = str(e)
                logging.error(f"Git克隆过程中发生未知错误 (尝试 {attempt + 1}/{max_clone_retries}): {error_msg}")
                
                if attempt == max_clone_retries - 1:
                    shutil.rmtree(tmp_dir, ignore_errors=True)
                    from fastapi import HTTPException
                    raise HTTPException(
                        status_code=500,
                        detail=f"Git克隆过程发生未知错误: {error_msg}"
                    )
                
                time.sleep(retry_delay * (attempt + 1))

        # 复制支持的文件到知识库源目录
        saved_paths: List[str] = []
        for root, _, files in os.walk(tmp_dir):
            for fn in files:
                ext = os.path.splitext(fn)[1].lower()
                if ext not in SUPPORTED_EXTS:
                    continue
                    
                src = os.path.join(root, fn)
                # 计算相对路径，保持仓库内的目录结构
                rel = os.path.relpath(src, tmp_dir)
                dst = os.path.join(self.kb_source_dir(name), rel)
                
                os.makedirs(os.path.dirname(dst), exist_ok=True)
                shutil.copy2(src, dst)
                saved_paths.append(dst)

        # 清理临时目录
        shutil.rmtree(tmp_dir, ignore_errors=True)

        # 处理文档并添加到向量存储
        docs = self._collect_docs(saved_paths)
        VectorStore(self.settings).add_documents(self.kb_vector_dir(name), docs)
        return len(docs)

    # ------------------------------
    # 内部辅助方法
    # ------------------------------
    
    def _collect_docs(self, paths: List[str]):
        """
        读取文件内容并分块处理，为向量存储准备文档数据
        
        Args:
            paths: 要处理的文件路径列表
            
        Returns:
            包含文本块和元数据的元组列表，格式为[(text_chunk, metadata), ...]
        """
        texts = []
        metadatas = []
        
        for p in paths:
            try:
                # 读取文件内容，使用utf-8编码并忽略解码错误
                with open(p, "r", encoding="utf-8", errors="ignore") as f:
                    content = f.read()
                    
                # 将文本分块并添加元数据
                for chunk in chunk_texts(content):
                    texts.append(chunk)
                    metadatas.append({"source": p})
                    
            except Exception:
                # 忽略处理失败的文件
                continue
                
        return list(zip(texts, metadatas))

    # ------------------------------
    # Git加速服务支持
    # ------------------------------
    
    def _get_accelerated_git_url(self, original_url: str) -> str:
        """
        获取Git加速服务URL（支持多加速器备选方案）
        
        支持多种Git加速服务：
        1. GitHub: ghproxy.com, fastgit.org, 等多种加速器
        2. GitLab: 使用mirror服务
        3. Gitee: 使用原生URL
        4. 其他: 保持原样
        
        Args:
            original_url: 原始Git仓库URL
            
        Returns:
            加速后的Git仓库URL
        """
        # 如果加速器被禁用，直接返回原URL
        if not self.settings.git_accelerator_enabled:
            return original_url
            
        # 如果URL已经是加速服务，直接返回
        if any(proxy in original_url for proxy in ["ghproxy.com", "fastgit.org", "mirror"]):
            return original_url
            
        # 获取加速器优先级配置
        accelerators = [acc.strip() for acc in self.settings.git_accelerator_priority.split(",")]
        
        # GitHub加速（多加速器备选）
        if "github.com" in original_url:
            for accelerator in accelerators:
                if accelerator == "ghproxy":
                    # 使用ghproxy.com加速GitHub
                    accelerated_url = original_url.replace(
                        "https://github.com/",
                        "https://ghproxy.com/https://github.com/"
                    )
                    if self._test_git_connection(accelerated_url):
                        import logging
                        logging.info(f"使用GitHub加速服务 (ghproxy): {accelerated_url}")
                        return accelerated_url
                        
                elif accelerator == "fastgit":
                    # 使用fastgit.org加速GitHub
                    accelerated_url = original_url.replace(
                        "https://github.com/",
                        "https://hub.fastgit.org/"
                    )
                    if self._test_git_connection(accelerated_url):
                        import logging
                        logging.info(f"使用GitHub加速服务 (fastgit): {accelerated_url}")
                        return accelerated_url
                        
                elif accelerator == "original":
                    # 使用原始URL
                    if self._test_git_connection(original_url):
                        return original_url
            
            # 所有加速器都失败，返回原始URL
            return original_url
            
        # GitLab加速（如果有镜像服务）
        elif "gitlab.com" in original_url:
            # 可以添加GitLab镜像服务，这里保持原样
            return original_url
            
        # Gitee和其他Git服务
        else:
            return original_url
            
    def _test_git_connection(self, url: str) -> bool:
        """
        测试Git连接是否可用
        
        Args:
            url: Git仓库URL
            
        Returns:
            bool: 连接是否可用
        """
        import subprocess
        import logging
        
        try:
            # 使用git ls-remote测试连接（轻量级操作）
            result = subprocess.run(
                ["git", "ls-remote", "--heads", url],
                capture_output=True,
                text=True,
                timeout=self.settings.git_connect_timeout
            )
            
            return result.returncode == 0
            
        except subprocess.TimeoutExpired:
            logging.warning(f"Git连接测试超时: {url}")
            return False
        except Exception as e:
            logging.warning(f"Git连接测试失败: {url}, 错误: {e}")
            return False

    def _fallback_to_git_cli(self, url: str, target_dir: str, branch: str | None = None) -> bool:
        """
        使用Git命令行工具作为备选方案克隆仓库
        
        当GitPython失败时，尝试使用系统Git命令
        
        Args:
            url: Git仓库URL
            target_dir: 目标目录
            branch: 分支名称
            
        Returns:
            bool: 是否成功克隆
        """
        import subprocess
        import logging
        
        try:
            # 构建Git命令
            cmd = ["git", "clone"]
            if branch:
                cmd.extend(["-b", branch])
            
            # 添加超时和重试参数
            cmd.extend(["--progress", "--verbose"])
            cmd.extend([url, target_dir])
            
            # 执行Git命令，使用配置的超时时间
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=self.settings.git_timeout
            )
            
            if result.returncode == 0:
                logging.info(f"Git命令行克隆成功: {url}")
                return True
            else:
                # 分析错误类型
                error_output = result.stderr
                logging.error(f"Git命令行克隆失败: {error_output}")
                
                # 如果是网络问题，可以尝试不同的加速器
                if "unable to access" in error_output or "Connection" in error_output:
                    logging.warning("检测到网络连接问题，尝试其他加速器...")
                    # 这里可以添加逻辑尝试其他加速器URL
                
                return False
                
        except subprocess.TimeoutExpired:
            logging.error(f"Git克隆超时（{self.settings.git_timeout}秒）")
            return False
        except Exception as e:
            logging.error(f"Git命令行执行错误: {e}")
            return False
            
    def _optimize_retry_strategy(self, attempt: int, max_retries: int) -> int:
        """
        优化重试策略（指数退避 + 随机抖动）
        
        Args:
            attempt: 当前尝试次数
            max_retries: 最大重试次数
            
        Returns:
            int: 下一次重试的等待时间（秒）
        """
        import random
        import math
        
        # 指数退避基础时间
        base_delay = 2
        max_delay = 60
        
        # 计算指数退避时间
        delay = min(max_delay, base_delay * math.pow(2, attempt))
        
        # 添加随机抖动（±20%）
        jitter = random.uniform(0.8, 1.2)
        delay = delay * jitter
        
        return int(delay)
