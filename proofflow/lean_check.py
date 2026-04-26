import ast
import asyncio
import json
import os
import re
import signal
import subprocess
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional

from kimina_client import (
    CheckResponse,
    KiminaClient,
    ReplResponse,
    Snippet,
    SnippetStatus,
)

from .utils import remove_imports


def extract_errors(text):
    errors = []
    # find all `response={...}` substrings
    for match in re.finditer(r"response=(\{.*?\}) diagnostics=None", text, re.DOTALL):
        response_str = match.group(1)
        try:
            # make it valid Python dict (single quotes -> double)
            response_str_jsonlike = response_str.replace("null", "None")
            response = ast.literal_eval(response_str_jsonlike)
            for msg in response.get("messages", []):
                if msg.get("severity") == "error":
                    errors.append(
                        {
                            "line": msg["pos"]["line"],
                            "column": msg["pos"]["column"],
                            "endLine": msg["endPos"]["line"],
                            "endColumn": msg["endPos"]["column"],
                            "data": msg["data"],
                        }
                    )
        except Exception as e:
            print("parse failed:", e)
    return errors


# --- Provided helper functions from the user's prompt ---

LEAN_LIBRARIES = """import Mathlib
import Aesop

set_option maxHeartbeats 0

open BigOperators Real Nat Topology Rat Filter"""


def process_lean_string(lean_string: str):
    """
    Process a Lean code string to ensure required imports and options are present.

    Adds the following in order if missing:
    1. import Mathlib (first line)
    2. import Aesop (second line, after Mathlib)
    3. set_option maxHeartbeats 0 (after Aesop)
    4. open BigOperators Real Nat Topology Rat Filter (after set_option)

    Args:
        lean_string (str): The Lean code as a string

    Returns:
        str: The processed Lean code with required imports/options added
    """

    # Split into lines and remove empty lines at the start for processing
    lines = lean_string.split("\n")

    # Required statements to check for
    required_statements = [
        "import Mathlib",
        "import Aesop",
        "set_option maxHeartbeats 0",
        "open BigOperators Real Nat Topology Rat Filter",
    ]

    # Check which statements are already present
    present_statements = set()
    for line in lines:
        line_stripped = line.strip()
        for stmt in required_statements:
            if stmt in line_stripped:
                present_statements.add(stmt)

    # Find the insertion point (after any existing imports/options)
    insert_index = 0

    # Skip any existing imports or set_option statements at the beginning
    for i, line in enumerate(lines):
        line_stripped = line.strip()
        if (
            line_stripped.startswith("import ")
            or line_stripped.startswith("set_option ")
            or line_stripped.startswith("open ")
            or line_stripped == ""
            or line_stripped.startswith("--")
        ):  # Also skip comments and empty lines
            insert_index = i + 1
        else:
            break

    # Build list of statements to insert
    statements_to_insert = []

    for stmt in required_statements:
        if stmt not in present_statements:
            statements_to_insert.append(stmt)

    # Insert missing statements at the appropriate position
    if statements_to_insert:
        # Insert in reverse order to maintain correct positioning
        for stmt in reversed(statements_to_insert):
            lines.insert(insert_index, stmt)

    # Ensure there's an empty line after the imports/options section
    # Find the end of the imports/options section
    imports_end_index = 0
    for i, line in enumerate(lines):
        line_stripped = line.strip()
        if (
            line_stripped.startswith("import ")
            or line_stripped.startswith("set_option ")
            or line_stripped.startswith("open ")
            or line_stripped.startswith("--")
        ):  # Skip comments too
            imports_end_index = i + 1
        elif line_stripped == "":  # Empty line
            continue
        else:
            break

    # Check if there's already an empty line after imports
    if (
        imports_end_index < len(lines)
        and imports_end_index > 0
        and lines[imports_end_index].strip() != ""
    ):
        lines.insert(imports_end_index, "")

    # Join lines back together
    result = "\n".join(lines)

    # Clean up any duplicate empty lines at the start
    while result.startswith("\n"):
        result = result[1:]

    return result


# create a function that picks a lean_strings and does the following
# check if imports are missing by the following steps:
# if "import Mathlib" does not exist add it as first line
# if "import Aesop" does not exists add it second line after Mathlib
# same thing for this set_option maxHeartbeats 0 -> add it after aesor
# same thing for this open BigOperators Real Nat Topology Rat


def _analyze_lean_output(output: str):
    error_patterns = [
        r'"severity"\s*:\s*"error"',
        r"'severity'\s*:\s*'error'",
    ]

    lean_pass = not any(re.search(pattern, output) for pattern in error_patterns)
    lean_verify = lean_pass and not any(
        [
            "declaration uses 'sorry'" in output,
            'declaration uses "sorry"' in output,
            "declaration uses `sorry`" in output,
            '"kind":"hasSorry"' in output,
            "failed" in output,
        ]
    )
    return lean_pass, lean_verify


def _analyze_lsp_diagnostics(diagnostics: List[Dict[str, Any]]):
    def _diag_severity(diag: Dict[str, Any]) -> Optional[int]:
        severity = diag.get("severity")
        if isinstance(severity, int):
            return severity
        if isinstance(severity, str):
            lowered = severity.lower()
            if lowered == "error":
                return 1
            if lowered == "warning":
                return 2
        return None

    def _diag_text(diag: Dict[str, Any]) -> str:
        try:
            return json.dumps(diag, ensure_ascii=False)
        except Exception:
            return str(diag)

    lean_pass = not any(_diag_severity(diag) == 1 for diag in diagnostics)
    lean_verify = lean_pass and not any(
        any(flag in _diag_text(diag).lower() for flag in ("sorry", "hassorry", "failed"))
        for diag in diagnostics
    )
    return lean_pass, lean_verify


def _build_temp_file_path(
    project_path: str,
    temp_root: str | None = None,
    job_id: str | None = None,
) -> Path:
    temp_dir = Path(temp_root) if temp_root else (Path(project_path) / "temp" / "lean_jobs")
    os.makedirs(temp_dir, exist_ok=True)
    unique_id = job_id or uuid.uuid4().hex
    return temp_dir / f"{unique_id}.lean"


def verify_lean_lemma_server(
    lean_string: str, client: KiminaClient, add_imports=False, timeout: int = 180
):
    """
    Verifies a Lean lemma using a remote server API.
    The client object is now passed to this function.
    """
    full_code = f"{LEAN_LIBRARIES}\n\n{lean_string}" if add_imports else lean_string

    snippets = [
        Snippet(id=str(idx), code=proof) for idx, proof in enumerate([full_code])
    ]

    timeout = 60
    compilation_result: CheckResponse = client.check(
        snips=snippets,
        timeout=timeout,
        max_workers=10,
    )

    results: list[ReplResponse] = compilation_result.results

    output = str(results[0])

    # Check for errors
    error_patterns = [
        r'"severity"\s*:\s*"error"',  # Double quotes with optional spaces
        r"'severity'\s*:\s*'error'",  # Single quotes with optional spaces
    ]

    lean_pass = not any(re.search(pattern, output) for pattern in error_patterns)

    # Check for verification (no errors, no sorries, no failures)
    lean_verify = lean_pass and not any(
        [
            "declaration uses 'sorry'" in output,
            'declaration uses "sorry"' in output,
            "declaration uses `sorry`" in output,
            '"kind":"hasSorry"' in output,
            "failed" in output,
        ]
    )

    output = extract_errors(output)

    return lean_pass, lean_verify, output


def verify_lean_lemma_local(
    lean_string: str,
    project_path: str,
    add_imports=False,
    timeout: int = 180,
    temp_root: str | None = None,
    job_id: str | None = None,
):
    """
    Verifies a single Lean lemma or theorem using the `lean` executable.
    (Function body as provided in the user's prompt)
    """
    temp_file_path = _build_temp_file_path(
        project_path=project_path,
        temp_root=temp_root,
        job_id=job_id,
    )
    full_code = f"{LEAN_LIBRARIES}\n\n{lean_string}" if add_imports else lean_string

    try:
        with open(temp_file_path, "w", encoding="utf-8") as f:
            f.write(full_code)

        result = subprocess.run(
            ["lake", "env", "lean", temp_file_path, "--json"],
            cwd=project_path,
            capture_output=True,
            text=True,
            encoding="utf-8",  # Force UTF-8 encoding
            errors="replace",  # Replace any problematic characters
            timeout=timeout,
        )

        # Check if process failed
        if result.returncode != 0 and not result.stdout:
            return (
                False,
                False,
                f"Process failed with exit code {result.returncode}: {result.stderr.strip()}",
            )

        # Check if no output (clean file)
        if not result.stdout.strip():
            return True, True, None

        # Parse the output
        output = result.stdout

        lean_pass, lean_verify = _analyze_lean_output(output)

        return lean_pass, lean_verify, output

    except FileNotFoundError:
        return (
            False,
            False,
            "Lean executable not found. Make sure it's in your system's PATH.",
        )
    except Exception as e:
        return (False, False, f"Verification failed with exception: {e}")

    finally:
        # 5. Clean up the temporary file
        if os.path.exists(temp_file_path):
            os.remove(temp_file_path)


async def verify_lean_lemma_local_async(
    lean_string: str,
    project_path: str,
    add_imports: bool = False,
    timeout: int = 180,
    temp_root: str | None = None,
    job_id: str | None = None,
):
    """Asynchronously verify a Lean snippet with a unique temp file per job."""
    temp_file_path = _build_temp_file_path(
        project_path=project_path,
        temp_root=temp_root,
        job_id=job_id,
    )
    full_code = f"{LEAN_LIBRARIES}\n\n{lean_string}" if add_imports else lean_string
    proc = None

    try:
        temp_file_path.write_text(full_code, encoding="utf-8")

        proc_kwargs = {
            "cwd": project_path,
            "stdout": asyncio.subprocess.PIPE,
            "stderr": asyncio.subprocess.PIPE,
        }
        if os.name == "nt":
            proc_kwargs["creationflags"] = subprocess.CREATE_NEW_PROCESS_GROUP
        else:
            proc_kwargs["start_new_session"] = True

        proc = await asyncio.create_subprocess_exec(
            "lake",
            "env",
            "lean",
            str(temp_file_path),
            "--json",
            **proc_kwargs,
        )
        stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=timeout)
        stdout_text = stdout.decode("utf-8", errors="replace")
        stderr_text = stderr.decode("utf-8", errors="replace")

        if proc.returncode != 0 and not stdout_text.strip():
            return (
                False,
                False,
                f"Process failed with exit code {proc.returncode}: {stderr_text.strip()}",
            )

        if not stdout_text.strip():
            return True, True, None

        lean_pass, lean_verify = _analyze_lean_output(stdout_text)
        return lean_pass, lean_verify, stdout_text
    except FileNotFoundError:
        return (
            False,
            False,
            "Lean executable not found. Make sure it's in your system's PATH.",
        )
    except asyncio.TimeoutError:
        if proc is not None and proc.returncode is None:
            try:
                if os.name == "nt":
                    proc.kill()
                else:
                    os.killpg(proc.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            await proc.wait()
        return False, False, f"Verification timed out after {timeout} seconds."
    except Exception as e:
        return False, False, f"Verification failed with exception: {e}"
    finally:
        if os.path.exists(temp_file_path):
            os.remove(temp_file_path)


class LocalLeanLspWorker:
    def __init__(
        self,
        project_path: str,
        temp_root: str | None,
        worker_id: int,
    ) -> None:
        self.project_path = Path(project_path).resolve()
        self.temp_root = (
            Path(temp_root).resolve()
            if temp_root
            else (self.project_path / "temp" / "lean_lsp_workers")
        )
        self.worker_id = worker_id
        self.file_path = self.temp_root / f"_persistent_worker_{worker_id}.lean"
        self.uri = self.file_path.as_uri()
        self.proc: Optional[asyncio.subprocess.Process] = None
        self.reader_task: Optional[asyncio.Task] = None
        self.stderr_task: Optional[asyncio.Task] = None
        self.pending_requests: Dict[int, asyncio.Future] = {}
        self.next_request_id = 0
        self.version = 0
        self.document_open = False
        self._job_future: Optional[asyncio.Future] = None
        self._job_version = 0
        self._job_diagnostics: List[Dict[str, Any]] = []
        self._job_seen_diagnostics = False
        self._job_seen_progress = False
        self._job_processing = False
        self._job_fallback_task: Optional[asyncio.Task] = None
        self._check_lock = asyncio.Lock()
        self._stderr_lines: List[str] = []

    async def start(self) -> None:
        if self.proc is not None and self.proc.returncode is None:
            return

        self.temp_root.mkdir(parents=True, exist_ok=True)
        warmup_text = self._build_document_text("namespace LeanWarmup\nend LeanWarmup")
        self.file_path.write_text(warmup_text, encoding="utf-8")

        proc_kwargs = {
            "cwd": str(self.project_path),
            "stdin": asyncio.subprocess.PIPE,
            "stdout": asyncio.subprocess.PIPE,
            "stderr": asyncio.subprocess.PIPE,
        }
        if os.name == "nt":
            proc_kwargs["creationflags"] = subprocess.CREATE_NEW_PROCESS_GROUP
        else:
            proc_kwargs["start_new_session"] = True

        self.proc = await asyncio.create_subprocess_exec(
            "lake",
            "env",
            "lean",
            "--server",
            **proc_kwargs,
        )
        self.reader_task = asyncio.create_task(self._reader_loop())
        self.stderr_task = asyncio.create_task(self._stderr_loop())

        root_uri = self.project_path.as_uri()
        await self._send_request(
            "initialize",
            {
                "processId": os.getpid(),
                "rootUri": root_uri,
                "workspaceFolders": [{"uri": root_uri, "name": self.project_path.name}],
                "capabilities": {},
                "clientInfo": {"name": "step-proof", "version": "local-persistent"},
            },
        )
        await self._send_notification("initialized", {})
        await self._publish_document(warmup_text, timeout=30)

    async def check(
        self,
        lean_string: str,
        *,
        add_imports: bool,
        timeout: int,
    ):
        async with self._check_lock:
            await self.start()
            text = self._build_document_text(lean_string if add_imports else remove_imports(lean_string))
            diagnostics = await self._publish_document(text, timeout=timeout)
            if not diagnostics:
                return True, True, None
            lean_pass, lean_verify = _analyze_lsp_diagnostics(diagnostics)
            return lean_pass, lean_verify, json.dumps(diagnostics, ensure_ascii=False)

    async def aclose(self) -> None:
        if self._job_fallback_task is not None:
            self._job_fallback_task.cancel()
            self._job_fallback_task = None

        if self.proc is not None and self.proc.returncode is None:
            try:
                if os.name == "nt":
                    self.proc.kill()
                else:
                    os.killpg(self.proc.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            await self.proc.wait()

        tasks = [task for task in (self.reader_task, self.stderr_task) if task is not None]
        for task in tasks:
            task.cancel()
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)

        self.proc = None
        self.reader_task = None
        self.stderr_task = None
        self.document_open = False
        self.pending_requests.clear()

    async def restart(self) -> None:
        await self.aclose()
        await self.start()

    def _build_document_text(self, lean_body: str) -> str:
        normalized = process_lean_string(lean_body)
        body = remove_imports(normalized).strip()
        if not body:
            body = "namespace LeanEmpty\nend LeanEmpty"
        return f"{LEAN_LIBRARIES}\n\n{body}\n"

    async def _publish_document(self, text: str, timeout: int) -> List[Dict[str, Any]]:
        if self.proc is None or self.proc.stdin is None:
            raise RuntimeError("Lean LSP worker is not running.")

        self.version += 1
        self.file_path.write_text(text, encoding="utf-8")
        loop = asyncio.get_running_loop()
        self._job_future = loop.create_future()
        self._job_version = self.version
        self._job_diagnostics = []
        self._job_seen_diagnostics = False
        self._job_seen_progress = False
        self._job_processing = True
        if self._job_fallback_task is not None:
            self._job_fallback_task.cancel()
            self._job_fallback_task = None

        if not self.document_open:
            await self._send_notification(
                "textDocument/didOpen",
                {
                    "textDocument": {
                        "uri": self.uri,
                        "languageId": "lean4",
                        "version": self.version,
                        "text": text,
                    }
                },
            )
            self.document_open = True
        else:
            await self._send_notification(
                "textDocument/didChange",
                {
                    "textDocument": {"uri": self.uri, "version": self.version},
                    "contentChanges": [{"text": text}],
                },
            )

        return await asyncio.wait_for(self._job_future, timeout=timeout)

    async def _send_request(self, method: str, params: Dict[str, Any]):
        loop = asyncio.get_running_loop()
        request_id = self.next_request_id
        self.next_request_id += 1
        future = loop.create_future()
        self.pending_requests[request_id] = future
        await self._write_message(
            {
                "jsonrpc": "2.0",
                "id": request_id,
                "method": method,
                "params": params,
            }
        )
        return await future

    async def _send_notification(self, method: str, params: Dict[str, Any]) -> None:
        await self._write_message(
            {
                "jsonrpc": "2.0",
                "method": method,
                "params": params,
            }
        )

    async def _write_message(self, payload: Dict[str, Any]) -> None:
        if self.proc is None or self.proc.stdin is None:
            raise RuntimeError("Lean LSP worker stdin is unavailable.")
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        header = f"Content-Length: {len(body)}\r\n\r\n".encode("ascii")
        self.proc.stdin.write(header + body)
        await self.proc.stdin.drain()

    async def _reader_loop(self) -> None:
        try:
            while True:
                message = await self._read_message()
                if message is None:
                    break
                self._handle_message(message)
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            self._fail_pending(exc)
        finally:
            self._fail_pending(RuntimeError("Lean LSP worker exited unexpectedly."))

    async def _stderr_loop(self) -> None:
        if self.proc is None or self.proc.stderr is None:
            return
        try:
            while True:
                line = await self.proc.stderr.readline()
                if not line:
                    return
                text = line.decode("utf-8", errors="replace").strip()
                if text:
                    self._stderr_lines.append(text)
                    if len(self._stderr_lines) > 20:
                        self._stderr_lines = self._stderr_lines[-20:]
        except asyncio.CancelledError:
            raise

    async def _read_message(self) -> Optional[Dict[str, Any]]:
        if self.proc is None or self.proc.stdout is None:
            return None

        content_length = None
        while True:
            line = await self.proc.stdout.readline()
            if not line:
                return None
            if line == b"\r\n":
                break
            decoded = line.decode("ascii", errors="replace").strip()
            if decoded.lower().startswith("content-length:"):
                content_length = int(decoded.split(":", 1)[1].strip())

        if content_length is None:
            raise RuntimeError("Lean LSP worker sent a message without Content-Length.")
        payload = await self.proc.stdout.readexactly(content_length)
        return json.loads(payload.decode("utf-8"))

    def _handle_message(self, message: Dict[str, Any]) -> None:
        if "id" in message and ("result" in message or "error" in message):
            future = self.pending_requests.pop(message["id"], None)
            if future is None or future.done():
                return
            if "error" in message:
                future.set_exception(RuntimeError(str(message["error"])))
            else:
                future.set_result(message.get("result"))
            return

        method = message.get("method")
        params = message.get("params", {})

        if method == "textDocument/publishDiagnostics":
            if params.get("uri") != self.uri or self._job_future is None:
                return
            version = params.get("version")
            if version is not None and version != self._job_version:
                return
            self._job_diagnostics = list(params.get("diagnostics") or [])
            self._job_seen_diagnostics = True
            self._schedule_job_completion_fallback()
            self._maybe_finish_job()
            return

        if method == "$/lean/fileProgress":
            text_document = params.get("textDocument") or {}
            if text_document.get("uri") != self.uri or self._job_future is None:
                return
            self._job_seen_progress = True
            self._job_processing = bool(params.get("processing"))
            self._maybe_finish_job()

    def _schedule_job_completion_fallback(self) -> None:
        if self._job_fallback_task is not None:
            self._job_fallback_task.cancel()
        self._job_fallback_task = asyncio.create_task(self._job_completion_fallback())

    async def _job_completion_fallback(self) -> None:
        try:
            await asyncio.sleep(0.1)
            self._maybe_finish_job(force=True)
        except asyncio.CancelledError:
            raise

    def _maybe_finish_job(self, force: bool = False) -> None:
        if self._job_future is None or self._job_future.done() or not self._job_seen_diagnostics:
            return
        if force or not self._job_seen_progress or not self._job_processing:
            self._job_future.set_result(self._job_diagnostics)

    def _fail_pending(self, exc: Exception) -> None:
        for future in list(self.pending_requests.values()):
            if not future.done():
                future.set_exception(exc)
        self.pending_requests.clear()
        if self._job_future is not None and not self._job_future.done():
            self._job_future.set_exception(exc)


class LocalLeanLspPool:
    def __init__(
        self,
        project_path: str,
        *,
        pool_size: int,
        temp_root: str | None = None,
    ) -> None:
        self.project_path = project_path
        self.pool_size = max(1, int(pool_size))
        self.temp_root = temp_root
        self.workers: List[LocalLeanLspWorker] = []
        self.available_workers: asyncio.Queue[LocalLeanLspWorker] = asyncio.Queue()
        self.started = False
        self.start_lock = asyncio.Lock()

    async def ensure_started(self) -> None:
        if self.started:
            return
        async with self.start_lock:
            if self.started:
                return
            for worker_id in range(self.pool_size):
                worker = LocalLeanLspWorker(
                    project_path=self.project_path,
                    temp_root=self.temp_root,
                    worker_id=worker_id,
                )
                await worker.start()
                self.workers.append(worker)
                await self.available_workers.put(worker)
            self.started = True

    async def check(
        self,
        lean_string: str,
        *,
        add_imports: bool = False,
        timeout: int = 180,
    ):
        await self.ensure_started()
        worker = await self.available_workers.get()
        worker_replaced = False
        try:
            return await worker.check(
                lean_string,
                add_imports=add_imports,
                timeout=timeout,
            )
        except asyncio.TimeoutError:
            worker_replaced = True
            replacement = await self._replace_worker(worker)
            await self.available_workers.put(replacement)
            return False, False, f"Verification timed out after {timeout} seconds."
        except Exception as exc:
            worker_replaced = True
            replacement = await self._replace_worker(worker)
            await self.available_workers.put(replacement)
            return False, False, f"Verification failed with exception: {exc}"
        finally:
            if not worker_replaced:
                await self.available_workers.put(worker)

    async def _replace_worker(self, worker: LocalLeanLspWorker) -> LocalLeanLspWorker:
        await worker.aclose()
        replacement = LocalLeanLspWorker(
            project_path=self.project_path,
            temp_root=self.temp_root,
            worker_id=worker.worker_id,
        )
        await replacement.start()
        self.workers[worker.worker_id] = replacement
        return replacement

    async def aclose(self) -> None:
        if not self.workers:
            return
        await asyncio.gather(*(worker.aclose() for worker in self.workers), return_exceptions=True)
        self.workers.clear()
        self.started = False
        while not self.available_workers.empty():
            try:
                self.available_workers.get_nowait()
            except asyncio.QueueEmpty:
                break


# --- The LeanServer class ---


class LeanServer:
    """
    A comprehensive Lean 4 verification server that supports both remote API and local execution.
    
    The LeanServer class provides a unified interface for verifying Lean 4 code snippets
    using either a remote Kimina server API or a local Lean installation. It automatically
    handles imports, error detection, and verification status reporting.
    
    Features:
        - Remote server verification via Kimina API
        - Local Lean 4 verification using lake/lean executables
        - Automatic import management (Mathlib, Aesop, etc.)
        - Comprehensive error detection and reporting
        - Support for both compilation and verification checks
        
    Modes:
        - Server Mode: Uses remote Kimina API for verification
        - Local Mode: Uses local Lean 4 installation for verification
        
    Example:
        >>> # Server mode
        >>> lean_server = LeanServer(api_url="http://localhost:14457")
        >>> lean_pass, lean_verify, errors = lean_server.check_lean_string("theorem test : 1 + 1 = 2 := by rfl")
        
        >>> # Local mode  
        >>> lean_server = LeanServer(project_path="/path/to/lean/project")
        >>> lean_pass, lean_verify, errors = lean_server.check_lean_string("theorem test : 1 + 1 = 2 := by rfl")
    """

    def __init__(
        self,
        api_url: str = None,
        project_path: str = None,
        *,
        backend: str = "subprocess",
        pool_size: int = 1,
        temp_root: str | None = None,
    ):
        """
        Initialize the LeanServer in either server or local mode.

        Args:
            api_url (str, optional): The URL for the remote Lean server API.
                When provided, the server will use Kimina API for verification.
                Example: "http://localhost:14457"
            project_path (str, optional): The local path to a Lean project directory.
                When provided, the server will use local Lean 4 installation.
                Example: "/path/to/lean/project"

        Raises:
            ValueError: If neither `api_url` nor `project_path` is provided.
            
        Note:
            - Server mode requires a running Kimina server at the specified URL
            - Local mode requires Lean 4 and lake to be installed and accessible
            - Local mode takes precedence if both parameters are provided
        """
        self.backend = backend
        self.pool_size = max(1, int(pool_size))
        self.temp_root = temp_root
        self.persistent_pool: Optional[LocalLeanLspPool] = None

        if project_path:
            # Prioritize local execution if a project path is provided
            self.mode = "local"
            self.path = project_path
            self.client = None  # Ensures client is not initialized in local mode
            if self.backend == "persistent_lsp":
                self.persistent_pool = LocalLeanLspPool(
                    project_path=self.path,
                    pool_size=self.pool_size,
                    temp_root=self.temp_root,
                )
            elif self.backend != "subprocess":
                raise ValueError(f"Unsupported local Lean backend: {self.backend}")
            print("LeanServer initialized in LOCAL mode.")
        elif api_url:
            self.mode = "server"
            self.path = api_url
            self.client = KiminaClient(
                api_url=self.path
            )  # The client is initialized once here
            print("LeanServer initialized in SERVER mode.")
        else:
            raise ValueError(
                "You must provide either an 'api_url' for server mode or a 'project_path' for local mode."
            )

    def check_lean_string(self, lean_string: str, add_imports: bool = False):
        """
        Verify a Lean 4 code string using the configured verification method.

        This method checks whether the provided Lean code compiles successfully
        and verifies without errors or 'sorry' statements. It automatically
        handles the verification process based on the server's mode (remote or local).

        Args:
            lean_string (str): The Lean 4 code to be verified.
                Can be a theorem, lemma, definition, or any valid Lean code.
            add_imports (bool, optional): Whether to automatically add standard
                Lean library imports (Mathlib, Aesop, etc.). Defaults to False.

        Returns:
            tuple: A tuple containing three elements:
                - lean_pass (bool): True if the code compiles without errors
                - lean_verify (bool): True if the code compiles AND verifies 
                  without 'sorry' statements or failures
                - output (list or str): Error details if compilation/verification fails,
                  None if successful

        Example:
            >>> lean_server = LeanServer(api_url="http://localhost:14457")
            >>> lean_pass, lean_verify, errors = lean_server.check_lean_string(
            ...     "theorem add_zero (n : ℕ) : n + 0 = n := by simp"
            ... )
            >>> print(f"Compiles: {lean_pass}, Verified: {lean_verify}")
            
        Note:
            - lean_pass=True means the code compiles without syntax/type errors
            - lean_verify=True means the code compiles AND has no 'sorry' statements
            - Both must be True for a fully verified proof
        """
        if self.mode == "server":
            return verify_lean_lemma_server(
                lean_string=lean_string, client=self.client, add_imports=add_imports
            )
        elif self.mode == "local":
            return verify_lean_lemma_local(
                lean_string=lean_string, project_path=self.path, add_imports=add_imports
            )
        else:
            # This case should not be reached due to the __init__ check
            return False, False, "Error: Invalid mode. This is an internal error."

    async def check_lean_string_async(
        self,
        lean_string: str,
        add_imports: bool = False,
        temp_root: str | None = None,
        job_id: str | None = None,
    ):
        """Async counterpart to check_lean_string for concurrency-friendly local checks."""
        if self.mode == "server":
            return await asyncio.to_thread(
                verify_lean_lemma_server,
                lean_string=lean_string,
                client=self.client,
                add_imports=add_imports,
            )
        elif self.mode == "local":
            if self.backend == "persistent_lsp":
                assert self.persistent_pool is not None
                return await self.persistent_pool.check(
                    lean_string=lean_string,
                    add_imports=add_imports,
                )
            return await verify_lean_lemma_local_async(
                lean_string=lean_string,
                project_path=self.path,
                add_imports=add_imports,
                temp_root=temp_root,
                job_id=job_id,
            )
        return False, False, "Error: Invalid mode. This is an internal error."

    async def aclose(self) -> None:
        if self.mode == "local" and self.persistent_pool is not None:
            await self.persistent_pool.aclose()

    def close(self) -> None:
        if self.mode != "local" or self.persistent_pool is None:
            return
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            asyncio.run(self.aclose())
