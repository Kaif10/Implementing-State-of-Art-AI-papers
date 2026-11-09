"""Screen-record original API-backed entry points on Windows, one at a time.

Requires windows-capture, imageio-ffmpeg and the project's normal dependencies.
The API key is read from the environment or a hidden prompt, never a file.
No local models are imported. Windows Graphics Capture records only the newly
created terminal window; frames are encoded in real time without reconstruction.
"""
import argparse
import ctypes
from ctypes import wintypes
from datetime import datetime, timezone
import getpass
import json
import os
from pathlib import Path
import subprocess
import sys
import threading
import time
import uuid

import cv2
import imageio_ffmpeg
from windows_capture import WindowsCapture

ROOT = Path(__file__).resolve().parents[1]
FPS = 5
PROJECTS = {
    "CAMEL_Paper": [["run.py", "--mode", "code", "--task", "Write a Python function add(a, b) and demonstrate add(2, 3).", "--max-messages", "6"]],
    "Chain_of_Agents": [["run.py", "--input_file", "doc_test.txt", "--task_requirements_file", "req_qa.txt", "--query", "What is Jonas Pike's security clearance tier?", "--k", "1350", "--debug"]],
    "FLARE-Active_RAG": [["main.py", "answer", "--data_dir", "data", "--question", "What is quantum computing and how does it differ from classical computing?", "--top_k", "2", "--max_steps", "2"]],
    "IRCOT-multihop": [["main.py", "answer", "--data_dir", "data", "--demos", "demos/demos_M1_seed0.json", "--question", "Which river flows through the city where Albert Einstein was born?", "--K", "2"]],
    "Multiagent_Debate": [["run_mad.py", "--task", "arithmetic", "--query", "(12 * 13) + (7 * 8)", "--agents", "2", "--rounds", "2", "--temperature", "0.7", "--print_agents"]],
    "Reflexion_Paper": [["-c", "import runpy; r=runpy.run_path('core.py',run_name='__main__'); print('Final evaluator decision:',r['ok']); print('Attempts:',len(r['attempts'])); raise SystemExit(0 if r['ok'] else 2)"]],
    "Self-RAG": [["main.py", "ingest", "--docs", "kb", "--out", "{index}"], ["main.py", "ask", "--index", "{index}", "--top-k", "3", "--per-step-passages", "2", "--max-steps", "3", "Which company services require MFA?"]],
}


def psquote(text):
    return "'" + str(text).replace("'", "''") + "'"


def free_memory_mb():
    class Memory(ctypes.Structure):
        _fields_ = [("length", wintypes.DWORD), ("load", wintypes.DWORD)] + [
            (name, ctypes.c_ulonglong) for name in ("total", "available", "page_total", "page_free", "virtual_total", "virtual_free", "extended")]
    memory = Memory()
    memory.length = ctypes.sizeof(memory)
    ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(memory))
    return memory.available / 1024**2


def visible_window(title):
    user = ctypes.windll.user32
    user.GetWindowTextLengthW.argtypes = [wintypes.HWND]
    user.GetWindowTextW.argtypes = [wintypes.HWND, wintypes.LPWSTR, ctypes.c_int]
    user.IsWindowVisible.argtypes = [wintypes.HWND]
    callback = ctypes.WINFUNCTYPE(wintypes.BOOL, wintypes.HWND, wintypes.LPARAM)
    found = []
    @callback
    def each(hwnd, parameter):
        length = user.GetWindowTextLengthW(hwnd)
        buf = ctypes.create_unicode_buffer(length + 1)
        user.GetWindowTextW(hwnd, buf, length + 1)
        if buf.value == title and user.IsWindowVisible(hwnd):
            found.append(hwnd)
        return True
    user.EnumWindows(each, 0)
    return found[0] if found else None


def record(project, credential):
    if free_memory_mb() < 400:
        raise RuntimeError("Less than 400 MB is free; recording not started")
    scratch = ROOT / ".venv" / "api-screen-scratch" / project
    scratch.mkdir(parents=True, exist_ok=True)
    out = ROOT / project / "media"
    out.mkdir(exist_ok=True)
    title = "AI Paper Demo - " + project + " - " + uuid.uuid4().hex[:6]
    ready, gate, done, release = [scratch / name for name in ("ready", "go", "done.json", "release")]
    for file in (ready, gate, done, release):
        if file.exists():
            file.unlink()
    commands = [[arg.replace("{index}", str(scratch / "index")) for arg in command] for command in PROJECTS[project]]
    script = ["$ErrorActionPreference = 'Continue'", "$Host.UI.RawUI.WindowTitle = " + psquote(title),
              "$Host.UI.RawUI.BackgroundColor = 'Black'", "$Host.UI.RawUI.ForegroundColor = 'Gray'", "Clear-Host",
              "Set-Location -LiteralPath " + psquote(ROOT/project),
              "Write-Host " + psquote(project + " | original implementation | OpenAI API"),
              "Set-Content -LiteralPath " + psquote(ready) + " -Value ready",
              "$deadline=(Get-Date).AddMinutes(5)",
              "while (!(Test-Path -LiteralPath " + psquote(gate) + ") -and (Get-Date) -lt $deadline) { Start-Sleep -Milliseconds 100 }",
              "$finalCode=0"]
    for command in commands:
        display = "python " + " ".join(psquote(arg) if " " in arg else arg for arg in command)
        script += ["if ($finalCode -eq 0) {", "Write-Host " + psquote("> " + display),
                   "& " + psquote(sys.executable) + " '-u' " + " ".join(psquote(arg) for arg in command) + " 2>&1 | Tee-Object -FilePath " + psquote(out/"terminal-transcript.txt") + " -Append",
                   "$finalCode=$LASTEXITCODE", "if ($null -eq $finalCode) { $finalCode=1 }", "}"]
    script += ["Write-Host ('Process exit code: ' + $finalCode)",
               "@{exit_code=$finalCode} | ConvertTo-Json | Set-Content -Encoding UTF8 -LiteralPath " + psquote(done),
               "$deadline=(Get-Date).AddSeconds(15)",
               "while (!(Test-Path -LiteralPath " + psquote(release) + ") -and (Get-Date) -lt $deadline) { Start-Sleep -Milliseconds 100 }",
               "exit $finalCode"]
    psfile = scratch / "run-original.ps1"
    psfile.write_text("\n".join(script), encoding="utf-8")
    transcript = out / "terminal-transcript.txt"
    if transcript.exists():
        transcript.unlink()
    env = os.environ.copy()
    env.update(OPENAI_API_KEY=credential, PYTHONIOENCODING="utf-8", PYTHONUNBUFFERED="1",
               OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1", NUMBA_NUM_THREADS="1")
    process = subprocess.Popen(["powershell.exe", "-NoProfile", "-File", str(psfile)],
                               env=env, creationflags=subprocess.CREATE_NEW_CONSOLE)
    last = [None]
    lock = threading.Lock()
    first = threading.Event()
    stop = threading.Event()
    failures = []
    writer = None
    capture_control = None
    worker = None
    start = time.monotonic()
    count = [0]
    try:
        for _ in range(100):
            hwnd = visible_window(title)
            if ready.exists() and hwnd:
                break
            time.sleep(.1)
        else:
            raise RuntimeError("The dedicated terminal window did not open")
        ctypes.windll.user32.SetProcessDPIAware()
        ctypes.windll.user32.SetWindowPos(wintypes.HWND(hwnd), 0, 80, 80, 1100, 650, 0x14)
        capture = WindowsCapture(cursor_capture=False, draw_border=True,
                                 secondary_window=False, minimum_update_interval=200, window_hwnd=hwnd)
        @capture.event
        def on_frame_arrived(frame, control):
            with lock:
                height, width = frame.frame_buffer.shape[:2]
                last[0] = frame.frame_buffer[:height//2*2, :width//2*2].copy()
            first.set()
        @capture.event
        def on_closed():
            stop.set()
        capture_control = capture.start_free_threaded()
        if not first.wait(8):
            raise RuntimeError("No native window frames were received")
        with lock:
            height, width = last[0].shape[:2]
        writer = imageio_ffmpeg.write_frames(str(out/"terminal-demo.mp4"), (width, height), fps=FPS,
            codec="libx264", pix_fmt_in="bgra", pix_fmt_out="yuv420p",
            output_params=["-preset", "ultrafast", "-crf", "24", "-threads", "1", "-movflags", "+faststart", "-fs", "20000000"],
            ffmpeg_log_level="error", macro_block_size=1)
        writer.send(None)
        def encode():
            next_frame = time.monotonic()
            try:
                while not stop.is_set():
                    with lock:
                        pixels = last[0]
                    if pixels.shape[:2] != (height, width):
                        raise RuntimeError("Terminal size changed during capture")
                    writer.send(pixels.tobytes())
                    count[0] += 1
                    next_frame += 1/FPS
                    stop.wait(max(0, next_frame-time.monotonic()))
            except Exception as error:
                failures.append(type(error).__name__)
                stop.set()
        worker = threading.Thread(target=encode, daemon=True)
        worker.start()
        time.sleep(.5)
        gate.write_text("go")
        while not done.exists():
            if free_memory_mb() < 180:
                raise RuntimeError("Recording stopped to preserve available RAM")
            if stop.is_set() or process.poll() is not None or time.monotonic()-start > 180:
                raise RuntimeError("The original command did not complete within the recording limit")
            time.sleep(.2)
        time.sleep(3)
        result = json.loads(done.read_text(encoding="utf-8-sig"))
        if failures:
            raise RuntimeError("Screen encoding failed")
    finally:
        stop.set()
        if worker:
            worker.join(timeout=5)
        if capture_control:
            capture_control.stop()
        if writer:
            writer.close()
        release.write_text("release")
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            subprocess.run(["taskkill", "/PID", str(process.pid), "/T", "/F"], capture_output=True)
        remaining = visible_window(title)
        if remaining:
            ctypes.windll.user32.PostMessageW(wintypes.HWND(remaining), 0x0010, 0, 0)
        env.pop("OPENAI_API_KEY", None)
    # Normalize PowerShell's UTF-16 transcript without changing the recorded pixels.
    raw = transcript.read_bytes()
    output = raw.decode("utf-16" if raw.startswith((b"\xff\xfe", b"\xfe\xff")) else "utf-8-sig", errors="replace")
    if credential in output:
        raise RuntimeError("Credential appeared in output; recording must not be published")
    transcript.write_text("\n".join(line.rstrip() for line in output.splitlines())+"\n", encoding="utf-8")
    metadata = dict(project=project, source_commit=subprocess.check_output(["git", "rev-parse", "HEAD"],cwd=ROOT,text=True).strip(),
                    commands=commands, backend="OpenAI / gpt-4o-mini", capture="Windows Graphics Capture / terminal window only",
                    playback="real time, no reconstructed frames", fps=FPS, frames=count[0], exit_code=result["exit_code"],
                    recorded_at=datetime.now(timezone.utc).isoformat())
    if project == "Self-RAG":
        metadata["embedding_model"] = "text-embedding-3-large"
        metadata["embedding_dimensions"] = 1024
    (out/"terminal-recording.json").write_text(json.dumps(metadata,indent=2)+"\n",encoding="utf-8")
    if result["exit_code"]:
        raise RuntimeError("The original command failed; recording will not be embedded")
    print(f"CAPTURED {project}: {count[0]/FPS:.1f}s native video", flush=True)
    return out


def finalize(project):
    """Convert captured pixels to an inline GIF; preserve real playback timing."""
    out = ROOT/project/"media"
    metadata = json.loads((out/"terminal-recording.json").read_text(encoding="utf-8"))
    if metadata["exit_code"] != 0 or metadata["frames"] < FPS*2:
        raise RuntimeError("Only successful native recordings can be embedded")
    ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
    palette = ROOT/".venv"/"api-screen-scratch"/project/"palette.png"
    common = [ffmpeg,"-hide_banner","-loglevel","error","-threads","1","-filter_threads","1"]
    subprocess.run(common+["-i",str(out/"terminal-demo.mp4"),"-vf",
        "fps=4,scale=960:-2:flags=lanczos,palettegen=max_colors=64", "-frames:v","1","-y",str(palette)],check=True)
    subprocess.run(common+["-i",str(out/"terminal-demo.mp4"),"-i",str(palette),"-filter_complex_threads","1",
        "-lavfi","[0:v]fps=4,scale=960:-2:flags=lanczos[v];[v][1:v]paletteuse=dither=bayer:bayer_scale=4",
        "-loop","0","-y",str(out/"terminal-demo.gif")],check=True)
    subprocess.run(common+["-sseof","-2","-i",str(out/"terminal-demo.mp4"),"-frames:v","1",
        "-y",str(out/"terminal-preview.png")],check=True)
    path = ROOT/project/"README.md"
    source = path.read_text(encoding="utf-8")
    if "<!-- native-api-terminal-demo -->" not in source:
        first,remainder = source.split("\n",1)
        section=("\n<!-- native-api-terminal-demo -->\n## Terminal demo\n\n"
            f"![Actual terminal screen recording of {project}](media/terminal-demo.gif)\n\n"
            "Screen recording of the existing implementation using OpenAI `gpt-4o-mini`, at normal playback speed. "
            "The animation plays inline. Commands, source revision, and terminal output are saved in `media/`.\n\n")
        path.write_text(first+"\n"+section+remainder,encoding="utf-8")
    print(f"EMBEDDED {project}: real-time native screen capture",flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--projects",nargs="+",choices=list(PROJECTS),default=list(PROJECTS))
    parser.add_argument("--finalize-only",action="store_true",help="Convert and embed existing successful captures; no API calls")
    args=parser.parse_args()
    cv2.setNumThreads(1)
    if args.finalize_only:
        for project in args.projects:
            finalize(project)
        return
    credential=os.environ.get("OPENAI_API_KEY") or getpass.getpass("OpenAI key (hidden): ")
    if not credential:
        raise SystemExit("An API key is required")
    failures=[]
    for project in args.projects:
        print("RECORDING "+project,flush=True)
        try:
            record(project,credential)
        except Exception as error:
            print("FAILED "+project+" / "+type(error).__name__,flush=True)
            failures.append(project)
    credential=None
    if failures:
        raise SystemExit("Recordings requiring inspection: "+", ".join(failures))


if __name__=="__main__":
    main()
