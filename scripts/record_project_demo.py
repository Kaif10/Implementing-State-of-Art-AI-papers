"""Record actual command output as a captioned terminal video for a project README.

Install Pillow and imageio-ffmpeg in the recording environment first.
Usage: python scripts/record_project_demo.py --project CAMEL_Paper --title CAMEL -- python run.py ...
The command runs inside the selected project. Output, timing and exit status are
saved alongside the video. Long idle periods are shortened in the video.
"""
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import re
import shlex
import subprocess
import sys
import textwrap
import threading
import time

import imageio_ffmpeg
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
WIDTH, HEIGHT, FPS = 1280, 800, 20


def font(size, mono=False, bold=False):
    windows = "consola" if mono else "segoeuib" if bold else "segoeui"
    linux = "DejaVuSansMono" if mono else "DejaVuSans-Bold" if bold else "DejaVuSans"
    for candidate in (Path("C:/Windows/Fonts") / f"{windows}.ttf",
                      Path("/usr/share/fonts/truetype/dejavu") / f"{linux}.ttf"):
        if candidate.is_file():
            return ImageFont.truetype(str(candidate), size)
    raise FileNotFoundError("Segoe UI/Consolas or DejaVu fonts are required")


TITLE, BODY, MONO, SMALL = font(34, bold=True), font(22), font(21, mono=True), font(17)
ANSI = re.compile(r"\x1b\[[0-?]*[ -/]*[@-~]")


def frame(title, project, command, lines, elapsed, duration, finished, exit_code, mode):
    im = Image.new("RGB", (WIDTH, HEIGHT), "#0b1220")
    d = ImageDraw.Draw(im)
    d.text((48, 24), "KAIF10 / PAPER IMPLEMENTATIONS", font=SMALL, fill="#82e6b3")
    d.text((48, 60), title, font=TITLE, fill="#eff5fc")
    d.text((48, 111), mode, font=BODY, fill="#ffbd75")
    d.rounded_rectangle((42, 159, 1238, 735), radius=16, fill="#121e31", outline="#2a3b53", width=2)
    for i, color in enumerate(("#fd7e81", "#ffcc7a", "#82e6b3")):
        d.ellipse((65+i*23, 179, 76+i*23, 190), fill=color)
    d.text((160, 174), project + " / actual recorded output", font=SMALL, fill="#a9bad0")
    content = ["$ " + command] + lines
    wrapped = []
    for line in content:
        wrapped.extend(textwrap.wrap(line.expandtabs(4), width=90, replace_whitespace=False, drop_whitespace=False) or [""])
    for i, line in enumerate(wrapped[-17:]):
        color = "#82e6b3" if line.startswith(("$ ", "DEMO COMPLETE", "PASS", "FINAL")) else "#e4edf8"
        d.text((66, 215+i*28), line, font=MONO, fill=color)
    status = f"Completed / exit {exit_code}" if finished else "Recorded command running"
    d.text((48, 757), status, font=SMALL, fill="#82e6b3" if finished and exit_code == 0 else "#a9bad0")
    d.text((625, 757), "Silent recording / idle time shortened / full transcript included", font=SMALL, fill="#a9bad0")
    d.rectangle((0, HEIGHT-4, int(WIDTH*elapsed/duration), HEIGHT), fill="#82e6b3")
    return im


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project", required=True)
    parser.add_argument("--title", required=True)
    parser.add_argument("--mode", default="Live model demonstration")
    parser.add_argument("--timeout", type=int, default=900)
    parser.add_argument("--replay", action="store_true", help="Render the saved recording again without running the model")
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    command = args.command[1:] if args.command[:1] == ["--"] else args.command
    project = (ROOT / args.project).resolve()
    if project.parent != ROOT or not project.is_dir():
        parser.error("project must be a direct project directory in this repository")
    (project / "media").mkdir(exist_ok=True)
    if args.replay:
        metadata = json.loads((project / "media" / "demo-recording.json").read_text(encoding="utf-8"))
        if metadata["exit_code"] != 0 or metadata["project"] != args.project:
            raise SystemExit("A successful matching recording is required")
        render_recording(metadata, project / "media")
        return
    if not command:
        parser.error("a command is required after --")
    env = os.environ.copy()
    if os.name == "nt" and not env.get("OPENAI_API_KEY"):
        # A newly configured user environment variable may not reach this parent
        # process. Read only this explicitly requested demo credential locally.
        import winreg
        try:
            with winreg.OpenKey(winreg.HKEY_CURRENT_USER, "Environment") as key:
                value, _ = winreg.QueryValueEx(key, "OPENAI_API_KEY")
                if value:
                    env["OPENAI_API_KEY"] = value
        except FileNotFoundError:
            pass
    env.update(PYTHONUNBUFFERED="1", PYTHONIOENCODING="utf-8", TOKENIZERS_PARALLELISM="false",
               TRANSFORMERS_VERBOSITY="error")
    # Never include configured credential values in committed transcripts.
    secrets = [env[k] for k in ("OPENAI_API_KEY", "ANTHROPIC_API_KEY", "HF_TOKEN") if env.get(k)]
    start = time.monotonic()
    process = subprocess.Popen(command, cwd=project, env=env, stdout=subprocess.PIPE,
                               stderr=subprocess.STDOUT, text=True, encoding="utf-8", errors="replace")
    timer = threading.Timer(args.timeout, process.kill)
    timer.daemon = True
    timer.start()
    events = []
    try:
        for raw in process.stdout:
            line = ANSI.sub("", raw.rstrip("\r\n"))
            for secret in secrets:
                line = line.replace(secret, "[REDACTED]")
            events.append({"seconds": round(time.monotonic()-start, 3), "line": line})
            print(line, flush=True)
            if time.monotonic()-start > args.timeout:
                process.kill()
                raise TimeoutError("Demo exceeded its recording timeout")
        exit_code = process.wait(timeout=10)
    finally:
        timer.cancel()
        if process.poll() is None:
            process.kill()
            process.wait()
    elapsed = round(time.monotonic()-start, 3)
    out = project / "media"
    out.mkdir(exist_ok=True)
    display_command = shlex.join([Path(command[0]).name] + command[1:])
    metadata = {"title": args.title, "mode": args.mode, "project": args.project,
                "command": display_command, "source_commit": subprocess.check_output(
                    ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
                "recorded_at": datetime.now(timezone.utc).isoformat(), "elapsed_seconds": elapsed,
                "exit_code": exit_code, "events": events}
    (out / "demo-recording.json").write_text(json.dumps(metadata, indent=2, ensure_ascii=False)+"\n", encoding="utf-8")
    (out / "demo-transcript.txt").write_text("\n".join(e["line"].rstrip() for e in events)+"\n", encoding="utf-8")
    if exit_code:
        raise SystemExit(f"Command failed ({exit_code}); inspect the transcript before publishing")
    render_recording(metadata, out)


def render_recording(metadata, out):
    events, exit_code = metadata["events"], metadata["exit_code"]
    # Preserve every output line, compress waiting, and retain enough reading time.
    timeline, t, previous = [], 2.0, 0.0
    for event in events:
        t += max(.15, min(1.5, event["seconds"]-previous))
        timeline.append((t, event["line"]))
        previous = event["seconds"]
    duration = max(18.0, t+6)
    writer = imageio_ffmpeg.write_frames(str(out / "demo.mp4"), (WIDTH, HEIGHT), fps=FPS,
        codec="libx264", pix_fmt_in="rgb24", pix_fmt_out="yuv420p",
        output_params=["-crf", "22", "-preset", "fast", "-movflags", "+faststart"], ffmpeg_log_level="error")
    writer.send(None)
    visible, index, gif_frames = [], 0, []
    try:
        for number in range(int(duration*FPS)):
            now = number/FPS
            while index < len(timeline) and timeline[index][0] <= now:
                visible.append(timeline[index][1])
                index += 1
            image = frame(metadata["title"], metadata["project"], metadata["command"], visible, now, duration,
                          index == len(timeline), exit_code, metadata["mode"])
            if number == int((duration-3)*FPS):
                image.save(out / "demo-preview.jpg", quality=92)
            if number % 5 == 0:
                gif_frames.append(image.resize((960, 600)).quantize(colors=96))
            writer.send(image.tobytes())
    finally:
        writer.close()
    gif_frames[0].save(out / "demo.gif", save_all=True, append_images=gif_frames[1:],
                       duration=250, loop=0, optimize=True)
    print(f"Rendered {metadata['project']}: {duration:.1f}s video from {metadata['elapsed_seconds']:.1f}s actual run", flush=True)


if __name__ == "__main__":
    main()
