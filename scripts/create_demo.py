"""Render a captioned README video from source excerpts and recorded test output.

Usage (Python 3.12):
    pip install -r docs/media/requirements.txt
    python scripts/create_demo.py --transcript path/to/test-output.txt
Fonts: Segoe UI / Consolas on Windows, or DejaVu Sans / Mono on Linux.
"""
import argparse
import json
import re
import subprocess
from pathlib import Path

import imageio_ffmpeg
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "docs" / "media"
W, H, FPS = 1280, 720, 24
BG, PANEL, LINE = "#0b1220", "#121e31", "#26374d"
WHITE, MUTED, GREEN, ORANGE = "#eff5fc", "#a9bad0", "#82e6b3", "#ffbd75"
DURATIONS = [7, 8, 10, 9, 8, 6]
LABELS = ["Overview", "12 papers", "Decoding", "Evaluation", "Validation", "Explore"]


def font(size, mono=False, bold=False):
    name = "consolab" if mono and bold else "consola" if mono else "segoeuib" if bold else "segoeui"
    linux = "DejaVuSansMono" if mono else "DejaVuSans"
    if bold:
        linux += "-Bold"
    for path in (Path("C:/Windows/Fonts") / f"{name}.ttf",
                 Path("/usr/share/fonts/truetype/dejavu") / f"{linux}.ttf"):
        if path.exists():
            return ImageFont.truetype(str(path), size)
    raise FileNotFoundError("Install Segoe UI/Consolas or DejaVu fonts")


FONTS = {"title": font(51, bold=True), "heading": font(34, bold=True),
         "body": font(26), "small": font(19), "mono": font(21, mono=True),
         "label": font(20, bold=True), "metric": font(82, bold=True)}


def text(draw, xy, value, style="body", fill=WHITE):
    draw.text(xy, value, font=FONTS[style], fill=fill)


def card(draw, box):
    draw.rounded_rectangle(box, radius=16, fill=PANEL, outline=LINE, width=2)


def code(draw, title, lines, reveal=1.0):
    card(draw, (62, 235, 1218, 571))
    for i, color in enumerate(("#fd7e81", "#ffcc7a", "#82e6b3")):
        draw.ellipse((85 + i * 22, 254, 96 + i * 22, 265), fill=color)
    text(draw, (175, 247), title, "small", MUTED)
    for i, line in enumerate(lines[:int(len(lines) * reveal + 0.99)]):
        text(draw, (88, 294 + 31 * i), line, "mono", GREEN if line.startswith(("108", "All", ">")) else WHITE)


def excerpt(path, start, stop, limit=8):
    source = (ROOT / path).read_text(encoding="utf-8")
    snippet = source[source.index(start):]
    snippet = snippet[:snippet.index(stop)]
    return snippet.strip().splitlines()[:limit]


def render(t, evidence, snippets):
    total = sum(DURATIONS)
    start, chapter = 0, 0
    for i, duration in enumerate(DURATIONS):
        chapter = i
        if t < start + duration:
            break
        start += duration
    local = t - start
    im = Image.new("RGB", (W, H), BG)
    d = ImageDraw.Draw(im)
    text(d, (64, 28), "KAIF10  /  AI PAPER IMPLEMENTATIONS", "label", GREEN)
    text(d, (1100, 30), f"{int(t):02d} / {total:02d}s", "small", MUTED)
    d.line((64, 69, 1216, 69), fill=LINE, width=2)
    reveal = min(1, max(.1, local / 2))
    if chapter == 0:
        text(d, (64, 115), "Research ideas. Explicit code.", "title")
        text(d, (64, 185), "A walkthrough of LLM inference, reasoning, retrieval and agents.", fill=MUTED)
        for x, metric, label in [(64, "12", "paper implementations"), (465, str(evidence["passed"]), "offline tests passing"), (866, "CI", "automated validation")]:
            card(d, (x, 280, x + 350, 505))
            text(d, (x + 24, 294), metric, "metric", GREEN)
            text(d, (x + 24, 435), label, "body")
        text(d, (64, 545), "Python / PyTorch  |  Core mechanisms + runnable examples", fill=ORANGE)
    elif chapter == 1:
        text(d, (64, 105), "One collection. Four research areas.", "title")
        groups = [("INFERENCE", ["Speculative Decoding", "s1 Budget Forcing", "Self-Consistency"]),
                  ("REASONING", ["Program of Thoughts", "Reflexion", "SkillOpt"]),
                  ("RETRIEVAL", ["FLARE", "IRCoT", "Self-RAG"]),
                  ("AGENTS", ["CAMEL", "Chain of Agents", "Multiagent Debate"])]
        for i, (name, papers) in enumerate(groups):
            x, y = 64 + (i % 2) * 584, 220 + (i // 2) * 179
            card(d, (x, y, x + 568, y + 161))
            text(d, (x + 22, y + 17), name, "label", GREEN)
            for j, paper in enumerate(papers):
                text(d, (x + 22, y + 49 + 31 * j), paper, "body")
    elif chapter == 2:
        text(d, (64, 105), "Inside speculative decoding", "title")
        text(d, (64, 177), "Draft > verify > accept or resample > crop the cache", fill=ORANGE)
        code(d, "Speculative_Decoding/speculative_decoding.py  |  source excerpt", snippets["decoding"], reveal)
        text(d, (64, 596), "The draft/verify loop is written explicitly, including cache rollback.", "body", MUTED)
    elif chapter == 3:
        text(d, (64, 105), "Follow an answer end to end", "title")
        text(d, (64, 177), "Program of Thoughts: generate > extract > execute > grade", fill=ORANGE)
        code(d, "Program_of_Thoughts/program_of_thought.py  |  source excerpt", snippets["evaluation"], reveal)
        text(d, (64, 596), "Per-example outputs record the code, prediction, status and correctness.", "body", MUTED)
    elif chapter == 4:
        text(d, (64, 105), "Checks you can reproduce", "title")
        text(d, (64, 177), "Recorded offline test output / no pretrained weights or API calls", fill=ORANGE)
        code(d, "python scripts/run_tests.py --include-models", [
            "Testing Program_of_Thoughts     19 passed",
            "Testing Multiagent_Debate       10 passed",
            "Testing Speculative_Decoding    27 passed",
            "Testing S1_Test_Time_Scaling     7 passed",
            "... plus 8 other project suites",
            f"{evidence['passed']} passed / {evidence['skipped']} optional model integration skipped",
            "All selected project suites passed.",
            "> GitHub Actions runs the same offline suite.",
        ], reveal)
        text(d, (64, 596), "Synthetic sampling, a tiny Transformer, grading, retrieval and agent state.", "body", MUTED)
    else:
        text(d, (64, 115), "Explore the implementation.", "title")
        text(d, (64, 188), "Read the algorithm. Run the example. Inspect the checks.", fill=MUTED)
        card(d, (64, 280, 1216, 489))
        text(d, (92, 313), "github.com/Kaif10", "heading", GREEN)
        text(d, (92, 372), "Implementing-State-of-Art-AI-papers", "heading")
        text(d, (64, 535), "Start with Speculative Decoding, Program of Thoughts or SkillOpt.", fill=ORANGE)
    d.line((64, 649, 1216, 649), fill=LINE, width=2)
    for i, label in enumerate(LABELS):
        text(d, (64 + i * 192, 665), label, "small", GREEN if i == chapter else MUTED)
    d.rectangle((0, H - 5, int(W * t / total), H), fill=GREEN)
    # Brief fades retain readable still frames throughout each chapter.
    fade = min(1.0, local / .25, (DURATIONS[chapter] - local) / .25)
    if fade < 1:
        im = Image.blend(Image.new("RGB", (W, H), BG), im, max(0, fade))
    return im


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--transcript", required=True, type=Path)
    args = parser.parse_args()
    raw = args.transcript.read_bytes()
    transcript = raw.decode("utf-16" if raw.startswith((b"\xff\xfe", b"\xfe\xff")) else "utf-8-sig")
    if "All selected project suites passed." not in transcript or re.search(r"\b(?:failed|ERRORS)\b", transcript):
        raise SystemExit("A successful full offline test transcript is required")
    evidence = {"source_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
                "passed": sum(map(int, re.findall(r"(\d+) passed", transcript))),
                "skipped": sum(map(int, re.findall(r"(\d+) skipped", transcript)))}
    if evidence["passed"] != 108 or "27 passed" not in transcript:
        raise SystemExit("Expected the complete 108-test suite; update captions if the suite changes")
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "test-output.txt").write_text(transcript, encoding="utf-8")
    (OUT / "evidence.json").write_text(json.dumps(evidence, indent=2) + "\n", encoding="utf-8")
    snippets = {
        "decoding": excerpt("Speculative_Decoding/speculative_decoding.py", "        if n_accepted < K:", "        # --- 4."),
        "evaluation": excerpt("Program_of_Thoughts/program_of_thought.py", "    raw = model.generate(question)", "\n\ndef load_gsm8k", limit=8),
    }
    render(3, evidence, snippets).save(OUT / "walkthrough-preview.jpg", quality=92)
    shots = Image.new("RGB", (960, 810), BG)
    start = 0
    for i, duration in enumerate(DURATIONS):
        shot = render(start + duration / 2, evidence, snippets).resize((480, 270))
        shots.paste(shot, ((i % 2) * 480, (i // 2) * 270))
        start += duration
    shots.save(OUT / "contact-sheet.jpg", quality=90)
    writer = imageio_ffmpeg.write_frames(str(OUT / "repository-walkthrough.mp4"), (W, H),
        fps=FPS, codec="libx264", pix_fmt_in="rgb24", pix_fmt_out="yuv420p",
        output_params=["-crf", "23", "-preset", "fast", "-movflags", "+faststart"],
        ffmpeg_log_level="error")
    writer.send(None)
    try:
        for frame in range(sum(DURATIONS) * FPS):
            if frame % (FPS * 8) == 0:
                print(f"Rendering {frame // FPS}/{sum(DURATIONS)} seconds", flush=True)
            writer.send(render(frame / FPS, evidence, snippets).tobytes())
    finally:
        writer.close()
    print(f"Video saved to {OUT / 'repository-walkthrough.mp4'}", flush=True)


if __name__ == "__main__":
    main()
