# Repository walkthrough

The 48-second MP4 uses captions, source excerpts and recorded offline test output.
It has no audio. `evidence.json` identifies the tested source commit;
`test-output.txt` contains the complete recorded run.

Rebuild from the repository root with Python 3.12 and the test dependencies:

```bash
python -m pip install -r requirements-test-models.txt -r docs/media/requirements.txt
python scripts/run_tests.py --include-models > test-output.txt
python scripts/create_demo.py --transcript test-output.txt
```

The renderer uses Segoe UI / Consolas on Windows or DejaVu Sans / Mono on Linux.
It writes the MP4, preview and six-frame contact sheet here. H.264/yuv420p encoding
and fast-start metadata support browser playback.
