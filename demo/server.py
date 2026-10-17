"""
server.py — stdlib ThreadingHTTPServer: JSON API + SSE + static files for the CT ->
Photoreal demo. No web framework (none is in the venv) — see demo/PLAN.md §3.8.

Binds 127.0.0.1 only. One run at a time. Every stage failure inside a Run becomes an
`error` SSE event with a hint (see pipeline.py) rather than a 500 — the process itself
should not crash because Blender or the model had a bad day.
"""
from __future__ import annotations

import argparse
import json
import mimetypes
import os
import subprocess
import sys
import threading
import time
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse, parse_qs, unquote

DEMO_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(DEMO_DIR))

import pipeline  # noqa: E402
cfg = pipeline.cfg

STATIC_DIR = DEMO_DIR / "static"

_MIME_EXTRA = {
    ".jsonl": "application/json", ".obj": "text/plain; charset=utf-8",
    ".bin": "application/octet-stream", ".exr": "application/octet-stream",
    ".npy": "application/octet-stream",
}

_current_run: pipeline.Run | None = None
_run_lock = threading.Lock()
_recon_backfill_lock = threading.Lock()  # serialises POST /api/runs/<id>/recon (PLAN_3D.md §4.4)


def _guess_type(path: Path) -> str:
    ext = path.suffix.lower()
    if ext in _MIME_EXTRA:
        return _MIME_EXTRA[ext]
    t, _ = mimetypes.guess_type(str(path))
    return t or "application/octet-stream"


def _safe_join(base: Path, rel: str) -> Path | None:
    """Resolve `rel` under `base`, refusing traversal outside it."""
    rel = unquote(rel).lstrip("/")
    candidate = (base / rel).resolve()
    try:
        candidate.relative_to(base.resolve())
    except ValueError:
        return None
    return candidate


class Handler(BaseHTTPRequestHandler):
    server_version = "CTPhotorealDemo/0.1"
    protocol_version = "HTTP/1.1"

    def log_message(self, fmt, *args):
        sys.stderr.write(f"[{self.log_date_time_string()}] {fmt % args}\n")

    # -- helpers -------------------------------------------------------------
    def _send_json(self, obj, status=200):
        body = json.dumps(obj).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(body)

    def _send_file(self, path: Path, download_name: str | None = None):
        if not path.is_file():
            self._send_json({"error": "not found", "path": str(path)}, 404)
            return
        data = path.read_bytes()
        self.send_response(200)
        self.send_header("Content-Type", _guess_type(path))
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Cache-Control", "no-cache")
        self.end_headers()
        self.wfile.write(data)

    def _read_json_body(self) -> dict:
        length = int(self.headers.get("Content-Length", 0) or 0)
        if length == 0:
            return {}
        raw = self.rfile.read(length)
        try:
            return json.loads(raw.decode("utf-8"))
        except Exception:
            return {}

    # -- routing ---------------------------------------------------------
    def do_GET(self):
        parsed = urlparse(self.path)
        path = parsed.path
        qs = parse_qs(parsed.query)
        try:
            if path == "/" or path == "":
                self._send_file(STATIC_DIR / "index.html")
            elif path.startswith("/static/"):
                p = _safe_join(STATIC_DIR, path[len("/static/"):])
                if p is None:
                    self._send_json({"error": "forbidden"}, 403)
                else:
                    self._send_file(p)
            elif path == "/api/status":
                self._api_status()
            elif path == "/api/browse":
                self._api_browse(qs)
            elif path == "/api/runs":
                self._send_json({"runs": pipeline.list_runs()})
            elif path.startswith("/api/runs/") and path.endswith("/events"):
                run_id = path[len("/api/runs/"):-len("/events")]
                self._api_run_events(run_id, qs)
            elif path.startswith("/runs/"):
                self._serve_run_file(path[len("/runs/"):])
            else:
                self._send_json({"error": "not found"}, 404)
        except BrokenPipeError:
            pass
        except Exception as e:
            try:
                self._send_json({"error": str(e)}, 500)
            except Exception:
                pass

    def do_POST(self):
        parsed = urlparse(self.path)
        path = parsed.path
        try:
            if path == "/api/pick":
                self._api_pick()
            elif path == "/api/inspect":
                self._api_inspect()
            elif path == "/api/run":
                self._api_run()
            elif path.startswith("/api/runs/") and path.endswith("/recon"):
                run_id = path[len("/api/runs/"):-len("/recon")]
                self._api_recon_backfill(run_id)
            else:
                self._send_json({"error": "not found"}, 404)
        except BrokenPipeError:
            pass
        except Exception as e:
            try:
                self._send_json({"error": str(e)}, 500)
            except Exception:
                pass

    # -- endpoints ---------------------------------------------------------
    def _api_status(self):
        snap = pipeline.MODEL.snapshot()
        busy = _current_run is not None and _current_run.status == "running"
        self._send_json({"model": snap["status"], "device": snap["device"],
                         "ckpt": snap["ckpt"], "preset": snap["preset"],
                         "params_m": snap["params_m"], "error": snap["error"], "busy": busy})

    def _api_browse(self, qs):
        raw = qs.get("path", [str(cfg.BROWSE_HOME)])[0]
        p = Path(raw).expanduser()
        if not p.exists() or not p.is_dir():
            self._send_json({"error": f"not a directory: {p}"}, 400)
            return
        dirs, files = [], []
        try:
            entries = sorted(p.iterdir(), key=lambda e: e.name.lower())
        except PermissionError:
            entries = []
        for e in entries:
            if e.name.startswith("."):
                continue
            try:
                if e.is_dir():
                    has_ct = (e / "ct.nii.gz").exists()
                    dirs.append({"name": e.name, "path": str(e), "has_ct": has_ct})
                elif e.is_file() and (e.name.endswith(".nii.gz") or e.name.endswith(".nii")):
                    files.append({"name": e.name, "path": str(e)})
            except OSError:
                continue
        self._send_json({"path": str(p), "parent": str(p.parent) if p.parent != p else None,
                         "dirs": dirs, "files": files})

    def _api_pick(self):
        script = (
            'POSIX path of (choose file with prompt '
            '"Choose a ct.nii.gz or a TotalSegmentator subject folder")'
        )
        try:
            res = subprocess.run(["osascript", "-e", script], capture_output=True,
                                 text=True, timeout=120)
            if res.returncode != 0:
                self._send_json({})
                return
            self._send_json({"path": res.stdout.strip()})
        except Exception:
            self._send_json({})

    def _api_inspect(self):
        body = self._read_json_body()
        path = body.get("path")
        if not path:
            self._send_json({"error": "missing 'path'"}, 400)
            return
        try:
            result = pipeline.inspect_ct(path)
        except FileNotFoundError as e:
            self._send_json({"error": str(e)}, 404)
            return
        except Exception as e:
            self._send_json({"error": str(e)}, 500)
            return
        info = dict(result["info"])
        sid = info["sid"]
        info["urls"] = {**{k: f"/runs/_inspect_{sid}/{v}" for k, v in result["slice_urls"].items()},
                        "axial_stack": f"/runs/_inspect_{sid}/{result['axial_stack_url']}"}
        info["sprite_idx"] = result["sprite_idx"]
        self._send_json(info)

    def _api_run(self):
        global _current_run
        body = self._read_json_body()
        path = body.get("path")
        if not path:
            self._send_json({"error": "missing 'path'"}, 400)
            return
        # Refuse before creating a run dir when the CT cannot go through (e.g. no masks).
        try:
            fails = [c for c in pipeline.inspect_ct(path)["checks"] if c["status"] == "fail"]
        except FileNotFoundError as e:
            self._send_json({"error": str(e)}, 404)
            return
        except Exception as e:
            self._send_json({"error": f"inspect failed: {e}"}, 400)
            return
        if fails:
            self._send_json({"error": "; ".join(f"{c['name']}: {c['reason']}" for c in fails)}, 400)
            return
        with _run_lock:
            if _current_run is not None and _current_run.status == "running":
                self._send_json({"error": "a run is already in progress",
                                 "run_id": _current_run.run_id}, 409)
                return
            try:
                run = pipeline.Run(path)
            except pipeline.Busy as e:
                self._send_json({"error": str(e)}, 409)
                return
            except FileNotFoundError as e:
                self._send_json({"error": str(e)}, 404)
                return
            except Exception as e:
                self._send_json({"error": str(e)}, 500)
                return
            _current_run = run
        t = threading.Thread(target=run.run, daemon=True)
        t.start()
        self._send_json({"run_id": run.run_id})

    def _api_recon_backfill(self, run_id: str):
        """POST /api/runs/<id>/recon — build the 3-D view assets for a run made before
        the `recon` stage existed (PLAN_3D.md §4.4). Never touches events.jsonl: replay
        order for a run must stay exactly what the pipeline wrote."""
        run_dir = cfg.RUNS / run_id
        if not run_dir.is_dir():
            self._send_json({"error": "no such run"}, 404)
            return
        viewer_url = f"/runs/{run_id}/recon/viewer.json"
        viewer_path = run_dir / "recon" / "viewer.json"
        if viewer_path.exists():
            # Cached result (R5: "a second POST returns the cached result") — still
            # includes stats so the card can show done/built + its stat tiles without
            # having to fetch viewer.json separately.
            try:
                stats = json.loads(viewer_path.read_text()).get("stats")
            except Exception:
                stats = None
            self._send_json({"viewer": viewer_url, "stats": stats, "cached": True})
            return
        run_json = run_dir / "run.json"
        if not run_json.exists():
            self._send_json({"error": "run has not finished"}, 400)
            return
        # Peek (not hold) the machine-wide run lock: refuse while a live pipeline run —
        # which can itself write this run's own recon/ — is in progress anywhere.
        lock_path = cfg.RUNS / ".run.lock"
        peek = pipeline._try_flock(lock_path)
        if peek is None:
            self._send_json({"error": f"a pipeline run is in progress (pid {pipeline._holder_pid(lock_path)})"}, 409)
            return
        peek.close()
        try:
            sid = json.loads(run_json.read_text()).get("subject")
        except Exception as e:
            self._send_json({"error": f"bad run.json: {e}"}, 500)
            return
        with _recon_backfill_lock:
            if viewer_path.exists():  # built while we waited
                try:
                    stats = json.loads(viewer_path.read_text()).get("stats")
                except Exception:
                    stats = None
                self._send_json({"viewer": viewer_url, "stats": stats, "cached": True})
                return
            try:
                stats = pipeline.recon.build(run_dir, sid)
            except Exception as e:
                self._send_json({"error": str(e)}, 500)
                return
        self._send_json({"viewer": viewer_url, "stats": stats})

    def _find_run(self, run_id: str) -> pipeline.Run | None:
        if _current_run is not None and _current_run.run_id == run_id:
            return _current_run
        return None

    def _api_run_events(self, run_id: str, qs):
        run_dir = cfg.RUNS / run_id
        events_path = run_dir / "events.jsonl"
        if not events_path.exists():
            self._send_json({"error": "no such run"}, 404)
            return
        replay = qs.get("replay", ["0"])[0] in ("1", "true", "yes")
        speed = float(qs.get("speed", ["1"])[0] or 1.0)

        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("Connection", "keep-alive")
        self.send_header("X-Accel-Buffering", "no")
        self.end_headers()

        def sse_write(line: str):
            self.wfile.write(f"data: {line}\n\n".encode("utf-8"))
            self.wfile.flush()

        run = self._find_run(run_id)
        try:
            if run is not None and run.status == "running" and not replay:
                backlog, q = run.subscribe()
                try:
                    for line in backlog:
                        sse_write(line)
                    while True:
                        try:
                            item = q.get(timeout=15)
                        except Exception:
                            self.wfile.write(b": keep-alive\n\n")
                            self.wfile.flush()
                            continue
                        if item is None:
                            break
                        sse_write(item)
                finally:
                    run.unsubscribe(q)
            else:
                lines = events_path.read_text().splitlines()
                prev_t = 0.0
                for line in lines:
                    if replay:
                        try:
                            t = json.loads(line).get("t", prev_t)
                        except Exception:
                            t = prev_t
                        dt = max(0.0, (t - prev_t)) / max(speed, 0.01)
                        if dt > 0:
                            time.sleep(min(dt, 5.0))
                        prev_t = t
                    sse_write(line)
        except BrokenPipeError:
            pass

    def _serve_run_file(self, rest: str):
        parts = rest.split("/", 1)
        if not parts or not parts[0]:
            self._send_json({"error": "missing run id"}, 400)
            return
        run_id = parts[0]
        rel = parts[1] if len(parts) > 1 else ""
        run_dir = cfg.RUNS / run_id
        if not run_dir.is_dir():
            self._send_json({"error": "no such run"}, 404)
            return
        p = _safe_join(run_dir, rel)
        if p is None:
            self._send_json({"error": "forbidden"}, 403)
            return
        self._send_file(p)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=cfg.PORT)
    ap.add_argument("--open", action="store_true")
    args = ap.parse_args()

    # One demo server per machine: each one holds the generator in memory, and several
    # at once (on different ports) is how the laptop ran out of RAM before.
    global _server_lock
    lock_path = cfg.RUNS / ".server.lock"
    _server_lock = pipeline._try_flock(lock_path)
    if _server_lock is None:
        print(f"[demo] another demo server is already running (pid {pipeline._holder_pid(lock_path)}) "
              f"— open http://{cfg.HOST}:{cfg.PORT}/ or stop that one first.")
        sys.exit(1)

    threading.Thread(target=pipeline.MODEL.load, daemon=True).start()

    httpd = ThreadingHTTPServer((cfg.HOST, args.port), Handler)
    url = f"http://{cfg.HOST}:{args.port}/"
    print(f"[demo] serving {url}")
    print(f"[demo] repo   = {cfg.REPO}")
    print(f"[demo] ckpt   = {cfg.CKPT}")
    print(f"[demo] runs   = {cfg.RUNS}")
    if args.open:
        threading.Timer(0.5, lambda: webbrowser.open(url)).start()
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        httpd.server_close()


if __name__ == "__main__":
    main()
