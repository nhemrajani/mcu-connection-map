"""review.py - a tiny local server for judging proposed connections.

    .venv/bin/python ml/review.py        then open http://127.0.0.1:8788

Serves ml/review.html and accepts judgements, appending each one straight into
data/connections.json as you go. Nothing is held in the browser, so closing the
tab loses nothing and you can stop whenever you like.

Every judgement is recorded, including the rejections. A rejected pair is not
deleted: it stays as a labelled negative example, which is what makes it
possible later to measure precision and to fine-tune the model on this
project's own notion of a connection.
"""
import json
import http.server
import socketserver
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
HERE = Path(__file__).resolve().parent
DATA = ROOT / "data"
OUT = HERE / "out"
PORT = 8788


def load(path, default):
    return json.loads(path.read_text()) if path.exists() else default


class Handler(http.server.SimpleHTTPRequestHandler):
    def _send(self, obj, code=200):
        body = json.dumps(obj).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if self.path in ("/", "/index.html"):
            html = (HERE / "review.html").read_bytes()
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(html)))
            self.end_headers()
            self.wfile.write(html)
            return

        if self.path == "/queue":
            moments = {m["id"]: m for m in load(DATA / "moments.json", [])}
            judged = {
                frozenset((c["source"], c["target"]))
                for c in load(DATA / "connections.json", [])
            }
            queue = [
                q for q in load(OUT / "review_queue.json", [])
                if frozenset((q["source"], q["target"])) not in judged
            ]
            self._send({
                "queue": queue,
                "moments": moments,
                "done": len(judged),
            })
            return

        self.send_error(404)

    def do_POST(self):
        if self.path != "/judge":
            self.send_error(404)
            return

        length = int(self.headers.get("Content-Length", 0))
        j = json.loads(self.rfile.read(length) or b"{}")

        path = DATA / "connections.json"
        connections = load(path, [])
        pair = frozenset((j["source"], j["target"]))
        # Re-judging a pair replaces the old verdict rather than duplicating it.
        connections = [
            c for c in connections if frozenset((c["source"], c["target"])) != pair
        ]

        record = {
            "source": j["source"],
            "target": j["target"],
            "verdict": j["verdict"],
            "proposed_by": j.get("proposed_by", "?"),
            "bucket": j.get("bucket"),
            "judged_at": date.today().isoformat(),
        }
        if j["verdict"] == "confirmed":
            record["type"] = j["type"]
        if j.get("note"):
            record["note"] = j["note"]

        connections.append(record)
        path.write_text(json.dumps(connections, indent=2) + "\n")
        self._send({"ok": True, "total": len(connections)})

    def log_message(self, *args):
        pass   # keep the terminal quiet while reviewing


if __name__ == "__main__":
    socketserver.TCPServer.allow_reuse_address = True
    with socketserver.TCPServer(("127.0.0.1", PORT), Handler) as httpd:
        print(f"Review tool running -> http://127.0.0.1:{PORT}")
        print("Judgements save straight into data/connections.json. Ctrl-C to stop.")
        httpd.serve_forever()
