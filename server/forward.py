"""Forwards port 8765 (the glasses' hand-server address) to the WiLoR server on 127.0.0.1:8767.

Windows Firewall only lets the system Python accept LAN connections without an admin prompt; the conda
python that runs WiLoR stays on loopback behind this forwarder.
Run:  python forward.py            (with  HAND_PORT=8767 python hands_server.py  in the wilor env)
Self-check: python forward.py --selftest
"""
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.error import HTTPError
from urllib.parse import urlparse
from urllib.request import Request, urlopen

LISTEN_PORT = 8765
TARGET = "http://127.0.0.1:8767"


class Forward(BaseHTTPRequestHandler):
    def relay(self, body=None):
        headers = {k: v for k, v in self.headers.items() if k.lower() in ("content-type",)}
        # The glasses send an absolute URL as the request path; keep only path + query.
        url = urlparse(self.path)
        path = url.path + ("?" + url.query if url.query else "")
        try:
            with urlopen(Request(TARGET + path, data=body, headers=headers, method=self.command), timeout=30) as r:
                status, ctype, data = r.status, r.headers.get("Content-Type", ""), r.read()
        except HTTPError as e:
            status, ctype, data = e.code, e.headers.get("Content-Type", ""), e.read()
        except OSError as e:  # WiLoR server not up yet
            status, ctype, data = 502, "application/json", ('{"error": "hand model server unreachable: %s"}' % e).encode()
        self.send_response(status)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):
        self.relay()

    def do_POST(self):
        self.relay(self.rfile.read(int(self.headers.get("Content-Length") or 0)))

    def log_message(self, *args):
        pass


def selftest():
    global TARGET

    class Echo(BaseHTTPRequestHandler):
        def do_POST(self):
            data = self.rfile.read(int(self.headers["Content-Length"]))
            self.send_response(200)
            self.send_header("Content-Type", "application/octet-stream")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data[::-1])

        def log_message(self, *args):
            pass

    echo = ThreadingHTTPServer(("127.0.0.1", 0), Echo)
    proxy = ThreadingHTTPServer(("127.0.0.1", 0), Forward)
    TARGET = "http://127.0.0.1:%d" % echo.server_port
    for s in (echo, proxy):
        threading.Thread(target=s.serve_forever, daemon=True).start()
    with urlopen(Request("http://127.0.0.1:%d/hands" % proxy.server_port, data=b"abc", method="POST")) as r:
        assert r.read() == b"cba", "body must round-trip through the forwarder"
    echo.shutdown()
    try:
        urlopen("http://127.0.0.1:%d/preview.jpg" % proxy.server_port)
        raise AssertionError("dead target should give 502")
    except HTTPError as e:
        assert e.code == 502
    print("selftest ok")


if __name__ == "__main__":
    if "--selftest" in sys.argv:
        selftest()
    else:
        print("forwarding :%d -> %s" % (LISTEN_PORT, TARGET))
        ThreadingHTTPServer(("0.0.0.0", LISTEN_PORT), Forward).serve_forever()
