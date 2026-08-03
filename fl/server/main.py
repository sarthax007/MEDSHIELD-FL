import http.server
import socketserver

PORT = 8080


class HealthHandler(http.server.SimpleHTTPRequestHandler):
    def do_GET(self):
        self.send_response(200)
        self.send_header("Content-type", "application/json")
        self.end_headers()
        self.wfile.write(b'{"status": "ok", "service": "fl_server"}')


with socketserver.TCPServer(("", PORT), HealthHandler) as httpd:
    print(f"Serving dummy FL server at port {PORT}")
    httpd.serve_forever()
