#!/usr/bin/env python3
import argparse
import json
import sqlite3
import time
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Dict, List
from urllib.parse import parse_qs, urlparse


HTML = """<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Posture Trends</title>
  <style>
    :root {
      --bg: #0d1b1e;
      --panel: #13272b;
      --ink: #f3f5ef;
      --muted: #8ca3a6;
      --good: #79c36a;
      --okay: #f0c35b;
      --bad: #e16b5c;
      --missing: #60757d;
      --grid: rgba(255,255,255,0.08);
    }
    body { margin: 0; background: radial-gradient(circle at top, #18353a 0, var(--bg) 55%); color: var(--ink); font: 16px/1.4 Georgia, serif; }
    .wrap { max-width: 1200px; margin: 0 auto; padding: 24px; }
    h1 { font-size: 34px; margin: 0 0 8px; }
    p { color: var(--muted); margin: 0 0 20px; }
    .controls, .cards, .charts { display: grid; gap: 16px; }
    .controls { grid-template-columns: repeat(auto-fit, minmax(180px, 1fr)); margin-bottom: 16px; }
    .cards { grid-template-columns: repeat(auto-fit, minmax(180px, 1fr)); margin-bottom: 16px; }
    .charts { grid-template-columns: 1.5fr 1fr; }
    .panel { background: rgba(19,39,43,0.9); border: 1px solid rgba(255,255,255,0.08); border-radius: 18px; padding: 16px; box-shadow: 0 20px 60px rgba(0,0,0,0.2); }
    .label { color: var(--muted); font-size: 12px; text-transform: uppercase; letter-spacing: 0.08em; }
    .value { font-size: 28px; margin-top: 6px; }
    select { width: 100%; padding: 12px; border-radius: 12px; border: 1px solid rgba(255,255,255,0.1); background: #0f2024; color: var(--ink); }
    svg { width: 100%; height: auto; display: block; }
    .legend { display: flex; gap: 12px; flex-wrap: wrap; margin-top: 8px; color: var(--muted); font-size: 13px; }
    .legend span::before { content: ""; display: inline-block; width: 10px; height: 10px; border-radius: 999px; margin-right: 6px; }
    .good::before { background: var(--good); }
    .okay::before { background: var(--okay); }
    .bad::before { background: var(--bad); }
    .missing::before { background: var(--missing); }
    table { width: 100%; border-collapse: collapse; font-size: 14px; }
    td, th { padding: 8px 0; border-bottom: 1px solid rgba(255,255,255,0.08); text-align: left; }
    @media (max-width: 900px) { .charts { grid-template-columns: 1fr; } }
  </style>
</head>
<body>
  <div class="wrap">
    <h1>Posture Trends</h1>
    <p>Always-on posture history from the Jetson runtime.</p>
    <div class="controls">
      <div class="panel">
        <div class="label">Window</div>
        <select id="hours">
          <option value="6">Last 6 hours</option>
          <option value="24" selected>Last 24 hours</option>
          <option value="72">Last 3 days</option>
          <option value="168">Last 7 days</option>
        </select>
      </div>
      <div class="panel">
        <div class="label">Bucket Size</div>
        <select id="bucket">
          <option value="1">1 minute</option>
          <option value="5" selected>5 minutes</option>
          <option value="15">15 minutes</option>
          <option value="60">60 minutes</option>
        </select>
      </div>
    </div>
    <div class="cards" id="cards"></div>
    <div class="charts">
      <div class="panel">
        <div class="label">Posture Score Over Time</div>
        <svg id="scoreChart" viewBox="0 0 860 300" preserveAspectRatio="none"></svg>
        <div class="legend">
          <span class="good">good = 1</span>
          <span class="okay">okay = 0</span>
          <span class="bad">bad = -1</span>
        </div>
      </div>
      <div class="panel">
        <div class="label">Posture Mix By Time Bucket</div>
        <svg id="mixChart" viewBox="0 0 500 300" preserveAspectRatio="none"></svg>
        <div class="legend">
          <span class="good">good</span>
          <span class="okay">okay</span>
          <span class="bad">bad</span>
          <span class="missing">not found</span>
        </div>
      </div>
    </div>
    <div class="panel" style="margin-top:16px;">
      <div class="label">Recent Status Changes</div>
      <table>
        <thead><tr><th>Time</th><th>Status</th><th>Confidence</th></tr></thead>
        <tbody id="events"></tbody>
      </table>
    </div>
  </div>
  <script>
    const scoreSvg = document.getElementById("scoreChart");
    const mixSvg = document.getElementById("mixChart");
    const cards = document.getElementById("cards");
    const eventsBody = document.getElementById("events");
    const hoursEl = document.getElementById("hours");
    const bucketEl = document.getElementById("bucket");

    function card(label, value) {
      return `<div class="panel"><div class="label">${label}</div><div class="value">${value}</div></div>`;
    }

    function formatTime(ts) {
      return new Date(ts * 1000).toLocaleString();
    }

    function renderCards(summary) {
      cards.innerHTML =
        card("Latest Status", summary.latest_status || "n/a") +
        card("Average Score", summary.avg_score === null ? "n/a" : summary.avg_score.toFixed(2)) +
        card("Samples", summary.sample_count) +
        card("Sessions", summary.session_count);
    }

    function renderScoreChart(series) {
      const w = 860, h = 300, pad = 30;
      const points = series.filter(row => row.avg_score !== null);
      if (!points.length) {
        scoreSvg.innerHTML = "";
        return;
      }
      const minTs = points[0].bucket_ts;
      const maxTs = points[points.length - 1].bucket_ts || minTs + 1;
      const lines = [];
      for (let idx = 0; idx < 3; idx += 1) {
        const value = 1 - idx;
        const y = pad + ((1 - value) / 2) * (h - pad * 2);
        lines.push(`<line x1="${pad}" y1="${y}" x2="${w - pad}" y2="${y}" stroke="rgba(255,255,255,0.08)" />`);
      }
      const coords = points.map(point => {
        const x = pad + ((point.bucket_ts - minTs) / Math.max(maxTs - minTs, 1)) * (w - pad * 2);
        const y = pad + ((1 - point.avg_score) / 2) * (h - pad * 2);
        return `${x},${y}`;
      });
      scoreSvg.innerHTML = `
        <rect x="0" y="0" width="${w}" height="${h}" fill="transparent" />
        ${lines.join("")}
        <polyline fill="none" stroke="#f3f5ef" stroke-width="3" points="${coords.join(" ")}" />
      `;
    }

    function renderMixChart(series) {
      const w = 500, h = 300, pad = 18;
      if (!series.length) {
        mixSvg.innerHTML = "";
        return;
      }
      const barWidth = Math.max(4, (w - pad * 2) / series.length);
      const colors = {good: "#79c36a", okay: "#f0c35b", bad: "#e16b5c", not_found: "#60757d"};
      const bars = [];
      series.forEach((row, idx) => {
        const total = Math.max(row.good_count + row.okay_count + row.bad_count + row.not_found_count, 1);
        let y = h - pad;
        [["good_count", "good"], ["okay_count", "okay"], ["bad_count", "bad"], ["not_found_count", "not_found"]].forEach(([field, colorKey]) => {
          const segmentH = ((row[field] || 0) / total) * (h - pad * 2);
          y -= segmentH;
          bars.push(`<rect x="${pad + idx * barWidth}" y="${y}" width="${barWidth - 1}" height="${segmentH}" fill="${colors[colorKey]}" />`);
        });
      });
      mixSvg.innerHTML = `<rect x="0" y="0" width="${w}" height="${h}" fill="transparent" />${bars.join("")}`;
    }

    function renderEvents(events) {
      eventsBody.innerHTML = events.map(event => `
        <tr>
          <td>${formatTime(event.ts_unix)}</td>
          <td>${event.oled_status_text}</td>
          <td>${event.posture_confidence === null ? "n/a" : event.posture_confidence.toFixed(2)}</td>
        </tr>
      `).join("");
    }

    async function refresh() {
      const hours = hoursEl.value;
      const bucket = bucketEl.value;
      const [summaryResp, seriesResp, eventsResp] = await Promise.all([
        fetch(`/api/summary?hours=${hours}`),
        fetch(`/api/series?hours=${hours}&bucket_minutes=${bucket}`),
        fetch(`/api/events?hours=${hours}`),
      ]);
      const summary = await summaryResp.json();
      const series = await seriesResp.json();
      const events = await eventsResp.json();
      renderCards(summary);
      renderScoreChart(series.rows);
      renderMixChart(series.rows);
      renderEvents(events.rows);
    }

    hoursEl.addEventListener("change", refresh);
    bucketEl.addEventListener("change", refresh);
    refresh();
    setInterval(refresh, 30000);
  </script>
</body>
</html>
"""


def query_summary(conn: sqlite3.Connection, hours: int) -> Dict[str, object]:
    since = time.time() - hours * 3600
    summary_row = conn.execute(
        """
        SELECT
          COUNT(*) AS sample_count,
          AVG(posture_value) AS avg_score,
          SUM(CASE WHEN status_label = 'good' THEN 1 ELSE 0 END) AS good_count,
          SUM(CASE WHEN status_label = 'okay' THEN 1 ELSE 0 END) AS okay_count,
          SUM(CASE WHEN status_label = 'bad' THEN 1 ELSE 0 END) AS bad_count,
          SUM(CASE WHEN status_label = 'not_found' THEN 1 ELSE 0 END) AS not_found_count
        FROM posture_samples
        WHERE ts_unix >= ?
        """,
        (since,),
    ).fetchone()
    latest_row = conn.execute(
        """
        SELECT oled_status_text
        FROM posture_samples
        ORDER BY ts_unix DESC
        LIMIT 1
        """
    ).fetchone()
    session_count = conn.execute(
        "SELECT COUNT(*) FROM sessions WHERE started_at >= ?",
        (since,),
    ).fetchone()[0]
    return {
        "sample_count": int(summary_row["sample_count"] or 0),
        "avg_score": summary_row["avg_score"],
        "good_count": int(summary_row["good_count"] or 0),
        "okay_count": int(summary_row["okay_count"] or 0),
        "bad_count": int(summary_row["bad_count"] or 0),
        "not_found_count": int(summary_row["not_found_count"] or 0),
        "session_count": int(session_count),
        "latest_status": latest_row["oled_status_text"] if latest_row else None,
    }


def query_series(conn: sqlite3.Connection, hours: int, bucket_minutes: int) -> Dict[str, List[Dict[str, object]]]:
    since = time.time() - hours * 3600
    bucket_seconds = max(bucket_minutes, 1) * 60
    rows = conn.execute(
        """
        WITH bucketed AS (
          SELECT
            CAST(ts_unix / ? AS INTEGER) * ? AS bucket_ts,
            AVG(posture_value) AS avg_score,
            SUM(CASE WHEN status_label = 'good' THEN 1 ELSE 0 END) AS good_count,
            SUM(CASE WHEN status_label = 'okay' THEN 1 ELSE 0 END) AS okay_count,
            SUM(CASE WHEN status_label = 'bad' THEN 1 ELSE 0 END) AS bad_count,
            SUM(CASE WHEN status_label = 'not_found' THEN 1 ELSE 0 END) AS not_found_count
          FROM posture_samples
          WHERE ts_unix >= ?
          GROUP BY bucket_ts
          ORDER BY bucket_ts
        )
        SELECT * FROM bucketed
        """,
        (bucket_seconds, bucket_seconds, since),
    ).fetchall()
    return {
        "rows": [
            {
                "bucket_ts": row["bucket_ts"],
                "avg_score": row["avg_score"],
                "good_count": row["good_count"],
                "okay_count": row["okay_count"],
                "bad_count": row["bad_count"],
                "not_found_count": row["not_found_count"],
            }
            for row in rows
        ]
    }


def query_events(conn: sqlite3.Connection, hours: int) -> Dict[str, List[Dict[str, object]]]:
    since = time.time() - hours * 3600
    rows = conn.execute(
        """
        SELECT ts_unix, oled_status_text, posture_confidence
        FROM posture_events
        WHERE ts_unix >= ?
        ORDER BY ts_unix DESC
        LIMIT 50
        """,
        (since,),
    ).fetchall()
    return {
        "rows": [
            {
                "ts_unix": row["ts_unix"],
                "oled_status_text": row["oled_status_text"],
                "posture_confidence": row["posture_confidence"],
            }
            for row in rows
        ]
    }


class DashboardHandler(BaseHTTPRequestHandler):
    db_path = ""

    def _open_db(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def _send_json(self, payload: Dict[str, object]) -> None:
        body = json.dumps(payload).encode("utf-8")
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:
        parsed = urlparse(self.path)
        params = parse_qs(parsed.query)
        hours = int(params.get("hours", ["24"])[0])
        conn = self._open_db()
        try:
            if parsed.path == "/":
                body = HTML.encode("utf-8")
                self.send_response(HTTPStatus.OK)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
                return
            if parsed.path == "/api/summary":
                self._send_json(query_summary(conn, hours))
                return
            if parsed.path == "/api/series":
                bucket_minutes = int(params.get("bucket_minutes", ["5"])[0])
                self._send_json(query_series(conn, hours, bucket_minutes))
                return
            if parsed.path == "/api/events":
                self._send_json(query_events(conn, hours))
                return
            self.send_error(HTTPStatus.NOT_FOUND, "not found")
        finally:
            conn.close()

    def log_message(self, format: str, *args: object) -> None:
        print(f"[dashboard] {self.address_string()} - {format % args}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--db", required=True)
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8787)
    args = parser.parse_args()

    DashboardHandler.db_path = args.db
    server = ThreadingHTTPServer((args.host, args.port), DashboardHandler)
    print(f"[dashboard] serving db={args.db} on http://{args.host}:{args.port}")
    server.serve_forever()


if __name__ == "__main__":
    main()
