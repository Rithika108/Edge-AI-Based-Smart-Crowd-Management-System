"""
MEM 2 - Live Dashboard + Graph + Alert Message
Edge-AI Based Smart Crowd Management System

Shows:  Total Count | Max Capacity | Remaining | Occupancy % | Status
        Live line graph of the last 10 minutes
        RED banner   "ALERT: CAPACITY EXCEEDED"  when OVERCROWDED
        GREEN banner "SAFE"                      when SAFE
        (ORANGE banner for NEAR CAPACITY)
        Email status line (filled in by Mem 3 / Mem 4)

Only the Python standard library is used. Run in IDLE with F5.

INPUT FILES (written by other members)
  max_capacity.json : from Mem 1. Any JSON with a key containing the word
                      "capacity" works, e.g. {"max_capacity": 40}
  live_count.txt    : current people count (a single number)

HOW MAIN.PY USES THIS MODULE
    import tkinter as tk
    from mem2 import Dashboard
    root = tk.Tk()
    dash = Dashboard(root, count_provider=my_function_returning_count)
    dash.set_email_status("Email sent to 4 guards at 10:32:05")
    print(dash.status, dash.count, dash.max_capacity)
    root.mainloop()
"""

import json
import os
import random
import time
import tkinter as tk
from collections import deque
from tkinter import ttk

# ---------------------------------------------------------------- settings
MAX_CAPACITY_FILE = "max_capacity.json"
LIVE_COUNT_FILE = "live_count.txt"
FALLBACK_CAPACITY = 50          # used only if max_capacity.json is missing
NEAR_PERCENT = 80               # >= this % (and <= 100) = NEAR CAPACITY
WINDOW_SEC = 600                # graph shows last 10 minutes
REFRESH_MS = 1000               # dashboard refresh every 1 second

COLORS = {
    "SAFE": "#1e9e4a",
    "NEAR CAPACITY": "#e67e22",
    "OVERCROWDED": "#d62828",
}
BANNER_TEXT = {
    "SAFE": "SAFE",
    "NEAR CAPACITY": "WARNING: NEAR CAPACITY",
    "OVERCROWDED": "ALERT: CAPACITY EXCEEDED",
}


# ---------------------------------------------------------------- helpers
def _find_capacity(obj):
    """Search JSON (dict/list) for a numeric value whose key has 'capacity'."""
    if isinstance(obj, dict):
        for k, v in obj.items():
            if "capacity" in str(k).lower() and isinstance(v, (int, float)):
                return int(v)
        for v in obj.values():
            found = _find_capacity(v)
            if found is not None:
                return found
    elif isinstance(obj, list):
        for v in obj:
            found = _find_capacity(v)
            if found is not None:
                return found
    return None


def load_max_capacity(path=MAX_CAPACITY_FILE):
    """Return max capacity from Mem 1's JSON, or None if unavailable."""
    try:
        with open(path, "r") as f:
            cap = _find_capacity(json.load(f))
        return cap if cap and cap > 0 else None
    except (OSError, ValueError):
        return None


def read_live_count(path=LIVE_COUNT_FILE):
    try:
        with open(path, "r") as f:
            return max(int(float(f.read().strip().split()[0])), 0)
    except (OSError, ValueError, IndexError):
        return None


def get_status(count, max_capacity):
    """Return (status, occupancy_percent)."""
    occ = 100.0 * count / max_capacity if max_capacity else 0.0
    if count > max_capacity:
        return "OVERCROWDED", occ
    if occ >= NEAR_PERCENT:
        return "NEAR CAPACITY", occ
    return "SAFE", occ


class Simulator:
    """Fake counter for demo when no camera / YOLO is running."""
    def __init__(self, cap):
        self.v = int(cap * 0.3)

    def next(self, cap):
        self.v += random.randint(-3, 4)
        if random.random() < 0.05:
            self.v += random.randint(5, 12)
        self.v = max(0, min(self.v, int(cap * 1.6)))
        return self.v


# ---------------------------------------------------------------- dashboard
class Dashboard:
    GW, GH = 720, 230          # graph size
    PAD_L, PAD_B, PAD_T = 40, 28, 12

    def __init__(self, root, count_provider=None, capacity_provider=None):
        """
        count_provider    : optional function returning current people count
        capacity_provider : optional function returning max capacity
        If not given, live_count.txt and max_capacity.json are read
        (or simulation mode is used if chosen in the dropdown).
        """
        self.root = root
        self.count_provider = count_provider
        self.capacity_provider = capacity_provider

        # public values other modules can read
        self.count = 0
        self.max_capacity = FALLBACK_CAPACITY
        self.remaining = FALLBACK_CAPACITY
        self.occupancy = 0.0
        self.status = "SAFE"

        self.history = deque()            # (timestamp, count)
        self.sim = Simulator(self.max_capacity)
        self.flash = False
        self.prev_status = "SAFE"

        root.title("Smart Crowd Management - Live Dashboard")
        root.configure(bg="#f4f6f8")
        root.resizable(False, False)
        self._build()
        self._tick()

    # ------------------------------------------------------------ build UI
    def _build(self):
        bg = "#f4f6f8"
        self.banner = tk.Label(self.root, text="SAFE", font=("Arial", 30, "bold"),
                               fg="white", bg=COLORS["SAFE"], pady=14)
        self.banner.pack(fill="x")

        top = tk.Frame(self.root, bg=bg)
        top.pack(pady=(8, 0))
        tk.Label(top, text="Data source:", bg=bg).pack(side="left")
        self.mode = tk.StringVar(value="Live (live_count.txt)")
        ttk.Combobox(top, textvariable=self.mode, width=22, state="readonly",
                     values=["Live (live_count.txt)", "Simulation (demo)"]
                     ).pack(side="left", padx=6)
        self.src_note = tk.Label(top, text="", bg=bg, fg="#7f8c8d", font=("Arial", 9))
        self.src_note.pack(side="left")

        cards = tk.Frame(self.root, bg=bg)
        cards.pack(pady=10, padx=10)
        self.vals = {}
        for i, name in enumerate(["Total Count", "Max Capacity", "Remaining",
                                  "Occupancy %", "Status"]):
            box = tk.Frame(cards, bg="white", padx=16, pady=10,
                           highlightbackground="#dfe4ea", highlightthickness=1)
            box.grid(row=0, column=i, padx=6)
            tk.Label(box, text=name, bg="white", fg="#7f8c8d",
                     font=("Arial", 10)).pack()
            lab = tk.Label(box, text="-", bg="white", fg="#2d3436",
                           font=("Arial", 20, "bold"), width=11 if name == "Status" else 7)
            lab.pack()
            self.vals[name] = lab

        self.bar = ttk.Progressbar(self.root, length=self.GW, maximum=100)
        self.bar.pack(pady=(0, 8))

        tk.Label(self.root, text="Live Crowd Count - Last 10 Minutes", bg=bg,
                 font=("Arial", 11, "bold")).pack()
        self.canvas = tk.Canvas(self.root, width=self.GW + self.PAD_L + 10,
                                height=self.GH + self.PAD_B + self.PAD_T,
                                bg="white", highlightbackground="#dfe4ea",
                                highlightthickness=1)
        self.canvas.pack(padx=10, pady=4)

        self.email_lbl = tk.Label(self.root, text="Email Status: Not sent yet",
                                  bg=bg, fg="#34495e", font=("Arial", 11))
        self.email_lbl.pack(pady=(6, 10))

    # ------------------------------------------------- public API (others)
    def set_email_status(self, text):
        """Called by Mem 3 / Mem 4 to show e.g. 'Email sent to 4 guards at 10:32'."""
        self.email_lbl.config(text="Email Status: " + text)

    def update(self, count):
        """Push a new count into the dashboard (can also be called by main.py)."""
        self.count = int(count)
        self.max_capacity = self._get_capacity()
        self.remaining = max(self.max_capacity - self.count, 0)
        self.status, self.occupancy = get_status(self.count, self.max_capacity)

        now = time.time()
        self.history.append((now, self.count))
        while self.history and now - self.history[0][0] > WINDOW_SEC:
            self.history.popleft()
        self._refresh_widgets()

    # -------------------------------------------------------------- internals
    def _get_capacity(self):
        if self.capacity_provider:
            try:
                return int(self.capacity_provider())
            except Exception:
                pass
        cap = load_max_capacity()
        if cap:
            self.src_note.config(text="")
            return cap
        self.src_note.config(
            text=f"(max_capacity.json not found - using {FALLBACK_CAPACITY})")
        return FALLBACK_CAPACITY

    def _get_count(self):
        if self.count_provider:
            try:
                return self.count_provider()
            except Exception:
                return None
        if self.mode.get().startswith("Simulation"):
            return self.sim.next(self.max_capacity)
        return read_live_count()

    def _tick(self):
        c = self._get_count()
        if c is not None:
            self.update(c)
        else:
            self.flash = False
            if not self.count_provider:
                self.src_note.config(text="(waiting for live_count.txt ...)")
        self.root.after(REFRESH_MS, self._tick)

    def _refresh_widgets(self):
        st = self.status
        color = COLORS[st]
        if st == "OVERCROWDED":              # flashing red banner
            self.flash = not self.flash
            color = "#ff1f1f" if self.flash else "#9b1c1c"
            if self.prev_status != "OVERCROWDED":
                self.root.bell()
        self.prev_status = st
        self.banner.config(text=BANNER_TEXT[st], bg=color)

        self.vals["Total Count"].config(text=str(self.count))
        self.vals["Max Capacity"].config(text=str(self.max_capacity))
        self.vals["Remaining"].config(text=str(self.remaining))
        self.vals["Occupancy %"].config(text=f"{self.occupancy:.1f}%")
        self.vals["Status"].config(text=st, fg=COLORS[st])
        self.bar["value"] = min(self.occupancy, 100)
        self._draw_graph()

    def _draw_graph(self):
        c = self.canvas
        c.delete("all")
        L, T = self.PAD_L, self.PAD_T
        W, H = self.GW, self.GH
        now = time.time()
        data = list(self.history)
        top = max(max((v for _, v in data), default=0), self.max_capacity) * 1.2
        top = max(top, 5)

        def X(t):
            return L + (t - (now - WINDOW_SEC)) / WINDOW_SEC * W

        def Y(v):
            return T + H - v / top * H

        # grid + y labels
        for i in range(5):
            val = top * i / 4
            y = Y(val)
            c.create_line(L, y, L + W, y, fill="#ecf0f1")
            c.create_text(L - 5, y, text=str(int(val)), anchor="e",
                          font=("Arial", 8), fill="#7f8c8d")
        # x labels
        for m in range(0, 11, 2):
            x = L + W * m / 10
            c.create_line(x, T + H, x, T + H + 4, fill="#7f8c8d")
            label = "now" if m == 10 else f"-{10 - m}m"
            c.create_text(x, T + H + 14, text=label, font=("Arial", 8), fill="#7f8c8d")
        c.create_rectangle(L, T, L + W, T + H, outline="#bdc3c7")

        # capacity line
        cy = Y(self.max_capacity)
        c.create_line(L, cy, L + W, cy, fill="#d62828", dash=(6, 4), width=2)
        c.create_text(L + 6, cy - 9, anchor="w", fill="#d62828",
                      font=("Arial", 8, "bold"),
                      text=f"Max capacity {self.max_capacity}")

        # data line
        if len(data) >= 2:
            pts = []
            for t, v in data:
                pts += [X(t), Y(v)]
            c.create_line(*pts, fill="#0984e3", width=2)
        if data:
            t, v = data[-1]
            c.create_oval(X(t) - 4, Y(v) - 4, X(t) + 4, Y(v) + 4,
                          fill=COLORS[self.status], outline="")


def main():
    root = tk.Tk()
    Dashboard(root)
    root.mainloop()


if __name__ == "__main__":
    main()
