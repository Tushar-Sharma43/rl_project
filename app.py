"""
FLOOD-OPS // FLOOD EMERGENCY OPERATIONS CENTER (evacuation command & control)
Standalone, fully interactive simulation (exercise data, not live telemetry).
Run: streamlit run app.py

Additions in this version
  * Shelter management  (create / edit / delete, capacity + occupancy validation)
  * Rescue team management (create / edit / delete, status + location + members)
  * Manual KPI override (citizens rescued, blocked nodes, simulation tick)
  * New KPIs, shelter capacity chart, shelter + team registries, map markers
All data lives in st.session_state and persists for the Streamlit session.
"""

import html
import time
from collections import deque
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import pydeck as pdk
import streamlit as st
from plotly.subplots import make_subplots

# ──────────────────────────────────────────────────────────────────────────────
# Page Configuration & SOC Styling
# ──────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="FLOOD-OPS | Emergency Operations Center",
    page_icon="🛟",
    layout="wide",
    initial_sidebar_state="expanded",
)

SOC_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500;700&display=swap');
:root{
  --bg:#0b0f14; --panel:#121820; --panel2:#171f29; --line:#243040; --line2:#1b2531;
  --txt:#d6dee8; --dim:#8091a5; --blue:#4c9aff; --green:#2fbf71; --amber:#f2a93b;
  --orange:#f0762e; --red:#e5484d; --violet:#a855f7;
}
/* typography (icons are left alone so Streamlit's Material icons keep rendering) */
.stApp, .stMarkdown, .stCaption, p, label, input, textarea, button,
[data-baseweb="tab"], [data-testid="stExpander"] summary p, [data-testid="stSidebar"] label {
  font-family:'Inter', system-ui, -apple-system, 'Segoe UI', sans-serif !important;
}
.mono, .kpi-v, .feed, .ev .t, .eoc-meta .v { font-family:'JetBrains Mono', Consolas, monospace !important; }

.stApp{ background:var(--bg); color:var(--txt); }
header[data-testid="stHeader"]{ background:transparent; }
.block-container{ padding-top:1.1rem; padding-bottom:1rem; max-width:100%; }
section[data-testid="stSidebar"]{ background:var(--panel); border-right:1px solid var(--line); }
section[data-testid="stSidebar"] *{ color:var(--txt); }

/* sidebar brand + section headings */
.brand{ display:flex; align-items:center; gap:.65rem; padding:.2rem 0 .9rem; border-bottom:1px solid var(--line); margin-bottom:.8rem; }
.brand-mark{ width:34px; height:34px; border-radius:6px; background:var(--blue); color:#06101c; font-weight:700;
  display:flex; align-items:center; justify-content:center; font-size:.85rem; letter-spacing:.04em; }
.brand-t{ font-weight:700; letter-spacing:.12em; font-size:.95rem; color:#fff; }
.brand-s{ font-size:.68rem; color:var(--dim); letter-spacing:.08em; text-transform:uppercase; }
.side-h{ font-size:.68rem; font-weight:600; color:var(--dim); letter-spacing:.14em; text-transform:uppercase;
  margin:1.1rem 0 .45rem; padding-bottom:.3rem; border-bottom:1px solid var(--line2); }

/* top bar */
.eoc-head{ display:flex; align-items:center; justify-content:space-between; gap:1rem; flex-wrap:wrap;
  background:var(--panel); border:1px solid var(--line); border-radius:6px 6px 0 0; padding:.75rem 1.1rem; }
.eoc-left{ display:flex; align-items:center; gap:.85rem; }
.eoc-badge{ background:var(--blue); color:#06101c; font-weight:700; font-size:.78rem; letter-spacing:.1em;
  padding:.35rem .55rem; border-radius:4px; }
.eoc-title{ font-size:1.12rem; font-weight:600; color:#fff; letter-spacing:.01em; }
.eoc-sub{ font-size:.72rem; color:var(--dim); margin-top:.1rem; }
.eoc-sub b{ color:var(--amber); font-weight:600; letter-spacing:.08em; }
.eoc-right{ display:flex; align-items:center; gap:1.3rem; flex-wrap:wrap; }
.eoc-meta{ display:flex; flex-direction:column; line-height:1.2; }
.eoc-meta .l{ font-size:.62rem; color:var(--dim); letter-spacing:.12em; text-transform:uppercase; }
.eoc-meta .v{ font-size:.98rem; color:#fff; font-weight:500; }
.alert{ padding:.32rem .8rem; border-radius:4px; font-size:.72rem; font-weight:700; letter-spacing:.12em; border:1px solid; }
.alert.normal{ color:var(--green); border-color:var(--green); background:rgba(47,191,113,.10); }
.alert.watch{ color:var(--amber); border-color:var(--amber); background:rgba(242,169,59,.10); }
.alert.warning{ color:var(--orange); border-color:var(--orange); background:rgba(240,118,46,.12); }
.alert.severe{ color:var(--red); border-color:var(--red); background:rgba(229,72,77,.14); }
.pill{ padding:.3rem .7rem; border-radius:999px; font-size:.7rem; font-weight:600; letter-spacing:.1em; border:1px solid; text-transform:uppercase; }
.pill.live{ color:var(--green); border-color:rgba(47,191,113,.5); }
.pill.live::first-letter{ animation:pulse 1.4s infinite; }
.pill.hold{ color:var(--dim); border-color:var(--line); }
@keyframes pulse{ 50%{ opacity:.35; } }
.prog{ height:3px; background:var(--line2); border-radius:0 0 6px 6px; margin-bottom:.8rem; overflow:hidden; }
.prog > div{ height:100%; background:var(--blue); }

/* KPI cards */
.kpi{ background:var(--panel); border:1px solid var(--line); border-left:3px solid var(--blue);
  border-radius:4px; padding:.6rem .85rem; height:100%; }
.kpi.green{ border-left-color:var(--green); } .kpi.red{ border-left-color:var(--red); }
.kpi.amber{ border-left-color:var(--amber); } .kpi.orange{ border-left-color:var(--orange); }
.kpi-l{ font-size:.66rem; color:var(--dim); letter-spacing:.12em; text-transform:uppercase; font-weight:500; }
.kpi-v{ font-size:1.55rem; font-weight:500; color:#fff; line-height:1.3; }
.kpi-s{ font-size:.7rem; color:var(--dim); }

/* panels */
.panel-t{ font-size:.7rem; color:var(--dim); font-weight:600; letter-spacing:.14em; text-transform:uppercase;
  margin:.15rem 0 .45rem; padding-bottom:.35rem; border-bottom:1px solid var(--line2); }
.legend{ display:flex; gap:1.2rem; flex-wrap:wrap; font-size:.72rem; color:var(--dim); margin-top:.35rem; }
.legend span{ display:inline-flex; align-items:center; gap:.4rem; }
.legend .dot{ width:10px; height:10px; border-radius:50%; display:inline-block; }
.legend .ln{ width:16px; height:3px; border-radius:2px; display:inline-block; }

/* event feed */
.feed{ background:var(--panel); border:1px solid var(--line); border-radius:4px; padding:.35rem .7rem;
  height:250px; overflow-y:auto; display:flex; flex-direction:column-reverse; font-size:.74rem; line-height:1.45; }
.ev{ display:grid; grid-template-columns:44px 52px 1fr; gap:.55rem; padding:.3rem 0; border-bottom:1px solid var(--line2); align-items:baseline; }
.ev .t{ color:var(--dim); }
.ev .sev{ font-size:.6rem; font-weight:700; letter-spacing:.08em; padding:.08rem .3rem; border-radius:3px; text-align:center; }
.ev .sev.info{ color:var(--blue); background:rgba(76,154,255,.14); }
.ev .sev.ok{ color:var(--green); background:rgba(47,191,113,.14); }
.ev .sev.warn{ color:var(--amber); background:rgba(242,169,59,.14); }
.ev .sev.crit{ color:var(--red); background:rgba(229,72,77,.16); }
.ev .m{ color:var(--txt); word-break:break-word; }
.ev.crit .m{ color:#ffb3b5; }

/* status chips */
.chips{ display:flex; gap:.45rem; flex-wrap:wrap; margin:.1rem 0 .55rem; }
.chip{ font-size:.7rem; font-weight:600; padding:.18rem .55rem; border-radius:999px; border:1px solid; letter-spacing:.03em; }

/* controls */
.stButton button, [data-testid="stFormSubmitButton"] button{
  background:var(--panel2); border:1px solid var(--line); color:var(--txt); border-radius:4px; font-weight:500; }
.stButton button:hover, [data-testid="stFormSubmitButton"] button:hover{ border-color:var(--blue); color:#fff; }
div[data-testid="stForm"]{ border:1px solid var(--line); background:var(--panel2); border-radius:4px; }
div[data-testid="stExpander"] details{ border:1px solid var(--line); background:var(--panel); border-radius:4px; }
.stTabs [data-baseweb="tab-list"]{ gap:0; border-bottom:1px solid var(--line); }
.stTabs [data-baseweb="tab"]{ padding:.55rem 1.1rem; color:var(--dim); font-size:.74rem; letter-spacing:.1em; text-transform:uppercase; font-weight:600; }
.stTabs [aria-selected="true"]{ color:#fff; }
.stTabs [data-baseweb="tab-highlight"]{ background:var(--blue); }
</style>
"""
st.markdown(SOC_CSS, unsafe_allow_html=True)

TEAM_STATUSES = ["Available", "En Route", "Rescuing", "Returning", "Offline"]


# ──────────────────────────────────────────────────────────────────────────────
# Robust Standalone Simulation Engine (Bengaluru Grid Mock)
# ──────────────────────────────────────────────────────────────────────────────
class StandaloneDemoEnv:
    SCHEMA = 2  # bump to force a clean session re-init when the class changes

    def __init__(self, side=25, n_vehicles=10, max_steps=72):
        self.side = side
        self.max_steps = max_steps
        rng = np.random.default_rng(42)

        # Generate grid coordinates centered around Bengaluru (12.9716° N, 77.5946° E)
        lon0, lat0, step = 77.52, 12.92, 0.003
        coords = []
        for i in range(side):
            for j in range(side):
                coords.append([lat0 + i * step + rng.normal(0, 1e-4), lon0 + j * step + rng.normal(0, 1e-4)])
        self.coords = np.array(coords)
        self.num_nodes = len(self.coords)

        # Safe zones (shelters) and flood epicenters
        # NOTE: shelter_idx is kept in sync with the shelter registry (st.session_state.shelters)
        self.shelter_idx = rng.choice(self.num_nodes, 5, replace=False)
        self.flood_centers = self.coords[rng.choice(self.num_nodes, 3, replace=False)]

        # State variables
        self.t = 0
        self.total_rescued = 0
        self.vehicle_positions = rng.choice(self.num_nodes, n_vehicles).tolist()
        self.prev_positions = list(self.vehicle_positions)
        self.water_depths = np.zeros(self.num_nodes)
        self.per_vehicle = np.zeros(n_vehicles)
        self.blocked_override = None   # manual override for the blocked-nodes KPI
        self.arrivals = []             # (node, evacuees) shelter arrivals during the last step

    def update_water(self):
        """Recompute flood depths for the current tick (also used when the tick is edited)."""
        d2 = ((self.coords[:, None, :] - self.flood_centers[None, :, :]) ** 2).sum(-1).min(1)
        self.water_depths = np.clip(self.t * 0.008 * np.exp(-d2 / (2 * 0.005**2)), 0, 1.8)

    def blocked_nodes(self):
        """Blocked-node count according to the flood model."""
        return int((self.water_depths > 0.4).sum())

    def reset(self, seed=None):
        self.t = 0
        self.total_rescued = 0
        rng = np.random.default_rng(seed)
        self.vehicle_positions = rng.choice(self.num_nodes, len(self.vehicle_positions)).tolist()
        self.prev_positions = list(self.vehicle_positions)
        self.per_vehicle = np.zeros(len(self.vehicle_positions))
        self.blocked_override = None
        self.arrivals = []
        self.update_water()
        return np.zeros(4), {}

    def step(self):
        self.t += 1
        rng = np.random.default_rng(self.t)
        self.arrivals = []

        # Simulate spreading flood water
        self.update_water()

        # Move vehicles towards shelters or random adjacent nodes
        self.prev_positions = list(self.vehicle_positions)
        for v in range(len(self.vehicle_positions)):
            curr = self.vehicle_positions[v]
            # Move towards a random neighbor
            neighbors = [curr - 1, curr + 1, curr - self.side, curr + self.side]
            valid_nbrs = [n for n in neighbors if 0 <= n < self.num_nodes]
            if valid_nbrs:
                nxt = rng.choice(valid_nbrs)
                # Avoid high flood if possible
                if self.water_depths[nxt] < 0.8 or rng.random() < 0.3:
                    self.vehicle_positions[v] = nxt

            # Check if vehicle reached a shelter
            if self.vehicle_positions[v] in self.shelter_idx:
                gain = int(rng.integers(4, 12))
                self.total_rescued += gain
                self.per_vehicle[v] += gain
                self.arrivals.append((int(self.vehicle_positions[v]), gain))
                self.vehicle_positions[v] = int(rng.choice(range(self.num_nodes)))

        blocked_count = self.blocked_override if self.blocked_override is not None else self.blocked_nodes()
        info = {
            "total_rescued": self.total_rescued,
            "blocked_nodes_count": blocked_count,
            "equity_variance": float(np.var(self.per_vehicle)),
        }
        reward = float(self.total_rescued * 1.5 - blocked_count * 0.05)
        return np.zeros(4), reward, self.t >= self.max_steps, False, info


# ──────────────────────────────────────────────────────────────────────────────
# Operational helpers (incident clock, alert levels, road network)
# ──────────────────────────────────────────────────────────────────────────────
SIM_START_MIN = 6 * 60   # incident starts 06:00
MIN_PER_TICK = 10        # one simulation tick = 10 minutes (72 ticks = 12 h)
ALERT_RANK = {"NORMAL": 0, "WATCH": 1, "WARNING": 2, "SEVERE": 3}


def clock(t):
    m = SIM_START_MIN + int(t) * MIN_PER_TICK
    return f"{(m // 60) % 24:02d}:{m % 60:02d}"


def alert_level(depth):
    if depth >= 0.40:
        return "SEVERE", "severe"
    if depth >= 0.25:
        return "WARNING", "warning"
    if depth >= 0.10:
        return "WATCH", "watch"
    return "NORMAL", "normal"


def build_edges(side):
    """Road segments between 4-neighbour grid nodes (right and up links)."""
    idx = np.arange(side * side).reshape(side, side)
    src = np.concatenate([idx[:, :-1].ravel(), idx[:-1, :].ravel()])
    dst = np.concatenate([idx[:, 1:].ravel(), idx[1:, :].ravel()])
    return src, dst


TEAM_COLORS = {"Available": "#2fbf71", "En Route": "#4c9aff", "Rescuing": "#f2a93b",
               "Returning": "#a855f7", "Offline": "#8091a5"}
SHELTER_COLORS = {"OPEN": "#2fbf71", "NEAR FULL": "#f2a93b", "FULL": "#e5484d"}


def chip_html(label, count, color):
    return (f'<span class="chip" style="color:{color};border-color:{color}55;background:{color}1a">'
            f'{html.escape(label)} · {count}</span>')


def style_status(df, colors):
    def _c(v):
        c = colors.get(v)
        return f"background-color:{c}26; color:{c}; font-weight:600" if c else ""
    styler = df.style
    fn = getattr(styler, "map", None) or styler.applymap
    return fn(_c, subset=["Status"])


# ──────────────────────────────────────────────────────────────────────────────
# Registry helpers (shelters / rescue teams)
# ──────────────────────────────────────────────────────────────────────────────
def _i(x, default=0):
    try:
        return int(x)
    except (TypeError, ValueError):
        return default


def rc_to_node(row, col):
    return int(row) * st.session_state.env.side + int(col)


def node_to_rc(node):
    return divmod(int(node), st.session_state.env.side)


def get_shelter(sid):
    return next((s for s in st.session_state.shelters if s["id"] == sid), None)


def get_team(tid):
    return next((t for t in st.session_state.teams if t["id"] == tid), None)


def shelter_pct(s):
    return (s["occupancy"] / s["capacity"] * 100.0) if s["capacity"] > 0 else 0.0


def shelter_totals():
    cap = sum(s["capacity"] for s in st.session_state.shelters)
    occ = sum(s["occupancy"] for s in st.session_state.shelters)
    return cap, occ, cap - occ, (occ / cap * 100.0 if cap > 0 else 0.0)


def sync_env_shelters():
    """Keep the simulation's shelter nodes identical to the shelter registry."""
    st.session_state.env.shelter_idx = np.array([s["node"] for s in st.session_state.shelters], dtype=int)


def flash(area, level, msg):
    st.session_state.setdefault("flash", {})[area] = (level, msg)


def show_flash(area):
    item = st.session_state.get("flash", {}).get(area)
    if item:
        level, msg = item
        {"success": st.success, "error": st.error, "warning": st.warning}.get(level, st.info)(msg)


def validate_shelter(name, row, col, cap, occ, exclude_id=None):
    side = st.session_state.env.side
    errs = []
    if not name:
        errs.append("Name is required.")
    if not (0 <= row < side and 0 <= col < side):
        errs.append(f"Grid row/column must be between 0 and {side - 1}.")
    else:
        node = row * side + col
        clash = next((s for s in st.session_state.shelters if s["node"] == node and s["id"] != exclude_id), None)
        if clash:
            errs.append(f"Grid cell ({row}, {col}) already hosts shelter {clash['id']}.")
    if cap < 0:
        errs.append("Capacity cannot be negative.")
    if occ < 0:
        errs.append("Occupancy cannot be negative.")
    if cap >= 0 and occ >= 0 and occ > cap:
        errs.append(f"Occupancy ({occ}) cannot exceed capacity ({cap}).")
    return errs


def validate_team(name, row, col, members, status):
    side = st.session_state.env.side
    errs = []
    if not name:
        errs.append("Name is required.")
    if not (0 <= row < side and 0 <= col < side):
        errs.append(f"Grid row/column must be between 0 and {side - 1}.")
    if members < 1:
        errs.append("A team needs at least 1 member.")
    if status not in TEAM_STATUSES:
        errs.append("Invalid status.")
    return errs


# ──────────────────────────────────────────────────────────────────────────────
# Session Initialization
# ──────────────────────────────────────────────────────────────────────────────
if "env" not in st.session_state or getattr(st.session_state.env, "SCHEMA", 0) != StandaloneDemoEnv.SCHEMA:
    st.session_state.env = StandaloneDemoEnv()
    st.session_state.step = 0
    st.session_state.playing = False
    st.session_state.history = []
    st.session_state.logs = deque(maxlen=200)
    st.session_state.total_reward = 0.0
    st.session_state.metrics = {"total_rescued": 0, "blocked_nodes_count": 0, "equity_variance": 0.0}
    st.session_state.prev_metrics = dict(st.session_state.metrics)
    st.session_state.logs.append((clock(0), "info", "Command center online. PPO routing policy loaded; awaiting simulation start."))

    # Seed registries from the simulation's initial layout
    _env = st.session_state.env
    _sh_names = ["Alpha", "Bravo", "Charlie", "Delta", "Echo"]
    _sh_caps = [500, 350, 400, 250, 600]
    st.session_state.shelters = [
        {"id": f"SH-{k + 1:02d}", "name": f"Shelter {_sh_names[k]}", "node": int(n), "capacity": _sh_caps[k], "occupancy": 0}
        for k, n in enumerate(_env.shelter_idx)
    ]
    st.session_state.next_shelter_n = len(st.session_state.shelters) + 1

    _tm_status = ["Available", "En Route", "Rescuing", "Returning", "Available"]
    _tm_members = [6, 8, 5, 7, 4]
    st.session_state.teams = [
        {"id": f"RT-{k + 1:02d}", "name": f"Rescue {_sh_names[k]}", "node": int(_env.vehicle_positions[k]),
         "members": _tm_members[k], "status": _tm_status[k]}
        for k in range(5)
    ]
    st.session_state.next_team_n = len(st.session_state.teams) + 1
    st.session_state.edges = build_edges(_env.side)
    st.session_state.alert_name = "NORMAL"
    st.session_state.flash = {}
    st.session_state.kpi_dirty = True
    st.session_state.sh_loaded = None
    st.session_state.tm_loaded = None


def reset_session():
    ss = st.session_state
    ss.env.reset(seed=int(ss.get("seed", 42)))
    ss.step = 0
    ss.total_reward = 0.0
    ss.history = []
    ss.playing = False
    ss.metrics = {"total_rescued": 0, "blocked_nodes_count": 0, "equity_variance": 0.0}
    ss.prev_metrics = dict(ss.metrics)
    if ss.get("clear_occ", True):
        for s in ss.shelters:
            s["occupancy"] = 0
        ss.sh_loaded = None
    ss.alert_name = "NORMAL"
    ss.kpi_dirty = True
    ss.logs.append((clock(0), "info", "Simulation reset by operator. Incident clock returned to start."))


def advance_step():
    ss = st.session_state
    env = ss.env
    if env.t >= env.max_steps:
        ss.playing = False
        ss.logs.append((clock(env.t), "info", "Mission window complete. Reset or lower the tick to continue."))
        return

    _, reward, term, trunc, info = env.step()
    ss.step = env.t
    ss.total_reward += reward
    ss.prev_metrics = ss.metrics
    ss.metrics = {k: float(info.get(k, 0)) for k in ("total_rescued", "blocked_nodes_count", "equity_variance")}

    d_res = int(ss.metrics["total_rescued"] - ss.prev_metrics["total_rescued"])
    d_blk = int(ss.metrics["blocked_nodes_count"] - ss.prev_metrics["blocked_nodes_count"])

    if d_blk > 0:
        ss.logs.append((clock(env.t), "warn", f"{d_blk} additional road nodes impassable. Routing agent re-planning."))
    new_alert, _ = alert_level(float(env.water_depths.max()))
    old_alert = ss.get("alert_name", "NORMAL")
    if new_alert != old_alert:
        up = ALERT_RANK[new_alert] > ALERT_RANK[old_alert]
        lvl = "crit" if (up and new_alert == "SEVERE") else "warn" if up else "info"
        verb = "raised" if up else "lowered"
        ss.logs.append((clock(env.t), lvl, f"Flood alert level {verb} to {new_alert} (peak depth {env.water_depths.max():.2f} m)."))
        ss.alert_name = new_alert

    # Shelter occupancy follows simulated arrivals (capped at capacity)
    if ss.get("auto_occ", True):
        for node, gain in env.arrivals:
            sh = next((s for s in ss.shelters if s["node"] == node), None)
            if sh is None:
                continue
            admitted = min(gain, max(0, sh["capacity"] - sh["occupancy"]))
            sh["occupancy"] += admitted
            if admitted:
                ss.logs.append((clock(env.t), "ok", f"{admitted} persons received at {sh['id']} {sh['name']} "
                                                   f"({sh['occupancy']}/{sh['capacity']})."))
            if admitted < gain:
                ss.logs.append((clock(env.t), "warn",
                                f"{sh['id']} {sh['name']} at capacity: {gain - admitted} persons could not be admitted."))
        if env.arrivals:
            ss.sh_loaded = None  # refresh the shelter edit form with the new occupancy
    elif env.arrivals:
        for node, gain in env.arrivals:
            ss.logs.append((clock(env.t), "ok", f"{gain} persons delivered to shelter at node {node}."))

    ss.history.append({
        "step": env.t,
        "total_rescued": ss.metrics["total_rescued"],
        "blocked_nodes_count": ss.metrics["blocked_nodes_count"],
        "reward": reward
    })
    ss.kpi_dirty = True
    if term or trunc:
        ss.playing = False
        ss.logs.append((clock(env.t), "info", "Mission objective completed successfully."))


# ──────────────────────────────────────────────────────────────────────────────
# Callbacks: manual KPI editing
# ──────────────────────────────────────────────────────────────────────────────
def sync_kpi_inputs():
    ss = st.session_state
    ss["kpi_in_rescued"] = int(ss.metrics["total_rescued"])
    ss["kpi_in_blocked"] = int(ss.metrics["blocked_nodes_count"])
    ss["kpi_in_tick"] = int(ss.env.t)


def _sync_last_history_point():
    ss = st.session_state
    if ss.history:
        ss.history[-1]["total_rescued"] = ss.metrics["total_rescued"]
        ss.history[-1]["blocked_nodes_count"] = ss.metrics["blocked_nodes_count"]


def apply_kpi_edits():
    ss = st.session_state
    env = ss.env
    new_res = max(0, _i(ss.kpi_in_rescued))
    new_blk = min(env.num_nodes, max(0, _i(ss.kpi_in_blocked)))
    new_tick = min(env.max_steps, max(0, _i(ss.kpi_in_tick)))

    old_res = int(ss.metrics["total_rescued"])
    old_blk = int(ss.metrics["blocked_nodes_count"])
    m = dict(ss.metrics)
    changes = []

    if new_tick != env.t:
        env.t = new_tick
        env.update_water()                       # map + peak-depth KPI follow the new tick
        ss.step = new_tick
        ss.alert_name = alert_level(float(env.water_depths.max()))[0]
        ss.history = [h for h in ss.history if h["step"] <= new_tick]
        if new_tick >= env.max_steps:
            ss.playing = False
        eff = env.blocked_override if env.blocked_override is not None else env.blocked_nodes()
        m["blocked_nodes_count"] = float(eff)
        changes.append(f"tick → {new_tick}")
    if new_res != old_res:
        env.total_rescued = new_res
        m["total_rescued"] = float(new_res)
        changes.append(f"rescued → {new_res:,}")
    if new_blk != old_blk:
        env.blocked_override = new_blk           # held until released or the sim is reset
        m["blocked_nodes_count"] = float(new_blk)
        changes.append(f"blocked nodes → {new_blk:,}")

    ss.metrics = m
    ss.prev_metrics = dict(m)                    # show +0 "this tick" after a manual edit
    _sync_last_history_point()
    ss.kpi_dirty = True
    if changes:
        ss.logs.append((clock(env.t), "warn", "Operator override applied: " + ", ".join(changes) + "."))
        flash("kpi", "success", "Applied: " + ", ".join(changes))
    else:
        flash("kpi", "warning", "No changes to apply.")


def release_blocked_override():
    ss = st.session_state
    env = ss.env
    env.blocked_override = None
    m = dict(ss.metrics)
    m["blocked_nodes_count"] = float(env.blocked_nodes())
    ss.metrics = m
    ss.prev_metrics = dict(m)
    _sync_last_history_point()
    ss.kpi_dirty = True
    ss.logs.append((clock(env.t), "info", "Road-status override released; flood model in control."))
    flash("kpi", "success", "Override released: blocked nodes follow the flood model again.")


# ──────────────────────────────────────────────────────────────────────────────
# Callbacks: shelters
# ──────────────────────────────────────────────────────────────────────────────
def add_shelter():
    ss = st.session_state
    name = str(ss.sh_c_name).strip()
    row, col, cap, occ = _i(ss.sh_c_row), _i(ss.sh_c_col), _i(ss.sh_c_cap), _i(ss.sh_c_occ)
    errs = validate_shelter(name, row, col, cap, occ)
    if errs:
        flash("shelter", "error", " ".join(errs))
        return
    sid = f"SH-{ss.next_shelter_n:02d}"
    ss.next_shelter_n += 1
    ss.shelters.append({"id": sid, "name": name, "node": rc_to_node(row, col), "capacity": cap, "occupancy": occ})
    sync_env_shelters()
    ss["sh_c_name"], ss["sh_c_row"], ss["sh_c_col"], ss["sh_c_cap"], ss["sh_c_occ"] = "", 0, 0, 300, 0
    ss.logs.append((clock(ss.env.t), "info", f"Shelter {sid} '{name}' registered (capacity {cap})."))
    flash("shelter", "success", f"Shelter {sid} '{name}' created.")


def save_shelter():
    ss = st.session_state
    s = get_shelter(ss.get("sh_sel"))
    if s is None:
        flash("shelter", "error", "Select a shelter first.")
        return
    name = str(ss.sh_e_name).strip()
    row, col, cap, occ = _i(ss.sh_e_row), _i(ss.sh_e_col), _i(ss.sh_e_cap), _i(ss.sh_e_occ)
    errs = validate_shelter(name, row, col, cap, occ, exclude_id=s["id"])
    if errs:
        flash("shelter", "error", f"{s['id']} not saved. " + " ".join(errs))
        return
    s.update(name=name, node=rc_to_node(row, col), capacity=cap, occupancy=occ)
    sync_env_shelters()
    ss.logs.append((clock(ss.env.t), "info", f"Shelter {s['id']} updated ({occ}/{cap})."))
    flash("shelter", "success", f"Shelter {s['id']} saved.")


def revert_shelter_form():
    st.session_state.sh_loaded = None


def delete_shelter():
    ss = st.session_state
    sid = ss.get("sh_sel")
    s = get_shelter(sid)
    if s is None:
        return
    if not ss.get("sh_del_ok"):
        flash("shelter", "warning", "Tick 'Confirm delete' first.")
        return
    ss.shelters = [x for x in ss.shelters if x["id"] != sid]
    sync_env_shelters()
    ss["sh_del_ok"] = False
    ss.logs.append((clock(ss.env.t), "warn", f"Shelter {sid} '{s['name']}' removed from registry."))
    flash("shelter", "success", f"Shelter {sid} deleted.")


# ──────────────────────────────────────────────────────────────────────────────
# Callbacks: rescue teams
# ──────────────────────────────────────────────────────────────────────────────
def add_team():
    ss = st.session_state
    name = str(ss.tm_c_name).strip()
    row, col, members, status = _i(ss.tm_c_row), _i(ss.tm_c_col), _i(ss.tm_c_members), ss.tm_c_status
    errs = validate_team(name, row, col, members, status)
    if errs:
        flash("team", "error", " ".join(errs))
        return
    tid = f"RT-{ss.next_team_n:02d}"
    ss.next_team_n += 1
    ss.teams.append({"id": tid, "name": name, "node": rc_to_node(row, col), "members": members, "status": status})
    ss["tm_c_name"], ss["tm_c_row"], ss["tm_c_col"], ss["tm_c_members"], ss["tm_c_status"] = "", 0, 0, 5, "Available"
    ss.logs.append((clock(ss.env.t), "info", f"Rescue team {tid} '{name}' registered ({members} members)."))
    flash("team", "success", f"Team {tid} '{name}' created.")


def save_team():
    ss = st.session_state
    t = get_team(ss.get("tm_sel"))
    if t is None:
        flash("team", "error", "Select a team first.")
        return
    name = str(ss.tm_e_name).strip()
    row, col, members, status = _i(ss.tm_e_row), _i(ss.tm_e_col), _i(ss.tm_e_members), ss.tm_e_status
    errs = validate_team(name, row, col, members, status)
    if errs:
        flash("team", "error", f"{t['id']} not saved. " + " ".join(errs))
        return
    t.update(name=name, node=rc_to_node(row, col), members=members, status=status)
    ss.logs.append((clock(ss.env.t), "info", f"Team {t['id']} updated: {status}, {members} members."))
    flash("team", "success", f"Team {t['id']} saved.")


def revert_team_form():
    st.session_state.tm_loaded = None


def delete_team():
    ss = st.session_state
    tid = ss.get("tm_sel")
    t = get_team(tid)
    if t is None:
        return
    if not ss.get("tm_del_ok"):
        flash("team", "warning", "Tick 'Confirm delete' first.")
        return
    ss.teams = [x for x in ss.teams if x["id"] != tid]
    ss["tm_del_ok"] = False
    ss.logs.append((clock(ss.env.t), "warn", f"Rescue team {tid} '{t['name']}' removed from registry."))
    flash("team", "success", f"Team {tid} deleted.")


def prime_shelter_form():
    """Load the selected shelter into the edit widgets (before they are instantiated)."""
    ss = st.session_state
    s = get_shelter(ss.get("sh_sel"))
    if s is None:
        return
    if ss.get("sh_loaded") != s["id"] or "sh_e_name" not in ss:
        r, c = node_to_rc(s["node"])
        ss["sh_e_name"], ss["sh_e_row"], ss["sh_e_col"] = s["name"], r, c
        ss["sh_e_cap"], ss["sh_e_occ"] = s["capacity"], s["occupancy"]
        ss["sh_loaded"] = s["id"]


def prime_team_form():
    ss = st.session_state
    t = get_team(ss.get("tm_sel"))
    if t is None:
        return
    if ss.get("tm_loaded") != t["id"] or "tm_e_name" not in ss:
        r, c = node_to_rc(t["node"])
        ss["tm_e_name"], ss["tm_e_row"], ss["tm_e_col"] = t["name"], r, c
        ss["tm_e_members"], ss["tm_e_status"] = t["members"], t["status"]
        ss["tm_loaded"] = t["id"]


# Defaults for widget-backed state (no `value=` args, so no session-state conflicts)
for _k, _v in {
    "sh_c_name": "", "sh_c_row": 0, "sh_c_col": 0, "sh_c_cap": 300, "sh_c_occ": 0,
    "tm_c_name": "", "tm_c_row": 0, "tm_c_col": 0, "tm_c_members": 5, "tm_c_status": "Available",
}.items():
    st.session_state.setdefault(_k, _v)

if st.session_state.get("kpi_dirty", True) or "kpi_in_tick" not in st.session_state:
    sync_kpi_inputs()
    st.session_state.kpi_dirty = False


# ──────────────────────────────────────────────────────────────────────────────
# Sidebar Controls
# ──────────────────────────────────────────────────────────────────────────────
def side_h(text):
    st.markdown(f'<div class="side-h">{html.escape(text)}</div>', unsafe_allow_html=True)


with st.sidebar:
    st.markdown(
        '<div class="brand"><div class="brand-mark">FO</div><div><div class="brand-t">FLOOD-OPS</div>'
        '<div class="brand-s">Evacuation command &amp; control</div></div></div>',
        unsafe_allow_html=True,
    )
    side_h("Simulation control")
    c1, c2 = st.columns(2)
    if c1.button("▶ Play", use_container_width=True):
        st.session_state.playing = True
    if c2.button("⏸ Pause", use_container_width=True):
        st.session_state.playing = False

    c3, c4 = st.columns(2)
    if c3.button("⟲ Reset", use_container_width=True):
        reset_session()
        st.rerun()
    if c4.button("⏭ Step", use_container_width=True):
        st.session_state.playing = False
        advance_step()
        st.rerun()

    st.slider("Tick interval (s)", 0.05, 2.0, 0.6, 0.05, key="tick")
    st.number_input("Scenario Seed", 0, 9999, 42, key="seed")

    side_h("Manual overrides")
    with st.expander("Edit incident figures", expanded=False):
        st.caption("Pause playback before editing. Fields refresh from live values on every tick.")
        _env_ = st.session_state.env
        with st.form("kpi_form"):
            st.number_input("Persons evacuated", min_value=0, step=1, key="kpi_in_rescued")
            st.number_input("Impassable road nodes", min_value=0, max_value=_env_.num_nodes, step=1, key="kpi_in_blocked")
            st.number_input("Simulation tick", min_value=0, max_value=_env_.max_steps, step=1, key="kpi_in_tick")
            st.form_submit_button("Apply to dashboard", on_click=apply_kpi_edits, use_container_width=True)
        if _env_.blocked_override is not None:
            st.warning(f"Impassable nodes pinned to {_env_.blocked_override:,} (manual).")
            st.button("Release road-status override", on_click=release_blocked_override, use_container_width=True)
        show_flash("kpi")

    side_h("Shelter link")
    st.checkbox("Sim updates shelter occupancy", True, key="auto_occ")
    st.checkbox("Clear occupancy on reset", True, key="clear_occ")

    side_h("Map layers")
    basemap = st.selectbox("Basemap", ["Dark", "Light", "Streets"], key="basemap")
    show_base = st.checkbox("Road network", True)
    show_flood = st.checkbox("Flood extent", True)
    show_shelters = st.checkbox("Shelters", True)
    show_teams = st.checkbox("Rescue teams", True)
    show_fleet = st.checkbox("Response units", True)
    show_arcs = st.checkbox("Routing vectors", True)
    show_labels = st.checkbox("Marker labels", True)
    pitch = st.slider("Map pitch", 0, 60, 30)


# ──────────────────────────────────────────────────────────────────────────────
# Main Header & KPIs
# ──────────────────────────────────────────────────────────────────────────────
env = st.session_state.env
peak_depth = float(env.water_depths.max()) if env.water_depths.size else 0.0
alert_name, alert_cls = alert_level(peak_depth)
status_cls = "live" if st.session_state.playing else "hold"
status_txt = "● Live" if st.session_state.playing else "Standby"
prog_pct = 100.0 * st.session_state.step / max(1, env.max_steps)

st.markdown(
    f"""<div class="eoc-head">
      <div class="eoc-left">
        <div class="eoc-badge">EOC</div>
        <div>
          <div class="eoc-title">Flood Emergency Operations Center</div>
          <div class="eoc-sub">Bengaluru Urban District · Incident FLD-BLR-2026-014 · Evacuation routing (PPO agent) · <b>EXERCISE / SIMULATION</b></div>
        </div>
      </div>
      <div class="eoc-right">
        <div class="eoc-meta"><span class="l">Incident time</span><span class="v">{clock(st.session_state.step)}</span></div>
        <div class="eoc-meta"><span class="l">Operational period</span><span class="v">{st.session_state.step:02d} / {env.max_steps}</span></div>
        <span class="alert {alert_cls}">FLOOD ALERT · {alert_name}</span>
        <span class="pill {status_cls}">{status_txt}</span>
      </div>
    </div>
    <div class="prog"><div style="width:{prog_pct:.1f}%"></div></div>""",
    unsafe_allow_html=True,
)

def kpi(label, value, sub="", color="blue"):
    return f'<div class="kpi {color}"><div class="kpi-l">{label}</div><div class="kpi-v">{value}</div><div class="kpi-s">{sub}</div></div>'

m, pm = st.session_state.metrics, st.session_state.prev_metrics
blk_sub = f"{int(m['blocked_nodes_count'] - pm['blocked_nodes_count']):+d} this tick"
if env.blocked_override is not None:
    blk_sub += " · manual"
depth_color = {"normal": "blue", "watch": "amber", "warning": "orange", "severe": "red"}[alert_cls]

k = st.columns(6)
k[0].markdown(kpi("Persons evacuated", f"{int(m['total_rescued']):,}", f"+{int(m['total_rescued'] - pm['total_rescued'])} this tick", "green"), unsafe_allow_html=True)
k[1].markdown(kpi("Impassable road nodes", f"{int(m['blocked_nodes_count']):,}", blk_sub, "red"), unsafe_allow_html=True)
k[2].markdown(kpi("Equity variance", f"{m['equity_variance']:.3f}", "lower = fairer coverage", "amber"), unsafe_allow_html=True)
k[3].markdown(kpi("Peak water depth", f"{peak_depth:.2f} m", "severe at 0.40 m", depth_color), unsafe_allow_html=True)
k[4].markdown(kpi("Response units", f"{len(env.vehicle_positions)}", "AI-routed, deployed", "green"), unsafe_allow_html=True)
k[5].markdown(kpi("Agent reward (cum.)", f"{st.session_state.total_reward:,.1f}", f"tick {st.session_state.step}", "blue"), unsafe_allow_html=True)

# Second KPI row: shelters & rescue teams
_cap, _occ, _rem, _pct = shelter_totals()
_teams = st.session_state.teams
_active_teams = [t for t in _teams if t["status"] != "Offline"]
_tot_members = sum(t["members"] for t in _teams)
_act_members = sum(t["members"] for t in _active_teams)

st.write("")
k2 = st.columns(4)
k2[0].markdown(kpi("Total shelters", f"{len(st.session_state.shelters)}", f"{_cap:,} total capacity", "blue"), unsafe_allow_html=True)
k2[1].markdown(kpi("Shelter occupancy", f"{_pct:.1f}%", f"{_occ:,} housed · {_rem:,} free",
                   "red" if _pct >= 90 else "amber" if _pct >= 70 else "green"), unsafe_allow_html=True)
k2[2].markdown(kpi("Active rescue teams", f"{len(_active_teams)}", f"{len(_teams)} registered", "green"), unsafe_allow_html=True)
k2[3].markdown(kpi("Rescue personnel", f"{_tot_members:,}", f"{_act_members:,} on active teams", "blue"), unsafe_allow_html=True)
st.write("")


# ──────────────────────────────────────────────────────────────────────────────
# PyDeck Map Rendering (basemap selectable: dark / light / streets)
# ──────────────────────────────────────────────────────────────────────────────
BASEMAPS = {"Dark": "dark", "Light": "light", "Streets": "road"}
FONT = dict(family="Inter, system-ui, sans-serif", color="#8091a5", size=11)


def build_deck() -> pdk.Deck:
    layers = []
    coords = env.coords
    lat, lon = coords[:, 0], coords[:, 1]
    shelters = st.session_state.shelters
    teams = st.session_state.teams
    depth = env.water_depths

    # 1. Flood extent (translucent, under the road network)
    if show_flood and depth.size > 0:
        idx = np.where(depth > 0.05)[0]
        if idx.size > 0:
            sub_d = depth[idx]
            sev = sub_d > 0.4
            flood_df = pd.DataFrame({
                "lat": lat[idx], "lon": lon[idx], "radius": sub_d * 350,
                "r": np.where(sev, 229, 76), "g": np.where(sev, 72, 154), "b": np.where(sev, 77, 255),
            })
            layers.append(pdk.Layer(
                "ScatterplotLayer", data=flood_df, get_position=["lon", "lat"],
                get_radius="radius", radius_min_pixels=3, get_fill_color="[r, g, b, 70]",
                pickable=False, stroked=False,
            ))

    # 2. Road network: clear / flooded / impassable segments
    if show_base:
        src, dst = st.session_state.edges
        de = np.maximum(depth[src], depth[dst])
        blocked = de > 0.4
        flooded = (de > 0.12) & ~blocked
        roads_df = pd.DataFrame({
            "s_lon": lon[src], "s_lat": lat[src], "t_lon": lon[dst], "t_lat": lat[dst],
            "r": np.where(blocked, 229, np.where(flooded, 76, 120)),
            "g": np.where(blocked, 72, np.where(flooded, 154, 140)),
            "b": np.where(blocked, 77, np.where(flooded, 255, 165)),
            "a": np.where(blocked, 245, np.where(flooded, 200, 80)),
            "w": np.where(blocked, 3.5, np.where(flooded, 2.5, 1.5)),
        })
        layers.append(pdk.Layer(
            "LineLayer", data=roads_df, get_source_position=["s_lon", "s_lat"],
            get_target_position=["t_lon", "t_lat"], get_color="[r, g, b, a]",
            get_width="w", width_min_pixels=1, pickable=False,
        ))

    # 3. Shelters: amber dot + capacity-coloured halo ring (green <70%, amber <90%, red >=90%)
    shelter_df = None
    if shelters:
        pts = coords[[s["node"] for s in shelters]]
        pcts = [shelter_pct(s) for s in shelters]
        ring = [(47, 191, 113) if p < 70 else (242, 169, 59) if p < 90 else (229, 72, 77) for p in pcts]
        shelter_df = pd.DataFrame({
            "lat": pts[:, 0], "lon": pts[:, 1],
            "label": [f"{s['id']} {p:.0f}%" for s, p in zip(shelters, pcts)],
            "tip": [f"{s['id']} · {s['name']} | {s['occupancy']}/{s['capacity']} ({p:.0f}%) | free {s['capacity'] - s['occupancy']}"
                    for s, p in zip(shelters, pcts)],
            "r": [c[0] for c in ring], "g": [c[1] for c in ring], "b": [c[2] for c in ring],
        })
    if show_shelters and shelter_df is not None:
        layers.append(pdk.Layer(
            "ScatterplotLayer", data=shelter_df, get_position=["lon", "lat"],
            get_radius=420, radius_min_pixels=14, filled=False, stroked=True,
            get_line_color="[r, g, b, 255]", line_width_min_pixels=3, pickable=False,
        ))
        layers.append(pdk.Layer(
            "ScatterplotLayer", data=shelter_df, get_position=["lon", "lat"],
            get_radius=220, radius_min_pixels=7, get_fill_color=[242, 169, 59, 255],
            get_line_color=[255, 255, 255, 220], stroked=True, line_width_min_pixels=2, pickable=True,
        ))

    # 4. Routing vectors (arcs from T-1 to T)
    if show_arcs and len(env.vehicle_positions) > 0:
        curr_pts = coords[env.vehicle_positions]
        prev_pts = coords[env.prev_positions]
        arcs_df = pd.DataFrame({
            "s_lat": prev_pts[:, 0], "s_lon": prev_pts[:, 1],
            "t_lat": curr_pts[:, 0], "t_lon": curr_pts[:, 1]
        })
        layers.append(pdk.Layer(
            "ArcLayer", data=arcs_df, get_source_position=["s_lon", "s_lat"],
            get_target_position=["t_lon", "t_lat"], get_source_color=[47, 191, 113, 70],
            get_target_color=[47, 191, 113, 240], get_width=3, width_min_pixels=1, pickable=False,
        ))

    # 5. AI-routed response units
    if show_fleet and len(env.vehicle_positions) > 0:
        f_coords = coords[env.vehicle_positions]
        fleet_df = pd.DataFrame({
            "lat": f_coords[:, 0], "lon": f_coords[:, 1],
            "tip": [f"UNIT-{v + 1:02d} | node {int(p)} | depth {depth[p]:.2f} m" for v, p in enumerate(env.vehicle_positions)],
        })
        layers.append(pdk.Layer(
            "ScatterplotLayer", data=fleet_df, get_position=["lon", "lat"],
            get_radius=250, radius_min_pixels=6, get_fill_color=[47, 191, 113, 255],
            get_line_color=[6, 16, 28, 255], stroked=True, line_width_min_pixels=1.5, pickable=True,
        ))

    # 6. Rescue teams: violet dot with blue outline (grey when offline)
    team_df = None
    if teams:
        pts = coords[[t["node"] for t in teams]]
        team_df = pd.DataFrame({
            "lat": pts[:, 0], "lon": pts[:, 1],
            "label": [t["id"] for t in teams],
            "tip": [f"{t['id']} · {t['name']} | {t['status']} | {t['members']} members" for t in teams],
            "r": [128 if t["status"] == "Offline" else 168 for t in teams],
            "g": [145 if t["status"] == "Offline" else 85 for t in teams],
            "b": [165 if t["status"] == "Offline" else 247 for t in teams],
        })
    if show_teams and team_df is not None:
        layers.append(pdk.Layer(
            "ScatterplotLayer", data=team_df, get_position=["lon", "lat"],
            get_radius=170, radius_min_pixels=6, get_fill_color="[r, g, b, 255]",
            get_line_color=[76, 154, 255, 255], stroked=True, line_width_min_pixels=2, pickable=True,
        ))

    # 7. Labels
    if show_labels:
        if show_shelters and shelter_df is not None:
            layers.append(pdk.Layer(
                "TextLayer", data=shelter_df, get_position=["lon", "lat"], get_text="label",
                get_size=12, get_color=[250, 205, 120, 255], get_pixel_offset=[0, -24],
                get_text_anchor='"middle"', get_alignment_baseline='"bottom"', pickable=False,
            ))
        if show_teams and team_df is not None:
            layers.append(pdk.Layer(
                "TextLayer", data=team_df, get_position=["lon", "lat"], get_text="label",
                get_size=11, get_color=[210, 175, 255, 255], get_pixel_offset=[0, 14],
                get_text_anchor='"middle"', get_alignment_baseline='"top"', pickable=False,
            ))

    return pdk.Deck(
        layers=layers,
        initial_view_state=pdk.ViewState(longitude=77.5946, latitude=12.9716, zoom=11.2, pitch=pitch, bearing=-10),
        map_style=BASEMAPS.get(basemap, "dark"),
        tooltip={
            "text": "{tip}",
            "style": {"backgroundColor": "#121820", "color": "#d6dee8", "fontSize": "12px",
                      "border": "1px solid #243040", "borderRadius": "4px"},
        },
    )


# ──────────────────────────────────────────────────────────────────────────────
# Telemetry & Event Feed UI Layout
# ──────────────────────────────────────────────────────────────────────────────
def build_chart() -> go.Figure:
    h = pd.DataFrame(st.session_state.history)
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    if not h.empty:
        fig.add_trace(go.Scatter(x=h["step"], y=h["total_rescued"], name="Persons evacuated", mode="lines",
                                 line=dict(color="#2fbf71", width=2.5), fill="tozeroy",
                                 fillcolor="rgba(47,191,113,0.10)"), secondary_y=False)
        fig.add_trace(go.Scatter(x=h["step"], y=h["blocked_nodes_count"], name="Impassable nodes", mode="lines",
                                 line=dict(color="#e5484d", width=2, dash="dot")), secondary_y=True)
    fig.update_layout(
        template="plotly_dark", height=280, margin=dict(l=0, r=0, t=28, b=0), font=FONT,
        plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
        legend=dict(orientation="h", yanchor="bottom", y=1.0, xanchor="left", x=0.0),
        xaxis=dict(title="Tick", gridcolor="#1b2531", zeroline=False),
        yaxis=dict(title="Evacuated", gridcolor="#1b2531", zeroline=False),
        yaxis2=dict(title="Impassable", overlaying="y", side="right", showgrid=False, zeroline=False),
    )
    return fig


def build_capacity_chart() -> go.Figure:
    sh = st.session_state.shelters
    fig = go.Figure()
    if sh:
        labels = [f"{s['id']} · {s['name']} ({shelter_pct(s):.0f}%)" for s in sh]
        colors = ["#2fbf71" if shelter_pct(s) < 70 else "#f2a93b" if shelter_pct(s) < 90 else "#e5484d" for s in sh]
        fig.add_trace(go.Bar(
            y=labels, x=[s["occupancy"] for s in sh], orientation="h", name="Occupied",
            marker_color=colors, hovertemplate="%{y}<br>Occupied: %{x:,}<extra></extra>",
        ))
        fig.add_trace(go.Bar(
            y=labels, x=[s["capacity"] - s["occupancy"] for s in sh], orientation="h", name="Remaining",
            marker_color="#243040", hovertemplate="%{y}<br>Remaining: %{x:,}<extra></extra>",
        ))
        fig.update_layout(barmode="stack")
    else:
        fig.add_annotation(text="No shelters registered", showarrow=False, font=dict(color="#8091a5"))
    fig.update_layout(
        template="plotly_dark", height=max(220, 80 + 42 * len(sh)), margin=dict(l=0, r=0, t=28, b=0), font=FONT,
        xaxis=dict(title="Beds", gridcolor="#1b2531", zeroline=False), yaxis=dict(autorange="reversed"),
        plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
        legend=dict(orientation="h", yanchor="bottom", y=1.0, xanchor="left", x=0.0),
    )
    return fig


def render_terminal():
    labels = {"info": "INFO", "ok": "OK", "warn": "WARN", "crit": "CRIT"}
    rows = []
    for t_, lvl, msg in st.session_state.logs:
        rows.append(
            f'<div class="ev {lvl}"><span class="t">{t_}</span>'
            f'<span class="sev {lvl}">{labels.get(lvl, lvl.upper())}</span>'
            f'<span class="m">{html.escape(msg)}</span></div>'
        )
    st.markdown(f'<div class="feed">{"".join(rows)}</div>', unsafe_allow_html=True)


def shelter_registry_df() -> pd.DataFrame:
    rows = []
    for s in st.session_state.shelters:
        r, c = node_to_rc(s["node"])
        pt = env.coords[s["node"]]
        rem, pct = s["capacity"] - s["occupancy"], shelter_pct(s)
        rows.append({
            "ID": s["id"], "Name": s["name"], "Row": r, "Col": c, "Node": s["node"],
            "Latitude": round(float(pt[0]), 5), "Longitude": round(float(pt[1]), 5),
            "Capacity": s["capacity"], "Occupancy": s["occupancy"], "Remaining": rem,
            "Occupancy %": round(pct, 1),
            "Status": "FULL" if rem <= 0 else "NEAR FULL" if pct >= 90 else "OPEN",
        })
    return pd.DataFrame(rows)


def team_registry_df() -> pd.DataFrame:
    rows = []
    for t in st.session_state.teams:
        r, c = node_to_rc(t["node"])
        pt = env.coords[t["node"]]
        rows.append({
            "ID": t["id"], "Name": t["name"], "Row": r, "Col": c, "Node": t["node"],
            "Latitude": round(float(pt[0]), 5), "Longitude": round(float(pt[1]), 5),
            "Members": t["members"], "Status": t["status"],
        })
    return pd.DataFrame(rows)


# Main Grid Layout
LEGEND = """<div class="legend">
<span><i class="dot" style="background:#f2a93b;box-shadow:0 0 0 3px rgba(47,191,113,.5)"></i>Shelter (ring = fill level)</span>
<span><i class="dot" style="background:#a855f7;border:2px solid #4c9aff"></i>Rescue team</span>
<span><i class="dot" style="background:#2fbf71"></i>Response unit</span>
<span><i class="ln" style="background:#4c9aff"></i>Flooded road</span>
<span><i class="ln" style="background:#e5484d"></i>Impassable road</span></div>"""

left_col, right_col = st.columns([3, 2], gap="medium")

with left_col:
    st.markdown('<div class="panel-t">Common operating picture · Bengaluru Urban</div>', unsafe_allow_html=True)
    st.pydeck_chart(build_deck(), height=680)
    st.markdown(LEGEND, unsafe_allow_html=True)

with right_col:
    st.markdown('<div class="panel-t">Evacuation &amp; road status</div>', unsafe_allow_html=True)
    st.plotly_chart(build_chart(), use_container_width=True, config={"displayModeBar": False})
    st.markdown('<div class="panel-t">Incident event feed</div>', unsafe_allow_html=True)
    render_terminal()

st.write("")
# Tabs for Bottom Data inspection
tab_w, tab_f, tab_s, tab_t = st.tabs(["PPO action weights", "Fleet telemetry", "Shelters", "Rescue teams"])

with tab_w:
    mock_weights = pd.DataFrame({
        "Head": [0, 1, 2, 3, 4, 5, 6, 7],
        "Action": [12, 45, 88, 23, 91, 14, 56, 72],
        "Weight": [0.892, 0.741, 0.954, 0.621, 0.883, 0.792, 0.912, 0.845],
        "Selected": [True, True, True, False, True, False, True, True]
    })
    st.dataframe(mock_weights, hide_index=True, use_container_width=True,
                 column_config={"Weight": st.column_config.ProgressColumn("Confidence", min_value=0.0, max_value=1.0, format="%.3f")})

with tab_f:
    fleet_rows = []
    for v, pos in enumerate(env.vehicle_positions):
        pt = env.coords[pos]
        fleet_rows.append({
            "Vehicle ID": f"UNIT-{v+1:02d}",
            "Node ID": int(pos),
            "Latitude": round(float(pt[0]), 5),
            "Longitude": round(float(pt[1]), 5),
            "Water Depth (m)": round(float(env.water_depths[pos]), 2),
            "Status": "Rerouting" if env.water_depths[pos] > 0.3 else "Nominal"
        })
    st.dataframe(style_status(pd.DataFrame(fleet_rows), {"Rerouting": "#f2a93b", "Nominal": "#2fbf71"}),
                 hide_index=True, use_container_width=True)

# ── Shelter management ────────────────────────────────────────────────────────
with tab_s:
    side = env.side
    st.caption(f"Grid location = (row, column), each 0–{side - 1}. Pause playback while editing.")
    show_flash("shelter")
    sc_a, sc_b = st.columns([2, 3], gap="medium")

    with sc_a:
        with st.expander("Create shelter", expanded=False):
            with st.form("sh_create_form"):
                st.text_input("Name", key="sh_c_name")
                g1, g2 = st.columns(2)
                g1.number_input("Grid row", min_value=0, max_value=side - 1, step=1, key="sh_c_row")
                g2.number_input("Grid column", min_value=0, max_value=side - 1, step=1, key="sh_c_col")
                g3, g4 = st.columns(2)
                g3.number_input("Capacity", min_value=0, max_value=1_000_000, step=10, key="sh_c_cap")
                g4.number_input("Occupancy", min_value=0, max_value=1_000_000, step=10, key="sh_c_occ")
                st.form_submit_button("Create shelter", on_click=add_shelter, use_container_width=True)

        with st.expander("Edit / delete shelter", expanded=False):
            shelters = st.session_state.shelters
            if shelters:
                ids = [s["id"] for s in shelters]
                names = {s["id"]: s["name"] for s in shelters}
                if st.session_state.get("sh_sel") not in ids:
                    st.session_state["sh_sel"] = ids[0]
                st.selectbox("Shelter", ids, key="sh_sel", format_func=lambda i: f"{i} · {names[i]}")
                prime_shelter_form()
                _s = get_shelter(st.session_state.sh_sel)
                st.caption(f"Stored: remaining capacity **{_s['capacity'] - _s['occupancy']:,}** · "
                           f"occupancy **{shelter_pct(_s):.1f}%**")
                with st.form("sh_edit_form"):
                    st.text_input("Name", key="sh_e_name")
                    e1, e2 = st.columns(2)
                    e1.number_input("Grid row", min_value=0, max_value=side - 1, step=1, key="sh_e_row")
                    e2.number_input("Grid column", min_value=0, max_value=side - 1, step=1, key="sh_e_col")
                    e3, e4 = st.columns(2)
                    e3.number_input("Capacity", min_value=0, max_value=1_000_000, step=10, key="sh_e_cap")
                    e4.number_input("Occupancy", min_value=0, max_value=1_000_000, step=10, key="sh_e_occ")
                    st.form_submit_button("Save changes", on_click=save_shelter, use_container_width=True)
                st.button("Revert fields", on_click=revert_shelter_form, key="sh_revert", use_container_width=True)
                d1, d2 = st.columns([2, 1])
                d1.checkbox("Confirm delete", key="sh_del_ok")
                d2.button("Delete", on_click=delete_shelter, key="sh_delete", use_container_width=True)
            else:
                st.info("No shelters registered. Create one above.")

    with sc_b:
        st.markdown('<div class="panel-t">Shelter capacity</div>', unsafe_allow_html=True)
        st.plotly_chart(build_capacity_chart(), use_container_width=True, config={"displayModeBar": False})

    st.markdown('<div class="panel-t">Shelter registry</div>', unsafe_allow_html=True)
    sdf = shelter_registry_df()
    if sdf.empty:
        st.info("Registry is empty.")
    else:
        _cnt = sdf["Status"].value_counts()
        st.markdown('<div class="chips">' + "".join(chip_html(k_, int(_cnt.get(k_, 0)), c_) for k_, c_ in SHELTER_COLORS.items()) + '</div>',
                    unsafe_allow_html=True)
        st.dataframe(
            style_status(sdf, SHELTER_COLORS), hide_index=True, use_container_width=True,
            column_config={
                "Occupancy %": st.column_config.ProgressColumn("Occupancy %", min_value=0.0, max_value=100.0, format="%.1f%%"),
                "Capacity": st.column_config.NumberColumn(format="%d"),
                "Occupancy": st.column_config.NumberColumn(format="%d"),
                "Remaining": st.column_config.NumberColumn(format="%d"),
            },
        )

# ── Rescue team management ───────────────────────────────────────────────────
with tab_t:
    side = env.side
    st.caption(f"Grid location = (row, column), each 0–{side - 1}. Teams are a registry layer; they do not alter the PPO fleet.")
    show_flash("team")
    tc_a, tc_b = st.columns([2, 3], gap="medium")

    with tc_a:
        with st.expander("Create rescue team", expanded=False):
            with st.form("tm_create_form"):
                st.text_input("Name", key="tm_c_name")
                h1, h2 = st.columns(2)
                h1.number_input("Grid row", min_value=0, max_value=side - 1, step=1, key="tm_c_row")
                h2.number_input("Grid column", min_value=0, max_value=side - 1, step=1, key="tm_c_col")
                h3, h4 = st.columns(2)
                h3.number_input("Members", min_value=1, max_value=500, step=1, key="tm_c_members")
                h4.selectbox("Status", TEAM_STATUSES, key="tm_c_status")
                st.form_submit_button("Create team", on_click=add_team, use_container_width=True)

        with st.expander("Edit / delete team", expanded=False):
            teams = st.session_state.teams
            if teams:
                tids = [t["id"] for t in teams]
                tnames = {t["id"]: t["name"] for t in teams}
                if st.session_state.get("tm_sel") not in tids:
                    st.session_state["tm_sel"] = tids[0]
                st.selectbox("Team", tids, key="tm_sel", format_func=lambda i: f"{i} · {tnames[i]}")
                prime_team_form()
                with st.form("tm_edit_form"):
                    st.text_input("Name", key="tm_e_name")
                    f1, f2 = st.columns(2)
                    f1.number_input("Grid row", min_value=0, max_value=side - 1, step=1, key="tm_e_row")
                    f2.number_input("Grid column", min_value=0, max_value=side - 1, step=1, key="tm_e_col")
                    f3, f4 = st.columns(2)
                    f3.number_input("Members", min_value=1, max_value=500, step=1, key="tm_e_members")
                    f4.selectbox("Status", TEAM_STATUSES, key="tm_e_status")
                    st.form_submit_button("Save changes", on_click=save_team, use_container_width=True)
                st.button("Revert fields", on_click=revert_team_form, key="tm_revert", use_container_width=True)
                x1, x2 = st.columns([2, 1])
                x1.checkbox("Confirm delete", key="tm_del_ok")
                x2.button("Delete", on_click=delete_team, key="tm_delete", use_container_width=True)
            else:
                st.info("No rescue teams registered. Create one above.")

    with tc_b:
        st.markdown('<div class="panel-t">Rescue team registry</div>', unsafe_allow_html=True)
        tdf = team_registry_df()
        if tdf.empty:
            st.info("Registry is empty.")
        else:
            counts = tdf["Status"].value_counts()
            st.markdown('<div class="chips">' + "".join(chip_html(k_, int(counts.get(k_, 0)), TEAM_COLORS[k_]) for k_ in TEAM_STATUSES) + '</div>',
                        unsafe_allow_html=True)
            st.dataframe(style_status(tdf, TEAM_COLORS), hide_index=True, use_container_width=True)

# ──────────────────────────────────────────────────────────────────────────────
# Auto-Play Execution Trigger
# ──────────────────────────────────────────────────────────────────────────────
if st.session_state.playing:
    advance_step()
    time.sleep(float(st.session_state.get("tick", 0.6)))
    st.rerun()