import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import skfuzzy as fuzz

from collections import deque
from skfuzzy import control as ctrl


# ============================================================
# 1. PAGE CONFIGURATION & STYLE
# ============================================================

st.set_page_config(
    page_title="Smart Traffic Light Control",
    page_icon="🚦",
    layout="wide"
)

st.markdown(
    """
    <style>
    .hero {
        background: linear-gradient(135deg, #0f172a 0%, #1e3a8a 55%, #10b981 140%);
        padding: 1.6rem 2rem;
        border-radius: 18px;
        color: white;
        margin-bottom: 1rem;
    }
    .hero h1 { margin: 0; font-size: 2.1rem; color: white; padding: 0; }
    .hero p  { margin: .5rem 0 0 0; opacity: .88; font-size: 1.02rem; }
    .pill {
        display: inline-block; padding: .15rem .7rem; margin: .6rem .4rem 0 0;
        border-radius: 999px; background: rgba(255,255,255,.16);
        font-size: .82rem;
    }
    div[data-testid="stMetric"] {
        background: rgba(120,120,120,.08);
        border: 1px solid rgba(120,120,120,.18);
        padding: .8rem 1rem; border-radius: 14px;
    }
    </style>
    """,
    unsafe_allow_html=True
)

plt.rcParams.update({
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.alpha": 0.25,
    "axes.titleweight": "bold",
    "font.size": 10,
})

C_FUZZY = "#10b981"
C_FIXED = "#f97316"
PHASE_COLORS = {"NS": "#3b82f6", "EW": "#a855f7", "CLEARANCE": "#ef4444"}

ROADS = ["North", "South", "East", "West"]
PHASE_ROADS = {"NS": ["North", "South"], "EW": ["East", "West"]}

# (vehicles, waiting, green)
RULES = [
    ("low", "short", "very_short"),
    ("low", "medium", "short"),
    ("low", "long", "medium"),
    ("medium", "short", "short"),
    ("medium", "medium", "medium"),
    ("medium", "long", "long"),
    ("high", "short", "medium"),
    ("high", "medium", "long"),
    ("high", "long", "very_long"),
]


def show(fig):
    st.pyplot(fig)
    plt.close(fig)


# ============================================================
# 2. FUZZY LOGIC CONTROLLER
# ============================================================

class FuzzyTrafficController:

    def __init__(self):

        self.vehicles = ctrl.Antecedent(np.arange(0, 51, 1), "vehicles")
        self.waiting = ctrl.Antecedent(np.arange(0, 61, 1), "waiting")
        self.green = ctrl.Consequent(np.arange(5, 61, 1), "green")

        self.vehicles["low"] = fuzz.trapmf(self.vehicles.universe, [0, 0, 5, 20])
        self.vehicles["medium"] = fuzz.trimf(self.vehicles.universe, [10, 25, 40])
        self.vehicles["high"] = fuzz.trapmf(self.vehicles.universe, [30, 40, 50, 50])

        self.waiting["short"] = fuzz.trapmf(self.waiting.universe, [0, 0, 5, 20])
        self.waiting["medium"] = fuzz.trimf(self.waiting.universe, [10, 30, 45])
        self.waiting["long"] = fuzz.trapmf(self.waiting.universe, [35, 45, 60, 60])

        self.green["very_short"] = fuzz.trapmf(self.green.universe, [5, 5, 8, 15])
        self.green["short"] = fuzz.trimf(self.green.universe, [10, 18, 28])
        self.green["medium"] = fuzz.trimf(self.green.universe, [20, 32, 44])
        self.green["long"] = fuzz.trimf(self.green.universe, [36, 46, 55])
        self.green["very_long"] = fuzz.trapmf(self.green.universe, [48, 55, 60, 60])

        rules = [
            ctrl.Rule(self.vehicles[v] & self.waiting[w], self.green[g])
            for v, w, g in RULES
        ]

        self.system = ctrl.ControlSystem(rules)

    def calculate(self, vehicles, waiting_time):

        vehicles = int(np.clip(vehicles, 0, 50))
        waiting_time = float(np.clip(waiting_time, 0, 60))

        sim = ctrl.ControlSystemSimulation(self.system)
        sim.input["vehicles"] = vehicles
        sim.input["waiting"] = waiting_time
        sim.compute()

        return float(sim.output["green"])

    def firing_strengths(self, vehicles, waiting_time):
        """Strength (0-1) with which each rule fires for the given inputs."""
        rows = []
        for v, w, g in RULES:
            dv = fuzz.interp_membership(
                self.vehicles.universe, self.vehicles[v].mf, vehicles
            )
            dw = fuzz.interp_membership(
                self.waiting.universe, self.waiting[w].mf, waiting_time
            )
            rows.append({
                "Rule": "IF vehicles = " + v + " AND waiting = " + w
                        + " THEN green = " + g.replace("_", " "),
                "Firing strength": float(min(dv, dw))
            })
        return pd.DataFrame(rows)


@st.cache_resource
def get_controller():
    return FuzzyTrafficController()


@st.cache_data(show_spinner=False)
def control_surface():
    ctl = get_controller()
    v = np.linspace(0, 50, 11)
    w = np.linspace(0, 60, 13)
    z = np.array([[ctl.calculate(a, b) for a in v] for b in w])
    return v, w, z


# ============================================================
# 3. TRAFFIC SIMULATION
# ============================================================

def run_simulation(
    controller_type,
    duration=1800,
    seed=42,
    arrival_rate=0.30,
    service_rate=1.0,
    fixed_green=30
):

    rng = np.random.default_rng(seed)

    # Pre-generated random numbers: both controllers face the
    # exact same arrivals and the same service randomness.
    arrivals = rng.poisson(arrival_rate, size=(duration, len(ROADS)))
    service_rand = rng.random((duration, len(ROADS)))

    fuzzy = get_controller() if controller_type == "fuzzy" else None

    queues = {road: deque() for road in ROADS}   # stores arrival times

    total_arrived = 0
    total_departed = 0
    total_delay = 0
    max_queue = 0
    served_waits = []

    base_service = int(np.floor(service_rate))
    frac_service = service_rate - base_service

    phase = "NS"
    elapsed_green = 0
    green_duration = fixed_green
    clearance_remaining = 0

    total_hist, phase_hist, served_hist = [], [], []
    road_hist = {road: [] for road in ROADS}
    cycle_log = []

    for t in range(duration):

        # Arrivals
        for i, road in enumerate(ROADS):
            n = int(arrivals[t, i])
            if n:
                queues[road].extend([t] * n)
                total_arrived += n

        # Delay accumulation (vehicle-seconds)
        total_delay += sum(len(q) for q in queues.values())

        # Decide green duration at the start of a phase
        if elapsed_green == 0 and clearance_remaining == 0:

            active = PHASE_ROADS[phase]
            total_vehicles = sum(len(queues[r]) for r in active)
            waits = [t - a + 1 for r in active for a in queues[r]]
            avg_wait = float(np.mean(waits)) if waits else 0.0

            if fuzzy is not None:
                green_duration = fuzzy.calculate(
                    min(total_vehicles, 50), min(avg_wait, 60)
                )
            else:
                green_duration = fixed_green

            green_duration = max(5, min(60, int(round(green_duration))))

            cycle_log.append({
                "Time (seconds)": t,
                "Phase": phase,
                "Vehicles Waiting": total_vehicles,
                "Avg Wait (s)": round(avg_wait, 1),
                "Green Time (s)": green_duration
            })

        in_clearance = clearance_remaining > 0

        if in_clearance:

            clearance_remaining -= 1

            if clearance_remaining == 0:
                phase = "EW" if phase == "NS" else "NS"
                elapsed_green = 0

        else:

            for road in PHASE_ROADS[phase]:

                q = queues[road]
                i = ROADS.index(road)

                departures = min(len(q), base_service)

                if len(q) > departures and service_rand[t, i] < frac_service:
                    departures += 1

                for _ in range(departures):
                    arrived_at = q.popleft()
                    served_waits.append(t - arrived_at + 1)

                total_departed += departures

            elapsed_green += 1

            if elapsed_green >= green_duration:
                clearance_remaining = 1

        # Statistics
        current_queue = sum(len(q) for q in queues.values())
        max_queue = max(max_queue, current_queue)

        total_hist.append(current_queue)
        phase_hist.append("CLEARANCE" if in_clearance else phase)
        served_hist.append(total_departed)

        for road in ROADS:
            road_hist[road].append(len(queues[road]))

    name = "Fuzzy Logic" if controller_type == "fuzzy" else "Fixed Time"

    results = {
        "Controller": name,
        "Average Waiting Time (s)": round(
            total_delay / total_arrived if total_arrived else 0.0, 2
        ),
        "95th Percentile Wait (s)": round(
            float(np.percentile(served_waits, 95)) if served_waits else 0.0, 2
        ),
        "Average Queue Length": round(float(np.mean(total_hist)), 2),
        "Maximum Queue Length": max_queue,
        "Vehicles Arrived": total_arrived,
        "Vehicles Served": total_departed,
        "Vehicles Remaining": sum(len(q) for q in queues.values()),
        "Throughput (%)": round(
            100 * total_departed / total_arrived, 2
        ) if total_arrived else 0.0
    }

    history = pd.DataFrame({
        "Time (seconds)": np.arange(duration),
        "Queue Length": total_hist,
        "Signal Phase": phase_hist,
        "Cumulative Vehicles Served": served_hist,
        **road_hist
    })

    return results, history, np.array(served_waits), pd.DataFrame(cycle_log)


@st.cache_data(show_spinner=False)
def run_sweep(rates, duration, seed, service_rate, fixed_green):
    rows = []
    for r in rates:
        f, _, _, _ = run_simulation("fuzzy", duration, seed, r, service_rate, fixed_green)
        x, _, _, _ = run_simulation("fixed", duration, seed, r, service_rate, fixed_green)
        rows.append({
            "Arrival Rate": r,
            "Fuzzy": f["Average Waiting Time (s)"],
            "Fixed": x["Average Waiting Time (s)"]
        })
    return pd.DataFrame(rows)


# ============================================================
# 4. DRAWING HELPERS
# ============================================================

def phase_segments(phases):
    segs, start = [], 0
    for i in range(1, len(phases) + 1):
        if i == len(phases) or phases[i] != phases[start]:
            segs.append((start, i - start, phases[start]))
            start = i
    return segs


def draw_intersection(ax, row, title, car_color):

    ax.set_xlim(-11, 11)
    ax.set_ylim(-11, 11)
    ax.set_aspect("equal")
    ax.axis("off")

    ax.add_patch(plt.Rectangle((-1.6, -11), 3.2, 22, color="#4b5563", zorder=1))
    ax.add_patch(plt.Rectangle((-11, -1.6), 22, 3.2, color="#4b5563", zorder=1))

    for xs, ys in [([0, 0], [1.6, 11]), ([0, 0], [-11, -1.6]),
                   ([1.6, 11], [0, 0]), ([-11, -1.6], [0, 0])]:
        ax.plot(xs, ys, color="white", lw=1, ls="--", zorder=2)

    phase = row["Signal Phase"]

    def light(road):
        if phase == "CLEARANCE":
            return "#ef4444"
        return "#22c55e" if road in PHASE_ROADS[phase] else "#ef4444"

    light_pos = {"North": (-2.4, 2.4), "South": (2.4, -2.4),
                 "East": (2.4, 2.4), "West": (-2.4, -2.4)}
    label_pos = {"North": (-4.6, 9.6), "South": (4.6, -9.6),
                 "East": (9.2, 3.6), "West": (-9.2, -3.6)}

    max_cars = 14

    for road in ROADS:

        lx, ly = light_pos[road]
        ax.scatter([lx], [ly], s=280, c=light(road),
                   edgecolors="black", linewidths=1.2, zorder=5)

        n = int(row[road])
        k = np.arange(min(n, max_cars))
        d = 2.5 + k * 0.6

        if road == "North":
            x, y = np.full_like(d, -0.8), d
        elif road == "South":
            x, y = np.full_like(d, 0.8), -d
        elif road == "East":
            x, y = d, np.full_like(d, 0.8)
        else:
            x, y = -d, np.full_like(d, -0.8)

        ax.scatter(x, y, marker="s", s=48, c=car_color, zorder=4)

        suffix = " (+" + str(n - max_cars) + ")" if n > max_cars else ""
        ax.text(label_pos[road][0], label_pos[road][1],
                road + ": " + str(n) + suffix,
                ha="center", va="center", fontsize=9, fontweight="bold")

    ax.set_title(title, fontsize=11)


# ============================================================
# 5. HEADER
# ============================================================

st.markdown(
    """
    <div class="hero">
        <h1>🚦 Smart Traffic Light Control Using Fuzzy Logic</h1>
        <p>A Mamdani fuzzy inference system that adapts green-signal time to live
        traffic, compared against a conventional fixed-time signal.</p>
        <span class="pill">🧠 Mamdani FIS</span>
        <span class="pill">📊 Poisson arrivals</span>
        <span class="pill">⚖️ Identical traffic for both controllers</span>
    </div>
    """,
    unsafe_allow_html=True
)


# ============================================================
# 6. SIDEBAR
# ============================================================

st.sidebar.header("⚙️ Simulation Settings")

simulation_minutes = st.sidebar.slider("Simulation duration (minutes)", 5, 60, 30, 5)

arrival_rate = st.sidebar.slider(
    "Arrival rate (vehicles/second/road)", 0.05, 0.80, 0.30, 0.05
)

service_rate = st.sidebar.slider(
    "Service rate (vehicles/second/road)", 0.5, 3.0, 1.0, 0.5
)

fixed_green = st.sidebar.slider("Fixed green duration (seconds)", 10, 60, 30, 5)

seed = st.sidebar.number_input("Random seed", 1, 10000, 42)

# Rough load estimate: each road is green about half the time
load = arrival_rate / (service_rate * 0.5)

if load < 0.7:
    st.sidebar.success("Estimated intersection load: " + f"{load:.0%}" + " (light)")
elif load < 1.0:
    st.sidebar.warning("Estimated intersection load: " + f"{load:.0%}" + " (busy)")
else:
    st.sidebar.error("Estimated intersection load: " + f"{load:.0%}" + " (oversaturated)")

run_button = st.sidebar.button("▶ Run Simulation", type="primary")


# ============================================================
# 7. RUN SIMULATIONS
# ============================================================

if run_button:

    with st.spinner("Simulating fuzzy and fixed-time controllers..."):

        duration = simulation_minutes * 60

        fz = run_simulation("fuzzy", duration, int(seed), arrival_rate,
                            service_rate, fixed_green)
        fx = run_simulation("fixed", duration, int(seed), arrival_rate,
                            service_rate, fixed_green)

        st.session_state["sim"] = {
            "fuzzy": fz,
            "fixed": fx,
            "duration": duration,
            "seed": int(seed),
            "arrival_rate": arrival_rate,
            "service_rate": service_rate,
            "fixed_green": fixed_green,
        }

    st.toast("Simulation completed!", icon="✅")


# ============================================================
# 8. TABS
# ============================================================

tab_lab, tab_sim, tab_data, tab_about = st.tabs([
    "🎛️ Controller Lab",
    "🚦 Simulation Results",
    "📘 Rules & Data",
    "ℹ️ About"
])


# ------------------------------------------------------------
# TAB 1 — CONTROLLER LAB
# ------------------------------------------------------------

with tab_lab:

    ctl = get_controller()

    st.subheader("Test the fuzzy controller")

    c1, c2 = st.columns(2)
    vehicles_input = c1.slider("Waiting vehicles", 0, 50, 25)
    waiting_input = c2.slider("Average waiting time (seconds)", 0, 60, 25)

    green_time = ctl.calculate(vehicles_input, waiting_input)

    m1, m2, m3 = st.columns(3)
    m1.metric("Recommended green time", f"{green_time:.1f} s")
    m2.metric("Fixed-time baseline", f"{fixed_green} s")
    m3.metric(
        "Difference",
        f"{green_time - fixed_green:+.1f} s",
        help="Positive: fuzzy gives more green than the fixed plan."
    )

    st.progress(
        float((green_time - 5) / 55),
        text="Green time on the 5 - 60 second scale"
    )

    # Membership functions with live input markers
    st.markdown("#### How the inputs are fuzzified")

    fig, axes = plt.subplots(1, 3, figsize=(15, 3.8))
    palette = ["#3b82f6", "#f59e0b", "#ef4444", "#8b5cf6", "#14b8a6"]

    for ax, var, value, title in [
        (axes[0], ctl.vehicles, vehicles_input, "Vehicles"),
        (axes[1], ctl.waiting, waiting_input, "Waiting time (s)"),
    ]:
        for color, (name, term) in zip(palette, var.terms.items()):
            ax.plot(var.universe, term.mf, label=name, color=color, lw=2)
            ax.fill_between(var.universe, term.mf, alpha=0.10, color=color)
        ax.axvline(value, color="black", ls="--", lw=1.3)
        ax.set_title(title)
        ax.set_ylim(0, 1.05)
        ax.legend(fontsize=8)

    ax = axes[2]
    for color, (name, term) in zip(palette, ctl.green.terms.items()):
        ax.plot(ctl.green.universe, term.mf,
                label=name.replace("_", " "), color=color, lw=2)
        ax.fill_between(ctl.green.universe, term.mf, alpha=0.10, color=color)
    ax.axvline(green_time, color="black", ls="--", lw=1.3)
    ax.set_title("Green time (s) — defuzzified output")
    ax.set_ylim(0, 1.05)
    ax.legend(fontsize=8)

    fig.tight_layout()
    show(fig)

    # Rule activation
    st.markdown("#### Which rules are firing?")

    fs = ctl.firing_strengths(vehicles_input, waiting_input)
    fs = fs[fs["Firing strength"] > 0].sort_values(
        "Firing strength", ascending=False
    )

    st.dataframe(
        fs,
        hide_index=True,
        column_config={
            "Firing strength": st.column_config.ProgressColumn(
                "Firing strength", min_value=0.0, max_value=1.0, format="%.2f"
            )
        }
    )

    # Control surface
    st.markdown("#### Control surface")

    v_axis, w_axis, z = control_surface()

    fig, ax = plt.subplots(figsize=(9, 4.5))
    cs = ax.contourf(v_axis, w_axis, z, levels=15, cmap="RdYlGn")
    fig.colorbar(cs, ax=ax, label="Green time (s)")
    ax.scatter([vehicles_input], [waiting_input], c="black", s=90,
               marker="*", zorder=5, label="Current input")
    ax.set_xlabel("Waiting vehicles")
    ax.set_ylabel("Average waiting time (s)")
    ax.set_title("More vehicles and longer waits give longer green")
    ax.legend()
    ax.grid(False)
    show(fig)

    st.caption(
        "This is the controller's recommendation for the selected inputs, "
        "not a field-validated signal timing."
    )


# ------------------------------------------------------------
# TAB 2 — SIMULATION RESULTS
# ------------------------------------------------------------

with tab_sim:

    if "sim" not in st.session_state:

        st.info("👈 Set your parameters in the sidebar and click **Run Simulation**.")

    else:

        sim = st.session_state["sim"]

        fr, fh, f_waits, f_cycles = sim["fuzzy"]
        xr, xh, x_waits, x_cycles = sim["fixed"]
        duration = sim["duration"]

        fuzzy_wait = fr["Average Waiting Time (s)"]
        fixed_wait = xr["Average Waiting Time (s)"]

        improvement = (
            (fixed_wait - fuzzy_wait) / fixed_wait * 100 if fixed_wait > 0 else 0.0
        )

        # Verdict
        if improvement > 5:
            st.success(
                "✅ Fuzzy control reduced average waiting time by "
                + f"{improvement:.1f}%" + " compared with the fixed-time signal."
            )
        elif improvement < -5:
            st.warning(
                "⚠️ Fuzzy control increased average waiting time by "
                + f"{abs(improvement):.1f}%" + " in this scenario."
            )
        else:
            st.info(
                "➖ Both controllers performed about the same here ("
                + f"{improvement:+.1f}%" + " change in waiting time)."
            )

        # KPI cards
        k1, k2, k3, k4 = st.columns(4)

        k1.metric(
            "Avg waiting time",
            f"{fuzzy_wait:.2f} s",
            f"{fuzzy_wait - fixed_wait:+.2f} s vs fixed",
            delta_color="inverse"
        )
        k2.metric(
            "95th percentile wait",
            f"{fr['95th Percentile Wait (s)']:.1f} s",
            f"{fr['95th Percentile Wait (s)'] - xr['95th Percentile Wait (s)']:+.1f} s vs fixed",
            delta_color="inverse"
        )
        k3.metric(
            "Avg queue length",
            f"{fr['Average Queue Length']:.2f}",
            f"{fr['Average Queue Length'] - xr['Average Queue Length']:+.2f} vs fixed",
            delta_color="inverse"
        )
        k4.metric(
            "Throughput",
            f"{fr['Throughput (%)']:.1f}%",
            f"{fr['Throughput (%)'] - xr['Throughput (%)']:+.1f}% vs fixed"
        )

        st.dataframe(pd.DataFrame([fr, xr]), hide_index=True)

        # Queue over time
        st.markdown("### Queue length over time")

        fig, ax = plt.subplots(figsize=(12, 4.2))

        for hist, color, label in [(fh, C_FUZZY, "Fuzzy Logic"),
                                   (xh, C_FIXED, "Fixed Time")]:
            raw = hist["Queue Length"]
            ax.plot(hist["Time (seconds)"], raw, color=color, alpha=0.18, lw=1)
            ax.plot(hist["Time (seconds)"], raw.rolling(30, min_periods=1).mean(),
                    color=color, lw=2.4, label=label + " (30 s average)")

        ax.set_xlabel("Time (seconds)")
        ax.set_ylabel("Waiting vehicles")
        ax.legend()
        show(fig)

        # Comparison bars + wait distribution
        st.markdown("### Performance at a glance")

        col_a, col_b = st.columns(2)

        with col_a:
            fig, axes = plt.subplots(1, 3, figsize=(9, 3.8))
            items = [
                ("Avg wait (s)", fuzzy_wait, fixed_wait),
                ("Avg queue", fr["Average Queue Length"], xr["Average Queue Length"]),
                ("Max queue", fr["Maximum Queue Length"], xr["Maximum Queue Length"]),
            ]
            for ax, (title, a, b) in zip(axes, items):
                bars = ax.bar(["Fuzzy", "Fixed"], [a, b],
                              color=[C_FUZZY, C_FIXED], width=0.6)
                ax.bar_label(bars, fmt="%.1f", fontsize=9)
                ax.set_title(title, fontsize=10)
                ax.grid(False)
            fig.tight_layout()
            show(fig)

        with col_b:
            fig, ax = plt.subplots(figsize=(6, 3.8))
            top = max(
                np.percentile(f_waits, 99) if len(f_waits) else 1,
                np.percentile(x_waits, 99) if len(x_waits) else 1,
            )
            bins = np.linspace(0, max(top, 1), 35)
            ax.hist(f_waits, bins=bins, alpha=0.6, color=C_FUZZY, label="Fuzzy Logic")
            ax.hist(x_waits, bins=bins, alpha=0.6, color=C_FIXED, label="Fixed Time")
            ax.set_xlabel("Wait of served vehicle (s)")
            ax.set_ylabel("Vehicles")
            ax.set_title("Distribution of waiting times")
            ax.legend()
            show(fig)

        # Adaptive green times
        st.markdown("### How the fuzzy controller adapts green time")

        fig, ax = plt.subplots(figsize=(12, 3.6))
        ax.plot(f_cycles["Time (seconds)"], f_cycles["Green Time (s)"],
                color=C_FUZZY, marker="o", ms=4, lw=2, label="Fuzzy green time")
        ax.axhline(sim["fixed_green"], color=C_FIXED, ls="--", lw=2,
                   label="Fixed green time")
        ax.set_xlabel("Time (seconds)")
        ax.set_ylabel("Green time (s)")
        ax.set_ylim(0, 65)
        ax.legend()
        show(fig)

        # Signal timeline (first 10 minutes)
        st.markdown("### Signal timeline (first 10 minutes)")

        window = min(duration, 600)

        fig, ax = plt.subplots(figsize=(12, 2.6))

        for y, hist in [(1, fh), (0, xh)]:
            phases = hist["Signal Phase"].tolist()[:window]
            for start, length, p in phase_segments(phases):
                ax.broken_barh([(start, length)], (y - 0.38, 0.76),
                               facecolors=PHASE_COLORS[p])

        ax.set_yticks([0, 1])
        ax.set_yticklabels(["Fixed", "Fuzzy"])
        ax.set_xlim(0, window)
        ax.set_xlabel("Time (seconds)")
        ax.grid(False)
        ax.legend(
            handles=[Patch(color=PHASE_COLORS["NS"], label="North-South green"),
                     Patch(color=PHASE_COLORS["EW"], label="East-West green"),
                     Patch(color=PHASE_COLORS["CLEARANCE"], label="All-red")],
            ncol=3, loc="upper center", bbox_to_anchor=(0.5, 1.35), frameon=False
        )
        show(fig)

        # Per-road queues
        st.markdown("### Where does the queue build up?")

        fig, axes = plt.subplots(1, 2, figsize=(13, 3.8), sharey=True)
        road_colors = ["#3b82f6", "#60a5fa", "#a855f7", "#c084fc"]

        for ax, hist, title in [(axes[0], fh, "Fuzzy Logic"),
                                (axes[1], xh, "Fixed Time")]:
            ax.stackplot(hist["Time (seconds)"],
                         [hist[r] for r in ROADS],
                         labels=ROADS, colors=road_colors, alpha=0.85)
            ax.set_title(title)
            ax.set_xlabel("Time (seconds)")

        axes[0].set_ylabel("Waiting vehicles")
        axes[1].legend(loc="upper left", fontsize=8)
        fig.tight_layout()
        show(fig)

        # Intersection snapshot
        st.markdown("### 🔍 Intersection snapshot")

        t_view = st.slider(
            "Move through time (seconds)",
            0, duration - 1, min(300, duration - 1), key="t_view"
        )

        s1, s2 = st.columns(2)

        with s1:
            fig, ax = plt.subplots(figsize=(5.2, 5.2))
            draw_intersection(ax, fh.iloc[t_view],
                              "Fuzzy Logic  |  t = " + str(t_view) + " s", C_FUZZY)
            show(fig)

        with s2:
            fig, ax = plt.subplots(figsize=(5.2, 5.2))
            draw_intersection(ax, xh.iloc[t_view],
                              "Fixed Time  |  t = " + str(t_view) + " s", C_FIXED)
            show(fig)

        st.caption("Green dot = go, red dot = stop. Each square is a waiting vehicle.")

        # Sensitivity sweep
        st.markdown("### 📈 Sensitivity: how does traffic volume change the result?")

        st.caption(
            "Runs both controllers across several arrival rates using your "
            "current seed, service rate and fixed green time."
        )

        if st.button("Run arrival-rate sweep"):

            rates = tuple(np.round(np.arange(0.10, 0.65, 0.05), 2))

            with st.spinner("Running sweep..."):
                sweep = run_sweep(
                    rates, duration, sim["seed"],
                    sim["service_rate"], sim["fixed_green"]
                )

            fig, ax = plt.subplots(figsize=(10, 4))
            ax.plot(sweep["Arrival Rate"], sweep["Fuzzy"], marker="o",
                    color=C_FUZZY, lw=2.4, label="Fuzzy Logic")
            ax.plot(sweep["Arrival Rate"], sweep["Fixed"], marker="o",
                    color=C_FIXED, lw=2.4, label="Fixed Time")
            ax.set_xlabel("Arrival rate (vehicles/second/road)")
            ax.set_ylabel("Average waiting time (s)")
            ax.set_title("Average waiting time vs traffic volume")
            ax.legend()
            show(fig)


# ------------------------------------------------------------
# TAB 3 — RULES & DATA
# ------------------------------------------------------------

with tab_data:

    st.subheader("Fuzzy rule base")

    rules_table = pd.DataFrame(
        [[v.title(), w.title(), g.replace("_", " ").title()] for v, w, g in RULES],
        columns=["Vehicle Count", "Waiting Time", "Green Duration"]
    )

    st.dataframe(rules_table, hide_index=True)

    rule_matrix = pd.DataFrame(
        [[g.replace("_", " ").title() for v2, w2, g in RULES if v2 == v and w2 == w][0]
         for w in ["short", "medium", "long"]]
        for v in ["low", "medium", "high"]
    )
    rule_matrix.index = ["Vehicles: Low", "Vehicles: Medium", "Vehicles: High"]
    rule_matrix.columns = ["Wait: Short", "Wait: Medium", "Wait: Long"]

    st.markdown("**Rule matrix**")
    st.dataframe(rule_matrix)

    st.subheader("Download results")

    if "sim" not in st.session_state:
        st.info("Run a simulation to enable downloads.")
    else:
        sim = st.session_state["sim"]
        fr, fh, _, f_cycles = sim["fuzzy"]
        xr, xh, _, x_cycles = sim["fixed"]

        d1, d2, d3 = st.columns(3)

        d1.download_button(
            "⬇ Performance comparison (CSV)",
            pd.DataFrame([fr, xr]).to_csv(index=False).encode("utf-8"),
            "traffic_performance_comparison.csv",
            "text/csv"
        )

        history_comparison = pd.DataFrame({
            "Time (seconds)": fh["Time (seconds)"],
            "Fuzzy Queue Length": fh["Queue Length"],
            "Fixed Queue Length": xh["Queue Length"],
            "Fuzzy Signal Phase": fh["Signal Phase"],
            "Fixed Signal Phase": xh["Signal Phase"],
            "Fuzzy Vehicles Served": fh["Cumulative Vehicles Served"],
            "Fixed Vehicles Served": xh["Cumulative Vehicles Served"],
        })

        d2.download_button(
            "⬇ Time-series data (CSV)",
            history_comparison.to_csv(index=False).encode("utf-8"),
            "traffic_simulation_history.csv",
            "text/csv"
        )

        d3.download_button(
            "⬇ Fuzzy signal cycle log (CSV)",
            f_cycles.to_csv(index=False).encode("utf-8"),
            "fuzzy_cycle_log.csv",
            "text/csv"
        )


# ------------------------------------------------------------
# TAB 4 — ABOUT
# ------------------------------------------------------------

with tab_about:

    st.markdown(
        """
        **Project:** Smart Traffic Light Control Using Fuzzy Logic for
        Traffic Congestion Reduction

        **Technique:** Mamdani Fuzzy Inference System (centroid defuzzification)

        **Inputs:** number of waiting vehicles and their average waiting time

        **Output:** green signal duration (5 - 60 seconds)

        **Baseline:** fixed-time signal controller

        **Metrics:** average waiting time, 95th percentile wait, average and
        maximum queue length, throughput, vehicles served

        **Simulation assumptions:** Poisson arrivals on four approaches, two
        non-conflicting phases (North-South and East-West), a one-second all-red
        clearance between phases, and a simplified vehicle service model. Both
        controllers receive exactly the same random arrivals for a fair
        comparison.

        **Limitations:** This is a research simulation, not a real-world traffic
        controller. Arrival model, service rates, membership functions and
        timing parameters need validation before any practical deployment.
        """
    )

    st.info(
        "Tip: Try several arrival rates and random seeds before drawing "
        "conclusions. Fuzzy control tends to show its strength under "
        "uneven or heavy traffic."
    )

st.caption("Smart Traffic Light Control | Fuzzy Logic | Python | Streamlit")