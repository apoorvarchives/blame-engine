"""
BlameEngine — FastF1 Real Telemetry Extractor
=============================================
Run this script locally (needs internet + fastf1 installed).

Install deps:
    pip install fastf1 pandas numpy

Run:
    python extract_f1_data.py

Output:
    f1_real_data.json  ← drop this in blame-engine/frontend/src/
    Then F1Simulator.jsx reads it automatically.

The JSON contains real:
  - Tyre degradation curves per compound per track
  - Lap time baselines per track
  - Driver pace ratings from actual quali/race data
  - Pit stop durations (stationary time)
  - Sector time benchmarks
  - Safety car frequency per track
  - DRS speed gain per track
"""

import fastf1
import fastf1.plotting
import pandas as pd
import numpy as np
import json
import os
from pathlib import Path

# ── CONFIG ────────────────────────────────────────────────────────────────────
SEASON = 2024
CACHE_DIR = "./f1_cache"
OUTPUT_FILE = "./f1_real_data.json"

# Which races to extract (match our 3 simulator tracks)
RACE_MAP = {
    "Monaco":      "Monaco",
    "Silverstone": "British",
    "Suzuka":      "Japanese",
}

fastf1.Cache.enable_cache(CACHE_DIR)
Path(CACHE_DIR).mkdir(exist_ok=True)

print("BlameEngine — FastF1 Real Data Extractor")
print("=" * 50)

output = {
    "season": SEASON,
    "tracks": {},
    "drivers": {},
    "meta": {}
}

# ── EXTRACT PER TRACK ─────────────────────────────────────────────────────────
for sim_name, gp_name in RACE_MAP.items():
    print(f"\n📡 Loading {gp_name} GP {SEASON}...")
    try:
        session = fastf1.get_session(SEASON, gp_name, "R")
        session.load(telemetry=True, laps=True, weather=True)

        laps = session.laps.pick_quicklaps()
        weather = session.weather_data

        # ── LAP TIME BASELINE ─────────────────────────────────────────────────
        fastest = float(laps["LapTime"].min().total_seconds())
        median_lap = float(laps["LapTime"].median().total_seconds())
        print(f"  ✓ Fastest lap: {fastest:.3f}s | Median: {median_lap:.3f}s")

        # ── SECTOR TIMES ─────────────────────────────────────────────────────
        s1_best = float(laps["Sector1Time"].min().total_seconds()) if "Sector1Time" in laps else 0
        s2_best = float(laps["Sector2Time"].min().total_seconds()) if "Sector2Time" in laps else 0
        s3_best = float(laps["Sector3Time"].min().total_seconds()) if "Sector3Time" in laps else 0
        print(f"  ✓ Sectors: S1={s1_best:.3f}s S2={s2_best:.3f}s S3={s3_best:.3f}s")

        # ── TYRE DEGRADATION CURVES ───────────────────────────────────────────
        # For each compound, fit lap time vs tyre age
        deg_curves = {}
        for compound in ["SOFT", "MEDIUM", "HARD", "INTERMEDIATE", "WET"]:
            comp_laps = laps[laps["Compound"] == compound].copy()
            if len(comp_laps) < 5:
                continue

            comp_laps["TyreAge"] = comp_laps["TyreLife"]
            comp_laps["LapSeconds"] = comp_laps["LapTime"].dt.total_seconds()

            # Filter outliers (pit in/out laps, SC laps)
            comp_laps = comp_laps[
                (comp_laps["LapSeconds"] < median_lap * 1.08) &
                (comp_laps["LapSeconds"] > fastest * 0.99) &
                (comp_laps["TyreAge"] > 0)
            ]

            if len(comp_laps) < 4:
                continue

            # Fit polynomial: lap_time = a + b*age + c*age^2 (deg curve)
            try:
                x = comp_laps["TyreAge"].values
                y = comp_laps["LapSeconds"].values
                coeffs = np.polyfit(x, y, 2)  # quadratic fit
                
                # Extract per-lap degradation rate (seconds lost per lap)
                # Derivative at age=10: b + 2c*10
                deg_per_lap = float(coeffs[1] + 2 * coeffs[2] * 10)
                base_time = float(np.polyval(coeffs, 1))  # time at age=1
                cliff_age = None

                # Detect cliff — where deg accelerates > 2x the early rate
                early_deg = float(coeffs[1] + 2 * coeffs[2] * 3)
                for age in range(5, 50):
                    late_deg = float(coeffs[1] + 2 * coeffs[2] * age)
                    if late_deg > early_deg * 2.2 and late_deg > 0.3:
                        cliff_age = age
                        break

                deg_curves[compound] = {
                    "deg_per_lap_seconds": round(max(0, deg_per_lap), 4),
                    "base_lap_time": round(base_time, 3),
                    "cliff_age": cliff_age,
                    "sample_count": len(comp_laps),
                    "poly_coeffs": [round(float(c), 6) for c in coeffs],
                }
                print(f"  ✓ {compound}: {deg_per_lap:.3f}s/lap deg | cliff@lap{cliff_age or 'none'}")
            except Exception as e:
                print(f"  ⚠ {compound} fit failed: {e}")

        # ── PIT STOP DURATIONS ────────────────────────────────────────────────
        pit_data = {"mean": 24.0, "min": 21.0, "std": 1.5}  # defaults
        try:
            pit_stops = session.laps[session.laps["PitOutTime"].notna() & session.laps["PitInTime"].notna()].copy()
            if len(pit_stops) > 3:
                # Stationary time = pit out - pit in (rough estimate from lap data)
                # Real stationary time requires timing data
                pit_stops["PitDuration"] = (
                    pit_stops["PitOutTime"] - pit_stops["PitInTime"]
                ).dt.total_seconds()
                valid = pit_stops[
                    (pit_stops["PitDuration"] > 18) &
                    (pit_stops["PitDuration"] < 45)
                ]["PitDuration"]
                if len(valid) > 2:
                    pit_data = {
                        "mean": round(float(valid.mean()), 2),
                        "min": round(float(valid.min()), 2),
                        "std": round(float(valid.std()), 2),
                        "count": len(valid),
                    }
                    print(f"  ✓ Pit stops: mean={pit_data['mean']}s min={pit_data['min']}s")
        except Exception as e:
            print(f"  ⚠ Pit data: {e}")

        # ── SAFETY CAR FREQUENCY ──────────────────────────────────────────────
        sc_laps = 0
        vsc_laps = 0
        try:
            track_status = session.track_status
            sc_laps = int((track_status["Status"] == "4").sum())   # SC
            vsc_laps = int((track_status["Status"] == "6").sum())  # VSC
            print(f"  ✓ SC laps: {sc_laps} | VSC laps: {vsc_laps}")
        except Exception as e:
            print(f"  ⚠ SC data: {e}")

        # ── WEATHER ───────────────────────────────────────────────────────────
        weather_info = {"had_rain": False, "max_rainfall": 0}
        try:
            if "Rainfall" in weather.columns:
                had_rain = bool(weather["Rainfall"].any())
                max_rain = float(weather["Rainfall"].max()) if had_rain else 0
                weather_info = {"had_rain": had_rain, "max_rainfall": round(max_rain, 3)}
                print(f"  ✓ Weather: rain={had_rain} max={max_rain:.1f}mm")
        except Exception as e:
            print(f"  ⚠ Weather: {e}")

        # ── DRS GAIN ─────────────────────────────────────────────────────────
        # Estimate DRS speed gain from telemetry on known DRS straights
        drs_gain_kmh = 10.0  # default
        try:
            # Sample fastest lap telemetry
            fastest_lap = session.laps.pick_fastest()
            tel = fastest_lap.get_telemetry()
            if "DRS" in tel.columns:
                drs_on = tel[tel["DRS"] >= 10]["Speed"]
                drs_off = tel[tel["DRS"] < 10]["Speed"]
                if len(drs_on) > 50 and len(drs_off) > 50:
                    # Compare top-speed segments
                    drs_gain_kmh = float(drs_on.quantile(0.9) - drs_off.quantile(0.9))
                    drs_gain_kmh = max(5, min(20, drs_gain_kmh))
                    print(f"  ✓ DRS gain: ~{drs_gain_kmh:.1f} km/h")
        except Exception as e:
            print(f"  ⚠ DRS telemetry: {e}")

        output["tracks"][sim_name] = {
            "gp_name": gp_name,
            "fastest_lap": round(fastest, 3),
            "median_lap": round(median_lap, 3),
            "sectors": {
                "s1_best": round(s1_best, 3),
                "s2_best": round(s2_best, 3),
                "s3_best": round(s3_best, 3),
            },
            "deg_curves": deg_curves,
            "pit_stops": pit_data,
            "sc_laps": sc_laps,
            "vsc_laps": vsc_laps,
            "total_laps": int(session.total_laps) if hasattr(session, "total_laps") else 0,
            "weather": weather_info,
            "drs_gain_kmh": round(drs_gain_kmh, 1),
        }

    except Exception as e:
        print(f"  ✗ Failed to load {gp_name}: {e}")

# ── DRIVER PACE RATINGS ───────────────────────────────────────────────────────
print("\n📡 Extracting driver pace ratings...")
try:
    # Use one session (Silverstone) to extract relative pace
    session = fastf1.get_session(SEASON, "British", "Q")
    session.load(laps=True, telemetry=False, weather=False)
    qlaps = session.laps.pick_quicklaps()

    # Get each driver's best Q lap vs overall fastest
    fastest_q = qlaps["LapTime"].min().total_seconds()
    driver_times = {}
    for drv in qlaps["Driver"].unique():
        drv_best = qlaps[qlaps["Driver"] == drv]["LapTime"].min().total_seconds()
        gap_pct = (drv_best - fastest_q) / fastest_q
        # Convert gap to pace rating 0.80-0.99
        rating = round(max(0.80, min(0.99, 0.99 - gap_pct * 8)), 3)
        driver_times[drv] = {
            "best_lap": round(drv_best, 3),
            "gap_to_fastest": round(drv_best - fastest_q, 3),
            "pace_rating": rating,
        }
        print(f"  {drv}: {drv_best:.3f}s (+{drv_best-fastest_q:.3f}s) → pace={rating}")

    output["drivers"] = driver_times

except Exception as e:
    print(f"  ✗ Driver ratings failed: {e}")

# ── WRITE OUTPUT ──────────────────────────────────────────────────────────────
output["meta"] = {
    "extracted_at": pd.Timestamp.now().isoformat(),
    "season": SEASON,
    "source": "FastF1",
    "note": "Real telemetry data — drop f1_real_data.json into frontend/src/"
}

with open(OUTPUT_FILE, "w") as f:
    json.dump(output, f, indent=2)

print(f"\n✅ Done! Written to {OUTPUT_FILE}")
print(f"   Drop it into blame-engine/frontend/src/f1_real_data.json")
print(f"   F1Simulator.jsx will pick it up automatically.\n")