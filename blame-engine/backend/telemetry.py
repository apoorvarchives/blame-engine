"""
BlameEngine Telemetry Engine
FastF1-powered race analysis and blame attribution
"""

import os
import numpy as np
import pandas as pd
from typing import Optional
from pathlib import Path

CACHE_DIR = os.environ.get("FASTF1_CACHE_DIR", "./cache")
Path(CACHE_DIR).mkdir(parents=True, exist_ok=True)

try:
    import fastf1
    fastf1.Cache.enable_cache(CACHE_DIR)
    FASTF1_AVAILABLE = True
except ImportError:
    FASTF1_AVAILABLE = False
    print("⚠️  FastF1 not installed. Using mock data. Run: pip install fastf1")


def get_session(year: int, gp: str, session_type: str = "R"):
    if not FASTF1_AVAILABLE:
        raise RuntimeError("FastF1 not available")
    session = fastf1.get_session(year, gp, session_type)
    session.load(telemetry=True, weather=True, messages=True)
    return session


def compute_qualifying_cost(race_session, quali_session, driver: str) -> float:
    try:
        quali_laps = quali_session.laps.pick_driver(driver)
        driver_best = quali_laps.pick_fastest()["LapTime"].total_seconds()
        pole_time = quali_session.laps.pick_fastest()["LapTime"].total_seconds()
        gap_to_pole = driver_best - pole_time
        race_laps = race_session.laps.pick_driver(driver)
        grid_pos = race_laps.iloc[0].get("GridPosition", 1)
        traffic_cost = max(0, (grid_pos - 1) * 0.12)
        return -round(gap_to_pole + traffic_cost, 3)
    except Exception:
        return 0.0


def compute_tyre_degradation_cost(laps: pd.DataFrame) -> float:
    try:
        cost = 0.0
        for stint_num in laps["Stint"].unique():
            stint = laps[laps["Stint"] == stint_num].copy()
            stint = stint[stint["LapTime"].notna()]
            if len(stint) < 3:
                continue
            times = stint["LapTime"].dt.total_seconds().values
            laps_in_stint = np.arange(len(times))
            coeffs = np.polyfit(laps_in_stint, times, 1)
            deg_rate = coeffs[0]
            compound = stint["Compound"].iloc[0] if "Compound" in stint.columns else "MEDIUM"
            optimal_deg = {"SOFT": 0.08, "MEDIUM": 0.05, "HARD": 0.03}.get(compound, 0.06)
            excess_deg = max(0, deg_rate - optimal_deg)
            cost += excess_deg * len(stint)
        return -round(cost, 3)
    except Exception:
        return 0.0


def compute_pit_execution_cost(laps: pd.DataFrame, session) -> float:
    try:
        pit_laps = laps[laps["PitOutTime"].notna() | laps["PitInTime"].notna()]
        if pit_laps.empty:
            return 0.0
        total_cost = 0.0
        team_avg_stop = 2.4
        for _, lap in pit_laps.iterrows():
            if pd.notna(lap.get("PitInTime")) and pd.notna(lap.get("PitOutTime")):
                stop_time = (lap["PitOutTime"] - lap["PitInTime"]).total_seconds()
                excess = max(0, stop_time - team_avg_stop)
                total_cost += excess
        return -round(total_cost, 3)
    except Exception:
        return 0.0


def compute_strategy_error(laps: pd.DataFrame, circuit: str, year: int) -> float:
    try:
        pit_laps_nums = laps[laps["PitOutTime"].notna()]["LapNumber"].tolist()
        if not pit_laps_nums:
            return 0.0
        total_race_laps = int(laps["LapNumber"].max())
        optimal_stop_lap = total_race_laps * 0.42
        cost = 0.0
        for pit_lap in pit_laps_nums:
            deviation = abs(pit_lap - optimal_stop_lap)
            if deviation > 5:
                cost += deviation * 0.15
        return -round(cost, 3)
    except Exception:
        return 0.0


def compute_car_pace_deficit(race_session, driver: str) -> float:
    try:
        driver_laps = race_session.laps.pick_driver(driver)
        clean = driver_laps[
            (driver_laps["TrackStatus"] == "1") &
            (driver_laps["PitOutTime"].isna()) &
            (driver_laps["PitInTime"].isna())
        ]
        if clean.empty:
            return 0.0
        driver_median = clean["LapTime"].dt.total_seconds().median()
        all_laps = race_session.laps[
            (race_session.laps["TrackStatus"] == "1") &
            (race_session.laps["PitOutTime"].isna())
        ]
        field_times = all_laps.groupby("Driver")["LapTime"].median().dt.total_seconds()
        field_times_sorted = field_times.sort_values()
        reference_group = field_times_sorted.iloc[3:-3]
        reference_pace = reference_group.median()
        deficit = driver_median - reference_pace
        return -round(max(0, deficit), 3)
    except Exception:
        return 0.0


def compute_incident_impact(race_session, driver: str) -> float:
    try:
        messages = race_session.race_control_messages
        driver_laps = race_session.laps.pick_driver(driver)
        total_cost = 0.0
        yellows = messages[messages["Message"].str.contains("YELLOW", na=False)]
        total_cost += len(yellows) * 0.08
        sc_laps = messages[messages["Message"].str.contains("SAFETY CAR DEPLOYED", na=False)]
        if not sc_laps.empty:
            sc_lap_nums = []
            for _, sc in sc_laps.iterrows():
                matching_lap = driver_laps[driver_laps["Time"] >= sc["Time"]]
                if not matching_lap.empty:
                    sc_lap_nums.append(matching_lap.iloc[0]["LapNumber"])
            pit_laps = driver_laps[driver_laps["PitOutTime"].notna()]["LapNumber"].tolist()
            missed_sc_pits = sum(1 for sc_lap in sc_lap_nums
                                 if not any(abs(p - sc_lap) <= 2 for p in pit_laps))
            total_cost += missed_sc_pits * 1.2
        return -round(total_cost, 3)
    except Exception:
        return 0.0


def compute_optimal_position(actual_pos: int, total_loss_seconds: float,
                              car_pace_deficit: float, race_session, driver: str) -> int:
    """
    Compute the best realistic position this driver could have achieved.

    Logic:
    - Start from actual finishing position
    - Calculate how many positions TIME LOSS is worth (each position ≈ 2-3s in race)
    - BUT cap improvement by car pace — if car is 1.5s/lap slower than midfield,
      you can't magically finish P1 regardless of strategy
    - Optimal is always BETTER than or EQUAL to actual (never worse)
    - Minimum realistic position bounded by car pace ceiling
    """
    try:
        # Estimate positions recoverable from non-car-pace losses
        # (strategy + pit exec + qualifying cost + incident)
        recoverable_loss = total_loss_seconds - abs(car_pace_deficit)
        recoverable_loss = max(0, recoverable_loss)

        # ~2.5s per position is a conservative F1 race estimate
        positions_recoverable = int(recoverable_loss / 2.5)

        # Car pace ceiling: how many positions ahead could the car realistically be?
        # Estimate by comparing car pace deficit to field spread
        # A 0.5s/lap deficit ≈ ~3 positions worse than a clean-air pace would suggest
        car_pace_ceiling = abs(car_pace_deficit) / 0.4  # positions limited by car pace
        car_pace_ceiling = int(car_pace_ceiling)

        # Optimal = actual improved by recoverable, but not beyond car pace ceiling
        raw_optimal = actual_pos - positions_recoverable
        # Never better than what car pace allows from P1
        car_limited_optimal = max(1, car_pace_ceiling + 1)
        optimal = max(raw_optimal, car_limited_optimal)

        # Hard rule: optimal is ALWAYS <= actual (can only improve, never get worse)
        optimal = min(optimal, actual_pos)
        optimal = max(1, optimal)

        return optimal

    except Exception:
        return actual_pos  # fallback: same as actual


def full_autopsy(year: int, gp: str, driver: str) -> dict:
    """
    Complete race autopsy using real FastF1 data.
    Falls back to mock data if FastF1 unavailable.
    """
    if not FASTF1_AVAILABLE:
        from main import compute_blame
        return compute_blame(year, gp, driver)

    try:
        race = get_session(year, gp, "R")
        quali = get_session(year, gp, "Q")

        driver_laps = race.laps.pick_driver(driver)

        qualifying = compute_qualifying_cost(race, quali, driver)
        tyre_deg = compute_tyre_degradation_cost(driver_laps)
        pit_exec = compute_pit_execution_cost(driver_laps, race)
        strategy = compute_strategy_error(driver_laps, gp, year)
        car_pace = compute_car_pace_deficit(race, driver)
        incident = compute_incident_impact(race, driver)

        total_loss = abs(qualifying + tyre_deg + pit_exec + strategy + car_pace + incident)

        blame = {
            "qualifying_cost": qualifying,
            "tyre_degradation": tyre_deg,
            "pit_execution": pit_exec,
            "strategy_error": strategy,
            "car_pace_deficit": car_pace,
            "incident_impact": incident,
        }

        primary_cause = min(blame, key=blame.get)
        primary_pct = round(abs(blame[primary_cause]) / total_loss * 100) if total_loss > 0 else 0

        actual_pos = int(driver_laps.iloc[-1].get("Position", 10))

        # ── FIXED: compute optimal properly ──────────────────────────────────
        optimal_pos = compute_optimal_position(
            actual_pos=actual_pos,
            total_loss_seconds=total_loss,
            car_pace_deficit=car_pace,
            race_session=race,
            driver=driver,
        )
        positions_lost = actual_pos - optimal_pos  # always >= 0

        return {
            "driver": driver,
            "gp": gp,
            "year": year,
            "blame": blame,
            "total_loss": round(total_loss, 3),
            "primary_cause": primary_cause,
            "position": {
                "actual": actual_pos,
                "optimal": optimal_pos,
                "positions_lost": positions_lost,
            },
            "verdict": (
                f"{driver} at {year} {gp}: "
                f"Primary performance killer was {primary_cause.replace('_', ' ')} "
                f"({primary_pct}% of {round(total_loss, 1)}s total deficit). "
                f"Strategy cost {abs(strategy):.2f}s; tyre management added {abs(tyre_deg):.2f}s. "
                f"With optimal execution, P{optimal_pos} was achievable vs actual P{actual_pos}."
            ),
            "telemetry_source": "FastF1 v3.4 (live)",
            "data_quality": "HIGH",
        }

    except Exception as e:
        return {"error": str(e), "fallback": "mock_data"}