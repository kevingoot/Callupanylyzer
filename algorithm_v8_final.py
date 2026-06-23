#!/usr/bin/env python3
"""
Proprietary MLB Call-Up Scoring Algorithm v8 (Final Consolidated Version)
- Learned multi-objective weights (WAR + Card Price Proxy)
- Position-specific next-gen breakdowns (normal + advanced stats with tailored weighting)
- Exact parallel pricing for Topps/Bowman sets (with set demand, velocity, monthly granularity)
- Support for rare cards in non-Topps sets via high multipliers
- Efficacy testing with wins/losses analysis
- Ready for daily/monthly runs and MVP/paid future price estimation

All iterations consolidated for maximum robustness and precision.

v8.1 fixes (no scoring/calibration weights were changed - see notes at each fix):
- compute_topps_parallel_price: removed a hard $3.00 output cap that was silently flattening
  every parallel/grade multiplier (up to 180x x 3.8x) down to the same few-dollar ceiling.
- get_social_hype_multiplier: removed hardcoded per-player-name boosts (it was matching on a
  handful of literal names, not measuring hype). Now returns a neutral no-op multiplier until
  a real data source is wired in.
- learn_weights_multi: added save=False option so evaluation code (efficacy_test, the new
  efficacy_test_kfold) no longer overwrites config/learned_multi_weights_v8.json with weights
  learned from a partial train split.
- efficacy_test: now correctly skips saving when learning fold/split weights internally.
- Added efficacy_test_kfold: a single 70/30 split gives a noisy correlation estimate on a small
  dataset; this reports mean +/- spread across multiple folds instead.
- compute_card_price_proxy: added an optional override so a real observed card price column
  can be used instead of the synthetic proxy if/when that data exists (see docstring - the
  synthetic proxy shares inputs with the scoring model, so it's a directional stand-in, not an
  independent validation signal).
"""

import pandas as pd
import numpy as np
import json
from pathlib import Path
from typing import Union
from scipy.optimize import minimize, Bounds, LinearConstraint

BASE_DIR = Path(__file__).parent.parent
DATA_DIR = BASE_DIR / "data"
CONFIG_DIR = BASE_DIR / "config"
OUTPUT_DIR = BASE_DIR / "output"

def ensure_directories():
    """Ensure required directories exist and create minimal templates if data is missing."""
    for directory in [DATA_DIR, CONFIG_DIR, OUTPUT_DIR]:
        directory.mkdir(parents=True, exist_ok=True)
    print(f"✅ Directories ensured: {DATA_DIR}, {CONFIG_DIR}, {OUTPUT_DIR}")

# ==================== LOOKUP TABLES ====================
_AGE_UPSIDE_TABLE = {21: 1.0, 22: 1.0, 23: 0.85, 24: 0.85, 25: 0.65}
_PRIOR_FACTOR_TABLE = {0: 0.7, 1: 0.85}
_POSITION_PREMIUM_TABLE = {
    "C": 1.18, "SS": 1.12, "CF": 1.08, "3B": 1.05, "2B": 1.02,
    "OF": 1.00, "LF": 0.98, "RF": 0.98, "1B": 0.92, "DH": 0.85,
    "RHP": 1.05, "LHP": 1.08
}
_LEVEL_FACTOR_TABLE = {"AAA": 1.12, "AA": 1.05, "A+": 1.00, "A": 0.95, "A-": 0.90}

# Exact parallel multipliers for Topps/Bowman (calibrated from real comps)
PARALLEL_MULTIPLIERS = {
    'base': 1.0,
    'refractor_499': 2.8,
    'purple_250': 5.0,
    'blue_150': 7.5,
    'gold_50': 22.0,
    'black_10': 45.0,
    'red_5': 65.0,
    'superfractor_1of1': 180.0,
    'lava': 3.5,
    'x_fractor': 9.0,
    'reptilian': 4.0,
}

def load_weights(config_path: Path = None) -> dict:
    if config_path is None:
        config_path = CONFIG_DIR / "weights.json"
    with open(config_path, "r") as f:
        weights = json.load(f)
    return {k: v for k, v in weights.items() if isinstance(v, (int, float))}

def load_team_markets() -> pd.DataFrame:
    path = DATA_DIR / "team_market_scores.csv"
    if not path.exists():
        print(f"⚠️  WARNING: {path} not found. Creating minimal template.")
        template = pd.DataFrame({"team": ["NYY", "LAD", "BOS"], "market_score": [9.5, 9.2, 8.8]})
        template.to_csv(path, index=False)
    return pd.read_csv(path)

def validate_dataset(df: pd.DataFrame, strict: bool = True) -> None:
    """Enforce minimum data quality. Fail loudly on toy or malformed datasets."""
    required_cols = [
        "player_name", "mlb_team", "position", "minor_league_recent_ops",
        "prospect_rank_overall", "age_at_callup", "highest_level",
        "post_callup_first_year_approx_war"
    ]
    
    missing = [col for col in required_cols if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}. Dataset is invalid.")

    n_rows = len(df)
    if n_rows < 50 and strict:
        raise ValueError(f"Dataset too small ({n_rows} rows). Minimum 50 prospects required for reliable evaluation. "
                        "Use strict=False for testing on dummy data.")

    if "callup_date" in df.columns:
        if pd.api.types.is_datetime64_any_dtype(df["callup_date"]):
            min_date = df["callup_date"].min()
            max_date = df["callup_date"].max()
            print(f"📅 Temporal coverage: {min_date.date()} to {max_date.date()}")
        else:
            print("⚠️  callup_date column exists but is not datetime.")

    print(f"✅ Dataset validation passed: {n_rows} prospects")


def load_historical_callups() -> pd.DataFrame:
    path = DATA_DIR / "historical_callups.csv"
    if not path.exists():
        print(f"⚠️  WARNING: {path} not found. Creating minimal template with dummy data.")
        # Minimal template for testing - include all required columns
        data = {
            "player_name": ["Dummy Prospect1", "Dummy Prospect2"],
            "mlb_team": ["NYY", "LAD"],
            "position": ["OF", "SS"],
            "minor_league_recent_ops": [0.82, 0.78],
            "prospect_rank_overall": [15, 25],
            "prospect_fv": [70, 65],
            "age_at_callup": [22, 23],
            "highest_level": ["AAA", "AAA"],
            "post_callup_first_year_approx_war": [2.1, 1.8],
            "injury_days_missed_last_2yrs": [0, 15],
            "surgery_history": [0, 0],
            "prior_mlb_stints": [0, 1],
            "defensive_tools": [55, 60],
            "team_market_score": [9.5, 9.2]
        }
        pd.DataFrame(data).to_csv(path, index=False)
    df = pd.read_csv(path)
    markets = load_team_markets()
    if "team_market_score" not in df.columns or df["team_market_score"].isna().any():
        df = df.merge(markets[["team", "market_score"]], left_on="mlb_team", right_on="team", how="left")
        df["team_market_score"] = df["market_score"].fillna(df.get("team_market_score", 5.0))
        df.drop(columns=["team", "market_score"], errors="ignore", inplace=True)
    
    # Validate after loading
    try:
        validate_dataset(df, strict=False)  # strict=False for template data
    except ValueError as e:
        print(f"⚠️ Validation warning: {e}")
    return df

# ==================== NORMALIZERS ====================
def normalize_stat(ops: float) -> float:
    if pd.isna(ops) or ops <= 0: return 0.5
    return round(min(1.0, max(0.0, (ops - 0.650) / 0.35)), 3)

def normalize_prospect(rank: float, fv: float = None) -> float:
    if pd.isna(rank) and pd.isna(fv): return 0.5
    if not pd.isna(fv): return round(min(1.0, max(0.0, (fv - 45) / 35)), 3)
    if not pd.isna(rank): return round(min(1.0, max(0.1, 1.1 - (rank / 80))), 3)
    return 0.5

def compute_injury_factor(days_missed: float, surgery: int) -> float:
    if pd.isna(days_missed): days_missed = 20.0
    risk = min(1.0, days_missed / 120 + (0.25 if surgery else 0.0))
    return round(max(0.1, 1.0 - risk), 3)

def compute_age_upside(age: float) -> float:
    if pd.isna(age): return 0.5
    age_int = int(age)
    if age_int in _AGE_UPSIDE_TABLE: return _AGE_UPSIDE_TABLE[age_int]
    if age_int <= 21: return 1.0
    return max(0.2, round(0.9 - (age - 25) * 0.1, 3))

def compute_prior_factor(prior_stints: int) -> float:
    if pd.isna(prior_stints): return 0.7
    return _PRIOR_FACTOR_TABLE.get(int(prior_stints), 0.6)

def compute_position_premium(position: str) -> float:
    if pd.isna(position): return 1.0
    return _POSITION_PREMIUM_TABLE.get(str(position).upper().strip(), 1.0)

def compute_level_factor(level: str) -> float:
    if pd.isna(level): return 1.0
    return _LEVEL_FACTOR_TABLE.get(str(level).upper().strip(), 1.0)

def normalize_defensive(tools: float) -> float:
    if pd.isna(tools) or tools <= 0: return 0.5
    return round(min(1.0, max(0.0, (tools - 40) / 40)), 3)

def normalize_pitcher_performance(row: pd.Series) -> float:
    if pd.isna(row.get("position")) or str(row.get("position", "")).upper() not in ["RHP", "LHP"]: return 0.5
    era = row.get("era"); whip = row.get("whip"); k9 = row.get("k9"); bb9 = row.get("bb9")
    era_s = 1.0 - min(1.0, max(0.0, (era - 2.0) / 4.0)) if era is not None and not pd.isna(era) else 0.5
    whip_s = 1.0 - min(1.0, max(0.0, (whip - 1.0) / 1.5)) if whip is not None and not pd.isna(whip) else 0.5
    k9_s = min(1.0, max(0.0, (k9 - 6.0) / 6.0)) if k9 is not None and not pd.isna(k9) else 0.5
    bb9_s = 1.0 - min(1.0, max(0.0, (bb9 - 2.0) / 4.0)) if bb9 is not None and not pd.isna(bb9) else 0.5
    composite = (0.35 * era_s + 0.25 * whip_s + 0.25 * k9_s + 0.15 * bb9_s)
    return round(min(1.0, max(0.0, composite)), 3)

def normalize_position_stats(row: pd.Series) -> float:
    """Position-specific with normal + next-gen stats (launch angle, hard-hit, spin, GB%, SwStr%, OAA, etc.)"""
    pos = str(row.get("position", "")).upper()
    if pos in ["RHP", "LHP"]:
        base = normalize_pitcher_performance(row)
        spin = row.get("spin_rate", 2200); gb = row.get("gb_pct", 42); swstr = row.get("swstr_pct", 11)
        spin_s = min(1.0, max(0.0, (spin - 2000) / 600))
        gb_s = min(1.0, max(0.0, (gb - 35) / 20))
        swstr_s = min(1.0, max(0.0, (swstr - 8) / 8))
        return round(0.6 * base + 0.15 * spin_s + 0.15 * gb_s + 0.10 * swstr_s, 3)
    
    # Hitter next-gen
    ops = row.get("minor_league_recent_ops", 0.75)
    hr_pa = row.get("hr_pa", 0.04)
    sb = row.get("sb", 10)
    xwoba = row.get("xwoba", 0.320)
    barrel = row.get("barrel_pct", 10)
    exit_v = row.get("exit_velo", 88)
    sprint = row.get("sprint_speed", 27)
    oaa = row.get("oaa", 5)
    launch = row.get("launch_angle", 12)
    hard_hit = row.get("hard_hit_rate", 42)

    ops_s = min(1.0, max(0.0, (ops - 0.650) / 0.35))
    hr_s = min(1.0, max(0.0, (hr_pa - 0.02) / 0.06)) if hr_pa else 0.5
    sb_s = min(1.0, max(0.0, sb / 40))
    xwoba_s = min(1.0, max(0.0, (xwoba - 0.300) / 0.08))
    barrel_s = min(1.0, max(0.0, barrel / 20))
    exit_s = min(1.0, max(0.0, (exit_v - 85) / 10))
    sprint_s = min(1.0, max(0.0, (sprint - 25) / 6))
    oaa_s = min(1.0, max(0.0, (oaa + 5) / 15))
    launch_s = min(1.0, max(0.0, (launch - 8) / 12))
    hard_hit_s = min(1.0, max(0.0, (hard_hit - 35) / 25))

    # Position-specific weights (more maintainable dict)
    POS_WEIGHTS = {
        "C":  [0.10, 0.07, 0.04, 0.10, 0.07, 0.07, 0.07, 0.30, 0.09, 0.09],
        "SS": [0.10, 0.07, 0.10, 0.09, 0.07, 0.07, 0.12, 0.20, 0.09, 0.09],
        "OF": [0.10, 0.10, 0.10, 0.09, 0.09, 0.10, 0.10, 0.08, 0.12, 0.12],
        "CF": [0.10, 0.10, 0.10, 0.09, 0.09, 0.10, 0.10, 0.08, 0.12, 0.12],
        "LF": [0.10, 0.10, 0.10, 0.09, 0.09, 0.10, 0.10, 0.08, 0.12, 0.12],
        "RF": [0.10, 0.10, 0.10, 0.09, 0.09, 0.10, 0.10, 0.08, 0.12, 0.12],
        "1B": [0.12, 0.15, 0.04, 0.10, 0.10, 0.12, 0.04, 0.13, 0.10, 0.10],
    }
    w = POS_WEIGHTS.get(pos, [0.11, 0.09, 0.09, 0.09, 0.09, 0.09, 0.10, 0.18, 0.08, 0.08])

    composite = (w[0]*ops_s + w[1]*hr_s + w[2]*sb_s + w[3]*xwoba_s + w[4]*barrel_s +
                 w[5]*exit_s + w[6]*sprint_s + w[7]*oaa_s + w[8]*launch_s + w[9]*hard_hit_s)
    return round(min(1.0, max(0.0, composite)), 3)

def compute_card_price_proxy(row: pd.Series, actual_price_col: str = "observed_card_price") -> float:
    """Card price proxy used as a training/eval target.

    **IMPORTANT FIX**: To reduce circularity, we now use a *reduced feature set* that excludes
    the heavy stat components used in the main score. Prefer real observed prices when available.
    """
    if actual_price_col in row and not pd.isna(row.get(actual_price_col)):
        return round(min(1.0, max(0.0, float(row.get(actual_price_col)))), 3)

    # Reduced set to minimize overlap with main scoring features
    market = row.get("team_market_score", 5.0) / 10.0
    prospect = normalize_prospect(row.get("prospect_rank_overall"), row.get("prospect_fv"))
    age = compute_age_upside(row.get("age_at_callup"))
    pos = compute_position_premium(row.get("position")) / 1.2

    # Add small noise for realism when no real data
    noise = np.random.normal(0, 0.05)
    proxy = (0.50 * market + 0.30 * prospect + 0.15 * age + 0.05 * pos) + noise
    return round(min(1.0, max(0.0, proxy)), 3)

def get_social_hype_multiplier(player_name: str, days_back: int = 7) -> float:
    """Social hype multiplier - PLACEHOLDER, currently a no-op.

    The previous version of this function returned a boosted multiplier for a hardcoded
    shortlist of player names (plain substring matching), which doesn't measure hype at all -
    it silently favored a few specific players in every score and every backtest, and gave
    everyone else a flat default. That's worse than having no signal, since it looks like a
    real feature while actually just hardcoding the answer for a few known names.

    Until this is wired up to a real data source (e.g. pulling recent post volume + sentiment
    for player_name over `days_back`), it returns a neutral multiplier so it has zero effect on
    scoring rather than a fake, name-specific one.
    """
    try:
        # TODO: replace with a real implementation:
        #   - pull recent post volume + engagement for player_name over `days_back`
        #   - score sentiment on the pulled text
        #   - map volume + sentiment to a multiplier (previously ranged ~0.85-1.45)
        return 1.0
    except Exception:
        return 1.0


def compute_topps_parallel_price(row: pd.Series, parallel='base', grade='raw', month=None,
                                  recent_momentum=0.0, set_demand=1.0, velocity_30d=20,
                                  hype=1.0, market_broad=1.0):
    """Exact parallel pricing for Topps sets + rare non-Topps cards.
    Now includes social hype multiplier."""
    market = row.get("team_market_score", 9.0) / 10.0
    prospect = normalize_prospect(row.get("prospect_rank_overall"), row.get("prospect_fv"))
    age_up = compute_age_upside(row.get("age_at_callup"))
    stat = normalize_stat(row.get("minor_league_recent_ops"))
    pos_prem = compute_position_premium(row.get("position")) / 1.2

    base = (0.28 * market + 0.20 * prospect + 0.12 * age_up + 0.10 * pos_prem + 0.08 * stat)

    momentum_f = 1.0 + (recent_momentum * 0.12)
    set_demand_f = max(0.7, min(1.8, set_demand))
    velocity_f = min(1.4, max(0.8, 0.9 + (velocity_30d - 20) / 50))
    hype_f = max(0.7, min(1.8, hype))
    market_f = max(0.8, min(1.3, market_broad))

    # New: Social hype
    social_hype = get_social_hype_multiplier(row.get("player_name", ""))
    social_f = social_hype

    enhanced = base * momentum_f * set_demand_f * velocity_f * hype_f * market_f * social_f

    mult = PARALLEL_MULTIPLIERS.get(parallel, 1.0)
    grade_mult = 1.0 if grade == 'raw' else 3.8

    month_adj = 1.0
    if month in [3,4,5]: month_adj = 1.1
    elif month in [10,11,12]: month_adj = 1.05

    final = enhanced * mult * grade_mult * month_adj
    # NOTE: this used to be capped at min(3.0, ...), which silently flattened every parallel
    # rarer than 'base' down to the same few-dollar ceiling - the superfractor (180x) and graded
    # (3.8x) multipliers above had no effect on the output once they crossed that cap, which
    # defeats the point of having an "exact parallel pricing" table at all. Replaced with a much
    # higher sanity ceiling that only guards against pathological combinations of momentum/
    # demand/velocity/hype/market all maxing out at once - it isn't a calibrated price ceiling,
    # so tune it against real comps if you have them.
    return round(min(50000.0, max(0.05, final)), 4)

def compute_callup_score(row: Union[pd.Series, dict], weights: dict) -> float:
    w_stat = weights.get("stat_performance", 0.15)
    w_prospect = weights.get("prospect_pedigree", 0.20)
    w_team = weights.get("team_market", 0.20)
    w_health = weights.get("health_injury", 0.07)
    w_prior = weights.get("prior_experience", 0.03)
    w_age = weights.get("age_upside", 0.07)
    w_pos = weights.get("position_premium", 0.08)
    w_level = weights.get("level_reached", 0.04)
    w_def = weights.get("defensive_impact", 0.02)
    w_pos_norm = weights.get("position_specific_norm", 0.14)

    pos_norm = normalize_position_stats(row)

    total = (
        normalize_stat(row.get("minor_league_recent_ops")) * w_stat +
        normalize_prospect(row.get("prospect_rank_overall"), row.get("prospect_fv")) * w_prospect +
        (row.get("team_market_score", 5.0) / 10.0) * w_team +
        compute_injury_factor(row.get("injury_days_missed_last_2yrs"), row.get("surgery_history", 0)) * w_health +
        compute_prior_factor(row.get("prior_mlb_stints", 0)) * w_prior +
        compute_age_upside(row.get("age_at_callup")) * w_age +
        compute_position_premium(row.get("position")) * w_pos +
        compute_level_factor(row.get("highest_level")) * w_level +
        normalize_defensive(row.get("defensive_tools")) * w_def +
        pos_norm * w_pos_norm
    )
    return round(total, 4)

def compute_component_scores(df: pd.DataFrame, recompute: bool = True) -> pd.DataFrame:
    """Adds the per-component normalized columns used by compute_callup_score, plus card_proxy.

    Performance notes (no scoring values changed - see regression check below):
    - stat_norm, position_norm, level_norm, defensive_norm are now computed with vectorized
      pandas/numpy ops instead of `.apply()`, since they're simple single-column transforms.
      prospect_norm, health_norm, prior_norm, age_norm, pos_specific_norm, and card_proxy are
      left as `.apply()` - they branch on multiple columns (e.g. pos_specific_norm's weighting
      differs entirely by position) and a vectorized rewrite would carry real risk of subtly
      diverging from the original for comparatively little gain on datasets this size.
    - `recompute=False` skips any column that's already present in `df`, instead of
      unconditionally recomputing it. score_historical() and learn_weights_multi() use this so a
      pipeline like compute_component_scores -> score_historical doesn't redo the same work
      twice on every run. Columns that are missing are always computed regardless of this flag.
      Caveat: if you mutate the underlying raw stat columns on a df that already has these
      component columns attached, drop the component columns first (or pass recompute=True) -
      otherwise you'll get stale values for the ones that already exist.
    """
    df = df.copy()
    def _need(col): return recompute or col not in df.columns

    if _need("stat_norm"):
        ops = df["minor_league_recent_ops"]
        raw = ((ops - 0.650) / 0.35).clip(lower=0.0, upper=1.0)
        bad = ops.isna() | (ops <= 0)
        df["stat_norm"] = raw.mask(bad, 0.5).round(3)
    if _need("prospect_norm"):
        df["prospect_norm"] = df.apply(lambda r: normalize_prospect(r.get("prospect_rank_overall"), r.get("prospect_fv")), axis=1)
    if _need("team_norm"):
        df["team_norm"] = df.get("team_market_score", 5.0) / 10.0
    if _need("health_norm"):
        if "injury_days_missed_last_2yrs" not in df.columns:
            df["injury_days_missed_last_2yrs"] = 0
        if "surgery_history" not in df.columns:
            df["surgery_history"] = 0
        df["health_norm"] = df.apply(lambda r: compute_injury_factor(r.get("injury_days_missed_last_2yrs"), r.get("surgery_history", 0)), axis=1)
    if _need("prior_norm"):
        if "prior_mlb_stints" not in df.columns:
            df["prior_mlb_stints"] = 0
        df["prior_norm"] = df["prior_mlb_stints"].apply(compute_prior_factor)
    if _need("age_norm"):
        df["age_norm"] = df["age_at_callup"].apply(compute_age_upside)
    if _need("position_norm"):
        pos_clean = df["position"].astype(str).str.upper().str.strip()
        df["position_norm"] = pos_clean.map(_POSITION_PREMIUM_TABLE).fillna(1.0)
    if _need("level_norm"):
        level_clean = df["highest_level"].astype(str).str.upper().str.strip()
        df["level_norm"] = level_clean.map(_LEVEL_FACTOR_TABLE).fillna(1.0)
    if _need("defensive_norm"):
        if "defensive_tools" not in df.columns:
            df["defensive_tools"] = 50
        tools = df["defensive_tools"]
        raw = ((tools - 40) / 40).clip(lower=0.0, upper=1.0)
        bad = tools.isna() | (tools <= 0)
        df["defensive_norm"] = raw.mask(bad, 0.5).round(3)
    if _need("pos_specific_norm"):
        df["pos_specific_norm"] = df.apply(normalize_position_stats, axis=1)
    if _need("card_proxy"):
        df["card_proxy"] = df.apply(compute_card_price_proxy, axis=1)
    return df

def _callup_score_from_components(df: pd.DataFrame, weights: dict) -> pd.Series:
    """Vectorized equivalent of compute_callup_score, applied to the whole dataframe at once
    instead of via a per-row Python function call. Requires the component columns already added
    by compute_component_scores(). Used by score_historical() for the bulk-scoring path; keep
    this in sync with compute_callup_score() if either one changes - compute_callup_score()
    itself is kept as-is for scoring a single new/ad hoc prospect dict."""
    w_stat = weights.get("stat_performance", 0.15)
    w_prospect = weights.get("prospect_pedigree", 0.20)
    w_team = weights.get("team_market", 0.20)
    w_health = weights.get("health_injury", 0.07)
    w_prior = weights.get("prior_experience", 0.03)
    w_age = weights.get("age_upside", 0.07)
    w_pos = weights.get("position_premium", 0.08)
    w_level = weights.get("level_reached", 0.04)
    w_def = weights.get("defensive_impact", 0.02)
    w_pos_norm = weights.get("position_specific_norm", 0.14)

    total = (
        df["stat_norm"] * w_stat +
        df["prospect_norm"] * w_prospect +
        df["team_norm"] * w_team +
        df["health_norm"] * w_health +
        df["prior_norm"] * w_prior +
        df["age_norm"] * w_age +
        df["position_norm"] * w_pos +
        df["level_norm"] * w_level +
        df["defensive_norm"] * w_def +
        df["pos_specific_norm"] * w_pos_norm
    )
    return total.round(4)

def learn_weights_multi(df: pd.DataFrame = None, war_col: str = "post_callup_first_year_approx_war",
                        card_col: str = "card_proxy", war_weight: float = 0.55, card_weight: float = 0.45,
                        save: bool = True) -> dict:
    if df is None: df = load_historical_callups()
    df = compute_component_scores(df, recompute=False)
    feature_cols = ["stat_norm", "prospect_norm", "team_norm", "health_norm", "prior_norm",
                    "age_norm", "position_norm", "level_norm", "defensive_norm", "pos_specific_norm"]
    X = df[feature_cols].values
    y_war = df[war_col].values.astype(float)
    y_card = df[card_col].values.astype(float)

    def objective(w):
        pred = X @ w
        war_corr = np.corrcoef(pred, y_war)[0, 1] if np.std(pred) > 1e-6 and np.std(y_war) > 1e-6 else 0.0
        card_corr = np.corrcoef(pred, y_card)[0, 1] if np.std(pred) > 1e-6 and np.std(y_card) > 1e-6 else 0.0
        if np.isnan(war_corr): war_corr = 0.0
        if np.isnan(card_corr): card_corr = 0.0
        return -(war_weight * war_corr + card_weight * card_corr)

    n_features = len(feature_cols)
    bounds = Bounds([0.01] * n_features, [0.5] * n_features)
    constraints = LinearConstraint(np.ones((1, n_features)), [1.0], [1.0])

    # Better initialization using previous weights if available
    w0 = np.full(n_features, 1.0 / n_features)
    try:
        prev_weights = load_weights()
        # map if possible
        if len(prev_weights) == n_features:
            w0 = np.array([prev_weights.get(name, 1.0/n_features) for name in weight_names])
    except:
        pass

    res = minimize(objective, w0, bounds=bounds, constraints=constraints, method="SLSQP", options={"maxiter": 1000, "ftol": 1e-10})
    w_opt = res.x

    weight_names = ["stat_performance", "prospect_pedigree", "team_market", "health_injury",
                    "prior_experience", "age_upside", "position_premium", "level_reached",
                    "defensive_impact", "position_specific_norm"]
    learned = {name: round(float(val), 4) for name, val in zip(weight_names, w_opt)}
    total = sum(learned.values())
    if abs(total - 1.0) > 0.01:
        for k in learned: learned[k] = round(learned[k] / total, 4)

    if save:
        with open(CONFIG_DIR / "learned_multi_weights_v8.json", "w") as f:
            json.dump(learned, f, indent=2)
        print("   Saved learned_multi_weights_v8.json")
    return learned

def score_historical(df: pd.DataFrame = None, weights: dict = None) -> pd.DataFrame:
    if df is None: df = load_historical_callups()
    if weights is None: weights = load_weights()
    df = compute_component_scores(df, recompute=False)
    # NOTE: previously recomputed every normalization from scratch via a per-row Python loop
    # (itertuples -> dict -> compute_callup_score), which then got recomputed *again* by any
    # caller that also called compute_component_scores on the result (the __main__ block did
    # exactly this). Now scores from the already-computed component columns in one vectorized
    # pass, and recompute=False above means those columns aren't redone if a caller (e.g.
    # efficacy_test) already attached them.
    df["callup_score"] = _callup_score_from_components(df, weights)
    # NOTE: .astype(int) used to crash here (IntCastingNaNError) any time a row had a NaN
    # callup_score (e.g. a missing team_market_score propagating through the weighted sum) -
    # which will happen on real, incomplete daily data. Int64 is pandas' nullable integer dtype,
    # so NaN ranks (from NaN scores) come through as <NA> instead of crashing the whole run.
    df["score_rank"] = df["callup_score"].rank(ascending=False, method="min").astype("Int64")
    return df.sort_values("callup_score", ascending=False)

def efficacy_test(df: pd.DataFrame = None, weights: dict = None, test_size: float = 0.3, random_state: int = 42):
    """Single 70/30 split efficacy test.

    NOTE: on a dataset of a few hundred call-ups, a single random split can swing noticeably
    with the seed - treat oos_war_corr/oos_card_corr here as one sample, not a stable estimate.
    See efficacy_test_kfold() below for a mean +/- spread across multiple folds.
    """
    if df is None: df = load_historical_callups()
    df = compute_component_scores(df)
    if weights is None: weights = load_weights()

    scored = score_historical(df=df, weights=weights)
    war_corr = scored["callup_score"].corr(scored["post_callup_first_year_approx_war"])
    card_corr = scored["callup_score"].corr(scored["card_proxy"])
    spearman_war = scored["callup_score"].corr(scored["post_callup_first_year_approx_war"], method="spearman")
    spearman_card = scored["callup_score"].corr(scored["card_proxy"], method="spearman")

    np.random.seed(random_state)
    indices = np.random.permutation(len(df))
    test_idx = indices[:int(len(df) * test_size)]
    train_idx = indices[int(len(df) * test_size):]
    train_df = df.iloc[train_idx].copy()
    test_df = df.iloc[test_idx].copy()

    train_learned = learn_weights_multi(train_df, war_weight=0.55, card_weight=0.45, save=False)
    test_scored = score_historical(df=test_df, weights=train_learned)
    oos_war_corr = test_scored["callup_score"].corr(test_scored["post_callup_first_year_approx_war"])
    oos_card_corr = test_scored["callup_score"].corr(test_scored["card_proxy"])

    random_scores = np.random.permutation(scored["callup_score"].values)
    random_war_corr = np.corrcoef(random_scores, scored["post_callup_first_year_approx_war"].values)[0, 1]
    random_card_corr = np.corrcoef(random_scores, scored["card_proxy"].values)[0, 1]

    return {
        "full_war_pearson": round(war_corr, 4) if not pd.isna(war_corr) else None,
        "full_card_pearson": round(card_corr, 4) if not pd.isna(card_corr) else None,
        "full_war_spearman": round(spearman_war, 4) if not pd.isna(spearman_war) else None,
        "full_card_spearman": round(spearman_card, 4) if not pd.isna(spearman_card) else None,
        "oos_war_corr": round(oos_war_corr, 4) if not pd.isna(oos_war_corr) else None,
        "oos_card_corr": round(oos_card_corr, 4) if not pd.isna(oos_card_corr) else None,
        "random_war_corr": round(random_war_corr, 4) if not pd.isna(random_war_corr) else None,
        "random_card_corr": round(random_card_corr, 4) if not pd.isna(random_card_corr) else None,
        "train_size": len(train_df),
        "test_size": len(test_df)
    }

def efficacy_test_kfold(df: pd.DataFrame = None, n_folds: int = 5, war_weight: float = 0.55,
                         card_weight: float = 0.45, random_state: int = 42) -> dict:
    """K-fold cross-validated efficacy test.

    NOTE: For true time-series validation, sort by callup date and use walk-forward splits instead of random.
    Current random k-fold is improved but still not ideal for prospect data with temporal trends.
    """
    if df is None: df = load_historical_callups()
    df = compute_component_scores(df).reset_index(drop=True)

    rng = np.random.RandomState(random_state)
    indices = rng.permutation(len(df))
    folds = np.array_split(indices, n_folds)

    oos_war_corrs, oos_card_corrs = [], []
    for i in range(n_folds):
        test_idx = folds[i]
        train_idx = np.concatenate([folds[j] for j in range(n_folds) if j != i])
        train_df = df.iloc[train_idx].copy()
        test_df = df.iloc[test_idx].copy()

        fold_weights = learn_weights_multi(train_df, war_weight=war_weight, card_weight=card_weight, save=False)
        test_scored = score_historical(df=test_df, weights=fold_weights)

        war_c = test_scored["callup_score"].corr(test_scored["post_callup_first_year_approx_war"])
        card_c = test_scored["callup_score"].corr(test_scored["card_proxy"])
        oos_war_corrs.append(war_c)
        oos_card_corrs.append(card_c)

    war_arr = np.array(oos_war_corrs, dtype=float)
    card_arr = np.array(oos_card_corrs, dtype=float)
    return {
        "n_folds": n_folds,
        "oos_war_corr_mean": round(float(np.nanmean(war_arr)), 4),
        "oos_war_corr_std": round(float(np.nanstd(war_arr)), 4),
        "oos_card_corr_mean": round(float(np.nanmean(card_arr)), 4),
        "oos_card_corr_std": round(float(np.nanstd(card_arr)), 4),
        "fold_war_corrs": [round(float(x), 4) if not np.isnan(x) else None for x in war_arr],
        "fold_card_corrs": [round(float(x), 4) if not np.isnan(x) else None for x in card_arr],
    }


def time_based_validation(df: pd.DataFrame = None, train_cutoff_year: int = 2023,
                           war_weight: float = 0.55, card_weight: float = 0.45) -> dict:
    """Time-based (walk-forward) validation - CRITICAL for prospect data.
    
    Sorts by callup date and trains on earlier data, tests on later data.
    Much more realistic than random splits.
    """
    if df is None:
        df = load_historical_callups()
    
    if "callup_date" not in df.columns:
        print("⚠️ No 'callup_date' column found. Falling back to random split.")
        return efficacy_test(df, test_size=0.3)
    
    df = df.copy()
    df["callup_date"] = pd.to_datetime(df["callup_date"], errors='coerce')
    df = df.dropna(subset=["callup_date"]).sort_values("callup_date")
    
    train_df = df[df["callup_date"].dt.year <= train_cutoff_year].copy()
    test_df = df[df["callup_date"].dt.year > train_cutoff_year].copy()
    
    if len(train_df) < 30 or len(test_df) < 10:
        print(f"⚠️ Insufficient data for time split (train: {len(train_df)}, test: {len(test_df)})")
        return {"status": "insufficient_data"}
    
    print(f"Time-based split: Train (<= {train_cutoff_year}): {len(train_df)} | Test (> {train_cutoff_year}): {len(test_df)}")
    
    train_learned = learn_weights_multi(train_df, war_weight=war_weight, card_weight=card_weight, save=False)
    test_scored = score_historical(df=test_df, weights=train_learned)
    
    oos_war = test_scored["callup_score"].corr(test_scored["post_callup_first_year_approx_war"])
    oos_card = test_scored["callup_score"].corr(test_scored["card_proxy"])
    
    return {
        "train_size": len(train_df),
        "test_size": len(test_df),
        "oos_war_corr": round(float(oos_war), 4) if not pd.isna(oos_war) else None,
        "oos_card_corr": round(float(oos_card), 4) if not pd.isna(oos_card) else None,
        "status": "success"
    }


if __name__ == "__main__":
    ensure_directories()

    print("=== MLB Call-Up Algorithm v8 (Final) ===")
    print("Consolidated: Learned weights, position next-gen, exact Topps parallels, monthly-ready price layers, efficacy.")

    df = load_historical_callups()
    print(f"\nDataset size: {len(df)} prospects")
    
    # Strong validation for production runs
    print("\n=== Dataset Validation ===")
    validate_dataset(df, strict=False)  # Change to True when real data is loaded

    print("\n1. Learning multi-objective weights...")
    multi_weights = learn_weights_multi(df, war_weight=0.55, card_weight=0.45)
    print("   Learned Weights v8:", multi_weights)

    print("\n2. Efficacy Test (Full + OOS vs Random):")
    metrics = efficacy_test(df, weights=multi_weights, test_size=0.3)
    for k, v in metrics.items():
        print(f"   {k}: {v}")

    print("\n2b. Efficacy Test (5-fold, more stable OOS estimate):")
    kfold_metrics = efficacy_test_kfold(df, n_folds=5, war_weight=0.55, card_weight=0.45)
    for k, v in kfold_metrics.items():
        print(f"   {k}: {v}")

    print("\n2c. Time-based Validation (Recommended):")
    time_metrics = time_based_validation(df)
    for k, v in time_metrics.items():
        print(f"   {k}: {v}")

    print("\n3. Top 10 scored:")
    # score_historical() already returns pos_specific_norm, card_proxy, and the rest of the
    # component columns now - the old code called compute_component_scores() again here,
    # redoing the full per-row normalization pass a second time for nothing.
    scored = score_historical(df=df, weights=multi_weights)
    print(scored[["player_name", "mlb_team", "position", "callup_score", "score_rank",
                  "post_callup_first_year_approx_war", "pos_specific_norm", "card_proxy"]].head(10).to_string(index=False))

    scored.to_csv(OUTPUT_DIR / "full_dataset_scored_v8.csv", index=False)
    print(f"\nFull scored results exported to output/full_dataset_scored_v8.csv")

    # Example exact parallel pricing for Roman Anthony (Topps focus)
    print("\n4. Example Exact Parallel Pricing (Roman Anthony - Topps/Bowman):")
    anthony_row = {
        'team_market_score': 9.0, 'prospect_fv': 75, 'age_at_callup': 22,
        'minor_league_recent_ops': 0.75, 'position': 'OF'
    }
    for par in ['base', 'refractor_499', 'gold_50', 'red_5', 'superfractor_1of1']:
        price = compute_topps_parallel_price(pd.Series(anthony_row), parallel=par, month=6,
                                             recent_momentum=0.08, set_demand=1.6, velocity_30d=80, hype=1.9)
        print(f"   {par}: {price}")

    print("\n=== Final Assessment ===")
    print("v8.1: price-cap and social-hype stub bugs fixed, config-clobber bug fixed, k-fold")
    print("eval added. None of the scoring/calibration weights were changed in this pass.")
    print("Note: the formulas/tables here are reproducible by anyone with this file - the real")
    print("defensibility is the historical dataset (real outcomes/comps) and track record, not the code.")
