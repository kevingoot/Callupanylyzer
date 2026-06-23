def load_weights(config_path: Path = None):
    if config_path is None:
        config_path = CONFIG_DIR / "weights.json"
    if not config_path.exists():
        # Create default if missing
        default = {
          "stat_performance": 0.15, "prospect_pedigree": 0.20, "team_market": 0.20,
          "health_injury": 0.07, "prior_experience": 0.03, "age_upside": 0.07,
          "position_premium": 0.08, "level_reached": 0.04, "defensive_impact": 0.02,
          "position_specific_norm": 0.14
        }
        config_path.parent.mkdir(parents=True, exist_ok=True)
        with open(config_path, "w") as f:
            json.dump(default, f, indent=2)
        return default
    with open(config_path, "r") as f:
        weights = json.load(f)
    return {k: v for k, v in weights.items() if isinstance(v, (int, float))}
