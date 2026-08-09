from pathlib import Path

from scripts.ops.laptop_pipeline import DEFAULT_CONFIG_PATH, load_pipeline_config


def test_load_pipeline_config_reads_safe_opt_in_defaults():
    config = load_pipeline_config(DEFAULT_CONFIG_PATH)
    assert config["pipeline"]["name"] == "laptop_pipeline"
    assert config["indexing"]["enabled"] is False
    assert config["connectivity"]["rounds"] == 3
    assert "ollama_tags_path" not in config["connectivity"]
    assert config["connectivity"]["hindsight_health_url"].endswith("/health")


def test_laptop_config_has_one_connectivity_mapping():
    raw = Path(DEFAULT_CONFIG_PATH).read_text(encoding="utf-8")
    assert raw.count("\nconnectivity:") == 1
