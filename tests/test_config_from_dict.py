"""Guards experiment_config_from_dict: the notebook/legacy CONFIG dict -> ExperimentConfig."""
from llm_culture.config import (
    experiment_config_from_dict,
    validate_experiment,
    Backend,
    Network,
)


def _base_config(**overrides):
    config = {
        "n_agents": 5,
        "n_timesteps": 3,
        "n_seeds": 2,
        "network_structure": "fully_connected",
        "n_cliques": 2,
        "prompt_init": {"name": "kid", "prompt": "..."},
        "prompt_update": {"name": "Combine2", "prompt": "..."},
        "personalities": [{"name": "empty", "prompt": ""}] * 5,
        "output_name": "test_experiment",
        "debug": False,
        "llm_backend": "llama.cpp",
        "model": "unsloth/SmolLM2-135M-Instruct-GGUF",
        "access_url": None,
        "temperature": 0.8,
    }
    config.update(overrides)
    return config


def test_maps_core_fields():
    cfg = experiment_config_from_dict(_base_config())
    assert cfg.backend.kind is Backend.llama_cpp
    assert cfg.backend.model == "unsloth/SmolLM2-135M-Instruct-GGUF"
    assert cfg.backend.access_url == ""              # None -> ""
    assert cfg.population.network_structure is Network.fully_connected
    assert cfg.population.n_agents == 5               # one agent per personality entry
    assert cfg.population.n_cliques == 2
    assert cfg.generation.temperature == 0.8
    assert cfg.n_timesteps == 3
    assert cfg.n_seeds == 2
    assert cfg.output == "results/experiments/test_experiment"
    assert cfg.debug is False
    # agents carry the per-entry persona + shared prompt names
    assert [a.personality for a in cfg.population.agents] == ["empty"] * 5
    assert all(a.prompt_init == "kid" and a.prompt_update == "Combine2"
               for a in cfg.population.agents)
    validate_experiment(cfg)                          # built config is self-consistent


def test_mixed_population_from_personalities_list():
    cfg = experiment_config_from_dict(
        _base_config(personalities=[{"name": "Fantasy", "prompt": ""}] * 3
                     + [{"name": "SciFi", "prompt": ""}] * 2)
    )
    assert cfg.population.n_agents == 5
    assert [a.personality for a in cfg.population.agents] == (
        ["Fantasy"] * 3 + ["SciFi"] * 2
    )


def test_defaults_and_fallbacks():
    config = _base_config()
    del config["temperature"]
    del config["n_seeds"]
    config["llm_backend"] = "something_unknown"
    cfg = experiment_config_from_dict(config)
    assert cfg.generation.temperature == 0.8          # dataclass default
    assert cfg.n_seeds == 1                            # dict default
    assert cfg.backend.kind is Backend.none            # unknown tag -> remote


def test_custom_network_name_uses_placeholder_enum():
    # A custom (non-enum) structure name is kept as a placeholder; the real graph
    # is built from the raw name elsewhere.
    cfg = experiment_config_from_dict(_base_config(network_structure="custom_graph"))
    assert cfg.population.network_structure is Network.fully_connected


def test_accepts_notebook_dataclass_equivalently():
    from llm_culture.config import NotebookConfig, Prompt
    nb = NotebookConfig(
        n_agents=5,
        n_timesteps=3,
        n_seeds=2,
        network_structure="fully_connected",
        prompt_init=Prompt("kid", "..."),
        prompt_update=Prompt("Combine2", "..."),
        personalities=[Prompt("empty", "")] * 5,
        output_name="test_experiment",
        llm_backend="llama.cpp",
        model="unsloth/SmolLM2-135M-Instruct-GGUF",
        temperature=0.8,
    )
    from_dc = experiment_config_from_dict(nb)
    from_dict = experiment_config_from_dict(_base_config())
    # the typed dataclass path yields the same ExperimentConfig as the dict path
    assert from_dc == from_dict
    assert from_dc.backend.kind is Backend.llama_cpp
    assert from_dc.population.n_agents == 5

