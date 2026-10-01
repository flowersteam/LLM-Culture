"""Run a full experiment (simulation + analysis) from a Hydra config.

Configuration is a structured config: the `ExperimentConfig` dataclass
(llm_culture/config.py) is registered as a Hydra schema, so the YAML in `conf/`
is validated against it and converted straight into a typed dataclass instance —
no argparse involved on this path.

Run from the default config, overriding fields as needed:

    uv run python run_experiment.py \\
        backend=llama_cpp model=unsloth/SmolLM2-135M-Instruct-GGUF \\
        n_agents=2 n_timesteps=2 n_seeds=1 output=results/my_test verbose=true

Or select a ready-made experiment preset from conf/experiment/ (copy one to make
your own), optionally overriding fields on top:

    uv run python run_experiment.py +experiment=base
    uv run python run_experiment.py +experiment=base n_timesteps=10

Sweep with -m (multirun):

    uv run python run_experiment.py -m n_agents=2,5,10 +experiment=base

The standalone scripts/run_simulation.py and scripts/run_analysis.py still work
with their original argparse flags (used by the reproduction scripts); this
Hydra entrypoint shares their code via run_simulation_from_config.
"""
import sys
from pathlib import Path

import hydra
from hydra.core.config_store import ConfigStore
from omegaconf import DictConfig, OmegaConf

# Ensure the repo root is importable (so `scripts` resolves) regardless of cwd.
sys.path.insert(0, str(Path(__file__).resolve().parent))

from llm_culture.config import ExperimentConfig, validate_experiment
from scripts.run_simulation import run_simulation_from_config
from scripts.run_analysis import main_analysis
from llm_culture.analysis.utils import initialize_nltk

cs = ConfigStore.instance()
cs.store(name="experiment_schema", node=ExperimentConfig)


@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(cfg: DictConfig) -> None:
    # Convert the validated DictConfig into a typed ExperimentConfig instance.
    exp: ExperimentConfig = OmegaConf.to_object(cfg)
    validate_experiment(exp)

    print("\n" + "#" * 64)
    print("# EXPERIMENT CONFIG")
    print("#" * 64)
    print(OmegaConf.to_yaml(cfg), end="")

    print("#" * 64)
    print("# STEP 1/2 — SIMULATION")
    print("#" * 64)
    run_simulation_from_config(exp)

    if not exp.analysis.run:
        print(f"\nDone (simulation only). Outputs in {Path(exp.output).resolve()}")
        return

    print("\n" + "#" * 64)
    print("# STEP 2/2 — ANALYSIS")
    print("#" * 64)
    initialize_nltk()
    main_analysis(
        str(exp.output),
        exp.analysis.font_sizes,
        exp.analysis.plot,
        force_recompute_cache=exp.analysis.recompute_cache,
    )

    folder = Path(exp.output).resolve()
    outputs = sorted(folder.glob("output*.json"))
    plots = sorted(folder.glob("*.png"))
    print("\n" + "=" * 64)
    print("EXPERIMENT COMPLETE")
    print(f"  folder: {folder}")
    print(f"  simulation outputs: {len(outputs)} file(s) ({', '.join(p.name for p in outputs) or 'none'})")
    print(f"  plots: {len(plots)} PNG(s)")
    print("=" * 64)


if __name__ == "__main__":
    main()
