"""Run a full experiment (simulation + analysis) in a single command.

This is a convenience wrapper. It does NOT replace the standalone scripts:
`scripts/run_simulation.py` and `scripts/run_analysis.py` keep working exactly
as before. This orchestrator just reuses their code so behavior stays identical
— it runs the simulation, then runs the analysis (which saves the plots) on the
same output folder.

Example (local llama.cpp backend on macOS, tiny model):

    uv run python run_experiment.py \\
        -na 2 -nt 2 -s 1 -o results/my_test \\
        --use_llama_cpp --model unsloth/SmolLM2-135M-Instruct-GGUF

All simulation flags are identical to scripts/run_simulation.py (run with -h to
see them). Analysis-side options are grouped under "analysis" below.
"""
import sys
from pathlib import Path

# Ensure the repo root is importable (so `scripts` resolves) regardless of cwd.
sys.path.insert(0, str(Path(__file__).resolve().parent))

from scripts.run_simulation import build_parser, main as run_simulation_main
from scripts.run_analysis import main_analysis
from llm_culture.analysis.utils import initialize_nltk


def parse_arguments():
    # Reuse the exact simulation flags, then add analysis-side options.
    parser = build_parser()
    parser.description = "Run a full experiment (simulation + analysis) in one command."
    analysis = parser.add_argument_group("analysis")
    analysis.add_argument("--no_analysis", action="store_true",
                          help="Only run the simulation; skip the analysis/plots step.")
    analysis.add_argument("--plot", action="store_true",
                          help="Also display plots interactively (they are saved to the folder either way).")
    analysis.add_argument("--no_recompute_cache", action="store_true",
                          help="Reuse an existing analysis cache instead of recomputing from fresh sim output.")
    analysis.add_argument("--ticks_font_size", type=int, default=12)
    analysis.add_argument("--labels_font_size", type=int, default=14)
    analysis.add_argument("--title_font_size", type=int, default=16)
    return parser.parse_args()


def main():
    args = parse_arguments()

    print("\n" + "#" * 64)
    print("# STEP 1/2 — SIMULATION")
    print("#" * 64)
    run_simulation_main(args)

    if args.no_analysis:
        print(f"\nDone (simulation only). Outputs in {Path(args.output).resolve()}")
        return

    print("\n" + "#" * 64)
    print("# STEP 2/2 — ANALYSIS")
    print("#" * 64)
    initialize_nltk()
    font_sizes = {
        "ticks": args.ticks_font_size,
        "labels": args.labels_font_size,
        "title": args.title_font_size,
    }
    main_analysis(
        str(args.output),
        font_sizes,
        args.plot,
        # Fresh sim output was just written, so recompute unless told otherwise.
        force_recompute_cache=not args.no_recompute_cache,
    )

    folder = Path(args.output).resolve()
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
