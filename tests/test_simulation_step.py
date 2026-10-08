"""Semantics of the per-timestep update, incl. the batched-generation path.

Generation is stubbed (no real LLM): server_answer.get_answer is monkeypatched to
echo its prompt, so each agent's output is a known function of ITS OWN prompt. That
lets us assert there is no cross-assignment between agents — the main risk when the
generations are run as one batch instead of one at a time. update_step routes all
generation through get_answers_batch, which (for the non-vLLM backends used here)
calls server_answer.get_answer, so patching it covers both batch and non-batch paths.
"""
import networkx as nx
import pytest

from llm_culture.config import ExperimentConfig
from llm_culture.simulation.utils import init_agents, update_step


def _echo_get_answer(access_url, prompt, generation, **kwargs):
    """Stand-in for server_answer.get_answer: echo the prompt back."""
    return f"ECHO::{prompt}"


def _agent_specs(personalities):
    # one (personality, prompt_init, prompt_update) triple per agent
    return [(p, "INIT", "UPD") for p in personalities]


def _build(cfg, graph, personalities):
    agents = init_agents(cfg, graph, _agent_specs(personalities))
    for a in agents:
        a.update_neighbours(graph, agents)
    return agents


@pytest.fixture
def cfg():
    # Defaults suffice: generation.instruct=True, backend.access_url="", debug=False,
    # backend.kind=none -> agents carry llm_backend=False (the remote/concurrent path).
    return ExperimentConfig()


@pytest.fixture(autouse=True)
def _stub_generation(monkeypatch):
    # get_answers_batch resolves get_answer as a server_answer module global.
    monkeypatch.setattr("llm_culture.simulation.server_answer.get_answer", _echo_get_answer)


def test_fully_connected_routing_is_positional(cfg):
    """Non-sequence network: every agent generates, each keeps its own output."""
    personalities = ["PERSONA_0", "PERSONA_1", "PERSONA_2"]
    graph = nx.complete_graph(len(personalities))  # undirected => non-sequence
    agents = _build(cfg, graph, personalities)

    new_stories = update_step(agents)

    assert len(new_stories) == len(agents)            # all eligible this step
    assert len({a.prompt for a in agents}) == len(agents)  # prompts distinct => routing is meaningful
    for a in agents:                                  # no cross-assignment
        assert a.story == f"ECHO::{a.prompt}"
    assert new_stories == [a.story for a in agents]   # returned in agent order


def test_sequence_advances_one_agent_per_step(cfg):
    """Sequence (transmission chain): exactly one agent generates per step, in order."""
    personalities = ["A", "B", "C"]
    graph = nx.path_graph(len(personalities), create_using=nx.DiGraph)  # directed => sequence
    agents = _build(cfg, graph, personalities)

    new1 = update_step(agents)                         # step 1 -> head of chain only
    assert len(new1) == 1
    assert agents[0].story is not None
    assert agents[1].story is None and agents[2].story is None
    assert new1 == [agents[0].story]

    new2 = update_step(agents)                         # step 2 -> advances to agent 1 only
    assert len(new2) == 1
    assert agents[1].story is not None
    assert agents[0].story is None and agents[2].story is None
    assert new2 == [agents[1].story]


def test_batched_equals_sequential(cfg):
    """Refactor guard: batched update_step reproduces the one-at-a-time path exactly
    (same per-step stories, same agent->story mapping, same wait bookkeeping)."""
    personalities = ["A", "B", "C", "D"]

    def run(batch):
        graph = nx.complete_graph(len(personalities))
        agents = _build(cfg, graph, personalities)
        step1 = update_step(agents, batch=batch)
        step2 = update_step(agents, batch=batch)       # 2nd step: prompts now include neighbour stories
        mapping = {a.agent_id: a.story for a in agents}
        waits = {a.agent_id: a.wait for a in agents}
        return step1, step2, mapping, waits

    assert run(batch=False) == run(batch=True)
