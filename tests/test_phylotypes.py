"""Distance-metric and end-to-end tests for the Phylotypes pipeline."""

import io
import json
from pathlib import Path

import pytest
import torch

from phylotypes.phylotypes import Phylotypes

FIXTURE = Path(__file__).parent / "fixture_jplace.json"


def _load(metric: str = "legacy") -> Phylotypes:
    p = Phylotypes(lwr_overlap=0.01, pd_threshold=1.0, distance=metric)
    with FIXTURE.open() as fh:
        p.load_jplace(fh)
    return p


def _legacy_loop_reference(p: Phylotypes, idx, *, distal_length: bool = True) -> torch.Tensor:
    """Ground truth for the ORIGINAL metric: a faithful per-pair loop, mirroring
    add_phylotypes.placement_pairwise_distance. The vectorized `_pairwise_legacy`
    must reproduce this to float precision."""
    names = [p.placement_names[i] for i in idx]
    n = len(names)
    out = torch.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            pa = {k: (v[p.lwr_idx], v[p.dl_idx] if distal_length else 0.0)
                  for k, v in p.sv_nodes[names[i]].items()}
            pb = {k: (v[p.lwr_idx], v[p.dl_idx] if distal_length else 0.0)
                  for k, v in p.sv_nodes[names[j]].items()}
            wa = sum(v[0] for v in pa.values())
            wb = sum(v[0] for v in pb.values())
            d = 0.0
            overlap = set(pa) & set(pb)
            d += sum(pa[k][1] * pa[k][0] / wa + pb[k][1] * pb[k][0] / wb for k in overlap)
            distant = (set(pa) | set(pb)) - overlap
            if distant:
                lca = p.tree.lowest_common_ancestor([p.name_node[str(k)] for k in distant])
                d += (sum((v[1] + lca.distance(p.name_node[str(k)])) * v[0]
                          for k, v in pa.items() if k in distant) / wa
                    + sum((v[1] + lca.distance(p.name_node[str(k)])) * v[0]
                          for k, v in pb.items() if k in distant) / wb)
            out[i, j] = out[j, i] = d
    return out


def test_fixture_loads():
    p = _load()
    assert p.tree is not None
    assert p.placement_names == ["sv1", "sv2", "sv3", "sv4"]


def test_legacy_matches_loop_reference():
    """Vectorized legacy metric must reproduce the original per-pair loop exactly."""
    p = _load("legacy")
    idx = [0, 1, 2, 3]
    fast = p.pairwise_distance(idx, metric="legacy")
    ref = _legacy_loop_reference(p, idx)
    assert torch.allclose(fast, ref, atol=1e-5), f"\nfast=\n{fast}\nref=\n{ref}"


def test_legacy_tree_distance_not_dropped():
    """Cross-clade pair must be much farther than siblings (guards the old bug)."""
    p = _load("legacy")
    emd = p.pairwise_distance([0, 1, 2, 3], metric="legacy")
    assert emd[0, 2] > emd[0, 1] + 0.3, (emd[0, 1].item(), emd[0, 2].item())


def test_kr_is_a_metric_and_matches_expected():
    """KR (true tree-Wasserstein): validated values + metric axioms."""
    p = _load("kr")
    emd = p.pairwise_distance([0, 1, 2, 3], metric="kr")
    expected = torch.tensor([
        [0.000, 0.100, 0.500, 0.508],
        [0.100, 0.000, 0.500, 0.508],
        [0.500, 0.500, 0.000, 0.068],
        [0.508, 0.508, 0.068, 0.000],
    ])
    assert torch.allclose(emd, expected, atol=1e-3), emd
    assert torch.allclose(emd, emd.t(), atol=1e-6)        # symmetric
    assert bool((torch.diagonal(emd) == 0).all())         # zero diagonal
    for a in range(4):                                    # triangle inequality
        for b in range(4):
            for c in range(4):
                assert emd[a, b] <= emd[a, c] + emd[c, b] + 1e-5


def test_generate_phylotypes_partitions_all_svs():
    for metric in ("legacy", "kr"):
        p = _load(metric)
        p.generate_phylotypes()
        grouped = {sv for grp in p.phylogroups for sv in grp}
        assert grouped == {"sv1", "sv2", "sv3", "sv4"}, metric


def test_missing_tree_raises():
    p = Phylotypes()
    bad = io.StringIO(json.dumps({"fields": ["edge_num", "like_weight_ratio",
                                              "distal_length"], "placements": []}))
    with pytest.raises(ValueError):
        p.load_jplace(bad)
