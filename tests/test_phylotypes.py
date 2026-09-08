"""Distance-metric and end-to-end tests for the Phylotypes pipeline."""

import io
import itertools
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
            pa = {k: (v[p.lwr_idx], v[p.dl_idx] if distal_length else 0.0) for k, v in p.sv_nodes[names[i]].items()}
            pb = {k: (v[p.lwr_idx], v[p.dl_idx] if distal_length else 0.0) for k, v in p.sv_nodes[names[j]].items()}
            wa = sum(v[0] for v in pa.values())
            wb = sum(v[0] for v in pb.values())
            d = 0.0
            overlap = set(pa) & set(pb)
            d += sum(pa[k][1] * pa[k][0] / wa + pb[k][1] * pb[k][0] / wb for k in overlap)
            distant = (set(pa) | set(pb)) - overlap
            if distant:
                lca = p.tree.lowest_common_ancestor([p.name_node[str(k)] for k in distant])
                d += (
                    sum((v[1] + lca.distance(p.name_node[str(k)])) * v[0] for k, v in pa.items() if k in distant) / wa
                    + sum((v[1] + lca.distance(p.name_node[str(k)])) * v[0] for k, v in pb.items() if k in distant) / wb
                )
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
    expected = torch.tensor(
        [
            [0.000, 0.100, 0.500, 0.508],
            [0.100, 0.000, 0.500, 0.508],
            [0.500, 0.500, 0.000, 0.068],
            [0.508, 0.508, 0.068, 0.000],
        ]
    )
    assert torch.allclose(emd, expected, atol=1e-3), emd
    assert torch.allclose(emd, emd.t(), atol=1e-6)  # symmetric
    assert bool((torch.diagonal(emd) == 0).all())  # zero diagonal
    for a in range(4):  # triangle inequality
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
    bad = io.StringIO(json.dumps({"fields": ["edge_num", "like_weight_ratio", "distal_length"], "placements": []}))
    with pytest.raises(ValueError):
        p.load_jplace(bad)


def test_load_jplace_handles_unnamed_nodes():
    """EPA-ng trees leave some nodes unnamed (e.g. the root carries no {edge_num}).
    The node-name cleanup must skip those rather than crash on node.name.replace()."""
    tree = "((A:0.1{0},B:0.1{1}):0.2{2},(C:0.1{3},D:0.1{4}):0.2{5});"
    jplace = {
        "version": 3,
        "tree": tree,
        "fields": ["edge_num", "like_weight_ratio", "distal_length"],
        "placements": [{"p": [[0, 1.0, 0.05]], "n": ["sv1"]}],
        "metadata": {},
    }
    p = Phylotypes()
    p.load_jplace(io.StringIO(json.dumps(jplace)))
    assert "sv1" in p.sv_nodes


def test_max_pregroup_size_splits_oversized_pregroups():
    """A max_pregroup_size of 0 forces every merged pregroup to be split back
    into its pre-merge groups; all SVs must still appear, exactly once."""
    p = _load("legacy")
    p.max_pregroup_size = 0
    p._pregroup_by_lwr()
    all_svs = [p.placement_names[i] for grp in p._pregrouped_sv for i in grp]
    assert sorted(all_svs) == ["sv1", "sv2", "sv3", "sv4"]

    p.generate_phylotypes()
    grouped = {sv for grp in p.phylogroups for sv in grp}
    assert grouped == {"sv1", "sv2", "sv3", "sv4"}


def test_generate_phylotypes_incremental_partitions_all_svs():
    """Mirrors test_generate_phylotypes_partitions_all_svs for the incremental path.

    seed_size=1 / expand_batch_size=1 forces every stage (seed, apply, expand,
    reconcile) to run on this 4-SV fixture.
    """
    for metric in ("legacy", "kr"):
        p = _load(metric)
        p.generate_phylotypes_incremental(seed_size=1, expand_batch_size=1)
        grouped = {sv for grp in p.phylogroups for sv in grp}
        assert grouped == {"sv1", "sv2", "sv3", "sv4"}, metric


def _make_star_jplace(n_leaves: int = 20, svs_per_leaf: int = 15) -> dict:
    """A star tree (root + n_leaves, each edge length 1.0) with svs_per_leaf
    SVs placed identically (full LWR, zero distal length) on each leaf.

    Distinct leaves are 2.0 apart and identical-leaf placements are 0 apart,
    so with the default pd_threshold=1.0 each leaf's SVs form their own
    phylotype, regardless of clustering strategy.
    """
    leaves = ",".join(f"L{i}:1.0[{i}]" for i in range(n_leaves))
    tree = f"({leaves}):0.0[{n_leaves}];"
    placements = [
        {"p": [[i, -10, 1.0, 0.0, 0.01]], "nm": [[f"sv_{i:03d}_{j:03d}", 1]]}
        for i in range(n_leaves)
        for j in range(svs_per_leaf)
    ]
    return {
        "version": 3,
        "tree": tree,
        "fields": ["edge_num", "likelihood", "like_weight_ratio", "distal_length", "pendant_length"],
        "placements": placements,
        "metadata": {},
    }


def _make_overlapping_jplace(
    n_pairs: int = 8,
    svs_per_leaf: int = 10,
    bridging_per_pair: int = 4,
) -> dict:
    """Sibling leaf pairs plus SVs whose placements overlap both siblings.

    Each leaf edge is 0.3 long, so pure placements on sibling leaves are 0.6
    apart. With ``pd_threshold=0.4`` they remain separate phylotypes, while a
    0.6/0.4 bridging placement is within the threshold of both. Long internal
    branches keep different sibling pairs well separated.
    """
    pairs = []
    placements = []
    for pair in range(n_pairs):
        left_edge = pair * 3
        right_edge = left_edge + 1
        internal_edge = left_edge + 2
        pairs.append(
            f"(L{pair}a:0.3[{left_edge}],L{pair}b:0.3[{right_edge}]):2.0[{internal_edge}]"
        )
        for side, edge in (("a", left_edge), ("b", right_edge)):
            placements.extend(
                {"p": [[edge, -10, 1.0, 0.0, 0.01]], "nm": [[f"pure_{pair}_{side}_{j}", 1]]}
                for j in range(svs_per_leaf)
            )
        placements.extend(
            {
                "p": [
                    [left_edge, -10, 0.6, 0.0, 0.01],
                    [right_edge, -10, 0.4, 0.0, 0.01],
                ],
                "nm": [[f"bridge_{pair}_{j}", 1]],
            }
            for j in range(bridging_per_pair)
        )

    return {
        "version": 3,
        "tree": f"({','.join(pairs)}):0.0[{n_pairs * 3}];",
        "fields": ["edge_num", "likelihood", "like_weight_ratio", "distal_length", "pendant_length"],
        "placements": placements,
        "metadata": {},
    }


def _load_overlapping(*, random_state: int = 7) -> Phylotypes:
    p = Phylotypes(pd_threshold=0.4, distance="kr", random_state=random_state)
    p.load_jplace(io.StringIO(json.dumps(_make_overlapping_jplace())))
    return p


def _overlapping_apply_state(p: Phylotypes):
    """Build a two-phylotype-per-pair pool and return the remaining SV indices."""
    pool = []
    edge_index = {}
    seeded = set()
    for pair in range(8):
        for side in ("a", "b"):
            idx = p.placement_idx[f"pure_{pair}_{side}_0"]
            seeded.add(idx)
            edges = p._sv_edges(idx)
            pt_i = len(pool)
            pool.append({"members": [idx], "edges": edges})
            for edge in edges:
                edge_index.setdefault(edge, set()).add(pt_i)
    remaining = [idx for idx in range(len(p.placement_names)) if idx not in seeded]
    return pool, edge_index, remaining


def _pairs_sharing_a_group(phylogroups):
    sv_to_group = {sv: gi for gi, grp in enumerate(phylogroups) for sv in grp}
    return {(a, b) for a, b in itertools.combinations(sorted(sv_to_group), 2) if sv_to_group[a] == sv_to_group[b]}


@pytest.mark.parametrize(
    "jplace,threshold,metric,seed_size,expand_batch_size",
    [
        (_make_star_jplace(n_leaves=20, svs_per_leaf=15), 1.0, "legacy", 10, 25),
        (_make_overlapping_jplace(), 0.4, "kr", 16, 25),
    ],
)
def test_incremental_close_to_batch_on_synthetic_data(
    jplace, threshold, metric, seed_size, expand_batch_size,
):
    """Medium synthetic test: incremental result should agree with batch on
    most SV pairs (>= 90%), though exact equality is not expected since the
    two paths use different linkage strategies."""
    p_batch = Phylotypes(lwr_overlap=0.1, pd_threshold=threshold, distance=metric, random_state=7)
    p_batch.load_jplace(io.StringIO(json.dumps(jplace)))
    p_batch.generate_phylotypes()

    p_inc = Phylotypes(lwr_overlap=0.1, pd_threshold=threshold, distance=metric, random_state=7)
    p_inc.load_jplace(io.StringIO(json.dumps(jplace)))
    p_inc.generate_phylotypes_incremental(seed_size=seed_size, expand_batch_size=expand_batch_size)

    batch_svs = {sv for grp in p_batch.phylogroups for sv in grp}
    inc_svs = {sv for grp in p_inc.phylogroups for sv in grp}
    assert inc_svs == batch_svs

    batch_pairs = _pairs_sharing_a_group(p_batch.phylogroups)
    inc_pairs = _pairs_sharing_a_group(p_inc.phylogroups)
    agreement = len(batch_pairs & inc_pairs) / len(batch_pairs)
    assert agreement >= 0.9, agreement


def test_multi_candidate_path_is_exercised():
    p = _load_overlapping()
    pool, edge_index, remaining = _overlapping_apply_state(p)
    candidate_counts = []
    for sv in remaining:
        candidates = set()
        for edge in p._sv_edges(sv):
            candidates.update(edge_index.get(edge, ()))
        candidate_counts.append(len(candidates))

    assert any(count > 1 for count in candidate_counts)
    assert pool  # sanity: candidates came from a populated pool


def test_batched_matches_sequential_apply():
    batched = _load_overlapping(random_state=17)
    batch_pool, batch_index, remaining = _overlapping_apply_state(batched)
    batch_orphans = batched._apply_svs_batched(
        remaining, batch_pool, batch_index, distal_length=True, chunk_size=1,
    )

    streaming = _load_overlapping(random_state=17)
    stream_pool, stream_index, stream_remaining = _overlapping_apply_state(streaming)
    stream_orphans = [
        sv for sv in stream_remaining
        if not streaming._apply_sv(sv, stream_pool, stream_index, distal_length=True)
    ]

    assert batch_orphans == stream_orphans
    assert [pt["members"] for pt in batch_pool] == [pt["members"] for pt in stream_pool]


def test_incremental_reproducible_with_seed():
    results = []
    for _ in range(2):
        p = _load_overlapping(random_state=23)
        p.generate_phylotypes_incremental(seed_size=16, expand_batch_size=25)
        results.append(sorted(sorted(group) for group in p.phylogroups))
    assert results[0] == results[1]


def test_chunk_size_effect_is_bounded():
    results = []
    for chunk_size in (1, 1000):
        p = _load_overlapping(random_state=31)
        p.generate_phylotypes_incremental(
            seed_size=16,
            expand_batch_size=25,
            apply_chunk_size=chunk_size,
        )
        results.append(_pairs_sharing_a_group(p.phylogroups))

    smaller, larger = results
    agreement = len(smaller & larger) / len(smaller)
    assert agreement >= 0.9, agreement
