"""Unit + end-to-end tests for add_phylotypes (adding new SVs to existing phylotypes)."""

import csv
import io
import json
from pathlib import Path

import pytest

from phylotypes.add_phylotypes import (
    assign_new_svs,
    build_combined,
    main,
    read_phylotype_csv,
)
from phylotypes.phylotypes import Phylotypes

FIXTURE = Path(__file__).parent / "fixture_jplace.json"

FIELDS = ["edge_num", "likelihood", "like_weight_ratio", "distal_length", "pendant_length"]


def _star_tree(n_leaves: int) -> str:
    """Star tree: root + n_leaves, each leaf edge length 1.0 (distinct leaves 2.0 apart)."""
    leaves = ",".join(f"L{i}:1.0[{i}]" for i in range(n_leaves))
    return f"({leaves}):0.0[{n_leaves}];"


def _jplace(tree: str, placements: list) -> dict:
    return {"version": 3, "tree": tree, "fields": FIELDS, "placements": placements, "metadata": {}}


def _on_edge(name: str, edge: int, lwr: float = 1.0, dl: float = 0.0) -> dict:
    """A placement putting all of `name`'s weight on a single edge."""
    return {"p": [[edge, -10, lwr, dl, 0.01]], "nm": [[name, 1]]}


def _make_previous(n_leaves: int = 4, used_leaves=(0, 1, 2), svs_per_leaf: int = 3):
    """A previous jplace + its phylotype CSV: one phylotype per used leaf.

    Leaf `n_leaves-1` style leaves left unused give us an edge with no phylotype
    (for the orphan case). Returns (previous_jplace_dict, sv_pt_map).
    """
    tree = _star_tree(n_leaves)
    placements = [_on_edge(f"prev_{leaf}_{j}", leaf) for leaf in used_leaves for j in range(svs_per_leaf)]
    prev = _jplace(tree, placements)

    p = Phylotypes(lwr_overlap=0.1, pd_threshold=1.0)
    p.load_jplace(io.StringIO(json.dumps(prev)))
    p.generate_phylotypes()
    csv_buf = io.StringIO()
    p.to_csv(csv_buf)
    csv_buf.seek(0)
    sv_pt = read_phylotype_csv(csv_buf)
    return prev, sv_pt


# ---------------------------------------------------------------------------
# Regression: the old Jplace loader crashed on standard edge-numbered jplace.
# ---------------------------------------------------------------------------
def test_build_combined_loads_standard_edge_numbered_jplace():
    """Guards the historic `int('A')` crash: loading `[N]`-annotated jplace must work."""
    prev = json.load(FIXTURE.open())
    new = _jplace(prev["tree"], [_on_edge("brand_new", 0)])
    combined, prev_names, new_names = build_combined(
        io.StringIO(json.dumps(prev)),
        io.StringIO(json.dumps(new)),
    )
    assert combined.tree is not None
    assert prev_names == {"sv1", "sv2", "sv3", "sv4"}
    assert new_names == {"brand_new"}
    # The combined instance holds both previous and new placements.
    assert "brand_new" in combined.sv_nodes
    assert "sv1" in combined.sv_nodes


# ---------------------------------------------------------------------------
# Assignment behavior
# ---------------------------------------------------------------------------
def test_single_candidate_assignment():
    """A new SV overlapping exactly one phylotype's edge joins that phylotype."""
    prev, sv_pt = _make_previous(n_leaves=4, used_leaves=(0, 1, 2))
    new = _jplace(prev["tree"], [_on_edge("new_on_L0", 0)])
    combined, _, new_names = build_combined(io.StringIO(json.dumps(prev)), io.StringIO(json.dumps(new)))
    assigned, orphans = assign_new_svs(combined, sv_pt, new_names)

    assert orphans == set()
    # It must land in the same phylotype as the existing SVs on leaf 0.
    l0_pt = sv_pt["prev_0_0"]
    assert assigned == {"new_on_L0": l0_pt}


def test_orphan_when_no_edge_overlap():
    """A new SV on an edge no phylotype uses is reported as an orphan."""
    prev, sv_pt = _make_previous(n_leaves=4, used_leaves=(0, 1, 2))  # leaf 3 unused
    new = _jplace(prev["tree"], [_on_edge("new_on_unused", 3)])
    combined, _, new_names = build_combined(io.StringIO(json.dumps(prev)), io.StringIO(json.dumps(new)))
    assigned, orphans = assign_new_svs(combined, sv_pt, new_names)

    assert assigned == {}
    assert orphans == {"new_on_unused"}


def test_multi_candidate_picks_nearest():
    """When a new SV overlaps several phylotypes, it joins the nearest by distance,
    and that choice agrees with the raw pairwise_distance metric.

    Uses the ``kr`` metric: the ``legacy`` metric is not a true metric and assigns
    distance 0 from a mixed placement to each of its pure-leaf overlaps (they tie),
    so it cannot discriminate this case. The true tree-Wasserstein (``kr``) metric
    can, and the multi-candidate branch selects on it.
    """
    prev, sv_pt = _make_previous(n_leaves=4, used_leaves=(0, 1, 2))
    # New SV split across leaf 0 (90%) and leaf 1 (10%): overlaps both phylotypes,
    # but is much closer to leaf 0's under a true metric.
    new = _jplace(
        prev["tree"],
        [{"p": [[0, -10, 0.9, 0.0, 0.01], [1, -10, 0.1, 0.0, 0.01]], "nm": [["straddler", 1]]}],
    )
    combined, _, new_names = build_combined(io.StringIO(json.dumps(prev)), io.StringIO(json.dumps(new)), distance="kr")
    assigned, orphans = assign_new_svs(combined, sv_pt, new_names)

    l0_pt = sv_pt["prev_0_0"]
    l1_pt = sv_pt["prev_1_0"]
    assert l0_pt != l1_pt  # sanity: they really are distinct phylotypes
    assert orphans == set()
    assert assigned == {"straddler": l0_pt}

    # Independently confirm the metric agrees: mean distance to leaf-0 members
    # is smaller than to leaf-1 members.
    s = combined.placement_idx["straddler"]
    l0_members = [combined.placement_idx["prev_0_0"], combined.placement_idx["prev_0_1"]]
    l1_members = [combined.placement_idx["prev_1_0"], combined.placement_idx["prev_1_1"]]
    d0 = combined.pairwise_distance([s, *l0_members])
    d1 = combined.pairwise_distance([s, *l1_members])
    assert float(d0[0, 1:].mean()) < float(d1[0, 1:].mean())


def test_assignment_covers_all_new_svs():
    """Every non-orphan new SV appears exactly once in the assignment."""
    prev, sv_pt = _make_previous(n_leaves=5, used_leaves=(0, 1, 2, 3))
    new = _jplace(
        prev["tree"],
        [_on_edge("n0", 0), _on_edge("n1", 1), _on_edge("n_orphan", 4)],
    )
    combined, _, new_names = build_combined(io.StringIO(json.dumps(prev)), io.StringIO(json.dumps(new)))
    assigned, orphans = assign_new_svs(combined, sv_pt, new_names)
    assert set(assigned) | orphans == new_names
    assert set(assigned).isdisjoint(orphans)
    assert orphans == {"n_orphan"}


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------
def test_read_phylotype_csv_missing_column_raises():
    bad = io.StringIO("phylotype,notsv\npt__00001,x\n")
    with pytest.raises(ValueError, match="sv"):
        read_phylotype_csv(bad)


def test_build_combined_mismatched_tree_raises():
    prev = _jplace(_star_tree(3), [_on_edge("a", 0)])
    new = _jplace(_star_tree(4), [_on_edge("b", 0)])
    with pytest.raises(ValueError, match="same reference tree"):
        build_combined(io.StringIO(json.dumps(prev)), io.StringIO(json.dumps(new)))


def test_build_combined_mismatched_fields_raises():
    prev = _jplace(_star_tree(3), [_on_edge("a", 0)])
    new = _jplace(_star_tree(3), [_on_edge("b", 0)])
    new["fields"] = ["edge_num", "like_weight_ratio", "distal_length"]  # different order/set
    with pytest.raises(ValueError, match="fields"):
        build_combined(io.StringIO(json.dumps(prev)), io.StringIO(json.dumps(new)))


def test_build_combined_missing_key_raises():
    prev = _jplace(_star_tree(3), [_on_edge("a", 0)])
    new = {"fields": FIELDS, "placements": []}  # no tree
    with pytest.raises(ValueError, match="tree"):
        build_combined(io.StringIO(json.dumps(prev)), io.StringIO(json.dumps(new)))


# ---------------------------------------------------------------------------
# End-to-end main()
# ---------------------------------------------------------------------------
def test_main_end_to_end(tmp_path, monkeypatch):
    prev, sv_pt = _make_previous(n_leaves=4, used_leaves=(0, 1, 2))
    new = _jplace(prev["tree"], [_on_edge("e0", 0), _on_edge("e1", 1)])

    prev_jp = tmp_path / "prev.jplace"
    prev_jp.write_text(json.dumps(prev))
    new_jp = tmp_path / "new.jplace"
    new_jp.write_text(json.dumps(new))
    prev_csv = tmp_path / "prev_pt.csv"
    with prev_csv.open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["phylotype", "sv"])
        for sv, pt in sv_pt.items():
            w.writerow([pt, sv])
    out_csv = tmp_path / "out.csv"

    monkeypatch.setattr(
        "sys.argv",
        ["add_phylotypes", "-P", str(prev_jp), "-p", str(prev_csv), "-N", str(new_jp), "-O", str(out_csv)],
    )
    main()

    rows = list(csv.DictReader(out_csv.open()))
    assert {r["sv"] for r in rows} == {"e0", "e1"}
    assert {r["phylotype"] for r in rows} <= set(sv_pt.values())


def test_main_mismatched_previous_csv_exits(tmp_path, monkeypatch):
    """A previous CSV whose SV set disagrees with the previous jplace exits non-zero."""
    prev, _sv_pt = _make_previous(n_leaves=4, used_leaves=(0, 1, 2))
    new = _jplace(prev["tree"], [_on_edge("e0", 0)])

    prev_jp = tmp_path / "prev.jplace"
    prev_jp.write_text(json.dumps(prev))
    new_jp = tmp_path / "new.jplace"
    new_jp.write_text(json.dumps(new))
    prev_csv = tmp_path / "prev_pt.csv"
    prev_csv.write_text("phylotype,sv\npt__00001,not_a_real_sv\n")  # wrong SV set
    out_csv = tmp_path / "out.csv"

    monkeypatch.setattr(
        "sys.argv",
        ["add_phylotypes", "-P", str(prev_jp), "-p", str(prev_csv), "-N", str(new_jp), "-O", str(out_csv)],
    )
    with pytest.raises(SystemExit) as exc:
        main()
    assert exc.value.code == 1
