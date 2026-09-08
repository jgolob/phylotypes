#!/usr/bin/env python3
"""Add a new set of placed sequence variants to an existing set of phylotypes.

Given a previous JPLACE + its phylotype assignments (as produced by
``phylotypes``), and a new JPLACE of sequence variants placed on the *same*
reference tree, assign each new SV into one of the existing phylotypes.

This module reuses :class:`phylotypes.phylotypes.Phylotypes` for both JPLACE
loading (which normalizes the SEPP/edge-numbered tree format and validates the
required fields) and pairwise distance computation, so there is a single,
tested code path for parsing and the legacy distance metric.
"""

import argparse
from collections import defaultdict
import csv
import json
import logging
from pathlib import Path
import random
import sys
from typing import Any, TextIO

from sklearn.cluster import AgglomerativeClustering

from phylotypes.phylotypes import Phylotypes


def read_phylotype_csv(fh: TextIO) -> dict[str, str]:
    """Read a two-column ``phylotype,sv`` CSV into an ``sv -> phylotype`` map.

    Parameters
    ----------
    fh : TextIO
        File handle for the phylotype CSV (as written by ``phylotypes``).

    Returns
    -------
    Dict[str, str]
        Mapping of each sequence-variant name to its phylotype id.

    Raises
    ------
    ValueError
        If the CSV is missing the ``phylotype`` or ``sv`` column.
    """
    reader = csv.DictReader(fh)
    fields = set(reader.fieldnames or [])
    for required in ("phylotype", "sv"):
        if required not in fields:
            msg = f"Phylotype CSV is missing the required '{required}' column."
            raise ValueError(msg)
    return {row["sv"]: row["phylotype"] for row in reader}


def _placement_names(jplace: dict[str, Any]) -> set[str]:
    """Collect every sequence-variant name declared in a JPLACE dict."""
    names: set[str] = set()
    for placement in jplace.get("placements", []):
        for sv, _ in placement.get("nm", []):
            names.add(sv)
        for sv in placement.get("n", []):
            names.add(sv)
    return names


def build_combined(
    previous_fh: TextIO,
    new_fh: TextIO,
    device: str = "cpu",
    distance: str = "legacy",
    random_state: int | None = None,
) -> tuple[Phylotypes, set[str], set[str]]:
    """Load the previous and new JPLACE into a single ``Phylotypes`` instance.

    Both sets of placements are loaded onto the same reference tree so that the
    tested :meth:`Phylotypes.pairwise_distance` can score a new SV against the
    existing phylotype members in one shared tensor space.

    Parameters
    ----------
    previous_fh : TextIO
        File handle for the previous JPLACE.
    new_fh : TextIO
        File handle for the new JPLACE (placed on the same reference tree).
    device : str, optional
        torch device passed through to ``Phylotypes`` (default: ``"cpu"``).
    distance : str, optional
        Pairwise distance metric, ``"legacy"`` or ``"kr"`` (default: ``"legacy"``).
    random_state : int or None, optional
        Seed for sampling-based operations performed by the combined
        ``Phylotypes`` instance (default: ``None``).

    Returns
    -------
    Tuple[Phylotypes, Set[str], Set[str]]
        The loaded ``Phylotypes`` instance, the set of previous SV names, and
        the set of new SV names.

    Raises
    ------
    ValueError
        If either JPLACE is missing required keys, or the two JPLACE files do
        not share the same ``fields`` and reference ``tree``.
    """
    previous = json.load(previous_fh)
    new = json.load(new_fh)

    for label, jplace in (("previous", previous), ("new", new)):
        for key in ("fields", "tree", "placements"):
            if key not in jplace:
                msg = f"Missing required '{key}' entry in {label} jplace."
                raise ValueError(msg)

    if previous["fields"] != new["fields"]:
        msg = "Previous and new jplace declare different 'fields'; they must match."
        raise ValueError(msg)
    if previous["tree"].strip() != new["tree"].strip():
        msg = "Previous and new jplace must be placed on the same reference tree."
        raise ValueError(msg)

    previous_names = _placement_names(previous)
    new_names = _placement_names(new)

    merged = {
        "fields": previous["fields"],
        "tree": previous["tree"],
        "placements": previous["placements"] + new["placements"],
    }
    combined = Phylotypes(device=device, distance=distance, random_state=random_state)
    combined.load_jplace_dict(merged)
    return combined, previous_names, new_names


def assign_new_svs(
    combined: Phylotypes,
    sv_pt: dict[str, str],
    new_names: set[str],
    *,
    distal_length: bool = True,
    sample_size: int = 10,
    pd_threshold: float | None = None,
    random_state: int | None = None,
    min_lwr: float = 0.0,
) -> tuple[dict[str, str], set[str]]:
    """Assign each new SV to an existing phylotype by edge overlap and distance.

    For each new SV, candidate phylotypes are those sharing at least one tree
    edge with the SV's placement. With a single candidate, the SV's mean
    distance to a sample of the candidate's members is checked against
    ``pd_threshold`` before assignment. With several candidates, the SV joins
    the phylotype with the smallest mean distance to a random sample of that
    phylotype's members, provided the distance is within ``pd_threshold``.
    With none, the SV is an orphan.

    Single-candidate SVs sharing the same candidate are batched into one
    distance-matrix call for efficiency. Multi-candidate SVs are evaluated
    sequentially after their candidate-member samples have been drawn.

    Parameters
    ----------
    combined : Phylotypes
        A ``Phylotypes`` holding both previous and new placements (from
        :func:`build_combined`).
    sv_pt : Dict[str, str]
        Mapping of previous SV name to phylotype id.
    new_names : Set[str]
        Names of the SVs to be added.
    distal_length : bool, optional
        Whether to include distal length in distance calculations (default: True).
    sample_size : int, optional
        Maximum number of members sampled per candidate phylotype when comparing
        distances (default: 10).
    pd_threshold : float or None, optional
        Maximum mean phylogenetic distance for assignment. SVs farther than this
        from all candidate phylotypes become orphans. ``None`` disables the
        threshold (legacy behaviour).
    random_state : int or None, optional
        Seed for the random number generator used when sampling phylotype
        members for distance estimation. Set for reproducible results.
        ``None`` (default) uses an unseeded generator.
    min_lwr : float, optional
        Minimum LWR weight on an edge for it to count in candidate lookup.
        ``0.0`` (default) includes all edges with any weight. Higher values
        filter out trace placements that would create spurious matches.

    Returns
    -------
    Tuple[Dict[str, str], Set[str]]
        A mapping of assigned new SV name to phylotype id, and the set of
        orphaned new SV names (no overlapping phylotype, or beyond threshold).
    """
    rng = random.Random(random_state) if random_state is not None else random.Random()

    # Build per-phylotype edge sets from existing members.  These use ALL
    # edges (no min_lwr filter) because the existing phylotype composition
    # was already validated during the original clustering.  Only the *new*
    # SV's edges are filtered by min_lwr below, matching the one-sided
    # design in Phylotypes._apply_sv.
    pt_edges: dict[str, set[int]] = defaultdict(set)
    pt_members: dict[str, list[str]] = defaultdict(list)
    for sv, pt in sv_pt.items():
        pt_edges[pt].update(combined.sv_nodes[sv].keys())
        pt_members[pt].append(sv)

    new_sv_pt: dict[str, str] = {}
    orphans: set[str] = set()

    lwr_idx = combined.lwr_idx

    # ---------- Phase 1: classify SVs by candidate set ----------
    # single_groups[pt_name] = list of new SV names whose sole candidate is pt_name
    single_groups: dict[str, list[str]] = defaultdict(list)
    # multi_svs: SVs with >1 candidates, handled individually
    multi_svs: list[tuple[str, list[str]]] = []  # (sv_name, candidate_list)

    for new_sv in sorted(new_names):
        if min_lwr > 0.0:
            new_edges = {
                edge for edge, data in combined.sv_nodes[new_sv].items()
                if data[lwr_idx] > min_lwr
            }
        else:
            new_edges = set(combined.sv_nodes[new_sv].keys())
        candidates = [pt for pt, edges in pt_edges.items() if edges & new_edges]

        if not candidates:
            orphans.add(new_sv)
        elif len(candidates) == 1:
            single_groups[candidates[0]].append(new_sv)
        else:
            multi_svs.append((new_sv, candidates))

    # ---------- Phase 2: batched single-candidate evaluation ----------
    for pt_name, sv_batch in single_groups.items():
        if pd_threshold is None:
            # No distance gating — assign all immediately.
            for sv in sv_batch:
                new_sv_pt[sv] = pt_name
            continue

        members = pt_members[pt_name]
        sample = members if len(members) <= sample_size else rng.sample(members, sample_size)
        ns = len(sample)
        sample_idxs = [combined.placement_idx[m] for m in sample]
        sv_idxs = [combined.placement_idx[sv] for sv in sv_batch]
        # One distance matrix: rows [0..ns-1] = sample, [ns..] = new SVs
        all_indices = [*sample_idxs, *sv_idxs]
        dist = combined.pairwise_distance(all_indices, distal_length=distal_length)
        # Per-SV mean distance to sample members
        sv_dists = dist[ns:, :ns].mean(dim=1)

        for k, sv in enumerate(sv_batch):
            if float(sv_dists[k]) <= pd_threshold:
                new_sv_pt[sv] = pt_name
            else:
                orphans.add(sv)

    # ---------- Phase 3: multi-candidate SVs ----------
    # Pre-sample members per candidate to keep sampling deterministic and avoid
    # drawing a different sample for every multi-candidate SV.
    _pt_samples: dict[str, list[str]] = {}
    for _, candidates in multi_svs:
        for pt in candidates:
            if pt not in _pt_samples:
                m = pt_members[pt]
                _pt_samples[pt] = m if len(m) <= sample_size else rng.sample(m, sample_size)

    def _evaluate_multi(new_sv: str, candidates: list[str]) -> tuple[str, str | None]:
        new_idx = combined.placement_idx[new_sv]
        best_pt: str | None = None
        best_dist = float("inf")
        for pt in candidates:
            sample = _pt_samples[pt]
            idxs = [new_idx, *(combined.placement_idx[m] for m in sample)]
            dist = combined.pairwise_distance(idxs, distal_length=distal_length)
            mean_dist = float(dist[0, 1:].mean())
            if mean_dist < best_dist:
                best_dist = mean_dist
                best_pt = pt
        if pd_threshold is not None and best_dist > pd_threshold:
            return new_sv, None
        return new_sv, best_pt

    results = [_evaluate_multi(sv, cands) for sv, cands in multi_svs]

    for sv, pt in results:
        if pt is not None:
            new_sv_pt[sv] = pt
        else:
            orphans.add(sv)

    return new_sv_pt, orphans


def cluster_orphans(
    combined: Phylotypes,
    orphan_names: set[str],
    *,
    distal_length: bool = True,
    pd_threshold: float = 1.0,
    batch_size: int = 200,
    prefix: str = "pt_new_",
    reserved_ids: set[str] | None = None,
) -> dict[str, str]:
    """Cluster orphaned SVs into new phylotypes.

    Uses AgglomerativeClustering with average linkage, the same algorithm
    as Stage D (EXPAND) of the incremental pipeline.

    Parameters
    ----------
    combined : Phylotypes
        A ``Phylotypes`` holding the orphan placements.
    orphan_names : set[str]
        SV names to cluster.
    distal_length : bool, optional
        Whether to include distal length (default: True).
    pd_threshold : float, optional
        Phylogenetic distance threshold for clustering (default: 1.0).
    batch_size : int, optional
        Maximum orphans to cluster at once (default: 200).
    prefix : str, optional
        Prefix for new phylotype IDs (default: ``"pt_new_"``).
    reserved_ids : set[str] or None, optional
        Existing phylotype IDs that must not be reused (default: none).

    Returns
    -------
    dict[str, str]
        Mapping of orphan SV name to new phylotype id.
    """
    if not orphan_names:
        return {}

    orphan_idxs = [combined.placement_idx[sv] for sv in sorted(orphan_names)]
    sorter = combined.primary_edge_sorter()
    if sorter is not None:
        orphan_idxs.sort(key=sorter)

    # Cluster in batches to bound memory, same as incremental EXPAND.
    groups: list[list[int]] = []

    for start in range(0, len(orphan_idxs), batch_size):
        batch_idxs = orphan_idxs[start : start + batch_size]

        if len(batch_idxs) == 1:
            groups.append(batch_idxs)
            continue

        dist = combined.pairwise_distance(batch_idxs, distal_length=distal_length)
        labels = AgglomerativeClustering(
            n_clusters=None,
            distance_threshold=pd_threshold,
            metric="precomputed",
            linkage="average",
        ).fit_predict(dist.cpu().numpy())

        batch_groups: dict[int, list[int]] = defaultdict(list)
        for idx, cl in zip(batch_idxs, labels, strict=True):
            batch_groups[int(cl)].append(idx)
        groups.extend(batch_groups.values())

    previous_threshold = combined.pd_threshold
    try:
        combined.pd_threshold = pd_threshold
        groups = combined._reconcile_groups(
            groups,
            distal_length=distal_length,
            sample_size=10,
        )
    finally:
        combined.pd_threshold = previous_threshold

    # Build sv -> phylotype-id mapping.
    result: dict[str, str] = {}
    reserved = reserved_ids or set()
    next_id = 1
    for members in groups:
        while f"{prefix}{next_id:05d}" in reserved:
            next_id += 1
        phylotype_id = f"{prefix}{next_id:05d}"
        next_id += 1
        for idx in members:
            result[combined.placement_names[idx]] = phylotype_id
    return result


def main() -> None:
    """Command-line entry point for adding new SVs into existing phylotypes."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)-8s [add_phylotypes] %(message)s",
    )

    args_parser = argparse.ArgumentParser(
        description="""Given a baseline set of placed sequence variants grouped into phylotypes,
        put a new set of sequence variants placed on the same reference tree into the existing
        phylotypes.""",
    )
    args_parser.add_argument(
        "--previous_jp",
        "-P",
        help="Previous JPLACE file, as created by pplacer or epa-ng",
        type=Path,
        required=True,
    )
    args_parser.add_argument(
        "--previous_phylotypes",
        "-p",
        help="CSV file with two columns: phylotype and sv. Represents the existing phylotypes",
        type=Path,
        required=True,
    )
    args_parser.add_argument(
        "--new_jp",
        "-N",
        help="NEW JPLACE file, as created by pplacer or epa-ng, containing placed sequence variants to be added",
        type=Path,
        required=True,
    )
    args_parser.add_argument(
        "--out",
        "-O",
        help="Output CSV file placing the new SVs into the existing phylotypes",
        type=Path,
        required=True,
    )
    args_parser.add_argument(
        "--orphans",
        help="Output CSV file listing SVs that could not be assigned to any "
        "phylotype (no overlapping edges or beyond pd_threshold). If not "
        "provided, orphans are logged as a warning but not saved.",
        type=Path,
        default=None,
    )
    args_parser.add_argument(
        "--no-distal-length",
        "-ndl",
        help="Ignore distal length to nodes. (Default: False)",
        action="store_true",
    )
    args_parser.add_argument(
        "--device",
        help="torch device for tensor computations, e.g. 'cpu' or 'cuda'. (Default: cpu).",
        default="cpu",
    )
    args_parser.add_argument(
        "--distance",
        "-D",
        help="Pairwise distance metric: 'legacy' (default; reproduces prior phylotypes) or "
        "'kr' (true tree-Wasserstein distance). (Default: legacy).",
        choices=["legacy", "kr"],
        default="legacy",
    )
    args_parser.add_argument(
        "--pd-threshold",
        type=float,
        default=None,
        help="Maximum mean phylogenetic distance for assignment. SVs farther than "
        "this from all candidate phylotypes become orphans instead of being "
        "force-assigned. (Default: None, no threshold — legacy behaviour).",
    )
    args_parser.add_argument(
        "--random-seed",
        type=int,
        default=None,
        help="Seed for the random number generator used by sampling-based distance "
        "estimation. Set for reproducible results. (Default: None, non-deterministic).",
    )
    args_parser.add_argument(
        "--min-lwr",
        type=float,
        default=0.0,
        help="Minimum LWR weight on an edge for it to count in candidate lookup. "
        "Filters out trace placements that would create spurious matches. "
        "(Default: 0.0, all edges with any weight).",
    )
    args_parser.add_argument(
        "--cluster-orphans",
        action="store_true",
        help="Cluster orphaned SVs into new phylotypes instead of leaving them "
        "unassigned. New phylotypes are appended to the output with IDs "
        "prefixed 'pt_new_' to distinguish them from existing phylotypes. "
        "Uses --pd-threshold as the clustering threshold, or 1.0 when it is unset.",
    )
    args = args_parser.parse_args()

    try:
        logging.info("Loading previous phylotype assignments")
        with args.previous_phylotypes.open() as pt_fh:
            sv_pt = read_phylotype_csv(pt_fh)

        logging.info("Loading previous and new jplace onto the shared reference tree")
        with args.previous_jp.open() as prev_fh, args.new_jp.open() as new_fh:
            combined, previous_names, new_names = build_combined(
                prev_fh,
                new_fh,
                device=args.device,
                distance=args.distance,
                random_state=args.random_seed,
            )

        if previous_names != set(sv_pt.keys()):
            msg = "Previous jplace and previous-phylotype CSV describe different sets of SVs."
            raise ValueError(msg)
    except ValueError as e:
        logging.error(e)
        sys.exit(1)

    logging.info("Assigning %d new SV into the existing phylotypes", len(new_names))
    new_sv_pt, orphans = assign_new_svs(
        combined,
        sv_pt,
        new_names,
        distal_length=not args.no_distal_length,
        pd_threshold=args.pd_threshold,
        random_state=args.random_seed,
        min_lwr=args.min_lwr,
    )

    total = len(new_sv_pt) + len(orphans)
    orphan_pt: dict[str, str] = {}
    if orphans:
        logging.warning(
            "Could not add %d of %d sequence variants (no overlapping phylotype"
            " or beyond pd_threshold).",
            len(orphans),
            total,
        )
        if args.cluster_orphans:
            threshold = args.pd_threshold
            if threshold is None:
                threshold = 1.0
                logging.info(
                    "--cluster-orphans without --pd-threshold: using default threshold 1.0",
                )
            logging.info("Clustering %d orphaned SVs into new phylotypes", len(orphans))
            orphan_pt = cluster_orphans(
                combined,
                orphans,
                distal_length=not args.no_distal_length,
                pd_threshold=threshold,
                reserved_ids=set(sv_pt.values()),
            )
            logging.info(
                "Created %d new phylotypes from %d orphaned SVs",
                len(set(orphan_pt.values())),
                len(orphan_pt),
            )
        if not args.orphans and not args.cluster_orphans:
            logging.info("Use --orphans <path> to save orphaned SV names to a file.")

    if args.orphans:
        # Always replace the requested output so results from an earlier run
        # cannot be mistaken for the current run's orphan set.
        remaining_orphans = orphans - set(orphan_pt)
        with args.orphans.open("w") as orphan_fh:
            orphan_writer = csv.writer(orphan_fh)
            orphan_writer.writerow(["sv"])
            for sv in sorted(remaining_orphans):
                orphan_writer.writerow([sv])
        logging.info("Wrote %d orphaned SVs to %s", len(remaining_orphans), args.orphans)
    logging.info(
        "Successfully integrated %d of %d sequence variants (%d into existing, %d into new phylotypes).",
        len(new_sv_pt) + len(orphan_pt),
        total,
        len(new_sv_pt),
        len(orphan_pt),
    )

    with args.out.open("w") as out_fh:
        writer = csv.writer(out_fh)
        writer.writerow(["phylotype", "sv"])
        for sv, pt in new_sv_pt.items():
            writer.writerow([pt, sv])
        for sv, pt in orphan_pt.items():
            writer.writerow([pt, sv])


if __name__ == "__main__":
    main()
