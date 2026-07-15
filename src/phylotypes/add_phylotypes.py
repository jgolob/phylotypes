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
import io
import json
import logging
from pathlib import Path
import random
import sys
from typing import Any, TextIO

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
    combined = Phylotypes(device=device, distance=distance)
    combined.load_jplace(io.StringIO(json.dumps(merged)))
    return combined, previous_names, new_names


def assign_new_svs(
    combined: Phylotypes,
    sv_pt: dict[str, str],
    new_names: set[str],
    *,
    distal_length: bool = True,
    sample_size: int = 10,
) -> tuple[dict[str, str], set[str]]:
    """Assign each new SV to an existing phylotype by edge overlap and distance.

    For each new SV, candidate phylotypes are those sharing at least one tree
    edge with the SV's placement. With a single candidate the SV joins it; with
    several, the SV joins the phylotype with the smallest mean distance to a
    random sample of that phylotype's members; with none, the SV is an orphan.

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

    Returns
    -------
    Tuple[Dict[str, str], Set[str]]
        A mapping of assigned new SV name to phylotype id, and the set of
        orphaned new SV names (no overlapping phylotype).
    """
    pt_edges: dict[str, set[int]] = defaultdict(set)
    pt_members: dict[str, list[str]] = defaultdict(list)
    for sv, pt in sv_pt.items():
        pt_edges[pt].update(combined.sv_nodes[sv].keys())
        pt_members[pt].append(sv)

    new_sv_pt: dict[str, str] = {}
    orphans: set[str] = set()

    for new_sv in sorted(new_names):
        new_edges = set(combined.sv_nodes[new_sv].keys())
        candidates = [pt for pt, edges in pt_edges.items() if edges & new_edges]

        if not candidates:
            orphans.add(new_sv)
            continue
        if len(candidates) == 1:
            new_sv_pt[new_sv] = candidates[0]
            continue

        new_idx = combined.placement_idx[new_sv]
        best_pt = candidates[0]
        best_dist = float("inf")
        for pt in candidates:
            members = pt_members[pt]
            sample = members if len(members) <= sample_size else random.sample(members, sample_size)
            idxs = [new_idx, *(combined.placement_idx[m] for m in sample)]
            dist = combined.pairwise_distance(idxs, distal_length=distal_length)
            mean_dist = float(dist[0, 1:].mean())
            if mean_dist < best_dist:
                best_dist = mean_dist
                best_pt = pt
        new_sv_pt[new_sv] = best_pt

    return new_sv_pt, orphans


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
    )

    total = len(new_sv_pt) + len(orphans)
    if orphans:
        logging.warning(
            "Could not add %d of %d sequence variants (no overlapping phylotype).",
            len(orphans),
            total,
        )
    logging.info("Successfully integrated %d of %d sequence variants.", len(new_sv_pt), total)

    with args.out.open("w") as out_fh:
        writer = csv.writer(out_fh)
        writer.writerow(["phylotype", "sv"])
        for sv, pt in new_sv_pt.items():
            writer.writerow([pt, sv])


if __name__ == "__main__":
    main()
