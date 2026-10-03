#!/usr/bin/env python3
"""
Phylotypes: A tool for generating phylogenetically grouped features from JPLACE files.

This module provides functionality to analyze phylogenetic placement data (JPLACE format)
and group features into phylotypes based on their phylogenetic relationships and
likelihood weight ratios.

Author: Jonathan Golob (j-dev@golob.org)
License: MIT
"""

import argparse
from collections import defaultdict
from collections.abc import Callable
import csv
import heapq
from io import StringIO
import itertools
import json
import logging
from pathlib import Path
import random
import re
import sys
from typing import (
    Any,
    ClassVar,
    TextIO,
)

from Bio import Phylo
import numpy as np
from skbio import TreeNode
from sklearn.cluster import AgglomerativeClustering
import torch

# Configure logging
root_logger = logging.getLogger()
root_logger.setLevel(logging.INFO)
log_formatter = logging.Formatter("%(asctime)s %(levelname)-8s [phylotypes] %(message)s")
console_handler = logging.StreamHandler()
console_handler.setFormatter(log_formatter)
root_logger.addHandler(console_handler)


class Phylotypes:
    """
    A class for generating phylotypes from phylogenetic placement data.

    This class handles the analysis of JPLACE files containing phylogenetic
    placements and groups features into phylotypes based on phylogenetic
    distance and likelihood weight ratios.

    Parameters
    ----------
    lwr_overlap : float, optional
        Minimum likelihood weight ratio overlap threshold (default: 0.1)
    pd_threshold : float, optional
        Phylogenetic distance threshold for clustering (default: 1.0)

    Attributes
    ----------
    lwr_overlap : float
        Minimum likelihood weight ratio overlap threshold
    pd_threshold : float
        Phylogenetic distance threshold for clustering
    phylogroups : List[Set[str]]
        List of phylotype groups (sets of feature names)
    sv_groups : List[List[str]]
        List of pre-grouped features by LWR overlap
    jplace : Dict[str, Any]
        Loaded JPLACE data dictionary
    tree : Optional[TreeNode]
        Phylogenetic tree as TreeNode object
    edge_idx : int
        Index of edge_num field in placement data
    lwr_idx : int
        Index of like_weight_ratio field in placement data
    dl_idx : int
        Index of distal_length field in placement data
    sv_nodes : Dict[str, Dict[int, List[float]]]
        Dictionary mapping feature names to placement nodes
    name_node : Dict[int, TreeNode]
        Dictionary mapping node IDs to TreeNode objects
    node_name : Dict[TreeNode, int]
        Dictionary mapping TreeNode objects to node IDs
    node_names : List[str]
        List of node names in consistent order for tensor alignment
    node_name_to_idx : Dict[str, int]
        Dictionary mapping node names to their index in node_names
    """

    required_fields: ClassVar[list[str]] = ["fields", "placements", "tree"]

    def __init__(
        self,
        lwr_overlap: float = 0.1,
        pd_threshold: float = 1.0,
        distance: str = "legacy",
        device: str = "cpu",
        max_pregroup_size: int = 5000,
        random_state: int | None = None,
    ) -> None:
        """
        Initialize Phylotypes instance with clustering parameters.

        Parameters
        ----------
        lwr_overlap : float, optional
            Minimum likelihood weight ratio overlap for initial grouping (default: 0.1)
        pd_threshold : float, optional
            Phylogenetic distance threshold for final clustering (default: 1.0)
        distance : str, optional
            Pairwise distance metric to use: "legacy" (default; exact vectorized
            reimplementation of the original metric, reproduces prior phylotypes) or
            "kr" (true tree-Wasserstein / Kantorovich-Rubinstein distance, opt-in)
        device : str, optional
            torch device to use for tensor computations, e.g. "cpu" or "cuda" (default: "cpu")
        max_pregroup_size : int, optional
            Maximum number of SVs allowed in a single pregroup after LCA-based
            re-clustering. Pregroups that would exceed this are split back into
            their pre-merge constituent groups, to bound the O(n^2) memory used
            by the pairwise distance matrix and clustering for that group
            (default: 5000)
        random_state : int or None, optional
            Seed for the random number generator used by sampling-based
            distance estimation. Set for reproducible results. ``None``
            (default) uses an unseeded generator.
        """
        if distance not in ("legacy", "kr"):
            msg = f"Unknown distance metric: {distance!r}. Must be 'legacy' or 'kr'."
            raise ValueError(msg)

        # Clustering parameters
        self.lwr_overlap: float = lwr_overlap
        self.pd_threshold: float = pd_threshold
        self.distance: str = distance
        self.device: torch.device = torch.device(device)
        self.max_pregroup_size: int = max_pregroup_size
        # Clustering samples require reproducibility, not cryptographic randomness.
        self._rng: random.Random = random.Random(random_state) if random_state is not None else random.Random()  # noqa: S311

        # Data containers
        self.phylogroups: list[set[str]] = []
        self.sv_groups: list[list[str]] = []
        self.jplace: dict[str, Any] = {}

        # Tree and indexing data
        self.tree: TreeNode | None = None
        self.edge_idx: int = 0
        self.lwr_idx: int = 0
        self.dl_idx: int = 0
        self.sv_nodes: dict[str, dict[int, list[float]]] = {}
        self.name_node: dict[str, TreeNode] = {}
        self.node_name: dict[TreeNode, str] = {}

        # Node indexing
        self.node_names: list[str] = []
        self.node_name_to_idx: dict[str, int] = {}

        # Groupings
        self._pregrouped_sv: list[list[int]] = []
        self._pregroup_lca: list[TreeNode] = []

    def load_jplace(self, jplace_fh: TextIO) -> None:
        """
        Load and validate JPLACE file data.

        Reads a JPLACE file handle, validates required fields, indexes field
        positions, loads the phylogenetic tree, and caches placement data.

        Parameters
        ----------
        jplace_fh : TextIO
            File handle for JPLACE file (opened in text mode)

        Notes
        -----
        This method validates the presence of required JPLACE fields:
        - 'fields': Column headers for placement data
        - 'placements': List of feature placements
        - 'tree': Newick-formatted phylogenetic tree

        Raises
        ------
        ValueError
            If the JPLACE file cannot be parsed, is missing required fields,
            or contains a tree that cannot be parsed.
        """
        logging.info("Loading jplace file")

        try:
            jplace = json.load(jplace_fh)
        except (json.JSONDecodeError, AttributeError) as e:
            msg = f"Failed to parse JPLACE file: {e}"
            raise ValueError(msg) from e

        self.load_jplace_dict(jplace)

    def load_jplace_dict(self, jplace: dict) -> None:
        """Load placement data from a pre-parsed JPLACE dictionary.

        This is the shared implementation for both :meth:`load_jplace` (which
        parses JSON from a file handle first) and direct callers that already
        have the dictionary in memory (e.g. ``add_phylotypes.build_combined``).

        Parameters
        ----------
        jplace : dict
            A parsed JPLACE dictionary with ``fields``, ``tree``, and
            ``placements`` keys.

        Raises
        ------
        ValueError
            If required keys/fields are missing or the tree cannot be parsed.
        """
        self.jplace = jplace

        # Validate required fields

        for field in self.required_fields:
            if field not in self.jplace:
                msg = f"Missing required '{field}' entry in jplace."
                raise ValueError(msg)

        logging.info("Indexing fields")
        try:
            self.edge_idx = self.jplace["fields"].index("edge_num")
            self.lwr_idx = self.jplace["fields"].index("like_weight_ratio")
            self.dl_idx = self.jplace["fields"].index("distal_length")
        except ValueError as e:
            msg = f"Missing required field: {e}. Required: edge_num, like_weight_ratio, distal_length"
            raise ValueError(msg) from e

        logging.info("Loading tree")
        self._load_tree()
        self._load_placements()

    def _load_tree(self) -> None:
        """
        Load and normalize phylogenetic tree from JPLACE data.

        Processes the Newick tree string from JPLACE data, normalizes edge
        names for consistent parsing, and creates TreeNode structures with
        mappings between node IDs and TreeNode objects.

        Raises
        ------
        ValueError
            If tree parsing fails
        """
        # Regular expression to match SEPP tree format
        re_sepp_tree = re.compile(r"(|\w+):(?P<edgelen>(\d+\.\d+)(|e[-+]\d+))\[(?P<edgeid>\d+)\]")

        def normalized_edges(m: re.Match[str]) -> str:
            """Normalize edge format for consistent parsing."""
            return f"{{{m['edgeid']}}}:{m['edgelen']}[{m['edgeid']}]"

        # Normalize tree string
        tree_norm = re_sepp_tree.sub(normalized_edges, self.jplace["tree"])

        try:
            # Parse tree with BioPython
            tp = Phylo.read(StringIO(tree_norm), "newick")

            # Convert to scikit-bio TreeNode format
            with StringIO() as th:
                Phylo.write(tp, th, "newick")
                th.seek(0)
                self.tree = TreeNode.read(th)
        except Exception as e:
            msg = f"Failed to parse phylogenetic tree: {e}"
            raise ValueError(msg) from e

        # Cleanup node names...
        for node in self.tree.traverse():
            if node.name is not None:
                node.name = node.name.replace("{", "").replace("}", "")

        self.name_node = {n.name: n for n in self.tree.traverse() if n.name is not None}
        self.node_name = {v: k for k, v in self.name_node.items()}

    def _load_placements(self) -> None:
        """
        Load and cache feature placement data.

        Processes placement data from JPLACE file and creates mappings
        between feature names (SVs) and their placement nodes.

        Generates
        --------
        sv_nodes : Dict[str, Dict[int, List[float]]]
            Dictionary mapping feature names to placement nodes
        """
        logging.info(
            "Indexing %d placements into SV-node map",
            len(self.jplace["placements"]),
        )
        self.sv_nodes = {}

        for pl in self.jplace["placements"]:
            # Create node mapping for this placement
            pl_nodes = {p[self.edge_idx]: p for p in pl["p"]}

            # Handle named features
            if "nm" in pl:
                for sv, _ in pl["nm"]:
                    self.sv_nodes[sv] = pl_nodes

            # Handle unnamed features
            if "n" in pl:
                for sv in pl["n"]:
                    self.sv_nodes[sv] = pl_nodes

        # And list / vector based lookups of node *names*
        self.node_names = sorted({str(node_name) for pl in self.sv_nodes.values() for node_name in pl})
        self.node_name_to_idx = {name: idx for idx, name in enumerate(self.node_names)}
        logging.info(
            "Indexed %d unique SVs across %d unique nodes",
            len(self.sv_nodes),
            len(self.node_names),
        )

        logging.info(
            "Building placement tensors (%d x %d)",
            len(self.sv_nodes),
            len(self.node_names),
        )
        self._build_placement_tensors()
        logging.info("Building tree geometry")
        self._build_tree_geometry()

    def _build_placement_tensors(self) -> None:
        """
        Build placement tensors.

        Generates tensors for placement data including names, indices, LWR, and DL.

        Generates
        --------
        placement_names : List[str]
            List of placement names
        placement_idx : Dict[str, int]
            Dictionary mapping placement names to their index
        placement_lwr : torch.Tensor
            Tensor of shape (n_placements, num_nodes) filled with LWR of nodes
        placement_dl : torch.Tensor
            Tensor of shape (n_placements, num_nodes) filled with distal length of placement at nodes
        """
        self.placement_names = sorted(self.sv_nodes.keys())
        self.placement_idx = {n: i for i, n in enumerate(self.placement_names)}
        n_placements = len(self.placement_names)
        n_nodes = len(self.node_names)

        # Collect all (row, col, lwr, dl) entries in a single pass, then
        # populate the tensors with two bulk index_put_ calls instead of
        # creating two small tensors per SV in a Python loop.
        rows: list[int] = []
        cols: list[int] = []
        lwr_vals: list[float] = []
        dl_vals: list[float] = []

        for placement_name, placement in self.sv_nodes.items():
            row_i = self.placement_idx[placement_name]
            for node, data in placement.items():
                rows.append(row_i)
                cols.append(self.node_name_to_idx[str(node)])
                lwr_vals.append(data[self.lwr_idx])
                dl_vals.append(data[self.dl_idx])

        row_idx = torch.tensor(rows, dtype=torch.long)
        col_idx = torch.tensor(cols, dtype=torch.long)

        self.placement_lwr = torch.zeros((n_placements, n_nodes), dtype=torch.float32, device=self.device)
        self.placement_dl = torch.zeros((n_placements, n_nodes), dtype=torch.float32, device=self.device)

        self.placement_lwr.index_put_(
            (row_idx, col_idx),
            torch.tensor(lwr_vals, dtype=torch.float32, device=self.device),
        )
        self.placement_dl.index_put_(
            (row_idx, col_idx),
            torch.tensor(dl_vals, dtype=torch.float32, device=self.device),
        )

        self.placement_present = self.placement_lwr > 0
        # Retain each placement's sparse support for incremental candidate
        # lookup and LCA calculations.  Reconstructing these sets with a
        # tensor-wide ``nonzero`` call was a substantial part of incremental
        # legacy-distance runtime.
        self.placement_edge_sets = [
            frozenset(self.node_name_to_idx[str(node)] for node in self.sv_nodes[name]) for name in self.placement_names
        ]

    def _build_tree_geometry(self) -> None:
        """
        Precompute reusable tree geometry for the distance metrics.

        Generates
        --------
        node_depth : Dict[int, float]
            Maps `id(TreeNode)` to that node's distance from the tree root, computed
            once via a single top-down traversal.
        node_depth_tensor : torch.Tensor
            Tensor of shape (n_nodes,) aligning `node_depth` to the placement
            tensors' node columns (the order of `self.node_names`).
        _lca_depth_cache : Dict[Tuple[int, ...], float]
            Memoization cache mapping a (sorted) tuple of node-column indices to the
            depth of their lowest common ancestor. Shared across pairwise-distance
            calls and pregroups.
        """
        self.node_depth: dict[int, float] = {}

        if self.tree is None:
            self.node_depth_tensor = torch.zeros(len(self.node_names), dtype=torch.float32, device=self.device)
            self._lca_depth_cache = {}
            return

        self.node_depth[id(self.tree)] = 0.0
        stack = [self.tree]
        while stack:
            node = stack.pop()
            depth = self.node_depth[id(node)]
            for child in node.children:
                self.node_depth[id(child)] = depth + (child.length or 0.0)
                stack.append(child)

        self.node_depth_tensor = torch.tensor(
            [self.node_depth[id(self.name_node[name])] for name in self.node_names],
            dtype=torch.float32,
            device=self.device,
        )
        self._lca_depth_cache = {}

    def _lca_depth_for_nodes(self, node_idx: tuple[int, ...]) -> float:
        """
        Depth of the lowest common ancestor of a set of node-column indices.

        Results are memoized in `_lca_depth_cache` since the same node sets recur
        across many placement pairs and pregroups.

        Parameters
        ----------
        node_idx : Tuple[int, ...]
            Sorted tuple of indices into `self.node_names`.

        Returns
        -------
        float
            Distance from the tree root to the lowest common ancestor of the nodes.
        """
        cached = self._lca_depth_cache.get(node_idx)
        if cached is not None:
            return cached
        if self.tree is None:
            msg = "Tree must be loaded"
            raise ValueError(msg)

        nodes = [self.name_node[self.node_names[i]] for i in node_idx]
        lca = nodes[0] if len(nodes) == 1 else self.tree.lowest_common_ancestor(nodes)
        depth = self.node_depth[id(lca)]
        self._lca_depth_cache[node_idx] = depth
        return depth

    def _lca_depth_matrix(self, placement_indices: list[int]) -> torch.Tensor:
        """
        Depth of the LCA of the symmetric-difference node set for each pair of placements.

        Parameters
        ----------
        placement_indices : list[int]
            Placement row indices. Their cached sparse edge supports are used to
            find each pair's symmetric difference.

        Returns
        -------
        torch.Tensor
            Tensor of shape (n, n) where entry [a, b] is the depth of the LCA of the
            nodes where placements `a` and `b` disagree (their symmetric difference).
            Pairs with identical support get 0 -- their contribution is zeroed out by
            the (1 - overlap) fractions in the caller regardless.
        """
        n = len(placement_indices)
        out = torch.zeros((n, n), dtype=torch.float32, device=self.device)
        for a in range(n):
            for b in range(a + 1, n):
                distant = (
                    self.placement_edge_sets[placement_indices[a]] ^ self.placement_edge_sets[placement_indices[b]]
                )
                if not distant:
                    continue
                depth = self._lca_depth_for_nodes(tuple(sorted(distant)))
                out[a, b] = depth
                out[b, a] = depth
        return out

    def _lca_depth_cross_matrix(
        self,
        left_indices: list[int],
        right_indices: list[int],
        *,
        dtype: torch.dtype,
        device: torch.device,
    ) -> torch.Tensor:
        """Return LCA depths for the cross product of two placement lists."""
        out = torch.zeros((len(left_indices), len(right_indices)), dtype=dtype, device=device)
        for left_i, left in enumerate(left_indices):
            left_support = self.placement_edge_sets[left]
            for right_i, right in enumerate(right_indices):
                distant = left_support ^ self.placement_edge_sets[right]
                if distant:
                    out[left_i, right_i] = self._lca_depth_for_nodes(tuple(sorted(distant)))
        return out

    def pairwise_distance(
        self,
        placement_indices: list[int] | None = None,
        *,
        distal_length: bool = True,
        metric: str | None = None,
    ) -> torch.Tensor:
        """
        Calculate the pairwise phylogenetic distance between placements.

        Dispatches to `_pairwise_legacy` (the default, exact vectorized
        reimplementation of the original metric) or `_pairwise_kr` (the true
        tree-Wasserstein / Kantorovich-Rubinstein distance, opt-in).

        Parameters
        ----------
        placement_indices : Optional[List[int]], optional
            List of placement indices to calculate distances for (default: all)
        distal_length : bool, optional
            Whether to include distal length in calculations (default: True)
        metric : Optional[str], optional
            Override `self.distance` for this call ("legacy" or "kr")

        Returns
        -------
        torch.Tensor
            A tensor of shape (n, n) containing the pairwise distances

        Raises
        ------
        ValueError
            If `metric` (or `self.distance`) is not "legacy" or "kr"
        """
        metric = metric or self.distance
        if metric == "kr":
            return self._pairwise_kr(placement_indices, distal_length=distal_length)
        if metric == "legacy":
            return self._pairwise_legacy(placement_indices, distal_length=distal_length)
        msg = f"Unknown distance metric: {metric!r}"
        raise ValueError(msg)

    def _cross_distance(
        self,
        left_indices: list[int],
        right_indices: list[int],
        *,
        distal_length: bool = True,
    ) -> torch.Tensor:
        """Calculate distances between, rather than within, two placement lists.

        Incremental APPLY and RECONCILE only use cross-group distances.  The
        legacy and KR implementations both avoid calculating the two unused
        within-list triangles.
        """
        if not left_indices or not right_indices:
            return torch.zeros(
                (len(left_indices), len(right_indices)),
                dtype=self.placement_lwr.dtype,
                device=self.placement_lwr.device,
            )
        if self.distance == "legacy":
            return self._cross_legacy(left_indices, right_indices, distal_length=distal_length)
        return self._cross_kr(left_indices, right_indices, distal_length=distal_length)

    def _cross_legacy(
        self,
        left_indices: list[int],
        right_indices: list[int],
        *,
        distal_length: bool = True,
    ) -> torch.Tensor:
        """Calculate the exact legacy metric for a rectangular pair set."""
        if self.tree is None:
            msg = "Tree must be loaded"
            raise ValueError(msg)

        dtype = self.placement_lwr.dtype
        device = self.placement_lwr.device
        left_lwr = self.placement_lwr[left_indices].to(dtype)
        right_lwr = self.placement_lwr[right_indices].to(dtype)
        left_ind = self.placement_present[left_indices].to(dtype)
        right_ind = self.placement_present[right_indices].to(dtype)
        eps = torch.finfo(dtype).eps
        left_p = left_lwr / left_lwr.sum(dim=1, keepdim=True).clamp_min(eps)
        right_p = right_lwr / right_lwr.sum(dim=1, keepdim=True).clamp_min(eps)
        if distal_length:
            left_dl = self.placement_dl[left_indices].to(dtype)
            right_dl = self.placement_dl[right_indices].to(dtype)
        else:
            left_dl = torch.zeros_like(left_lwr)
            right_dl = torch.zeros_like(right_lwr)
        depth = self.node_depth_tensor.to(dtype=dtype, device=device)

        left_s = (left_dl * left_p).sum(dim=1)
        right_s = (right_dl * right_p).sum(dim=1)
        left_dep_all = (depth.unsqueeze(0) * left_p).sum(dim=1)
        right_dep_all = (depth.unsqueeze(0) * right_p).sum(dim=1)
        left_overlap = left_p @ right_ind.T
        right_overlap = right_p @ left_ind.T
        left_dep_overlap = (left_p * depth.unsqueeze(0)) @ right_ind.T
        right_dep_overlap = (right_p * depth.unsqueeze(0)) @ left_ind.T
        lca_depth = self._lca_depth_cross_matrix(left_indices, right_indices, dtype=dtype, device=device)

        result = (
            left_s.unsqueeze(1)
            + right_s.unsqueeze(0)
            + (left_dep_all.unsqueeze(1) - left_dep_overlap)
            + (right_dep_all.unsqueeze(0) - right_dep_overlap.T)
            - lca_depth * (2.0 - left_overlap - right_overlap.T)
        )
        for left_i, left in enumerate(left_indices):
            for right_i, right in enumerate(right_indices):
                if left == right:
                    result[left_i, right_i] = 0.0
        return result

    def _cross_kr(
        self,
        left_indices: list[int],
        right_indices: list[int],
        *,
        distal_length: bool = True,
    ) -> torch.Tensor:
        """Calculate exact KR distances for a rectangular pair set."""
        phi = self._kr_embedding([*left_indices, *right_indices], distal_length=distal_length)
        left_count = len(left_indices)
        return (phi[:left_count].unsqueeze(1) - phi[left_count:].unsqueeze(0)).abs().sum(dim=2)

    def _pairwise_legacy(
        self,
        placement_indices: list[int] | None = None,
        *,
        distal_length: bool = True,
    ) -> torch.Tensor:
        """
        Exact vectorized reimplementation of the original LCA-weighted-average distance.

        For a pair of placements (a, b), this reproduces
        `add_phylotypes.placement_pairwise_distance` to float precision:

            legacy(a, b) = s_a + s_b
                         + (DepAll_a - DepOv_ab) + (DepAll_b - DepOv_ba)
                         - depth(L_ab) * (fracA_ab + fracB_ab)

        where `p` is the row-normalized LWR, `Ind = (placement_lwr > 0)`, `depth` is
        each node's distance from the tree root, `dl` is the distal length, and
        `L_ab` is the lowest common ancestor of the symmetric-difference node set for
        the pair (its only genuinely pairwise term).

        Parameters
        ----------
        placement_indices : Optional[List[int]], optional
            List of placement indices to calculate distances for (default: all)
        distal_length : bool, optional
            Whether to include distal length in calculations (default: True)

        Returns
        -------
        torch.Tensor
            A tensor of shape (n, n) containing the pairwise distances

        Raises
        ------
        ValueError
            If required placement tensors or the tree are not built/loaded
        """
        if not hasattr(self, "placement_lwr"):
            msg = "Placement tensor must be built first"
            raise ValueError(msg)
        if distal_length and not hasattr(self, "placement_dl"):
            msg = "Placement distal length tensor must be built"
            raise ValueError(msg)
        if self.tree is None:
            msg = "Tree must be loaded"
            raise ValueError(msg)

        idx = list(range(self.placement_lwr.shape[0])) if placement_indices is None else list(placement_indices)
        dtype = self.placement_lwr.dtype
        device = self.placement_lwr.device
        n = len(idx)
        if n <= 1:
            return torch.zeros((n, n), dtype=dtype, device=device)

        lwr = self.placement_lwr[idx].to(dtype)
        ind = self.placement_present[idx].to(dtype)
        row_sum = lwr.sum(dim=1, keepdim=True).clamp_min(torch.finfo(dtype).eps)
        p = lwr / row_sum
        dl = self.placement_dl[idx].to(dtype) if distal_length else torch.zeros_like(lwr)
        depth = self.node_depth_tensor.to(dtype=dtype, device=device)

        s = (dl * p).sum(dim=1)
        dep_all = (depth.unsqueeze(0) * p).sum(dim=1)
        wov = p @ ind.T
        dep_ov = (p * depth.unsqueeze(0)) @ ind.T

        frac_a = 1.0 - wov
        frac_b = 1.0 - wov.T
        lca_depth = self._lca_depth_matrix(idx).to(dtype=dtype, device=device)

        result = (
            s.unsqueeze(1)
            + s.unsqueeze(0)
            + (dep_all.unsqueeze(1) - dep_ov)
            + (dep_all.unsqueeze(0) - dep_ov.T)
            - lca_depth * (frac_a + frac_b)
        )
        torch.diagonal(result).fill_(0.0)
        return result

    def _pairwise_kr(
        self,
        placement_indices: list[int] | None = None,
        *,
        distal_length: bool = True,
        batch_size: int = 64,
    ) -> torch.Tensor:
        """
        Exact pairwise tree-Wasserstein (Kantorovich-Rubinstein) distance.

        Each placement is a probability measure on the tree (LWR normalised to sum 1,
        mass on edge e sitting `distal_length` from e's distal node). KR = integral
        |F_a - F_b| over the tree, which is an L1 distance in the embedding
        phi_p[seg] = len(seg) * F_p(seg) over tree segments on which every F_p is
        constant. A true metric: symmetric, zero-diagonal, valid for precomputed
        clustering.

        Parameters
        ----------
        placement_indices : Optional[List[int]], optional
            List of placement indices to calculate distances for (default: all)
        distal_length : bool, optional
            Whether to include distal length in calculations (default: True)
        batch_size : int, optional
            Row-batch size for the final pairwise L1 computation (default: 64)

        Returns
        -------
        torch.Tensor
            A tensor of shape (n, n) containing the pairwise tree-Wasserstein distances

        Raises
        ------
        ValueError
            If required placement tensors or the tree are not built/loaded
        """
        if not hasattr(self, "placement_lwr"):
            msg = "Placement tensor must be built first"
            raise ValueError(msg)
        if distal_length and not hasattr(self, "placement_dl"):
            msg = "Placement distal length tensor must be built"
            raise ValueError(msg)
        if self.tree is None:
            msg = "Tree must be loaded"
            raise ValueError(msg)

        idx = list(range(self.placement_lwr.shape[0])) if placement_indices is None else list(placement_indices)
        dtype = self.placement_lwr.dtype
        device = self.placement_lwr.device
        n = len(idx)
        if n <= 1:
            return torch.zeros((n, n), dtype=dtype, device=device)

        phi = self._kr_embedding(idx, distal_length=distal_length)
        emd = torch.zeros((n, n), dtype=dtype, device=device)
        for s in range(0, n, batch_size):
            emd[s : s + batch_size] = (phi[s : s + batch_size].unsqueeze(1) - phi.unsqueeze(0)).abs().sum(dim=2)
        emd.clamp_min_(0.0)
        torch.diagonal(emd).fill_(0.0)
        return emd

    def _kr_embedding(self, placement_indices: list[int], *, distal_length: bool) -> torch.Tensor:
        """Build the exact KR L1 embedding for the supplied placements."""
        if not hasattr(self, "placement_lwr"):
            msg = "Placement tensor must be built first"
            raise ValueError(msg)
        if distal_length and not hasattr(self, "placement_dl"):
            msg = "Placement distal length tensor must be built"
            raise ValueError(msg)
        if self.tree is None:
            msg = "Tree must be loaded"
            raise ValueError(msg)

        dtype = self.placement_lwr.dtype
        device = self.placement_lwr.device
        n = len(placement_indices)
        a = self.placement_lwr[placement_indices].to(dtype).clone()
        row_sum = a.sum(dim=1, keepdim=True)
        if bool((row_sum.squeeze(1) <= 0).any()):
            logging.warning(
                "%d placement(s) have zero total LWR; KR is undefined for them.",
                int((row_sum.squeeze(1) <= 0).sum()),
            )
        a = a / row_sum.clamp_min(torch.finfo(dtype).eps)

        dl = self.placement_dl[placement_indices].to(dtype)
        zero_vec = torch.zeros(n, dtype=dtype, device=device)

        incl: dict[int, torch.Tensor] = {}
        phi_cols: list[torch.Tensor] = []
        for node in self.tree.postorder(include_self=True):
            col = self.node_name_to_idx.get(node.name)
            own = a[:, col] if col is not None else zero_vec
            below = zero_vec
            for child in node.children:
                below = below + incl.pop(id(child))
            incl[id(node)] = own + below
            length = node.length
            if node.is_root() or not length or length <= 0:
                continue
            length = float(length)
            if col is None or not bool((own > 0).any()):
                phi_cols.append(below * length)
                continue
            if not distal_length:
                phi_cols.append((below + own) * length)
                continue
            offsets = dl[:, col].clamp(0.0, length)
            on = own > 0
            cuts = sorted({0.0, length} | {float(o) for o in offsets[on].tolist() if 0.0 < o < length})
            for lo, hi in itertools.pairwise(cuts):
                seg_len = hi - lo
                if seg_len <= 0:
                    continue
                included = own * (offsets <= lo).to(dtype)
                phi_cols.append((below + included) * seg_len)

        if not phi_cols:
            return torch.zeros((n, 0), dtype=dtype, device=device)
        phi = torch.stack(phi_cols, dim=1)
        keep = (phi.amax(dim=0) - phi.amin(dim=0)) > 0
        return phi[:, keep]

    def _get_lca_for_group(self, group: list[int]) -> TreeNode:
        """
        Get the lowest common ancestor for a group of placements.

        Parameters
        ----------
        group : List
            List of placement indices

        Returns
        -------
        TreeNode
            The lowest common ancestor TreeNode for the group
        """
        if self.tree is None:
            msg = "Tree not loaded; call load_jplace first."
            raise ValueError(msg)
        return self.tree.lowest_common_ancestor(
            {
                self.name_node[self.node_names[nid]]
                for g_i in group
                for nid in self.placement_present[g_i].nonzero().flatten().tolist()
            }
        )

    def _pregroup_by_lwr(self) -> None:
        """
        Pre-group features by likelihood weight ratio overlap.

        Performs initial clustering of features based on LWR overlap,
        then clusters groups by phylogenetic distance to create
        preliminary feature groupings.

        Notes
        -----
        Uses pd_threshold and lwr_overlap instance attributes for clustering parameters.
        """
        # Our holder for groups
        groups = []
        # Sort features by number of placements (ascending)
        node_counts = self.placement_present.sum(dim=1).cpu().numpy()
        sv_to_group = [
            sv
            for sv, _ in sorted(
                zip(self.placement_names, node_counts, strict=True),
                key=lambda v: v[1],
            )
        ]

        logging.info("Grouping %d SV", len(sv_to_group))
        while len(sv_to_group) > 0:
            seed_sv = sv_to_group.pop()
            seed_sv_idx = self.placement_idx[seed_sv]
            group_svs_idx = {seed_sv_idx}
            # Create a mask on the nodes axis where the seed has placement
            seed_sv_mask = self.placement_present[seed_sv_idx]
            # find sv with overlap probability with the seed above threshold
            # And add to group_svs
            group_svs_idx.update(
                {
                    idx.item()
                    for idx in torch.nonzero(self.placement_lwr[:, seed_sv_mask].sum(dim=1) > self.lwr_overlap)
                    if self.placement_names[idx] in sv_to_group
                }
            )
            group_svs = {self.placement_names[i] for i in group_svs_idx}
            # Remove these from the svs to be grouped
            sv_to_group = [sv for sv in sv_to_group if sv not in group_svs]
            # And get rid of sv
            groups.append(list(group_svs_idx))

        logging.info(
            "Done pre-grouping SV into %d groups, of which the largest is %d items",
            len(groups),
            max(len(svg) for svg in groups),
        )

        # Get the LCA for each group
        logging.info("Obtaining lowest common ancestor for each group")
        group_lca = [self._get_lca_for_group(grp) for grp in groups]

        # Calculate pairwise phylogenetic distances between groups
        logging.info("Calculating pairwise phylogenetic distance between groups")
        g_lca_mat = np.zeros(shape=(len(group_lca), len(group_lca)), dtype=np.float64)
        for i in range(len(group_lca)):
            for j in range(i + 1, len(group_lca)):
                ij_pd = group_lca[i].distance(group_lca[j])
                g_lca_mat[i, j] = ij_pd
                g_lca_mat[j, i] = ij_pd

        # It is possible that these groups may be closer together
        # then our desired lumping level. That can lead to SVs
        # being inappropriately split into phylotypes.
        # So we use the LCA and clustering to combine those pregroups together.
        # From a computational perpective we are better off with *more* *smaller* clusters
        # But alas...

        # ``AgglomerativeClustering`` requires at least two samples.  A single
        # LWR pregroup is already the complete pregrouping result.
        if len(groups) == 1:
            g_lca_clusters = np.zeros(1, dtype=int)
        else:
            # Cluster groups by phylogenetic distance
            logging.info("Clustering groups by phylogenetic distance")
            g_lca_clusters = AgglomerativeClustering(
                n_clusters=None,
                distance_threshold=self.pd_threshold,
                metric="precomputed",
                linkage="average",
            ).fit_predict(g_lca_mat)

        # Regroup SVs based on group clusters
        logging.info("Regrouping SV based on group-clusters")
        new_old_sv_groups = defaultdict(set)
        for old_cluster_idx, new_cluster_idx in enumerate(g_lca_clusters):
            new_old_sv_groups[new_cluster_idx].add(old_cluster_idx)

        new_sv_groups = []
        for olds in new_old_sv_groups.values():
            merged = list({sv for idx in olds for sv in groups[idx]})
            if len(merged) > self.max_pregroup_size:
                logging.warning(
                    "Pregroup of %d exceeds max_pregroup_size=%d; keeping it unmerged.",
                    len(merged),
                    self.max_pregroup_size,
                )
                new_sv_groups.extend(list(groups[idx]) for idx in olds)
            else:
                new_sv_groups.append(merged)

        logging.debug(
            "Now %d groups, with the largest %d items",
            len(new_sv_groups),
            max(len(svg) for svg in new_sv_groups),
        )
        self._pregrouped_sv = new_sv_groups
        self._pregroup_lca = [self._get_lca_for_group(grp) for grp in new_sv_groups]

    def generate_phylotypes(
        self,
        *,
        distal_length: bool = True,
    ) -> None:
        """
        Group features into phylotypes based on phylogenetic distance.

        Performs two-stage clustering: first by LWR overlap, then by
        phylogenetic distance within overlapping groups. Can optionally
        ignore distal length calculations.

        Parameters
        ----------
        distal_length : bool, optional
            Whether to include distal length in calculations (default: True)

        Returns
        -------
        List[Set[str]]
            List of phylotype groups, each as a set of feature names

        Notes
        -----
        This method updates the instance's phylogroups attribute and
        returns the complete list of phylotypes.
        """
        if distal_length:
            logging.info("Using Distal Length")
        else:
            logging.info("Ignoring distal length")

        logging.info("Pregrouping based on overlapping LWR for SV")
        self._pregroup_by_lwr()

        logging.info("Starting phylogrouping")

        for g_i, g_sv in enumerate(self._pregrouped_sv):
            if (g_i + 1) % 100 == 0:
                logging.debug("Group %d of %d", g_i, len(self._pregrouped_sv))

            if len(g_sv) == 1:
                self.phylogroups.append({self.placement_names[g_sv[0]]})
                continue
            # Implict else, this isn't a singleton group.
            # Calculate pairwise phylogenetic distances within group
            g_sv_dist_mat = self.pairwise_distance(g_sv, distal_length=distal_length)
            # Cluster features by phylogenetic distance
            g_sv_clusters = AgglomerativeClustering(
                n_clusters=None,
                distance_threshold=self.pd_threshold,
                metric="precomputed",
                linkage="average",
            ).fit_predict(g_sv_dist_mat.cpu().numpy())

            # Map clusters to feature sets
            g_phylotype_svs = defaultdict(set)
            for sv, cl in zip(g_sv, g_sv_clusters, strict=True):
                g_phylotype_svs[cl].add(sv)

            # Add clusters to phylogroups
            self.phylogroups.extend(
                [{self.placement_names[sv_i] for sv_i in phylotype_svs} for phylotype_svs in g_phylotype_svs.values()]
            )

    def _sv_edges(self, sv_idx: int, *, min_lwr: float = 0.0) -> set[int]:
        """Tree-edge (node-column) indices on which placement ``sv_idx`` has weight.

        Parameters
        ----------
        sv_idx : int
            Row index into ``placement_lwr``.
        min_lwr : float, optional
            Minimum LWR value for an edge to be included. ``0.0`` (default)
            returns all edges with any weight (legacy behaviour).
        """
        if min_lwr > 0.0:
            return set((self.placement_lwr[sv_idx] > min_lwr).nonzero(as_tuple=True)[0].tolist())
        return set(self.placement_edge_sets[sv_idx])

    def _group_edges(self, members: list[int]) -> set[int]:
        """Union of tree-edge (node-column) indices used by any placement in `members`."""
        edges: set[int] = set()
        for member in members:
            edges.update(self.placement_edge_sets[member])
        return edges

    @staticmethod
    def _attach_sv(
        sv: int,
        pt_i: int,
        sv_edges: set[int],
        pool: list[dict[str, Any]],
        edge_index: dict[int, set[int]],
    ) -> None:
        """Commit ``sv`` to phylotype ``pt_i``, updating pool and edge index."""
        pool[pt_i]["members"].append(sv)
        new_edges = sv_edges - pool[pt_i]["edges"]
        pool[pt_i]["edges"].update(new_edges)
        for edge in new_edges:
            edge_index[edge].add(pt_i)

    def _apply_sv(
        self,
        sv: int,
        pool: list[dict[str, Any]],
        edge_index: dict[int, set[int]],
        *,
        distal_length: bool,
        sample_size: int = 10,
        min_lwr: float = 0.0,
    ) -> bool:
        """
        Try to assign placement `sv` to an existing phylotype in `pool`.

        Candidate phylotypes are found via `edge_index` (tree-edge overlap). If
        there is exactly one candidate, `sv` is assigned to it provided its
        mean distance to a sample of the candidate's members is within
        ``pd_threshold``. If there are several, `sv` is assigned to the
        candidate with the smallest mean distance to a sample of its members,
        provided that distance is within `pd_threshold`.

        On assignment, `pool[pt_i]["members"]` and `pool[pt_i]["edges"]` (and
        `edge_index`) are updated in place.

        Parameters
        ----------
        sv : int
            Placement index to assign.
        pool : List[Dict[str, Any]]
            Current phylotype pool; each entry has "members" (List[int]) and
            "edges" (Set[int]).
        edge_index : Dict[int, Set[int]]
            Inverted index mapping tree-edge index to the set of phylotype
            indices (into `pool`) placed on that edge.
        distal_length : bool
            Whether to include distal length in distance calculations.
        sample_size : int, optional
            Number of members to sample per candidate phylotype when comparing
            distances (default: 10).
        min_lwr : float, optional
            Minimum LWR weight on an edge for it to count in candidate lookup.
            ``0.0`` (default) includes all edges with any weight. Higher values
            filter out trace placements that would create spurious matches.

        Returns
        -------
        bool
            True if `sv` was assigned to a phylotype, False if it is an orphan.
        """
        # Use min_lwr-filtered edges for candidate lookup.  Unfiltered edges
        # are computed lazily (only on assignment) for updating the pool's edge
        # set so the inverted index stays complete.
        sv_edges_lookup = self._sv_edges(sv, min_lwr=min_lwr)
        candidates: set[int] = set()
        for edge in sv_edges_lookup:
            candidates.update(edge_index.get(edge, ()))

        if not candidates:
            return False

        if len(candidates) == 1:
            pt_i = next(iter(candidates))
            # Distance-gate: verify the SV is within pd_threshold of the sole
            # candidate, matching the multi-candidate path's threshold check.
            members = pool[pt_i]["members"]
            sample = members if len(members) <= sample_size else self._rng.sample(members, sample_size)
            dist = self._cross_distance([sv], sample, distal_length=distal_length)
            mean_dist = float(dist[0].mean())
            if mean_dist >= self.pd_threshold:
                return False
        else:
            best: tuple[float, int] | None = None
            for cand in candidates:
                members = pool[cand]["members"]
                sample = members if len(members) <= sample_size else self._rng.sample(members, sample_size)
                dist = self._cross_distance([sv], sample, distal_length=distal_length)
                mean_dist = float(dist[0].mean())
                if best is None or mean_dist < best[0]:
                    best = (mean_dist, cand)
            if best is None or best[0] >= self.pd_threshold:
                return False
            pt_i = best[1]

        # Compute full (unfiltered) edge set only now that assignment is confirmed.
        sv_edges = sv_edges_lookup if min_lwr == 0.0 else self._sv_edges(sv)
        self._attach_sv(sv, pt_i, sv_edges, pool, edge_index)
        return True

    def _apply_svs_batched(
        self,
        svs: list[int],
        pool: list[dict[str, Any]],
        edge_index: dict[int, set[int]],
        *,
        distal_length: bool,
        sample_size: int = 10,
        min_lwr: float = 0.0,
        chunk_size: int = 10_000,
    ) -> list[int]:
        """Assign multiple SVs to existing phylotypes in streaming order.

        Each SV is evaluated after every earlier accepted SV has updated both
        the candidate edge index and the candidate's membership.  This is
        necessary for average-distance gating: accepting a batch against the
        membership at the start of the batch can admit placements whose mean
        distance from the grown group exceeds ``pd_threshold``.

        ``chunk_size`` remains an API compatibility parameter.  It controls
        only the outer iteration, not assignment visibility; every assignment
        is immediately visible to the next one.

        Parameters
        ----------
        svs : list[int]
            Placement indices to try assigning.
        pool, edge_index, distal_length, sample_size, min_lwr
            Same semantics as :meth:`_apply_sv`.
        chunk_size : int, optional
            Compatibility-only outer iteration size; every assignment is applied
            sequentially regardless of this value (default: 10_000).

        Returns
        -------
        list[int]
            Placement indices that could not be assigned (orphans).
        """
        del chunk_size  # accepted for API compatibility, no longer used
        orphans: list[int] = []

        for sv in svs:
            if not self._apply_sv(
                sv,
                pool,
                edge_index,
                distal_length=distal_length,
                sample_size=sample_size,
                min_lwr=min_lwr,
            ):
                orphans.append(sv)

        return orphans

    def primary_edge_sorter(self) -> Callable[[int], int] | None:
        """Return a sort-key function ordering SVs by their primary edge's postorder rank.

        Returns ``None`` when no tree is loaded.  Used by Stage D (EXPAND) and
        by :func:`~phylotypes.add_phylotypes.cluster_orphans` to sort orphans
        before batching so phylogenetically close SVs land in the same batch.
        """
        if self.tree is None:
            return None
        postorder_rank = {
            self.node_name_to_idx.get(node.name, -1): rank
            for rank, node in enumerate(self.tree.postorder(include_self=True))
            if node.name is not None
        }

        def _key(sv_idx: int) -> int:
            primary = int(self.placement_lwr[sv_idx].argmax())
            return postorder_rank.get(primary, 0)

        return _key

    def _reconcile_groups(
        self,
        groups: list[list[int]],
        *,
        distal_length: bool = True,
        sample_size: int = 10,
    ) -> list[list[int]]:
        """Merge close groups using sampled, size-weighted average linkage.

        For each pair of groups, samples up to ``sample_size`` members from
        each, computes the mean pairwise distance between the two samples,
        and builds initial inter-group distances.  The merge updates then use
        the number of SVs in each group as the average-linkage weight.  Treating
        every preliminary group as one observation would overweight singleton
        groups and can produce a different cut from SV-level average linkage.

        Parameters
        ----------
        groups : list[list[int]]
            Groups of placement indices to consider merging.
        distal_length : bool, optional
            Whether to include distal length (default: True).
        sample_size : int, optional
            Maximum members sampled per group (default: 10).

        Returns
        -------
        list[list[int]]
            Merged groups (may be fewer than the input).
        """
        if len(groups) <= 1:
            return groups

        # Pre-sample deterministically before the O(K^2) loop.
        samples = [g if len(g) <= sample_size else self._rng.sample(g, sample_size) for g in groups]

        k = len(groups)
        distances: dict[tuple[int, int], float] = {}
        queue: list[tuple[float, int, int]] = []
        for i in range(k):
            for j in range(i + 1, k):
                si, sj = samples[i], samples[j]
                distance = float(self._cross_distance(si, sj, distal_length=distal_length).mean())
                distances[(i, j)] = distance
                queue.append((distance, i, j))

        # Perform UPGMA directly so subsequent merge distances are weighted by
        # the SV counts represented by the two groups.  sklearn's precomputed
        # agglomeration considers each *input group* a single observation and
        # therefore cannot preserve those weights.
        active = set(range(k))
        members = {i: list(group) for i, group in enumerate(groups)}
        weights = {i: len(group) for i, group in enumerate(groups)}
        heapq.heapify(queue)
        next_id = k

        while queue:
            distance, left, right = heapq.heappop(queue)
            # Stale heap entries (referencing already-merged clusters) are
            # skipped here; the ``distances`` dict is pruned via ``.pop()``
            # inside the merge loop, so only the heap grows monotonically.
            if left not in active or right not in active:
                continue
            if distance >= self.pd_threshold:
                break

            merged_id = next_id
            next_id += 1
            members[merged_id] = [*members.pop(left), *members.pop(right)]
            weights[merged_id] = weights[left] + weights[right]
            active.remove(left)
            active.remove(right)
            distances.pop((left, right) if left < right else (right, left), None)

            for other in active:
                left_key = (left, other) if left < other else (other, left)
                right_key = (right, other) if right < other else (other, right)
                w_left = weights[left] * distances.pop(left_key)
                w_right = weights[right] * distances.pop(right_key)
                updated = (w_left + w_right) / weights[merged_id]
                key = (merged_id, other) if merged_id < other else (other, merged_id)
                distances[key] = updated
                heapq.heappush(queue, (updated, *key))
            active.add(merged_id)

            # The heap retains stale entries after a UPGMA merge.  Compact it
            # when they outnumber live distances, keeping peak memory bounded
            # by a small multiple of the active O(K^2) state.
            if len(queue) > 2 * len(distances):
                queue = [(distance, *key) for key, distance in distances.items()]
                heapq.heapify(queue)

        return [members[group_id] for group_id in sorted(active)]

    def generate_phylotypes_incremental(
        self,
        *,
        distal_length: bool = True,
        seed_size: int = 200,
        expand_batch_size: int = 200,
        min_lwr: float = 0.0,
        sample_size: int = 10,
        apply_chunk_size: int = 10_000,
    ) -> None:
        """
        Group features into phylotypes incrementally (seed -> apply -> expand -> reconcile).

        An alternative to `generate_phylotypes` for datasets too large for a single
        O(n^2) distance matrix and clustering pass. Builds an initial phylotype pool
        from a small "seed" of the most specific placements (Stage A), indexes each
        phylotype by the tree edges its members are placed on (Stage B), assigns the
        remaining placements in streaming order (Stage C),
        clusters and absorbs orphans into new phylotypes (Stage D), and finally
        merges close phylotypes using sampled inter-phylotype distance (Stage E).

        Design note
        -----------
        The batch path (`generate_phylotypes`) clusters each pregroup with average
        linkage over its full pairwise distance matrix. This streaming scheme instead
        assigns each placement by its distance to a *sample* of an existing
        phylotype's members -- behavior closer to single/centroid linkage. Near
        phylotype boundaries, some placements may therefore be grouped differently
        than they would be by the batch path. This is an accepted, explicit
        trade-off in exchange for avoiding a full SV-by-SV distance matrix in
        the SEED and EXPAND stages. RECONCILE retains only active sampled
        inter-phylotype distances, but its time and memory use are still
        quadratic in the number of provisional phylotypes.

        Every APPLY decision observes the members and edge index created by all
        earlier accepted SVs. ``apply_chunk_size`` is retained for command-line
        and API compatibility, but it no longer changes assignment behavior.
        RECONCILE draws one sample per phylotype rather than per pair; this
        reduces sampling variance while retaining a sampled estimate for large
        groups.

        Parameters
        ----------
        distal_length : bool, optional
            Whether to include distal length in distance calculations (default: True)
        seed_size : int, optional
            Number of (most specific) placements used to seed the initial phylotype
            pool via the batch clusterer (default: 200)
        expand_batch_size : int, optional
            Maximum number of orphaned placements clustered together per EXPAND pass
            (default: 200)
        min_lwr : float, optional
            Minimum LWR weight on an edge for it to count in candidate lookup
            during the APPLY and EXPAND stages. Higher values filter out trace
            placements that would create spurious candidate matches. ``0.0``
            (default) retains all edges with any weight (legacy behaviour).
        sample_size : int, optional
            Number of members to sample per candidate phylotype when comparing
            distances during APPLY, EXPAND, and RECONCILE stages (default: 10).
        apply_chunk_size : int, optional
            Compatibility-only outer iteration size for Stage C and Stage D
            re-application passes. Assignments are always applied sequentially,
            so this value does not change grouping (default: 10_000).

        Raises
        ------
        ValueError
            If required placement tensors or the tree are not built/loaded
        """
        if not hasattr(self, "placement_lwr"):
            msg = "Placement tensor must be built first"
            raise ValueError(msg)
        if self.tree is None:
            msg = "Tree must be loaded"
            raise ValueError(msg)
        if apply_chunk_size < 1:
            msg = "apply_chunk_size must be at least 1"
            raise ValueError(msg)
        if seed_size < 1:
            msg = "seed_size must be at least 1"
            raise ValueError(msg)
        if expand_batch_size < 1:
            msg = "expand_batch_size must be at least 1"
            raise ValueError(msg)
        if sample_size < 1:
            msg = "sample_size must be at least 1"
            raise ValueError(msg)

        # Most specific placements (fewest placement nodes) first.
        order = torch.argsort(self.placement_present.sum(dim=1)).tolist()
        seed_idx = order[:seed_size]
        remaining = order[seed_size:]

        # ---- Stage A: SEED ----
        logging.info(
            "Incremental Stage A (SEED): clustering %d seed placements",
            len(seed_idx),
        )
        pool: list[dict[str, Any]] = []
        if len(seed_idx) == 1:
            pool.append({"members": [seed_idx[0]], "edges": self._sv_edges(seed_idx[0])})
        elif len(seed_idx) > 1:
            seed_dist = self.pairwise_distance(seed_idx, distal_length=distal_length)
            seed_clusters = AgglomerativeClustering(
                n_clusters=None,
                distance_threshold=self.pd_threshold,
                metric="precomputed",
                linkage="average",
            ).fit_predict(seed_dist.cpu().numpy())
            seed_groups: dict[int, list[int]] = defaultdict(list)
            for local_i, cl in enumerate(seed_clusters):
                seed_groups[cl].append(seed_idx[local_i])
            for members in seed_groups.values():
                pool.append({"members": members, "edges": self._group_edges(members)})

        # ---- Stage B: edge -> phylotype inverted index ----
        logging.info(
            "Incremental Stage B (INDEX): indexing %d seed phylotypes",
            len(pool),
        )
        edge_index: dict[int, set[int]] = defaultdict(set)
        for pt_i, pt in enumerate(pool):
            for edge in pt["edges"]:
                edge_index[edge].add(pt_i)

        # ---- Stage C: APPLY ----
        logging.info(
            "Incremental Stage C (APPLY): assigning %d remaining placements (chunk_size=%d)",
            len(remaining),
            apply_chunk_size,
        )
        orphans = self._apply_svs_batched(
            remaining,
            pool,
            edge_index,
            distal_length=distal_length,
            sample_size=sample_size,
            min_lwr=min_lwr,
            chunk_size=apply_chunk_size,
        )

        # ---- Stage D: EXPAND ----
        logging.info(
            "Incremental Stage D (EXPAND): %d orphans to cluster in batches of %d",
            len(orphans),
            expand_batch_size,
        )
        # Sort orphans by their primary (highest-LWR) edge's postorder position
        # in the tree so that phylogenetically close orphans land in the same
        # batch, reducing batch-boundary artifacts.  Re-applying preserves the
        # input order of surviving orphans, so this is needed only once.
        _sorter = self.primary_edge_sorter()
        if orphans and _sorter is not None:
            orphans.sort(key=_sorter)

        expand_round = 0
        while orphans:
            expand_round += 1
            batch, orphans = orphans[:expand_batch_size], orphans[expand_batch_size:]
            logging.info(
                "EXPAND round %d: clustering batch of %d orphans (%d still queued, %d phylotypes so far)",
                expand_round,
                len(batch),
                len(orphans),
                len(pool),
            )
            if len(batch) == 1:
                new_groups = [batch]
            else:
                batch_dist = self.pairwise_distance(batch, distal_length=distal_length)
                batch_clusters = AgglomerativeClustering(
                    n_clusters=None,
                    distance_threshold=self.pd_threshold,
                    metric="precomputed",
                    linkage="average",
                ).fit_predict(batch_dist.cpu().numpy())
                batch_groups: dict[int, list[int]] = defaultdict(list)
                for local_i, cl in enumerate(batch_clusters):
                    batch_groups[cl].append(batch[local_i])
                new_groups = list(batch_groups.values())

            logging.info(
                "EXPAND round %d: batch produced %d new phylotypes",
                expand_round,
                len(new_groups),
            )
            for members in new_groups:
                pt_i = len(pool)
                edges = self._group_edges(members)
                pool.append({"members": members, "edges": edges})
                for edge in edges:
                    edge_index[edge].add(pt_i)

            pre_reapply = len(orphans)
            orphans = self._apply_svs_batched(
                orphans,
                pool,
                edge_index,
                distal_length=distal_length,
                sample_size=sample_size,
                min_lwr=min_lwr,
                chunk_size=apply_chunk_size,
            )
            absorbed = pre_reapply - len(orphans)
            if pre_reapply > 0:
                logging.info(
                    "EXPAND round %d: re-apply absorbed %d/%d queued orphans, %d remain",
                    expand_round,
                    absorbed,
                    pre_reapply,
                    len(orphans),
                )

        # ---- Stage E: RECONCILE ----
        logging.info(
            "Incremental Stage E (RECONCILE): merging %d phylotypes by sampled inter-phylotype distance",
            len(pool),
        )
        final_groups = self._reconcile_groups(
            [pt["members"] for pt in pool],
            distal_length=distal_length,
            sample_size=sample_size,
        )

        self.phylogroups.extend({self.placement_names[sv_i] for sv_i in members} for members in final_groups)

    def to_long(self) -> list[tuple[str, str]]:
        """
        Convert phylogroups to long format for output.

        Creates a list of (phylotype_id, feature_name) tuples sorted
        by phylotype size (largest first).

        Returns
        -------
        List[Tuple[str, str]]
            List of (phylotype_id, feature_name) tuples in long format
        """
        # Sort phylogroups by size (descending)
        pg_size = sorted(
            [(pg_i, len(pg)) for pg_i, pg in enumerate(self.phylogroups)],
            key=lambda pgc: -1 * pgc[1],
        )

        return [
            (f"pt__{pg_i + 1:05d}", sv)
            for pg_i, (pg_idx, pg_size) in enumerate(pg_size)
            for sv in self.phylogroups[pg_idx]
        ]

    def to_csv(self, out_h: TextIO) -> None:
        """
        Write phylogroups to CSV file in long format.

        Parameters
        ----------
        out_h : TextIO
            File handle for output CSV file (opened in text mode)
        """
        pg_long = self.to_long()
        writer = csv.writer(out_h)
        writer.writerow(["phylotype", "sv"])
        writer.writerows(pg_long)

    def __repr__(self) -> str:
        """
        Return a string representation of the Phylotypes instance.

        Returns
        -------
        str
            String representation indicating if tree is loaded
        """
        return f"Phylotype Generator. Tree loaded: {self.tree is not None}."


def main() -> None:
    """
    Run the phylotypes command-line interface.

    Parse command-line arguments, load JPLACE data, perform phylotype
    clustering, and write results to a CSV file.
    """
    args_parser = argparse.ArgumentParser(
        description="""Given a JPLACE file of placed features on a phylogenetic tree,
        generate phylotypes or phylogenetically grouped features."""
    )

    args_parser.add_argument(
        "--jplace",
        "-J",
        help="JPLACE file, as created by pplacer or epa-ng",
        type=Path,
        required=True,
    )

    args_parser.add_argument(
        "--out",
        "-O",
        help="Where to place the phylogroups (in csv long format)?",
        type=Path,
        required=True,
    )

    args_parser.add_argument(
        "--lwr-overlap",
        "-L",
        help="minimum like-weight ratio for grouping of features. (Default: 0.1).",
        default=0.1,
        type=float,
    )

    args_parser.add_argument(
        "--threshold_pd",
        "-T",
        help="Phylogenetic distance threshold for clustering. (Default: 1.0). Calibrated for "
        "--distance legacy; if using --distance kr, re-calibrate this on your data -- the "
        "legacy default will over- or under-split.",
        default=1.0,
        type=float,
    )

    args_parser.add_argument(
        "--no-distal-length",
        "-ndl",
        help="Ignore distal length to nodes. (Default: False)",
        action="store_true",
    )

    args_parser.add_argument(
        "--distance",
        "-D",
        help="Pairwise distance metric: 'legacy' (default; reproduces prior phylotypes) or "
        "'kr' (true tree-Wasserstein distance, opt-in). (Default: legacy). Note: "
        "--threshold_pd defaults are calibrated for 'legacy'; 'kr' is a different "
        "(smaller-valued) scale and needs its own threshold, re-calibrated on your data.",
        choices=["legacy", "kr"],
        default="legacy",
    )

    args_parser.add_argument(
        "--device",
        help="torch device for tensor computations, e.g. 'cpu' or 'cuda'. (Default: cpu).",
        default="cpu",
    )

    args_parser.add_argument(
        "--max-pregroup-size",
        help="Maximum SVs in a single pregroup after LCA-based re-clustering; larger "
        "pregroups are split back into their pre-merge groups to bound memory use. "
        "(Default: 5000).",
        default=5000,
        type=int,
    )

    args_parser.add_argument(
        "--incremental",
        help="Use the incremental seed->apply->expand->reconcile clustering path "
        "instead of the batch path. Scales to larger inputs at the cost of an "
        "approximate, order-dependent linkage near phylotype boundaries; it does "
        "not enforce batch LWR-pregroup partitions. "
        "(Default: False).",
        action="store_true",
    )

    args_parser.add_argument(
        "--seed-size",
        help="Number of (most specific) placements used to seed the initial phylotype "
        "pool when --incremental is set. (Default: 200).",
        default=200,
        type=int,
    )

    args_parser.add_argument(
        "--expand-batch-size",
        help="Maximum number of orphaned placements clustered together per EXPAND pass "
        "when --incremental is set. (Default: 200).",
        default=200,
        type=int,
    )

    args_parser.add_argument(
        "--min-lwr",
        help="Minimum LWR weight on an edge for it to count in candidate lookup "
        "during the --incremental APPLY and EXPAND stages. Filters out trace "
        "placements that would create spurious candidate matches. (Default: 0.0).",
        default=0.0,
        type=float,
    )

    args_parser.add_argument(
        "--random-seed",
        help="Seed for the random number generator used by sampling-based distance "
        "estimation. Set for reproducible results. (Default: None, non-deterministic).",
        default=None,
        type=int,
    )

    args_parser.add_argument(
        "--sample-size",
        help="Number of members to sample per candidate phylotype when comparing "
        "distances during --incremental APPLY, EXPAND, and RECONCILE stages. "
        "(Default: 10).",
        default=10,
        type=int,
    )

    args_parser.add_argument(
        "--apply-chunk-size",
        help="Compatibility-only outer iteration size for --incremental APPLY passes. "
        "Assignments are always applied sequentially, so it does not change grouping. "
        "(Default: 10000).",
        default=10_000,
        type=int,
    )

    args = args_parser.parse_args()

    phylotypes = Phylotypes(
        lwr_overlap=args.lwr_overlap,
        pd_threshold=args.threshold_pd,
        distance=args.distance,
        device=args.device,
        max_pregroup_size=args.max_pregroup_size,
        random_state=args.random_seed,
    )
    try:
        with args.jplace.open() as jplace_fh:
            phylotypes.load_jplace(jplace_fh)
    except ValueError as e:
        logging.error(e)
        sys.exit(1)
    if args.incremental:
        phylotypes.generate_phylotypes_incremental(
            distal_length=not args.no_distal_length,
            seed_size=args.seed_size,
            expand_batch_size=args.expand_batch_size,
            min_lwr=args.min_lwr,
            sample_size=args.sample_size,
            apply_chunk_size=args.apply_chunk_size,
        )
    else:
        phylotypes.generate_phylotypes(distal_length=not args.no_distal_length)

    logging.info("Done Phylogrouping. Outputting.")
    with args.out.open("w") as out_fh:
        phylotypes.to_csv(out_fh)
    logging.info("DONE!")
