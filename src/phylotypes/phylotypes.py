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
import csv
import itertools
import json
import logging
import random
import re
import sys
from collections import defaultdict
from io import StringIO
from typing import (
    Any,
    Dict,
    List,
    Optional,
    Set,
    Tuple,
    Union,
    TextIO,
)

import numpy as np
import torch
from Bio import Phylo
from sklearn.cluster import AgglomerativeClustering
from skbio import TreeNode


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

    required_fields = ["fields", "placements", "tree"]

    def __init__(
        self,
        lwr_overlap: float = 0.1,
        pd_threshold: float = 1.0,
        distance: str = "legacy",
        device: str = "cpu",
        max_pregroup_size: int = 5000,
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

        # Data containers
        self.phylogroups: List[Set[str]] = []
        self.sv_groups: List[List[str]] = []
        self.jplace: Dict[str, Any] = {}

        # Tree and indexing data
        self.tree: Optional[TreeNode] = None
        self.edge_idx: int = 0
        self.lwr_idx: int = 0
        self.dl_idx: int = 0
        self.sv_nodes: Dict[str, Dict[int, List[float]]] = {}
        self.name_node: Dict[str, TreeNode] = {}
        self.node_name: Dict[TreeNode, str] = {}

        # Node indexing
        self.node_names: List[str] = []
        self.node_name_to_idx: Dict[str, int] = {}

        # Groupings
        self._pregrouped_sv: List[List[int]] = []
        self._pregroup_lca: List[TreeNode] = []

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
            self.jplace = json.load(jplace_fh)
        except (json.JSONDecodeError, AttributeError) as e:
            msg = f"Failed to parse JPLACE file: {e}"
            raise ValueError(msg) from e

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
        logging.info("Loading and caching placements")
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
            tp = Phylo.read(StringIO(tree_norm), "newick")  # type: ignore

            # Convert to scikit-bio TreeNode format
            with StringIO() as th:
                Phylo.write(tp, th, "newick")  # type: ignore
                th.seek(0)
                self.tree = TreeNode.read(th)
        except Exception as e:
            msg = f"Failed to parse phylogenetic tree: {e}"
            raise ValueError(msg) from e

        # Cleanup node names...
        for node in self.tree.traverse():
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
        self.node_names = sorted({str(node_name) for pl in self.sv_nodes.values() for node_name in pl.keys()})
        self.node_name_to_idx = {name: idx for idx, name in enumerate(self.node_names)}

        self._build_placement_tensors()
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
        self.placement_lwr = torch.zeros((n_placements, n_nodes), dtype=torch.float32, device=self.device)
        self.placement_dl = torch.zeros((n_placements, n_nodes), dtype=torch.float32, device=self.device)

        for placement_name, placement in self.sv_nodes.items():
            placement_i = self.placement_idx[placement_name]
            placement_nodes = list(placement.keys())
            self.placement_lwr[placement_i, [self.node_name_to_idx[str(node)] for node in placement_nodes]] = (
                torch.tensor(
                    [placement[node][self.lwr_idx] for node in placement_nodes],
                    dtype=torch.float32,
                    device=self.device,
                )
            )
            self.placement_dl[placement_i, [self.node_name_to_idx[str(node)] for node in placement_nodes]] = (
                torch.tensor(
                    [placement[node][self.dl_idx] for node in placement_nodes],
                    dtype=torch.float32,
                    device=self.device,
                )
            )

        self.placement_present = self.placement_lwr > 0

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
        self.node_depth: Dict[int, float] = {}

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

    def _lca_depth_for_nodes(self, node_idx: Tuple[int, ...]) -> float:
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

    def _lca_depth_matrix(self, present: torch.Tensor) -> torch.Tensor:
        """
        Depth of the LCA of the symmetric-difference node set for each pair of placements.

        Parameters
        ----------
        present : torch.Tensor
            Boolean tensor of shape (n, n_nodes); `present[i]` is the set of node
            columns on which placement `i` has any weight.

        Returns
        -------
        torch.Tensor
            Tensor of shape (n, n) where entry [a, b] is the depth of the LCA of the
            nodes where placements `a` and `b` disagree (their symmetric difference).
            Pairs with identical support get 0 -- their contribution is zeroed out by
            the (1 - overlap) fractions in the caller regardless.
        """
        n = present.shape[0]
        out = torch.zeros((n, n), dtype=torch.float32, device=self.device)
        for a in range(n):
            for b in range(a + 1, n):
                distant = (present[a] ^ present[b]).nonzero(as_tuple=True)[0]
                if distant.numel() == 0:
                    continue
                depth = self._lca_depth_for_nodes(tuple(sorted(distant.tolist())))
                out[a, b] = depth
                out[b, a] = depth
        return out

    def pairwise_distance(
        self,
        placement_indices: Optional[List[int]] = None,
        distal_length: bool = True,
        metric: Optional[str] = None,
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
            return self._pairwise_kr(placement_indices, distal_length)
        if metric == "legacy":
            return self._pairwise_legacy(placement_indices, distal_length)
        msg = f"Unknown distance metric: {metric!r}"
        raise ValueError(msg)

    def _pairwise_legacy(
        self,
        placement_indices: Optional[List[int]] = None,
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
        lca_depth = self._lca_depth_matrix(self.placement_present[idx]).to(dtype=dtype, device=device)

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
        placement_indices: Optional[List[int]] = None,
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

        a = self.placement_lwr[idx].to(dtype).clone()
        row_sum = a.sum(dim=1, keepdim=True)
        if bool((row_sum.squeeze(1) <= 0).any()):
            logging.warning(
                "%d placement(s) have zero total LWR; KR is undefined for them.",
                int((row_sum.squeeze(1) <= 0).sum()),
            )
        a = a / row_sum.clamp_min(torch.finfo(dtype).eps)

        dl = self.placement_dl[idx].to(dtype)
        zero_vec = torch.zeros(n, dtype=dtype, device=device)

        incl: Dict[int, torch.Tensor] = {}
        phi_cols: List[torch.Tensor] = []
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
            return torch.zeros((n, n), dtype=dtype, device=device)
        phi = torch.stack(phi_cols, dim=1)
        keep = (phi.amax(dim=0) - phi.amin(dim=0)) > 0
        phi = phi[:, keep]
        emd = torch.zeros((n, n), dtype=dtype, device=device)
        for s in range(0, n, batch_size):
            emd[s : s + batch_size] = (phi[s : s + batch_size].unsqueeze(1) - phi.unsqueeze(0)).abs().sum(dim=2)
        emd.clamp_min_(0.0)
        torch.diagonal(emd).fill_(0.0)
        return emd

    def _get_lca_for_group(self, group: List[int]) -> TreeNode:
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
        return self.tree.lowest_common_ancestor(  # type: ignore
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
        sv_to_group = [
            v[0]
            for v in sorted(
                [
                    (sv, num_nodes)
                    for (sv, num_nodes) in zip(self.placement_names, self.placement_present.sum(dim=1).cpu().numpy())
                ],
                key=lambda v: v[1],
            )
        ]

        logging.info("Grouping {} SV".format(len(sv_to_group)))
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
            "Done pre-grouping SV into {} groups, of which the largest is {} items".format(
                len(groups), max([len(svg) for svg in groups])
            )
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
            "Now {} groups, with the largest {} items".format(
                len(new_sv_groups), max([len(svg) for svg in new_sv_groups])
            )
        )
        self._pregrouped_sv = new_sv_groups
        self._pregroup_lca = [self._get_lca_for_group(grp) for grp in new_sv_groups]

    def generate_phylotypes(
        self,
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
                self.phylogroups.append(set([self.placement_names[g_sv[0]]]))
                continue
            # Implict else, this isn't a singleton group.
            # Calculate pairwise phylogenetic distances within group
            g_sv_dist_mat = self.pairwise_distance(g_sv)
            # Cluster features by phylogenetic distance
            g_sv_clusters = AgglomerativeClustering(
                n_clusters=None,
                distance_threshold=self.pd_threshold,
                metric="precomputed",
                linkage="average",
            ).fit_predict(g_sv_dist_mat.cpu().numpy())

            # Map clusters to feature sets
            g_phylotype_svs = defaultdict(set)
            for sv, cl in zip(g_sv, g_sv_clusters):
                g_phylotype_svs[cl].add(sv)

            # Add clusters to phylogroups
            self.phylogroups.extend(
                [{self.placement_names[sv_i] for sv_i in phylotype_svs} for phylotype_svs in g_phylotype_svs.values()]
            )

    def _sv_edges(self, sv_idx: int) -> Set[int]:
        """Tree-edge (node-column) indices on which placement `sv_idx` has any weight."""
        return set(self.placement_present[sv_idx].nonzero(as_tuple=True)[0].tolist())

    def _group_edges(self, members: List[int]) -> Set[int]:
        """Union of tree-edge (node-column) indices used by any placement in `members`."""
        return set(self.placement_present[members].any(dim=0).nonzero(as_tuple=True)[0].tolist())

    def _apply_sv(
        self,
        sv: int,
        pool: List[Dict[str, Any]],
        edge_index: Dict[int, Set[int]],
        *,
        distal_length: bool,
        sample_size: int = 10,
    ) -> bool:
        """
        Try to assign placement `sv` to an existing phylotype in `pool`.

        Candidate phylotypes are found via `edge_index` (tree-edge overlap). If
        there is exactly one candidate, `sv` is assigned to it. If there are
        several, `sv` is assigned to the candidate with the smallest mean
        distance to a sample of its members, provided that distance is within
        `pd_threshold`.

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

        Returns
        -------
        bool
            True if `sv` was assigned to a phylotype, False if it is an orphan.
        """
        sv_edges = self._sv_edges(sv)
        candidates: Set[int] = set()
        for edge in sv_edges:
            candidates.update(edge_index.get(edge, ()))

        if not candidates:
            return False

        if len(candidates) == 1:
            pt_i = next(iter(candidates))
        else:
            best: Optional[Tuple[float, int]] = None
            for cand in candidates:
                members = pool[cand]["members"]
                sample = members if len(members) <= sample_size else random.sample(members, sample_size)
                dist = self.pairwise_distance([sv, *sample], distal_length=distal_length)
                mean_dist = float(dist[0, 1:].mean())
                if best is None or mean_dist < best[0]:
                    best = (mean_dist, cand)
            if best is None or best[0] > self.pd_threshold:
                return False
            pt_i = best[1]

        pool[pt_i]["members"].append(sv)
        new_edges = sv_edges - pool[pt_i]["edges"]
        pool[pt_i]["edges"].update(new_edges)
        for edge in new_edges:
            edge_index[edge].add(pt_i)
        return True

    def generate_phylotypes_incremental(
        self,
        distal_length: bool = True,
        seed_size: int = 200,
        expand_batch_size: int = 200,
    ) -> None:
        """
        Group features into phylotypes incrementally (seed -> apply -> expand -> reconcile).

        An alternative to `generate_phylotypes` for datasets too large for a single
        O(n^2) distance matrix and clustering pass. Builds an initial phylotype pool
        from a small "seed" of the most specific placements (Stage A), indexes each
        phylotype by the tree edges its members are placed on (Stage B), streams the
        remaining placements through that index -- assigning each to a matching
        phylotype within `pd_threshold` or marking it an orphan (Stage C), repeatedly
        clusters and absorbs orphans into new phylotypes (Stage D), and finally does
        one cheap pass merging phylotype representatives that ended up close together
        (Stage E).

        Design note
        -----------
        The batch path (`generate_phylotypes`) clusters each pregroup with average
        linkage over its full pairwise distance matrix. This streaming scheme instead
        assigns each placement by its distance to a *sample* of an existing
        phylotype's members -- behavior closer to single/centroid linkage. Near
        phylotype boundaries, some placements may therefore be grouped differently
        than they would be by the batch path. This is an accepted, explicit
        trade-off in exchange for bounded (`O(K*m + seed_size^2 + expand_batch_size^2)`)
        memory instead of `O(n^2)`.

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

        # Most specific placements (fewest placement nodes) first.
        order = torch.argsort(self.placement_present.sum(dim=1)).tolist()
        seed_idx = order[:seed_size]
        remaining = order[seed_size:]

        # ---- Stage A: SEED ----
        pool: List[Dict[str, Any]] = []
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
            seed_groups: Dict[int, List[int]] = defaultdict(list)
            for local_i, cl in enumerate(seed_clusters):
                seed_groups[cl].append(seed_idx[local_i])
            for members in seed_groups.values():
                pool.append({"members": members, "edges": self._group_edges(members)})

        # ---- Stage B: edge -> phylotype inverted index ----
        edge_index: Dict[int, Set[int]] = defaultdict(set)
        for pt_i, pt in enumerate(pool):
            for edge in pt["edges"]:
                edge_index[edge].add(pt_i)

        # ---- Stage C: APPLY (stream the rest) ----
        orphans: List[int] = [
            sv for sv in remaining if not self._apply_sv(sv, pool, edge_index, distal_length=distal_length)
        ]

        # ---- Stage D: EXPAND ----
        while orphans:
            batch, orphans = orphans[:expand_batch_size], orphans[expand_batch_size:]
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
                batch_groups: Dict[int, List[int]] = defaultdict(list)
                for local_i, cl in enumerate(batch_clusters):
                    batch_groups[cl].append(batch[local_i])
                new_groups = list(batch_groups.values())

            for members in new_groups:
                pt_i = len(pool)
                edges = self._group_edges(members)
                pool.append({"members": members, "edges": edges})
                for edge in edges:
                    edge_index[edge].add(pt_i)

            orphans = [sv for sv in orphans if not self._apply_sv(sv, pool, edge_index, distal_length=distal_length)]

        # ---- Stage E: RECONCILE ----
        if len(pool) > 1:
            lcas = [self._get_lca_for_group(pt["members"]) for pt in pool]
            lca_mat = np.zeros((len(pool), len(pool)), dtype=np.float64)
            for i in range(len(pool)):
                for j in range(i + 1, len(pool)):
                    pd_ij = lcas[i].distance(lcas[j])
                    lca_mat[i, j] = pd_ij
                    lca_mat[j, i] = pd_ij
            rep_clusters = AgglomerativeClustering(
                n_clusters=None,
                distance_threshold=self.pd_threshold,
                metric="precomputed",
                linkage="average",
            ).fit_predict(lca_mat)
            merged: Dict[int, List[int]] = defaultdict(list)
            for pt_i, cl in enumerate(rep_clusters):
                merged[cl].extend(pool[pt_i]["members"])
            final_groups = list(merged.values())
        else:
            final_groups = [pt["members"] for pt in pool]

        self.phylogroups.extend({self.placement_names[sv_i] for sv_i in members} for members in final_groups)

    def to_long(self) -> List[Tuple[str, str]]:
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

        # Convert to long format
        pg_sv_long = [
            ("pt__{:05d}".format(pg_i + 1), sv)
            for pg_i, (pg_idx, pg_size) in enumerate(pg_size)
            for sv in self.phylogroups[pg_idx]
        ]

        return pg_sv_long

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
    Main entry point for command-line interface.

    Parses command-line arguments, loads JPLACE data, performs phylotype
    clustering, and outputs results to CSV file.

    Notes
    -----
    This function handles the complete workflow from command line arguments
    to final output generation.
    """
    args_parser = argparse.ArgumentParser(
        description="""Given a JPLACE file of placed features on a phylogenetic tree,
        generate phylotypes or phylogenetically grouped features."""
    )

    args_parser.add_argument(
        "--jplace",
        "-J",
        help="JPLACE file, as created by pplacer or epa-ng",
        type=argparse.FileType("r"),
        required=True,
    )

    args_parser.add_argument(
        "--out",
        "-O",
        help="Where to place the phylogroups (in csv long format)?",
        type=argparse.FileType("wt"),
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
        "approximate (single/centroid-like) linkage near phylotype boundaries. "
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

    args = args_parser.parse_args()

    phylotypes = Phylotypes(
        lwr_overlap=args.lwr_overlap,
        pd_threshold=args.threshold_pd,
        distance=args.distance,
        device=args.device,
        max_pregroup_size=args.max_pregroup_size,
    )
    try:
        phylotypes.load_jplace(args.jplace)
    except ValueError as e:
        logging.error(e)
        sys.exit(1)
    if args.incremental:
        phylotypes.generate_phylotypes_incremental(
            seed_size=args.seed_size,
            expand_batch_size=args.expand_batch_size,
        )
    else:
        phylotypes.generate_phylotypes()

    logging.info("Done Phylogrouping. Outputting.")
    phylotypes.to_csv(args.out)
    logging.info("DONE!")
