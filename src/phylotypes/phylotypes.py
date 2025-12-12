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
import json
import logging
import re
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
        List of node names in consistent order for tree node distance matrix
    node_name_to_idx : Dict[str, int]
        Dictionary mapping node names to their index in node_names
    tree_node_distance_matrix : Optional[torch.Tensor]
        torch tensor of shape (n_nodes, n_nodes) with pairwise tree node distances
    """

    required_fields = ["fields", "placements", "tree"]

    def __init__(self, lwr_overlap: float = 0.1, pd_threshold: float = 1.0) -> None:
        """
        Initialize Phylotypes instance with clustering parameters.

        Parameters
        ----------
        lwr_overlap : float, optional
            Minimum likelihood weight ratio overlap for initial grouping (default: 0.1)
        pd_threshold : float, optional
            Phylogenetic distance threshold for final clustering (default: 1.0)
        """
        # Clustering parameters
        self.lwr_overlap: float = lwr_overlap
        self.pd_threshold: float = pd_threshold

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

        # Tree node distance matrix and node indexing
        self.node_names: List[str] = []
        self.node_name_to_idx: Dict[str, int] = {}
        self.tree_node_distance_matrix: Optional[torch.Tensor] = None

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
        """
        logging.info("Loading jplace file")

        try:
            self.jplace = json.load(jplace_fh)
        except (json.JSONDecodeError, AttributeError) as e:
            logging.error(f"Failed to parse JPLACE file: {e}")
            return

        # Validate required fields

        for field in self.required_fields:
            if field not in self.jplace:
                logging.error(f"Missing required '{field}' entry in jplace. Exiting")
                return

        logging.info("Indexing fields")
        try:
            self.edge_idx = self.jplace["fields"].index("edge_num")
            self.lwr_idx = self.jplace["fields"].index("like_weight_ratio")
            self.dl_idx = self.jplace["fields"].index("distal_length")
        except ValueError as e:
            logging.error(f"Missing required field: {e}. Required: edge_num, like_weight_ratio, distal_length")
            return

        logging.info("Loading and caching placements")
        self._load_placements()
        logging.info("Loading tree")
        self._load_tree()

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
            logging.error(f"Failed to parse phylogenetic tree: {e}")
            return

        # Cleanup node names...
        for node in self.tree.traverse():
            node.name = node.name.replace("{", "").replace("}", "")

        self.name_node = {n.name: n for n in self.tree.traverse() if n.name is not None}
        self.node_name = {v: k for k, v in self.name_node.items()}

    def _get_tree_node_distance_matrix(
        self,
        node_indices: Optional[List[int]] = None,
    ) -> Optional[torch.Tensor]:
        """
        Generate pairwise tree node distances

        Generates:
        tree_node_distance_matrix: Tensor of shape (n_nodes, n_nodes)
            Contains pairwise distances


        """
        if self.tree is None or self.node_names is None or self.node_name_to_idx is None:
            logging.warning("No tree loaded, cannot generate distance matrix")
            return None
        # Implict else

        if node_indices is not None:
            node_names = [self.node_names[nid] for nid in node_indices]
        else:
            node_names = self.node_names

        n_nodes = len(node_names)
        logging.info(
            "Starting pairwise tree node distance matrix generation on %d nodes, or %d pairs", n_nodes, n_nodes**2
        )

        tree_node_distance_matrix = torch.zeros((n_nodes, n_nodes), dtype=torch.float32)

        # Simple O(n^2) approach
        for i, node_name_i in enumerate(node_names):
            node_i = self.name_node[node_name_i]
            for j, node_name_j in enumerate(node_names[i + 1 :], start=i + 1):
                node_j = self.name_node[node_name_j]
                pdist = node_i.distance(node_j)
                tree_node_distance_matrix[i, j] = pdist
                tree_node_distance_matrix[j, i] = pdist
        return tree_node_distance_matrix

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
        self.placement_lwr = torch.zeros((n_placements, n_nodes), dtype=torch.float32)
        self.placement_dl = torch.zeros((n_placements, n_nodes), dtype=torch.float32)

        for placement_name, placement in self.sv_nodes.items():
            placement_i = self.placement_idx[placement_name]
            placement_nodes = list(placement.keys())
            self.placement_lwr[placement_i, [self.node_name_to_idx[str(node)] for node in placement_nodes]] = (
                torch.tensor(
                    [placement[node][self.lwr_idx] for node in placement_nodes],
                    dtype=torch.float32,
                )
            )
            self.placement_dl[placement_i, [self.node_name_to_idx[str(node)] for node in placement_nodes]] = (
                torch.tensor(
                    [placement[node][self.dl_idx] for node in placement_nodes],
                    dtype=torch.float32,
                )
            )

        self.placement_present = self.placement_lwr > 0

    def pairwise_emd(
        self,
        placement_indices: Optional[List[int]] = None,
        distal_length: bool = True,
        batch_size: int = 10,
    ) -> torch.Tensor:
        """
        Calculate the pairwise Earth Mover's Distance between all pairs of placements in a fully vectorized manner.

        Requires placement and tree node pairwise distance tensors to be generated and populated.

        Parameters
        ----------
        placement_indices : Optional[List[int]], optional
            List of placement indices to calculate EMD for (default: None)
        distal_length : bool, optional
            Whether to include distal length in calculations (default: True)

        Returns
        -------
        torch.Tensor
            A tensor of shape (n_placements, n_placements) containing the EMD values between all pairs of placements

        Raises
        ------
        ValueError
            If required placement tensors are not built
        """
        if not hasattr(self, "placement_lwr"):
            raise ValueError("Placement Tensor need to be built first")
        if not hasattr(self, "placement_present"):
            raise ValueError("Placement present tensor must be build")
        if distal_length and not hasattr(self, "placement_dl"):
            raise ValueError("Placement distal length tensor need to be built")
            # Implicit else we have needed attributes

        # Subset based on provided indices if provided
        if placement_indices is not None:
            subset_placement_lwr = self.placement_lwr[placement_indices]
            subset_dl = self.placement_dl[placement_indices]
        else:
            subset_placement_lwr = self.placement_lwr
            subset_dl = self.placement_dl

        # And the nodes we want to work with..
        subset_node_idx_w_weight = subset_placement_lwr.amax(dim=0).nonzero().squeeze()
        subset_placement_lwr = subset_placement_lwr[:, subset_node_idx_w_weight]
        subset_dl = subset_dl[:, subset_node_idx_w_weight]

        # Handle the edge case where all the weight for these placements is on *one* node
        # i.e., the dim of subset_node_idx_w_weight is 0
        if subset_node_idx_w_weight.dim() == 0:
            logging.info(f"Calculating EMD for edge case of {subset_placement_lwr.shape[0]} placements on one node.")
            # NO need to bother with flows. It's all here
            if not distal_length:
                # If there is no distal length, distance is zero
                return torch.zeros(
                    subset_placement_lwr.shape[0], subset_placement_lwr.shape[0], dtype=subset_placement_lwr.dtype
                )
            else:  # There *is* distal length
                emd_matrix = subset_dl.unsqueeze(1) + subset_dl.unsqueeze(0)
                torch.diagonal(emd_matrix).fill_(0.0)
                return emd_matrix

        logging.info(f"Building tree distance matrix {subset_node_idx_w_weight.shape[0]} nodes.")
        tree_node_distance_matrix = self._get_tree_node_distance_matrix(list(subset_node_idx_w_weight))

        n_placements = subset_placement_lwr.shape[0]
        emd_matrix = torch.zeros((n_placements, n_placements), dtype=torch.float32)
        # Expand dimensions for broadcasting
        logging.info(
            f"Calculating flows for {subset_placement_lwr.shape[0]} placements over {subset_placement_lwr.shape[1]} nodes."
        )

        for i in range(0, n_placements, batch_size):
            for j in range(0, n_placements, batch_size):
                # Extract batches
                batch_i = subset_placement_lwr[i : i + batch_size]
                batch_j = subset_placement_lwr[j : j + batch_size]
                # And limit to notes in this batch
                batch_nodes_idx = torch.vstack([batch_i, batch_j]).amax(dim=0).nonzero().squeeze()
                # And then limit again!
                batch_i = batch_i[:, batch_nodes_idx]
                batch_j = batch_j[:, batch_nodes_idx]
                # Compute flows for the batch
                P_a_i = batch_i.unsqueeze(1).unsqueeze(3)  # Shape: [batch_i, 1, n_nodes, 1]
                P_b_j = batch_j.unsqueeze(0).unsqueeze(2)  # Shape: [1, batch_j, 1, n_nodes]
                flows = P_a_i * P_b_j  # Shape: [batch_i, batch_j, n_nodes, n_nodes]
                # Compute adjusted tree distances
                if distal_length:
                    distal_length_a_i = subset_dl[i : i + batch_size, batch_nodes_idx].unsqueeze(1).unsqueeze(3)
                    distal_length_b_j = subset_dl[j : j + batch_size, batch_nodes_idx].unsqueeze(0).unsqueeze(2)
                    adjusted_tree_distances = (
                        tree_node_distance_matrix[batch_nodes_idx, batch_nodes_idx]  # type: ignore
                        + distal_length_a_i
                        + distal_length_b_j
                    )
                else:
                    adjusted_tree_distances = tree_node_distance_matrix[batch_nodes_idx, batch_nodes_idx]  # type: ignore
                # Compute weighted distances and sum
                weighted_dist = flows * adjusted_tree_distances
                emd_batch = torch.sum(weighted_dist, dim=(2, 3))  # Shape: [batch_i, batch_j]
                # Update the EMD matrix
                emd_matrix[i : i + batch_size, j : j + batch_size] = emd_batch

        torch.diagonal(emd_matrix).fill_(0.0)

        return emd_matrix

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
                self.name_node[self.node_names[nid.item()]]
                for g_i in group
                for nid in self.placement_present[g_i].nonzero().numpy().flatten()
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

        new_sv_groups = [list(set([sv for idx in olds for sv in groups[idx]])) for olds in new_old_sv_groups.values()]

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
                logging.debug("Group {} of {}".format(g_i, len(self.sv_groups)))

            if len(g_sv) == 1:
                self.phylogroups.append(set([self.placement_names[g_sv[0]]]))
                continue
            # Implict else, this isn't a singleton group.
            # Calculate pairwise phylogenetic distances within group
            g_sv_dist_mat = self.pairwise_emd(g_sv)
            # Cluster features by phylogenetic distance
            g_sv_clusters = AgglomerativeClustering(
                n_clusters=None,
                distance_threshold=self.pd_threshold,
                metric="precomputed",
                linkage="average",
            ).fit_predict(g_sv_dist_mat)

            # Map clusters to feature sets
            g_phylotype_svs = defaultdict(set)
            for sv, cl in zip(g_sv, g_sv_clusters):
                g_phylotype_svs[cl].add(sv)

            # Add clusters to phylogroups
            self.phylogroups.extend(
                [{self.placement_names[sv_i] for sv_i in phylotype_svs} for phylotype_svs in g_phylotype_svs.values()]
            )

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
        help="minimum like-weight ratio for grouping of features. (Default: 0.01).",
        default=0.01,
        type=float,
    )

    args_parser.add_argument(
        "--threshold_pd",
        "-T",
        help="Phylogenetic distance threshold for clustering. (Default: 1.0)",
        default=0.1,
        type=float,
    )

    args_parser.add_argument(
        "--no-distal-length",
        "-ndl",
        help="Ignore distal length to nodes. (Default: False)",
        action="store_true",
    )

    args = args_parser.parse_args()

    phylotypes = Phylotypes(
        lwr_overlap=args.lwr_overlap,
        pd_threshold=args.threshold_pd,
    )
    phylotypes.load_jplace(args.jplace)
    phylotypes.generate_phylotypes()

    logging.info("Done Phylogrouping. Outputting.")
    phylotypes.to_csv(args.out)
    logging.info("DONE!")
