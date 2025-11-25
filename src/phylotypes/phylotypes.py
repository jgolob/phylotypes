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


def LWR_Filter(mat: Union[np.ndarray, torch.Tensor], threshold: float) -> np.ndarray:
    """
    Filter matrix rows based on likelihood weight ratio threshold.

    This function converts input to PyTorch tensor, calculates row sums,
    and creates a boolean mask where rows meeting the threshold are marked.

    Args:
        mat: Input matrix as numpy array
        threshold: Minimum row sum threshold for inclusion

    Returns:
        Binary mask as numpy array (0s and 1s)

    Raises:
        TypeError: If mat cannot be converted to tensor
        ValueError: If threshold is negative
    """
    # Convert numpy array to PyTorch tensor if needed
    if not isinstance(mat, torch.Tensor):
        mat = torch.from_numpy(mat)

    # Calculate row sums
    row_sums = torch.sum(mat, dim=1)

    # Create boolean mask for rows where sum >= threshold
    result_mask = row_sums >= threshold

    # Convert to int (0 or 1) and return as numpy array
    return result_mask.int().cpu().numpy()


class Phylotypes:
    """
    A class for generating phylotypes from phylogenetic placement data.

    This class handles the analysis of JPLACE files containing phylogenetic
    placements and groups features into phylotypes based on phylogenetic
    distance and likelihood weight ratios.

    Attributes:
        lwr_overlap: Minimum likelihood weight ratio overlap threshold
        pd_threshold: Phylogenetic distance threshold for clustering
        phylogroups: List of phylotype groups (sets of feature names)
        sv_groups: List of pre-grouped features by LWR overlap
        jplace: Loaded JPLACE data dictionary
        tree: Phylogenetic tree as TreeNode object
        edge_idx: Index of edge_num field in placement data
        lwr_idx: Index of like_weight_ratio field in placement data
        dl_idx: Index of distal_length field in placement data
        sv_nodes: Dictionary mapping feature names to placement nodes
        name_node: Dictionary mapping node IDs to TreeNode objects
        node_name: Dictionary mapping TreeNode objects to node IDs
        node_names: List of node names in consistent order for tree node distance matrix
        node_name_to_idx: Dictionary mapping node names to their index in node_names
        tree_node_distance_matrix: torch tensor of shape (n_nodes, n_nodes) with pairwise tree node distances
    """

    def __init__(self, lwr_overlap: float = 0.1, pd_threshold: float = 1.0) -> None:
        """
        Initialize Phylotypes instance with clustering parameters.

        Args:
            lwr_overlap: Minimum likelihood weight ratio overlap for initial grouping
                         (default: 0.1)
            pd_threshold: Phylogenetic distance threshold for final clustering
                         (default: 1.0)
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
        self.name_node: Dict[int, TreeNode] = {}
        self.node_name: Dict[TreeNode, int] = {}

        # Tree node distance matrix and node indexing
        self.node_names: List[str] = []
        self.node_name_to_idx: Dict[str, int] = {}
        self.tree_node_distance_matrix: Optional[torch.Tensor] = None

    def load_jplace(self, jplace_fh: TextIO) -> None:
        """
        Load and validate JPLACE file data.

        Reads a JPLACE file handle, validates required fields, indexes field
        positions, loads the phylogenetic tree, and caches placement data.

        Args:
            jplace_fh: File handle for JPLACE file (opened in text mode)

        Note:
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
        required_fields = ["fields", "placements", "tree"]
        for field in required_fields:
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

        Raises:
            ValueError: If tree parsing fails
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

        # Generate tree node distance matrix and node indexing
        #self._generate_tree_node_distance_matrix()

    def _generate_tree_node_distance_matrix(self) -> None:
        """
        Generate pairwise tree node distances

        Generates:
        tree_node_distance_matrix: Tensor of shape (n_nodes, n_nodes)
            Contains pairwise distances


        """
        if self.tree is None or self.node_names is None or self.node_name_to_idx is None:
            logging.warning("No tree loaded, cannot generate distance matrix")
            return
        # Implict else
        
        n_nodes = len(self.node_names)
        logging.info("Starting pairwise tree node distance matrix generation on %d nodes, or %d pairs", n_nodes, n_nodes**2)

        # Initialize tree node distance matrix on CPU
        self.tree_node_distance_matrix = torch.zeros((n_nodes, n_nodes), dtype=torch.float32)

        # Simple O(n^2) approach
        for i, node_name_i in enumerate(self.node_names):
            node_i = self.name_node[node_name_i]
            for j, node_name_j in enumerate(self.node_names[i + 1 :], start=i + 1):
                node_j = self.name_node[node_name_j]
                pdist = node_i.distance(node_j)
                self.tree_node_distance_matrix[i, j] = pdist
                self.tree_node_distance_matrix[j, i] = pdist

        # Log statistics for monitoring
        logging.info(
            f"Generated tree node distance matrix of shape {self.tree_node_distance_matrix.shape} "
            f"for {n_nodes} nodes with max distance {self.tree_node_distance_matrix.max():.4f}"
        )

    def _load_placements(self) -> None:
        """
        Load and cache feature placement data.

        Processes placement data from JPLACE file and creates mappings
        between feature names (SVs) and their placement nodes.

        Generates sv_nodes a dictionary
            Key: sv_id (string)
            Value: Dict
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
        self.node_names = sorted({
            str(node_name)
            for pl in self.sv_nodes.values()
            for node_name in pl.keys()
        })
        self.node_name_to_idx = {name: idx for idx, name in enumerate(self.node_names)}


        self._build_placement_tensors()

    def _build_placement_tensors(self) -> None:
        """Build placement tensors
        Generates
            placement_names[List]
            placement_idx[Dict]: Name -> index in placement_names
            placement_lwr tensor (float32) of shape (n_placements, num_nodes)
                filled with LWR of nodes
            placement_dl tensor (float32) of shape (n_placements, num_nodes)
                filled with distal length of placement at nodes
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
        ) -> torch.Tensor:
        """
        Calculate the pairwise Earth Mover's Distance between all pairs of placements in a fully vectorized manner.

        Requires placement and tree node pairwise distance tensors to be generated and populated.

        Generates:
            torch.Tensor: A tensor of shape (n_placements, n_placements) containing the EMD values between all pairs of placements
        """
        if not hasattr(self, "placement_lwr"):
            raise ValueError("Placement Tensor need to be built first")
        if not hasattr(self, "placement_present"):
            raise ValueError("Placement present tensor must be build")
        if distal_length and not hasattr(self, "placement_dl"):
            raise ValueError("Placement distal length tensor need to be built")

        # Implicit else we have needed attributes
        # Expand dimensions for broadcasting
        P_a_i = self.placement_lwr.unsqueeze(1).unsqueeze(3)  # Shape ([n_placement, 1, n_nodes, 1])
        P_b_j = self.placement_lwr.unsqueeze(0).unsqueeze(2)  # Shape ([1, n_placement, 1, n_nodes])

        flows = P_a_i * P_b_j  # Shape ([n_placement, n_placement, n_node, n_node])

        if distal_length:
            # distal_length_a_i shape: (n_placements, 1, n_nodes, 1)
            distal_length_a_i = self.placement_dl.unsqueeze(1).unsqueeze(3)
            # distal_length_b_j shape: (1, n_placements, 1, n_nodes)
            distal_length_b_j = self.placement_dl.unsqueeze(0).unsqueeze(2)
            adjusted_tree_distances = self.tree_node_distance_matrix + distal_length_a_i + distal_length_b_j  # type: ignore[operator]
        else:
            adjusted_tree_distances = self.tree_node_distance_matrix  # type: ignore[assignment]

        weighted_dist = flows * adjusted_tree_distances  # Shape ([n_placement, n_placement, n_node, n_node])

        emd_matrix = torch.sum(weighted_dist, dim=(2, 3))  # shape ([n_placement, n_placement])
        torch.diagonal(emd_matrix).fill_(0.0)

        return emd_matrix

    def _pregroup_by_lwr(self, pd_threshold: float, lwr_overlap: float) -> None:
        """
        Pre-group features by likelihood weight ratio overlap.

        Performs initial clustering of features based on LWR overlap,
        then clusters groups by phylogenetic distance to create
        preliminary feature groupings.

        Args:
            pd_threshold: Phylogenetic distance threshold for group clustering
            lwr_overlap: Minimum LWR overlap threshold for inclusion
        """
        # Sort features by number of placements (ascending)
        sv_to_group = [
            v[0]
            for v in 
            sorted([
                (sv, num_nodes)
                for (sv, num_nodes) in zip(
                    self.placement_names,
                    self.placement_present.sum(dim=1).cpu().numpy()
                )
            ],
            key=lambda v: v[1]
            )
        ]

        sv_groups: List[List[str]] = []
        logging.info("Grouping {} features".format(len(sv_to_group)))

        # Group features by LWR overlap
        while len(sv_to_group) > 0:
            seed_sv = sv_to_group.pop()
            group_svs = {seed_sv}
            group_node_ids = set(self.sv_nodes[seed_sv])

            if len(sv_to_group) == 0:
                sv_groups.append(list(group_svs))
                continue

            # Create overlap matrix for current group
            empty_mat = [0] * (self.lwr_idx + 1)
            g_overlap_mat = np.array(
                [[self.sv_nodes[sv].get(n, empty_mat)[self.lwr_idx] for n in group_node_ids] for sv in sv_to_group]
            )

            # Filter features by overlap threshold
            group_mask = LWR_Filter(g_overlap_mat, lwr_overlap)

            # Add matching features to group
            group_svs.update({sv for (sv, m) in zip(sv_to_group, group_mask.astype(bool)) if m})

            # Remove grouped features from consideration
            sv_to_group = [sv for sv in sv_to_group if sv not in group_svs]
            sv_groups.append(list(group_svs))

            logging.debug(f"{len(sv_to_group)} SVs remain to pregroup, with {len(group_node_ids)} nodes at this step")

        logging.info(
            "Done pre-grouping features into {} groups, of which the largest is {} items".format(
                len(sv_groups), max([len(svg) for svg in sv_groups])
            )
        )

        # Calculate LCAs for each group
        logging.info("Obtaining lowest common ancestor for each group")
        sv_group_lca = [
            self.tree.lowest_common_ancestor([self.name_node[nid] for sv in gsv for nid in self.sv_nodes[sv]])  # type: ignore[union-attr]
            for gsv in sv_groups
        ]

        # Calculate pairwise phylogenetic distances between groups
        logging.info("Calculating pairwise phylogenetic distance between groups")
        g_lca_mat = np.zeros(shape=(len(sv_group_lca), len(sv_group_lca)), dtype=np.float64)

        for i in range(len(sv_group_lca)):
            for j in range(i + 1, len(sv_group_lca)):
                ij_pd = sv_group_lca[i].distance(sv_group_lca[j])
                g_lca_mat[i, j] = ij_pd
                g_lca_mat[j, i] = ij_pd

        # Cluster groups by phylogenetic distance
        logging.info("Clustering groups by phylogenetic distance")
        g_lca_clusters = AgglomerativeClustering(
            n_clusters=None,
            distance_threshold=pd_threshold,
            metric="precomputed",
            linkage="average",
        ).fit_predict(g_lca_mat)

        # Regroup SVs based on group clusters
        logging.info("Regrouping SV based on group-clusters")
        new_old_sv_groups = defaultdict(set)
        for old_cluster_idx, new_cluster_idx in enumerate(g_lca_clusters):
            new_old_sv_groups[new_cluster_idx].add(old_cluster_idx)

        new_sv_groups = [
            list(set([sv for idx in olds for sv in sv_groups[idx]])) for olds in new_old_sv_groups.values()
        ]

        logging.debug(
            "Now {} groups, with the largest {} items".format(
                len(new_sv_groups), max([len(svg) for svg in new_sv_groups])
            )
        )

        self.sv_groups = new_sv_groups

    def group_features(
        self, lwr_overlap: float = 0.95, pd_threshold: float = 0.1, no_dl: bool = False
    ) -> List[Set[str]]:
        """
        Group features into phylotypes based on phylogenetic distance.

        Performs two-stage clustering: first by LWR overlap, then by
        phylogenetic distance within overlapping groups. Can optionally
        ignore distal length calculations.

        Args:
            lwr_overlap: Minimum likelihood weight ratio for initial grouping
                        (default: 0.95)
            pd_threshold: Phylogenetic distance threshold for final clustering
                         (default: 0.1)
            no_dl: If True, ignore distal length to nodes (default: False)

        Returns:
            List of phylotype groups, each as a set of feature names

        Note:
            This method updates the instance's phylogroups attribute and
            returns the complete list of phylotypes.
        """
        if no_dl:
            logging.info("Ignoring distal length.")

        self.lwr_overlap = lwr_overlap
        self.pd_threshold = pd_threshold

        logging.info("Pregrouping based on overlapping LWR for SV")
        self._pregroup_by_lwr(pd_threshold, lwr_overlap)

        logging.info("Starting phylogrouping")

        for g_i, g_sv in enumerate(self.sv_groups):
            if (g_i + 1) % 100 == 0:
                logging.debug("Group {} of {}".format(g_i, len(self.sv_groups)))

            if len(g_sv) == 1:
                self.phylogroups.append(set(g_sv))
                continue

            # Calculate pairwise phylogenetic distances within group
            g_sv_dist_mat = np.zeros(shape=(len(g_sv), len(g_sv)), dtype=np.float64)

            for i in range(len(g_sv)):
                sv1 = g_sv[i]

                # Build placement profiles
                if no_dl:
                    sv1_p = {nid: (npl[self.lwr_idx], 0) for nid, npl in self.sv_nodes[sv1].items()}
                else:
                    sv1_p = {nid: (npl[self.lwr_idx], npl[self.dl_idx]) for nid, npl in self.sv_nodes[sv1].items()}  # type: ignore

                sv1_lwr_total = sum((p[0] for p in sv1_p.values()))

                for j in range(i + 1, len(g_sv)):
                    sv2 = g_sv[j]

                    # Build placement profiles for second feature
                    if no_dl:
                        sv2_p = {nid: (npl[self.lwr_idx], 0) for nid, npl in self.sv_nodes[sv2].items()}
                    else:
                        sv2_p = {nid: (npl[self.lwr_idx], npl[self.dl_idx]) for nid, npl in self.sv_nodes[sv2].items()}  # type: ignore

                    sv2_lwr_total = sum((p[0] for p in sv2_p.values()))

                    # Calculate phylogenetic distance
                    paired_dist = 0.0

                    # Overlapped nodes contribution
                    overlap_nodes = set(sv1_p).intersection(set(sv2_p))
                    paired_dist += sum(
                        [
                            sv1_p[n][1] * sv1_p[n][0] / sv1_lwr_total + sv2_p[n][1] * sv2_p[n][0] / sv2_lwr_total
                            for n in overlap_nodes
                        ]
                    )

                    # Distant nodes contribution
                    distant_nodes = set(sv1_p).union(set(sv2_p)) - set(sv1_p).intersection(set(sv2_p))
                    if len(distant_nodes) > 0:
                        # Find LCA of distant nodes
                        svp_lca = self.tree.lowest_common_ancestor([self.name_node.get(nid) for nid in distant_nodes])  # type: ignore[union-attr]

                        # Add weighted distance to LCA
                        paired_dist += (
                            sum(
                                [
                                    (np[1] + svp_lca.distance(self.name_node[nid])) * np[0]
                                    for nid, np in sv1_p.items()
                                    if nid in distant_nodes
                                ]
                            )
                            / sv1_lwr_total
                            + sum(
                                [
                                    (np[1] + svp_lca.distance(self.name_node[nid])) * np[0]
                                    for nid, np in sv2_p.items()
                                    if nid in distant_nodes
                                ]
                            )
                            / sv2_lwr_total
                        )

                    g_sv_dist_mat[i, j] = paired_dist
                    g_sv_dist_mat[j, i] = paired_dist

            # Cluster features by phylogenetic distance
            g_sv_clusters = AgglomerativeClustering(
                n_clusters=None,
                distance_threshold=pd_threshold,
                metric="precomputed",
                linkage="average",
            ).fit_predict(g_sv_dist_mat)

            # Map clusters to feature sets
            g_phylotype_svs = defaultdict(set)
            for sv, cl in zip(g_sv, g_sv_clusters):
                g_phylotype_svs[cl].add(sv)

            # Add clusters to phylogroups
            self.phylogroups.extend(list(g_phylotype_svs.values()))

        return self.phylogroups

    def to_long(self) -> List[Tuple[str, str]]:
        """
        Convert phylogroups to long format for output.

        Creates a list of (phylotype_id, feature_name) tuples sorted
        by phylotype size (largest first).

        Returns:
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

        Args:
            out_h: File handle for output CSV file (opened in text mode)
        """
        pg_long = self.to_long()
        writer = csv.writer(out_h)
        writer.writerow(["phylotype", "sv"])
        writer.writerows(pg_long)

    def __repr__(self) -> str:
        return f"Phylotype Generator. Tree loaded: {self.tree is not None}."


def main() -> None:
    """
    Main entry point for command-line interface.

    Parses command-line arguments, loads JPLACE data, performs phylotype
    clustering, and outputs results to CSV file.
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

    # Fixed class name bug: was Jplace, should be Phylotypes
    phylotypes = Phylotypes()
    phylotypes.load_jplace(args.jplace)
    phylotypes.group_features(args.lwr_overlap, args.threshold_pd, no_dl=args.no_distal_length)

    logging.info("Done Phylogrouping. Outputting.")
    phylotypes.to_csv(args.out)
    logging.info("DONE!")
