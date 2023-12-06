#!/usr/bin/env python3
import argparse
import logging
import numpy as np
from Bio import Phylo
from skbio import TreeNode
import json
from io import StringIO
from collections import defaultdict
from sklearn.cluster import AgglomerativeClustering
import csv
import taichi as ti
from multiprocessing import Pool
import re
from itertools import combinations
import os

# Set up logging
rootLogger = logging.getLogger()
rootLogger.setLevel(logging.INFO)
logFormatter = logging.Formatter(
    '%(asctime)s %(levelname)-8s [phylotypes] %(message)s'
)
consoleHandler = logging.StreamHandler()
consoleHandler.setFormatter(logFormatter)
rootLogger.addHandler(consoleHandler)

LARGE_GROUP = 1000

# Taichi functions
ti.init()


@ti.kernel
def LWR_Filter(mat: ti.types.ndarray(), res: ti.types.ndarray(), threshold: float):
    for i in range(mat.shape[0]):
        rsum = 0.0
        for j in range(mat.shape[1]):
            rsum += mat[i, j]
        if rsum >= threshold:
            res[i] = 1
        else:
            res[i] = 0

@ti.kernel
def OverlappedNodePairedDistance(n_overlap: int, sv1_lwr_total: float, sv2_lwr_total: float, sv1_p_0: ti.types.ndarray(), sv1_p_1: ti.types.ndarray(), sv2_p_0: ti.types.ndarray(), sv2_p_1: ti.types.ndarray()) -> ti.f64:
    pd_1 = 0.0
    pd_2 = 0.0

    for i in range(n_overlap):
        pd_1 += sv1_p_1[i] * sv1_p_0[i]
        pd_2 += sv2_p_1[i] * sv2_p_0[i]
    return pd_1 / sv1_lwr_total + pd_2 / sv2_lwr_total

@ti.kernel
def DistantNodePairedDistance(lwr_total: float, sv_p1: ti.types.ndarray(), sv_lca_p0: ti.types.ndarray()) -> ti.f64:
    pd = 0.0
    for i in range(sv_p1.shape[0]):
        pd += sv_p1[i] + sv_lca_p0[i]
    return pd / lwr_total

class Jplace():
    lwr_overlap = 0.1
    pd_threshold = 1.0

    def __init__(self, jplace_fh):
        # Set up some reasonable instance defaults

        self.phylogroups = []
        self.sv_groups = []
        self.jplace = {}

        logging.info("Loading jplace file")
        self.jplace = json.load(
            jplace_fh
        )
        if 'fields' not in self.jplace:
            logging.error("Missing required 'fields' entry in jplace. Exiting")
            return
        if 'placements' not in self.jplace:
            logging.error("Missing required 'placements' entry in jplace. Exiting")
            return
        if 'tree' not in self.jplace:
            logging.error("Missing required 'tree' entry in jplace. Exiting")
            return
        logging.info("Indexing fields")
        try:
            self.edge_idx = self.jplace['fields'].index('edge_num')
            self.lwr_idx = self.jplace['fields'].index('like_weight_ratio')
            self.dl_idx = self.jplace['fields'].index('distal_length')
        except ValueError:
            logging.error("Missing a needed field (edge_num, like_weight_ratio, distal length")
            return

        logging.info("Loading tree")
        self.__load_tree__()
        logging.info("Loading and caching placements")
        self.__load_placements__()

    def __load_tree__(self):
        # Have to be sure the "NAME" of each edge is not gunk but perfectly matched to the jplace
        re_SEPP_tree = re.compile(r'(|\w+):(?P<edgelen>(\d+\.\d+)(|e[-+]\d+))\[(?P<edgeid>\d+)\]')
        
        def normalized_edges(m):
            return f"{{{m['edgeid']}}}:{m['edgelen']}[{m['edgeid']}]"
        
        
        tree_norm = re_SEPP_tree.sub(
            normalized_edges,
            self.jplace['tree']
        )
        tp = Phylo.read(
            StringIO(tree_norm),
            'newick'
        )
        with StringIO() as th:
            Phylo.write(tp, th, 'newick')
            th.seek(0)
            self.tree = TreeNode.read(th)
        self.name_node = {
            int(n.name.replace('{', "").replace('}', '')): n
            for n in self.tree.traverse() if n.name is not None
        }
        self.node_name = {
            v: k
            for k, v in self.name_node.items()
        }

    def __load_placements__(self):
        self.sv_nodes = {}
        for pl in self.jplace['placements']:
            pl_nodes = {
                p[self.edge_idx]: p
                for p in pl['p']
            }
            if 'nm' in pl:
                for sv, w in pl['nm']:
                    self.sv_nodes[sv] = pl_nodes
            if 'n' in pl:
                for sv in pl['n']:
                    self.sv_nodes[sv] = pl_nodes

    def __pregroup_by_LWR__(self, pd_threshold, lwr_overlap):
        sv_to_group = [
            sv
            for sv, svp_l in
            sorted([
                (sv, len(svp))
                for sv, svp in
                self.sv_nodes.items()
            ], key=lambda v: v[1])
        ]
        # Reset to empty list
        sv_groups = []
        logging.info("Grouping {} features".format(len(sv_to_group)))
        while len(sv_to_group) > 0:
            seed_sv = sv_to_group.pop()
            group_svs = set([seed_sv])
            group_node_ids = set(self.sv_nodes[seed_sv])
            if len(sv_to_group) == 0:
                sv_groups.append(
                    group_svs
                )
                continue
            # Make a matrix for this group
            empty_mat = [0] * (self.lwr_idx + 1)
            g_overlap_mat = np.array([
                [
                    self.sv_nodes[sv].get(
                        n, empty_mat
                    )[self.lwr_idx]
                    for n in group_node_ids
                ]
                for sv in sv_to_group
            ])
            # Use some matrix math to find svs to add to this group
            group_mask = np.zeros(
                g_overlap_mat.shape[0],
                dtype=int
            )
            LWR_Filter(
                g_overlap_mat,
                group_mask,
                lwr_overlap
            )
            group_svs.update(
                {
                    sv
                    for (sv, m) in
                    zip(
                        sv_to_group, group_mask.astype(bool)
                    ) if m
                }
            )
            sv_to_group = [sv for sv in sv_to_group if sv not in group_svs]
            sv_groups.append(list(group_svs))
            logging.debug(f'{len(sv_to_group)} SVs remain to pregroup, with {len(group_node_ids)} nodes at this step')

        logging.info("Done pre-grouping features into {} groups, of which the largest is {} items".format(
            len(sv_groups),
            max([len(svg) for svg in sv_groups])
        ))
        logging.info("Obtaining lowest common ancestor for each group")
        sv_group_lca = [
            self.tree.lowest_common_ancestor([self.name_node[nid] for sv in gsv for nid in self.sv_nodes[sv]])
            for gsv in sv_groups
        ]

        logging.info("Calculating pairwise phylogenetic distance between groups")
        g_lca_mat = np.zeros(
            shape=(len(sv_group_lca), len(sv_group_lca)),
            dtype=np.float32
        )
        for i in range(len(sv_group_lca)):
            for j in range(i + 1, len(sv_group_lca)):
                ij_pd = sv_group_lca[i].distance(sv_group_lca[j])
                g_lca_mat[i, j] = ij_pd
                g_lca_mat[j, i] = ij_pd

        logging.info('Clusting groups by phylogenetic distance')
        g_lca_clusters = AgglomerativeClustering(
            n_clusters=None,
            distance_threshold=pd_threshold,
            metric='precomputed',
            linkage='average'
        ).fit_predict(g_lca_mat)

        logging.info("Regrouping SV based on group-clusters")
        new_old_sv_groups = defaultdict(set)
        for old_cluster_idx, new_cluster_idx in enumerate(g_lca_clusters):
            new_old_sv_groups[new_cluster_idx].add(old_cluster_idx)
        new_sv_groups = [
            list(set([
                sv for idx in olds for sv in sv_groups[idx]
            ]))
            for olds in new_old_sv_groups.values()
        ]
        logging.info("Now {} groups, with the largest {} items".format(
            len(new_sv_groups),
            max([
                len(svg) for svg in new_sv_groups
            ])
        ))
        self.sv_groups = new_sv_groups

    def __phylotypes_from_groups__(self, g_sv, pd_threshold, no_dl=False):
        # Simple case of a single sv in this group, just assign it.
        if len(g_sv) == 1:
            return [set(g_sv)]
        # Implicit else we have some work to do...
        # If the length of the SV_group is greater than zero, cluster by pairwise PD
        g_sv_dist_mat = np.zeros(
            shape=(
                len(g_sv),
                len(g_sv)
            ),
            dtype=np.float32
        )

        for i in range(len(g_sv)):
            sv1 = g_sv[i]
            if no_dl:
                sv1_p = {
                    nid:
                    (
                        npl[self.lwr_idx],
                        0
                    )
                    for nid, npl in self.sv_nodes[sv1].items()
                }
            else:
                sv1_p = {
                    nid:
                    (
                        npl[self.lwr_idx],
                        npl[self.dl_idx]
                    )
                    for nid, npl in self.sv_nodes[sv1].items()
                }
            sv1_lwr_total = sum((p[0] for p in sv1_p.values()))
            for j in range(i + 1, len(g_sv)):
                sv2 = g_sv[j]
                if no_dl:
                    sv2_p = {
                        nid:
                        (
                            npl[self.lwr_idx],
                            0
                        )
                        for nid, npl in self.sv_nodes[sv2].items()
                    }
                else:
                    sv2_p = {
                        nid:
                        (
                            npl[self.lwr_idx],
                            npl[self.dl_idx]
                        )
                        for nid, npl in self.sv_nodes[sv2].items()
                    }
                sv2_lwr_total = sum((p[0] for p in sv2_p.values()))
                # Initialize the distance as zero
                paired_dist = 0
                # Overlapped nodes
                overlap_nodes = set(sv1_p).intersection(set(sv2_p))
                paired_dist += OverlappedNodePairedDistance(
                    len(overlap_nodes),
                    sv1_lwr_total,
                    sv2_lwr_total,
                    np.array([sv1_p[n][0] for n in overlap_nodes], dtype=np.float32),
                    np.array([sv1_p[n][1] for n in overlap_nodes], dtype=np.float32),
                    np.array([sv2_p[n][0] for n in overlap_nodes], dtype=np.float32),
                    np.array([sv2_p[n][1] for n in overlap_nodes], dtype=np.float32),
                )
                distant_nodes = set(sv1_p).union(set(sv2_p)) - set(sv1_p).intersection(set(sv2_p))
                if len(distant_nodes) > 0:
                    # Determine the lowest common ancestor of the distant nodes for this pair
                    svp_lca = self.tree.lowest_common_ancestor(
                        [
                            self.name_node.get(nid)
                            for nid in distant_nodes
                        ]
                    )
                    # Add weighted average distance of SV1 to the LCA
                    # and SV2 to the LCA to the paired_distace                   
                    paired_dist += DistantNodePairedDistance(
                        sv1_lwr_total,
                        np.array([
                            np[1] 
                            for nid, np in sv1_p.items()
                            if nid in distant_nodes
                        ], dtype=np.float32),
                        np.array([
                            svp_lca.distance(self.name_node[nid]) * np[0]
                            for nid, np in sv1_p.items()
                            if nid in distant_nodes
                        ], dtype=np.float32),                        
                    )
                    paired_dist += DistantNodePairedDistance(
                        sv2_lwr_total,
                        np.array([
                            np[1] 
                            for nid, np in sv2_p.items()
                            if nid in distant_nodes
                        ], dtype=np.float32),
                        np.array([
                            svp_lca.distance(self.name_node[nid]) * np[0]
                            for nid, np in sv2_p.items()
                            if nid in distant_nodes
                        ], dtype=np.float32),                        
                    )

                g_sv_dist_mat[i, j] = paired_dist
                g_sv_dist_mat[j, i] = paired_dist

        # And use it for agglomerative clustering
        g_sv_clusters = AgglomerativeClustering(
            n_clusters=None,
            distance_threshold=pd_threshold,
            metric='precomputed',
            linkage='average'
        ).fit_predict(g_sv_dist_mat)
        # Map
        g_phylotype_svs = defaultdict(set)
        for sv, cl in zip(g_sv, g_sv_clusters):
            g_phylotype_svs[cl].add(sv)
        
        return [
            set(svs) for svs in g_phylotype_svs.values()
        ]

    # Need a wrapper to handle edge case of a zero length list of nodes      
    def lca_wrapper(self, dn):
        if len(dn) == 0:
            return None
        else:
                self.tree.lowest_common_ancestor(dn)

    # Public methods
    def group_features(self, lwr_overlap=0.95, pd_threshold=0.1, no_dl=False, threads=None):
        # no_dl if true will ignore the remaining distance to nodes.
        if no_dl:
            logging.info("Ignoring distal length.")
        self.lwr_overlap = lwr_overlap
        self.pd_threshold = pd_threshold

        logging.info("Pregrouping based on overlapping LWR for SV")
        self.__pregroup_by_LWR__(pd_threshold, lwr_overlap)

        logging.info("Identifying lonely SV (in groups of one)")
        # For lonely SV (in groups of one) make them phylotypes
        self.phylogroups = [
            set(svg) for svg in self.sv_groups if len(svg) == 1
        ]
        logging.info("Phylotyping start")
        # Set up the pool to handle processing...
        with Pool(threads) as pool:
            for svg in self.sv_groups:
                if len(svg) == 1:
                    # Handled above... so just continue
                    continue
                # Implicit else
                if len(svg) > LARGE_GROUP:
                    logging.info(f'Working on a large group with {len(svg):,d} members')
                
                # Make an index of sv <-> position in the svg array
                sv_i = {
                    sv: i 
                    for (i, sv) in enumerate(svg)
                }
                # Generate all the permutations for this group in long format and store as indicies to save space (as opposed to strings)
                sv_pairs = [
                        (
                            sv_i.get(sv[0]), # SV0 index
                            sv_i.get(sv[1]), # SV1 index
                            #set(self.sv_nodes[sv[0]].keys()).intersection(set(self.sv_nodes[sv[1]].keys())), # Overlapped Nodes
                            #set(self.sv_nodes[sv[0]].keys()).union(set(self.sv_nodes[sv[1]].keys())) - set(self.sv_nodes[sv[0]].keys()).intersection(set(self.sv_nodes[sv[1]].keys())), # Distant Nodes
                            #np.sum([npl[self.lwr_idx] for npl in self.sv_nodes[sv[0]].values()]), # sv0 LWR total
                            #np.sum([npl[self.lwr_idx] for npl in self.sv_nodes[sv[1]].values()]), # sv1 LWR total,

                        )
                        for sv in
                        combinations(svg, 2)
                    ]

                if len(svg) > LARGE_GROUP:
                    logging.info(f"Identifying lowest common ancestors between SV {len(sv_pairs):,d} permutations")
                sv_pairs_lca = list(
                    pool.imap(
                        self.lca_wrapper,
                        (
                            [self.name_node.get(nid) for nid in dn]
                            for dn in (  # Distant nodes generator
                                set(
                                    self.sv_nodes[svg[sv0]].keys()).union(
                                        set(self.sv_nodes[svg[sv1]].keys()
                                    )
                                ) - set(
                                    self.sv_nodes[svg[sv0]].keys()).intersection(set(self.sv_nodes[svg[sv1]].keys())
                                ) # Distant Nodes
                                for (sv0, sv1) in sv_pairs
                            )
                        ),
                        chunksize=100,
                    ),
                )

                if len(svg) > LARGE_GROUP:
                    logging.info("Overlapped distance calculation for SV pairs")
                
                pwd_oln = pool.starmap(
                    OverlappedNodePairedDistance, 
                    (
                        (
                            len(oln),
                            sv0_lwr_total,
                            sv1_lwr_total,
                            np.array([
                                pl[self.lwr_idx]
                                for nid, pl in self.sv_nodes[sv0].items()
                                if nid in oln
                            ], dtype=np.float32),
                            np.array([
                                pl[self.dl_idx]
                                for nid, pl in self.sv_nodes[sv0].items()
                                if nid in oln
                            ], dtype=np.float32),
                            np.array([
                                pl[self.lwr_idx]
                                for nid, pl in self.sv_nodes[sv1].items()
                                if nid in oln
                            ], dtype=np.float32),
                            np.array([
                                pl[self.dl_idx]
                                for nid, pl in self.sv_nodes[sv1].items()
                                if nid in oln
                            ], dtype=np.float32),
                        )
                        for sv0, sv1, oln, sv0_lwr_total, sv1_lwr_total in (
                            (
                                svg[sv0],
                                svg[sv1],
                                set(self.sv_nodes[svg[sv0]].keys()).intersection(set(self.sv_nodes[svg[sv1]].keys())), # Overlapped nodes
                                np.sum([npl[self.lwr_idx] for npl in self.sv_nodes[svg[sv0]].values()]), # sv0 LWR total
                                np.sum([npl[self.lwr_idx] for npl in self.sv_nodes[svg[sv1]].values()]), # sv1 LWR total
                            )
                            for sv0, sv1 in sv_pairs
                        )
                    ),
                    chunksize=100
                )

                if len(svg) > LARGE_GROUP:
                    logging.info("Distant nodes distance for first SV")

                pwd_dn_0 = pool.starmap(
                    DistantNodePairedDistance, 
                    (
                        (
                            sv0_lwr_total,
                            np.array([
                                pl[self.dl_idx]
                                for nid, pl in self.sv_nodes[sv0].items()
                                if nid in dn
                            ], dtype=np.float32),
                            np.array([
                                sv_pairs_lca[p_i].distance(self.name_node[nid]) * pl[self.lwr_idx]
                                for nid, pl in self.sv_nodes[sv0].items()
                                if nid in dn
                            ], dtype=np.float32),
                        ) if sv_pairs_lca[p_i] is not None else (
                            sv0_lwr_total,
                            np.array([], dtype=np.float32),
                            np.array([], dtype=np.float32),
                        )
                        for p_i, (sv0, dn, sv0_lwr_total, ) in enumerate((
                            (
                                svg[sv0],
                                set(
                                    self.sv_nodes[svg[sv0]].keys()).union(
                                        set(self.sv_nodes[svg[sv1]].keys()
                                    )
                                ) - set(
                                    self.sv_nodes[svg[sv0]].keys()).intersection(set(self.sv_nodes[svg[sv1]].keys())
                                ), # Distant Nodes                            
                                np.sum([npl[self.lwr_idx] for npl in self.sv_nodes[svg[sv0]].values()]), # sv0 LWR total
                            )
                            for sv0, sv1 in sv_pairs
                        ))
                    ),
                    chunksize=100
                )

                if len(svg) > LARGE_GROUP:
                    logging.info("Distant nodes distance for second SV")

                pwd_dn_1 = pool.starmap(
                    DistantNodePairedDistance, 
                    (
                        (
                            sv1_lwr_total,
                            np.array([
                                pl[self.dl_idx]
                                for nid, pl in self.sv_nodes[sv1].items()
                                if nid in dn
                            ], dtype=np.float32),
                            np.array([
                                sv_pairs_lca[p_i].distance(self.name_node[nid]) * pl[self.lwr_idx]
                                for nid, pl in self.sv_nodes[sv1].items()
                                if nid in dn
                            ], dtype=np.float32),
                        ) if sv_pairs_lca[p_i] is not None else (
                            sv1_lwr_total,
                            np.array([], dtype=np.float32),
                            np.array([], dtype=np.float32),
                        )
                        for p_i, (sv1, dn, sv1_lwr_total) in enumerate((
                            (
                                svg[sv1],
                                set(
                                    self.sv_nodes[svg[sv0]].keys()).union(
                                        set(self.sv_nodes[svg[sv1]].keys()
                                    )
                                ) - set(
                                    self.sv_nodes[svg[sv0]].keys()).intersection(set(self.sv_nodes[svg[sv1]].keys())
                                ), # Distant Nodes 
                                np.sum([npl[self.lwr_idx] for npl in self.sv_nodes[svg[sv1]].values()]),                        
                            )
                            for sv0, sv1 in sv_pairs
                        ))
                    ),
                    chunksize=100
                )            
                
                if len(svg) > LARGE_GROUP:
                    logging.info("Building PWD Matrix")
                
                g_sv_dist_mat = np.zeros(
                    shape=(
                        len(svg),
                        len(svg)
                    ),
                    dtype=np.float32
                )
                for svs, pwd_o, pwd_d0, pwd_d1 in zip(
                    sv_pairs,
                    pwd_oln,
                    pwd_dn_0,
                    pwd_dn_1
                ):
                    d = pwd_o + pwd_d0 + pwd_d1
                    g_sv_dist_mat[svs[0], svs[1]] = d
                    g_sv_dist_mat[svs[1], svs[0]] = d

                g_sv_clusters = AgglomerativeClustering(
                    n_clusters=None,
                    distance_threshold=pd_threshold,
                    metric='precomputed',
                    linkage='average'
                ).fit_predict(g_sv_dist_mat)
                g_phylotype_svs = defaultdict(set)
                for sv, cl in zip(svg, g_sv_clusters):
                    g_phylotype_svs[cl].add(sv)
                self.phylogroups += [
                    set(svs) for svs in g_phylotype_svs.values()
                ]         

        return self.phylogroups

    def to_long(self):
        # For each phylogroup calculate the size of the group (in SV) and sort by size
        pg_size = sorted([
            (pg_i, len(pg))
            for pg_i, pg in enumerate(self.phylogroups)
        ], key=lambda pgc: -1 * pgc[1])
        # Convert to long format for output
        pg_sv_long = [
            (
                'pt__{:05d}'.format(pg_i + 1),
                sv
            )
            for pg_i, (pg_idx, pg_size) in enumerate(pg_size)
            for sv in self.phylogroups[pg_idx]
        ]
        return pg_sv_long

    def to_csv(self, out_h):
        pg_long = self.to_long()
        writer = csv.writer(out_h)
        writer.writerow(['phylotype', 'sv'])
        writer.writerows(pg_long)


def main():
    args_parser = argparse.ArgumentParser(
        description="""Given a JPLACE file of placed features on a phylogenetic tree,
        generate phylotypes or phylogenetically grouped features.
        """
    )
    args_parser.add_argument(
        '--jplace', '-J',
        help='JPLACE file, as created by pplacer or epa-ng',
        type=argparse.FileType('r'),
        required=True
    )
    args_parser.add_argument(
        '--out', '-O',
        help='Where to place the phylogroups (in csv long format)?',
        type=argparse.FileType('wt'),
        required=True
    )
    args_parser.add_argument(
        '--lwr-overlap', '-L',
        help='minimum like-weight ratio for grouping of features. (Default: 0.01).',
        default=0.01,
        type=float,
    )
    args_parser.add_argument(
        '--threshold_pd', '-T',
        help='Phylogenetic distance threshold for clustering. (Default: 1.0)',
        default=0.1,
        type=float,
    )
    args_parser.add_argument(
        '--cpus', '-C',
        help='Number of CPUs / threads to use. Default is all available.',
        default=os.cpu_count(),
    )    
    args_parser.add_argument(
        '--no-distal-length', '-ndl',
        help='Ignore distal length to nodes. (Default: False)',
        action='store_true'
    )
    args = args_parser.parse_args()
    jplace = Jplace(args.jplace)
    jplace.group_features(args.lwr_overlap, args.threshold_pd, no_dl=args.no_distal_length)
    logging.info("Done Phylogrouping. Outputting.")
    jplace.to_csv(args.out)
    logging.info("DONE!")


if __name__ == "__main__":
    main()
