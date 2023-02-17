#!/usr/bin/env python
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
import multiprocessing

# Set up logging
logging.basicConfig(level=logging.INFO)


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
        tp = Phylo.read(
            StringIO(self.jplace['tree']),
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
            logging.info(f'{len(sv_to_group)} SVs remain to pregroup, with {len(group_node_ids)} nodes at this step')

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
            dtype=np.float64
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
            affinity='precomputed',
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

    # Public methods
    def group_features(self, lwr_overlap=0.95, pd_threshold=0.1, no_dl=False):
        # no_dl if true will ignore the remaining distance to nodes.
        if no_dl:
            logging.info("Ignoring distal length.")
        self.lwr_overlap = lwr_overlap
        self.pd_threshold = pd_threshold

        logging.info("Pregrouping based on overlapping LWR for SV")
        self.__pregroup_by_LWR__(pd_threshold, lwr_overlap)

        # ----
        logging.info("Starting phylogrouping")
        for g_i, g_sv in enumerate(self.sv_groups):
            if (g_i + 1) % 100 == 0:
                logging.info("Group {} of {}".format(g_i, len(self.sv_groups)))
            if len(g_sv) == 1:
                self.phylogroups.append(set(g_sv))
                continue
            # If the length of the SV_group is greater than zero, cluster by pairwise PD
            g_sv_dist_mat = np.zeros(
                shape=(
                    len(g_sv),
                    len(g_sv)
                ),
                dtype=np.float64
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
                    paired_dist += sum([
                        sv1_p[n][1] * sv1_p[n][0] / sv1_lwr_total + sv2_p[n][1] * sv2_p[n][0] / sv2_lwr_total
                        for n in overlap_nodes
                    ])
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
                        # and SV2 to the LCA to the paired_distacne
                        paired_dist += (sum([
                            (np[1] + svp_lca.distance(self.name_node[nid])) * np[0]
                            for nid, np in sv1_p.items()
                            if nid in distant_nodes
                        ]) / sv1_lwr_total + sum([
                            (np[1] + svp_lca.distance(self.name_node[nid])) * np[0]
                            for nid, np in sv2_p.items()
                            if nid in distant_nodes
                        ]) / sv2_lwr_total)
                    g_sv_dist_mat[i, j] = paired_dist
                    g_sv_dist_mat[j, i] = paired_dist

            # And use it for agglomerative clustering
            g_sv_clusters = AgglomerativeClustering(
                n_clusters=None,
                distance_threshold=pd_threshold,
                affinity='precomputed',
                linkage='average'
            ).fit_predict(g_sv_dist_mat)
            # Map
            g_phylotype_svs = defaultdict(set)
            for sv, cl in zip(g_sv, g_sv_clusters):
                g_phylotype_svs[cl].add(sv)
            # And append the groups to the phylogroups list
            self.phylogroups += [
                svs for svs in g_phylotype_svs.values()
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
