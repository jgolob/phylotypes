#!/usr/bin/env python3

# # Objective: Add a new set of placements to an existing set of phylotypes.

import argparse
import logging
import numpy as np
from Bio import Phylo
from skbio import TreeNode
import json
from io import StringIO
from collections import defaultdict
import csv
import sys


def placement_pairwise_distance(pl_1, pl_2, jplace):
    pl_1_lwr_total = sum((p[0] for p in pl_1.values()))
    pl_2_lwr_total = sum((p[0] for p in pl_2.values()))
    # Initialize the distance as zero
    paired_dist = 0
    # Overlapped nodes, we to do a bit of weighted averaging
    overlap_nodes = set(pl_1).intersection(set(pl_2))
    paired_dist += sum([
        pl_1[n][1] * pl_1[n][0] / pl_1_lwr_total + pl_2[n][1] * pl_2[n][0] / pl_2_lwr_total
        for n in overlap_nodes
    ])

    distant_nodes = set(pl_1).union(set(pl_2)) - set(pl_1).intersection(set(pl_2))
    if len(distant_nodes) > 0:
        # Determine the lowest common ancestor of the distant nodes for this pair
        svp_lca = jplace.tree.lowest_common_ancestor(
            [
                jplace.name_node.get(nid)
                for nid in distant_nodes
            ]
        )
        # Add weighted average distance of SV1 to the LCA
        # and SV2 to the LCA to the paired_distacne
        paired_dist += (sum([
            (np[1] + svp_lca.distance(jplace.name_node[nid])) * np[0]
            for nid, np in pl_1.items()
            if nid in distant_nodes
        ]) / pl_1_lwr_total + sum([
            (np[1] + svp_lca.distance(jplace.name_node[nid])) * np[0]
            for nid, np in pl_2.items()
            if nid in distant_nodes
        ]) / pl_2_lwr_total)
    return paired_dist


def calculate_sv_pt_distance(new_sv, new_jplace, pt_svs, old_jplace):
    # First placement is the new SV

    pl_1 = {
        nid:
        (
            npl[new_jplace.lwr_idx],
            npl[new_jplace.dl_idx]
        )
        for nid, npl in new_jplace.sv_nodes[new_sv].items()
    }
    return [
        placement_pairwise_distance(pl_1, {
        nid: (
            npl[old_jplace.lwr_idx],
            npl[old_jplace.dl_idx]
        )
        for nid, npl in old_jplace.sv_nodes[sv].items()
        }, old_jplace)
        for sv in pt_svs
    ]


class Jplace():
    lwr_overlap = 0.1
    pd_threshold = 1.0

    def __init__(self, jplace_fh):
        # Set up some reasonable instance defaults
        self.jplace = None
        self.phylogroups = []
        self.sv_groups = []

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


def main():
    args_parser = argparse.ArgumentParser(
        description="""Given a baseline set of placed sequence variants grouped into phylotypes,
        put a new set of sequence variants placed on the same reference tree into the existing phylotypes. 
        """
    )
    args_parser.add_argument(
        '--previous_jp', '-P',
        help='Previous JPLACE file, as created by pplacer or epa-ng',
        type=argparse.FileType('r'),
        required=True
    )
    args_parser.add_argument(
        '--previous_phylotypes', '-p',
        help='CSV file with two columns: phylotype and sv. Represents the existing phylotypes',
        type=argparse.FileType('r'),
        required=True
    )
    args_parser.add_argument(
        '--new_jp', '-N',
        help='NEW JPLACE file, as created by pplacer or epa-ng, containing placed sequence variants to be added',
        type=argparse.FileType('r'),
        required=True
    )
    args_parser.add_argument(
        '--out', '-O',
        help='Output CSV file with the new SV into the existing phylotypes',
        type=argparse.FileType('wt'),
        required=True
    )
    args = args_parser.parse_args()

    logging.info("Loading Previous placements and phylogroups")
    old_jp = Jplace(args.previous_jp)
    cur_pt = {
        r['sv']: r['phylotype']
        for r in csv.DictReader(args.previous_phylotypes)
    }
    pt_sv = defaultdict(set)
    for sv, pt in cur_pt.items():
        pt_sv[pt].add(sv)

    # Verify these overlap
    try:
        assert set(old_jp.sv_nodes.keys()) == set(cur_pt.keys())
    except AssertionError:
        logging.error("Phylotypes and jplace do not match")
        sys.exit(-1)
    # Convert to a phylotype -> contained SV
    pt_sv = defaultdict(set)
    for sv, pt in cur_pt.items():
        pt_sv[pt].add(sv)

    # And lookup edges
    pt_edges = defaultdict(set)
    for sv, pt_id in cur_pt.items():
        pt_edges[pt_id].update(
            old_jp.sv_nodes[sv].keys()
        )

    logging.info("Loading NEW jplace and contained placements")
    new_jp = Jplace(args.new_jp)

    new_sv_pt = {}
    orphaned_new_sv = set()

    for new_sv, new_sv_nodes in new_jp.sv_nodes.items():
        overlap_edges = sorted([
            (
                pt,
                len(
                    pte.intersection(new_sv_nodes.keys())
                )
            )
            for pt, pte in pt_edges.items()
            if len(pte.intersection(new_sv_nodes.keys())) > 0
        ], key=lambda v: -1*v[1])
        if len(overlap_edges) == 1:
            new_sv_pt[new_sv] = overlap_edges[0][0]
        elif len(overlap_edges) == 0:
            # No overlaps. Think about making new phylotype(s) for these SV later
            orphaned_new_sv.add(new_sv)
        else:
            # There are multiple overlapping phylotypes.
            # pick the phylogroup withe shortest average distance
            # to the new sv
            # Take a random sample of only 10 of the existing SV in each phylotype to speed things up.
            nsv_pt_dist = sorted([
                (
                    pt,
                    np.mean(calculate_sv_pt_distance(
                        new_sv,
                        new_jp,
                        np.random.choice(
                            list(pt_sv[pt]),
                            size=min(len(pt_sv[pt]), 10),
                            replace=False
                        ),
                        old_jp
                    ))
                )
                for pt, pt_olc in overlap_edges
            ], key=lambda v: v[1])
            new_sv_pt[new_sv] = nsv_pt_dist[0][0]

    if len(orphaned_new_sv) > 0:
        logging.error(f"Could not add {len(orphaned_new_sv)} of {len(orphaned_new_sv) + len(new_sv_pt)} sequence variants into the existing phylotypes.")

    print(
        f"Successfully integrated {len(new_sv_pt)} of {len(orphaned_new_sv) + len(new_sv_pt)} sequence variants into the existing phylotypes."
    )
    w = csv.writer(args.out)
    w.writerow(['phylotype', 'sv'])
    for sv, pt in new_sv_pt.items():
        w.writerow([pt, sv])


if __name__ == "__main__":
    main()
