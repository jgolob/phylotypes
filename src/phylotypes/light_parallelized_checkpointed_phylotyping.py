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
import re
import time
import os
import sqlite3
from typing import Optional

# -------- Optional memory logging via psutil (if available) --------
try:
    import psutil
    _PSUTIL = True
except Exception:
    _PSUTIL = False


def _log_mem(note=""):
    if _PSUTIL:
        rss = psutil.Process(os.getpid()).memory_info().rss / (1024**3)
        logging.debug("[mem] RSS={:.2f} GB {}".format(rss, note))


# -------- Logging setup helpers --------
def setup_logging(log_file: Optional[str], console_level: str, file_level: str = "DEBUG"):
    rootLogger = logging.getLogger()
    rootLogger.handlers = []

    ch = logging.StreamHandler()
    ch.setLevel(getattr(logging, console_level.upper(), logging.INFO))
    ch.setFormatter(logging.Formatter('%(asctime)s %(levelname)-8s [phylotypes] %(message)s'))
    rootLogger.addHandler(ch)

    if log_file:
        fh = logging.FileHandler(log_file, mode='w')
        fh.setLevel(getattr(logging, file_level.upper(), logging.DEBUG))
        fh.setFormatter(logging.Formatter(
            '%(asctime)s %(levelname)-8s [phylotypes] pid=%(process)d %(filename)s:%(lineno)d — %(message)s'
        ))
        rootLogger.addHandler(fh)

    rootLogger.setLevel(logging.DEBUG if log_file else getattr(logging, console_level.upper(), logging.INFO))


class Timer:
    def __init__(self, label: str):
        self.label = label
        self.t0 = None

    def __enter__(self):
        self.t0 = time.perf_counter()
        logging.info("▶ {} ...".format(self.label))
        return self

    def __exit__(self, exc_type, exc, tb):
        dt = time.perf_counter() - self.t0
        status = "OK" if exc_type is None else "ERROR: {}".format(exc_type.__name__)
        logging.info("✔ {} done in {:.2f}s ({})".format(self.label, dt, status))


def log_every(i: int, total: int, interval: int, prefix: str):
    if interval <= 0:
        return
    if (i % interval == 0) or (i == total - 1):
        pct = (i + 1) / total * 100.0 if total else 100.0
        logging.debug("{}: {}/{} ({:.1f}%)".format(prefix, i + 1, total, pct))


# -------- ensure on-disk durability right after writing --------
def _flush_close_and_report(fh, label: str):
    """
    Flush + fsync the file handle, log absolute path & size, and close it if it's a real file.
    If it's stdout ('-'), we only flush and report.
    """
    path = getattr(fh, 'name', None)
    try:
        fh.flush()
    except Exception as e:
        logging.debug("flush failed for {}: {}".format(label, e))
    try:
        # may fail for e.g. stdout on some platforms; best-effort
        os.fsync(fh.fileno())
    except Exception as e:
        logging.debug("fsync failed for {}: {}".format(label, e))

    try:
        if path and path not in ('<stdout>', 'stdout'):
            abspath = os.path.abspath(path)
            size = os.path.getsize(path) if os.path.exists(path) else 0
            logging.info("Wrote {} to {} ({} bytes)".format(label, abspath, size))
        else:
            logging.info("Wrote {} to STDOUT".format(label))
    except Exception as e:
        logging.debug("stat failed for {}: {}".format(label, e))

    # Close only if it's a real file handle (not stdout)
    try:
        if path and path not in ('<stdout>', 'stdout'):
            fh.close()
    except Exception as e:
        logging.debug("close failed for {}: {}".format(label, e))


# -------- Taichi init (used only in pregroup) --------
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


# -------- sklearn compatibility helper --------
def _agglom_precomputed(distance_threshold, linkage="average"):
    """Handle metric/affinity API change across sklearn versions."""
    try:
        return AgglomerativeClustering(
            n_clusters=None,
            distance_threshold=distance_threshold,
            metric='precomputed',
            linkage=linkage
        )
    except TypeError:
        return AgglomerativeClustering(
            n_clusters=None,
            distance_threshold=distance_threshold,
            affinity='precomputed',
            linkage=linkage
        )


class Jplace():
    lwr_overlap = 0.1
    pd_threshold = 1.0

    def __init__(self, jplace_fh, log_interval: int = 1000):
        self._workdir = None
        self._resume = False
        self._log_interval = max(1, int(log_interval))
        self.phylogroups = []
        self.sv_groups = []
        self.jplace = {}
        self._sv_dist_cache = {}
        self._sv_to_group_centroid_cache = {}
        # disk-backed SV distance cache (lazy-opened if self._workdir is set)
        self._sv_dist_db_path = None
        self._sv_dist_db_conn = None
        self._sv_dist_db_writes = 0

        with Timer("Load JPLACE JSON"):
            self.jplace = json.load(jplace_fh)
        for key in ('fields', 'placements', 'tree'):
            if key not in self.jplace:
                logging.error("Missing required '{}' in jplace.".format(key))
                raise ValueError("Missing '{}' in jplace".format(key))

        with Timer("Index JPLACE fields"):
            try:
                self.edge_idx = self.jplace['fields'].index('edge_num')
                self.lwr_idx = self.jplace['fields'].index('like_weight_ratio')
                self.dl_idx = self.jplace['fields'].index('distal_length')
            except ValueError:
                logging.error("Missing a needed field (edge_num, like_weight_ratio, distal_length)")
                raise

        with Timer("Normalize & load tree"):
            self.__load_tree__()
        with Timer("Load placements & build sv_nodes"):
            self.__load_placements__()

        logging.info("SV count: {} | fields: {}".format(len(self.sv_nodes), self.jplace['fields']))
        _log_mem("after load")

    def __load_tree__(self):
        re_SEPP_tree = re.compile(r'(|\w+):(?P<edgelen>(\d+\.\d+)(|e[-+]\d+))\[(?P<edgeid>\d+)\]')

        def normalized_edges(m):
            return "{{{}}}:{}[{}]".format(m.group('edgeid'), m.group('edgelen'), m.group('edgeid'))

        tree_norm = re_SEPP_tree.sub(normalized_edges, self.jplace['tree'])

        tp = Phylo.read(StringIO(tree_norm), 'newick')
        with StringIO() as th:
            Phylo.write(tp, th, 'newick')
            th.seek(0)
            self.tree = TreeNode.read(th)

        self.name_node = {
            int(n.name.replace('{', "").replace('}', '')): n
            for n in self.tree.traverse() if n.name is not None
        }
        self.node_name = {v: k for k, v in self.name_node.items()}
        logging.debug("Tree loaded: {} named edges".format(len(self.name_node)))

    def __load_placements__(self):
        self.sv_nodes = {}
        placements = self.jplace['placements']
        for idx, pl in enumerate(placements):
            pl_nodes = {p[self.edge_idx]: p for p in pl['p']}
            if 'nm' in pl:
                for sv, _w in pl['nm']:
                    self.sv_nodes[sv] = pl_nodes
            if 'n' in pl:
                for sv in pl['n']:
                    self.sv_nodes[sv] = pl_nodes
            log_every(idx, len(placements), self._log_interval, "loading placements")
        logging.debug("Built sv_nodes for {} SVs".format(len(self.sv_nodes)))

    # ---------------------- distance helpers & caches ----------------------

    def __sv_profile__(self, sv, no_dl: bool):
        if no_dl:
            return {nid: (npl[self.lwr_idx], 0.0) for nid, npl in self.sv_nodes[sv].items()}
        else:
            return {nid: (npl[self.lwr_idx], npl[self.dl_idx]) for nid, npl in self.sv_nodes[sv].items()}

    def sv_pair_distance(self, sv1, sv2, no_dl: bool = False) -> float:
        key = (sv1, sv2, 1 if no_dl else 0) if sv1 <= sv2 else (sv2, sv1, 1 if no_dl else 0)
        # 1) in-memory cache
        if key in self._sv_dist_cache:
            return self._sv_dist_cache[key]

        # 2) on-disk cache (SQLite), if workdir is set
        no_dl_int = 1 if no_dl else 0
        k0, k1 = (sv1, sv2) if sv1 <= sv2 else (sv2, sv1)
        d_from_db = self._dist_db_get(k0, k1, no_dl_int)
        if d_from_db is not None:
            self._sv_dist_cache[key] = d_from_db
            return d_from_db

        # 3) compute if not cached
        sv1_p = self.__sv_profile__(sv1, no_dl)
        sv2_p = self.__sv_profile__(sv2, no_dl)
        sv1_lwr_total = sum(p[0] for p in sv1_p.values()) or 1e-12
        sv2_lwr_total = sum(p[0] for p in sv2_p.values()) or 1e-12

        paired_dist = 0.0

        overlap_nodes = set(sv1_p).intersection(sv2_p)
        if overlap_nodes:
            paired_dist += sum(
                sv1_p[n][1] * (sv1_p[n][0] / sv1_lwr_total) +
                sv2_p[n][1] * (sv2_p[n][0] / sv2_lwr_total)
                for n in overlap_nodes
            )

        distant_nodes = set(sv1_p).union(sv2_p) - overlap_nodes
        if distant_nodes:
            svp_lca = self.tree.lowest_common_ancestor(
                [self.name_node.get(nid) for nid in distant_nodes]
            )
            paired_dist += sum(
                (npv[1] + svp_lca.distance(self.name_node[nid])) * (npv[0] / sv1_lwr_total)
                for nid, npv in sv1_p.items() if nid in distant_nodes
            )
            paired_dist += sum(
                (npv[1] + svp_lca.distance(self.name_node[nid])) * (npv[0] / sv2_lwr_total)
                for nid, npv in sv2_p.items() if nid in distant_nodes
            )

        d_val = float(paired_dist)
        # Update both caches
        self._sv_dist_cache[key] = d_val
        self._dist_db_put(k0, k1, no_dl_int, d_val)
        return d_val

    def _ensure_dist_db(self):
        """
        Ensure a SQLite db is available for persistent SV-SV distance caching.
        Open lazily using self._workdir; safe to call from child processes.
        """
        workdir = getattr(self, "_workdir", None)
        if not workdir:
            return None
        try:
            os.makedirs(workdir, exist_ok=True)
        except Exception:
            pass
        if self._sv_dist_db_path is None:
            self._sv_dist_db_path = os.path.join(workdir, "sv_dist_cache.sqlite")
        if (self._sv_dist_db_conn is None):
            conn = sqlite3.connect(self._sv_dist_db_path, timeout=60.0, check_same_thread=False)
            cur = conn.cursor()
            # WAL for better concurrency; NORMAL sync for speed (acceptable for a cache)
            try:
                cur.execute("PRAGMA journal_mode=WAL;")
                cur.execute("PRAGMA synchronous=NORMAL;")
                cur.execute("PRAGMA temp_store=MEMORY;")
                cur.execute("PRAGMA cache_size=-200000;")  # ~200MB page cache, negative means KB units
            except Exception:
                pass
            cur.execute("""
                CREATE TABLE IF NOT EXISTS distances(
                    sv1 TEXT NOT NULL,
                    sv2 TEXT NOT NULL,
                    no_dl INTEGER NOT NULL,
                    d REAL NOT NULL,
                    PRIMARY KEY (sv1, sv2, no_dl)
                );
            """)
            cur.execute("CREATE INDEX IF NOT EXISTS idx_dist_no_dl ON distances(no_dl);")
            conn.commit()
            self._sv_dist_db_conn = conn
            self._sv_dist_db_writes = 0
        return self._sv_dist_db_conn

    def _dist_db_get(self, sv1, sv2, no_dl_int):
        conn = self._ensure_dist_db()
        if conn is None:
            return None
        try:
            cur = conn.cursor()
            cur.execute("SELECT d FROM distances WHERE sv1=? AND sv2=? AND no_dl=?;", (sv1, sv2, no_dl_int))
            row = cur.fetchone()
            return (None if row is None else float(row[0]))
        except Exception:
            return None

    def _dist_db_put(self, sv1, sv2, no_dl_int, d):
        conn = self._ensure_dist_db()
        if conn is None:
            return
        try:
            cur = conn.cursor()
            cur.execute(
                "INSERT OR IGNORE INTO distances(sv1, sv2, no_dl, d) VALUES(?,?,?,?);",
                (sv1, sv2, no_dl_int, float(d))
            )
            self._sv_dist_db_writes = (self._sv_dist_db_writes or 0) + 1
            # Commit in batches to reduce contention
            if (self._sv_dist_db_writes % 1000) == 0:
                conn.commit()
        except Exception:
            # best-effort cache; ignore write errors
            pass

    def close(self):
        try:
            if self._sv_dist_db_conn is not None:
                try:
                    # final commit if pending writes
                    self._sv_dist_db_conn.commit()
                except Exception:
                    pass
                self._sv_dist_db_conn.close()
                self._sv_dist_db_conn = None
        except Exception:
            pass

    def __phylotype_labels_and_groups__(self):
        pg_size = sorted(
            [(pg_i, len(pg)) for pg_i, pg in enumerate(self.phylogroups)],
            key=lambda x: -x[1]
        )
        labels = ['pt__{:05d}'.format(i + 1) for i in range(len(pg_size))]
        groups = [sorted(list(self.phylogroups[idx])) for idx, _ in pg_size]  # stabilize
        sv_to_gid = {}
        for gi, members in enumerate(groups):
            for sv in members:
                sv_to_gid[sv] = gi
        return labels, groups, sv_to_gid

    # ---------------------- pregrouping & clustering ----------------------

    def __pregroup_by_LWR__(self, pd_threshold, lwr_overlap):
        sv_to_group = [
            sv for sv, svp_l in
            sorted(
                [(sv, len(svp)) for sv, svp in self.sv_nodes.items()],
                key=lambda v: v[1]
            )
        ]
        sv_groups = []
        total = len(sv_to_group)
        logging.info("Pregrouping {} SVs (LWR overlap >= {})".format(total, lwr_overlap))
        step = 0
        while len(sv_to_group) > 0:
            seed_sv = sv_to_group.pop()
            group_svs = set([seed_sv])
            group_node_ids = set(self.sv_nodes[seed_sv])

            if len(sv_to_group) == 0:
                sv_groups.append(group_svs)
                break

            empty_mat = [0] * (self.lwr_idx + 1)
            g_overlap_mat = np.array([
                [
                    self.sv_nodes[sv].get(n, empty_mat)[self.lwr_idx]
                    for n in group_node_ids
                ]
                for sv in sv_to_group
            ])
            group_mask = np.zeros(g_overlap_mat.shape[0], dtype=int)
            LWR_Filter(g_overlap_mat, group_mask, lwr_overlap)
            group_svs.update(
                {sv for (sv, m) in zip(sv_to_group, group_mask.astype(bool)) if m}
            )
            sv_to_group = [sv for sv in sv_to_group if sv not in group_svs]
            sv_groups.append(list(group_svs))
            step += len(group_svs)
            log_every(step, total, self._log_interval, "pregrouping")
        logging.info("Pregrouped into {} provisional groups".format(len(sv_groups)))
        _log_mem("after pregroup")

        logging.info("Computing LCA per provisional group")
        sv_group_lca = [
            self.tree.lowest_common_ancestor([self.name_node[nid] for sv in gsv for nid in self.sv_nodes[sv]])
            for gsv in sv_groups
        ]

        logging.info("Pairwise distance between group LCAs (for merging)")
        g = len(sv_group_lca)
        g_lca_mat = np.zeros((g, g), dtype=np.float64)
        pairs = 0
        for i in range(g):
            for j in range(i + 1, g):
                ij_pd = sv_group_lca[i].distance(sv_group_lca[j])
                g_lca_mat[i, j] = ij_pd
                g_lca_mat[j, i] = ij_pd
                pairs += 1
                log_every(pairs, g * (g - 1) // 2, self._log_interval, "LCA distance pairs")
        logging.info("Agglomerative clustering of provisional groups")
        g_lca_clusters = _agglom_precomputed(pd_threshold).fit_predict(g_lca_mat)

        logging.info("Merging provisional groups by cluster label")
        new_old_sv_groups = defaultdict(set)
        for old_cluster_idx, new_cluster_idx in enumerate(g_lca_clusters):
            new_old_sv_groups[new_cluster_idx].add(old_cluster_idx)
        new_sv_groups = [
            list(set([sv for idx in olds for sv in sv_groups[idx]]))
            for olds in new_old_sv_groups.values()
        ]
        self.sv_groups = new_sv_groups
        logging.info("After merge: {} coarse groups; max size={}".format(
            len(self.sv_groups), max(len(gx) for gx in self.sv_groups)))
        _log_mem("after coarse grouping")

    def group_features(self, lwr_overlap=0.95, pd_threshold=0.1, no_dl=False, n_jobs=1):
        if no_dl:
            logging.info("Ignoring distal length (no_dl=True).")
        self.lwr_overlap = lwr_overlap
        self.pd_threshold = pd_threshold

        self._sv_dist_cache.clear()
        self._sv_to_group_centroid_cache.clear()
        # checkpoint paths
        import csv
        workdir = getattr(self, '_workdir', None)
        pregroups_path = os.path.join(workdir, 'pregroups.csv') if workdir else None
        partial_path = os.path.join(workdir, 'phylogroups_partial.csv') if workdir else None
        marker_path = os.path.join(workdir, '_last_completed_group.txt') if workdir else None
        if workdir:
            os.makedirs(workdir, exist_ok=True)

        with Timer("Pregroup by LWR overlap"):
            self.__pregroup_by_LWR__(pd_threshold, lwr_overlap)
        # write pregroups checkpoint
        if workdir:
            if not (getattr(self, '_resume', False) and os.path.exists(pregroups_path)):
                with open(pregroups_path, 'w', newline='') as fh:
                    w = csv.writer(fh)
                    w.writerow(['group_id', 'sv'])
                    for gid, g in enumerate(self.sv_groups):
                        for sv in g:
                            w.writerow([gid, sv])
                logging.info(f"Checkpoint: wrote pregroups -> {pregroups_path}")

        # --- parallel refine across coarse groups (Linux fork), else serial ---

        if n_jobs > 1:
            # Parallel, checkpointed via per-shard part files.
            if not workdir:
                logging.warning("Parallel mode without --workdir disables checkpointing; proceeding.")
            from multiprocessing import Process
            total_groups = len(self.sv_groups)

            # Partition groups deterministically across shards
            shards = max(1, int(n_jobs))
            shard_specs = []
            for sid in range(shards):
                shard_indices = [i for i in range(total_groups) if (i % shards) == sid]
                shard_specs.append((sid, shard_indices))

            procs = []
            for (sid, idxs) in shard_specs:
                # Skip empty shards
                if not idxs:
                    continue
                # If resuming and marker exists, skip this shard
                part_path = os.path.join(workdir, f"phylogroups_partial.part-{sid:02d}.csv") if workdir else None
                done_marker = os.path.join(workdir, f"phylogroups_partial.part-{sid:02d}.done") if workdir else None
                if workdir and getattr(self, '_resume', False) and os.path.exists(done_marker):
                    logging.info(f"[resume] skipping shard {sid} (done marker present)")
                    continue

                def _run_shard(jself, sid, idxs, part_path, done_marker):
                    import csv
                    # write header if new
                    if part_path and (not os.path.exists(part_path)):
                        with open(part_path, "w", newline="") as fh:
                            csv.writer(fh).writerow(["phylotype", "sv", "group_index"])
                    for g_i in idxs:
                        g_sv = jself.sv_groups[g_i]
                        if len(g_sv) == 1:
                            # single-member group
                            if part_path:
                                with open(part_path, "a", newline="") as fh:
                                    csv.writer(fh).writerow([f"pt__{g_i:05d}_001", next(iter(g_sv)), g_i])
                            continue
                        g_sv = sorted(g_sv)
                        n = len(g_sv)
                        g_sv_dist_mat = np.zeros((n, n), dtype=np.float64)
                        for i in range(n):
                            for j in range(i + 1, n):
                                d = jself.sv_pair_distance(g_sv[i], g_sv[j], no_dl=no_dl)
                                g_sv_dist_mat[i, j] = d
                                g_sv_dist_mat[j, i] = d
                        g_sv_clusters = AgglomerativeClustering(
                            n_clusters=None, distance_threshold=pd_threshold,
                            metric='precomputed', linkage='average'
                        ).fit_predict(g_sv_dist_mat)
                        g_phylotype_svs = defaultdict(set)
                        for sv, cl in zip(g_sv, g_sv_clusters):
                            g_phylotype_svs[cl].add(sv)
                        if part_path:
                            with open(part_path, "a", newline="") as fh:
                                w = csv.writer(fh)
                                local_counter = 0
                                for svs in g_phylotype_svs.values():
                                    local_counter += 1
                                    for sv in svs:
                                        w.writerow([f"pt__{g_i:05d}_{local_counter:03d}", sv, g_i])
                    if done_marker:
                        open(done_marker, "w").close()

                p = Process(target=_run_shard, args=(self, sid, idxs, part_path, done_marker))
                p.start()
                procs.append((sid, p))

            # Wait for shards
            for sid, p in procs:
                p.join()

            # Merge parts into in-memory phylogroups for consistency (optional) and return
            self.phylogroups = []
            if workdir:
                import csv
                part_files = sorted(
                    [f for f in os.listdir(workdir)
                     if f.startswith("phylogroups_partial.part-") and f.endswith(".csv")]
                )
                # group by group_index
                groups = {}
                for pf in part_files:
                    with open(os.path.join(workdir, pf), newline="") as fh:
                        r = csv.DictReader(fh)
                        for row in r:
                            gi = int(row["group_index"])
                            pt = row["phylotype"]
                            sv = row["sv"]
                            if gi not in groups:
                                groups[gi] = {}
                            groups[gi].setdefault(pt, set()).add(sv)
                # flatten by group index order
                for gi in sorted(groups.keys()):
                    for pt, svs in groups[gi].items():
                        self.phylogroups.append(set(svs))
            return self.phylogroups

        # serial refine
        with Timer("Refine groups by within-group clustering"):
            total_groups = len(self.sv_groups)
            # resume support
            start_idx = 0
            if workdir and getattr(self, '_resume', False) and marker_path and os.path.exists(marker_path):
                try:
                    start_idx = int(open(marker_path).read().strip()) + 1
                    logging.info(f"Resuming from group index {start_idx}")
                except Exception:
                    start_idx = 0
            # ensure partial header
            if workdir and partial_path and not os.path.exists(partial_path):
                with open(partial_path, 'w', newline='') as _fh:
                    csv.writer(_fh).writerow(["phylotype", "sv"])
            for g_i in range(start_idx, total_groups):
                g_sv = self.sv_groups[g_i]
                log_every(g_i, total_groups, max(1, self._log_interval // 10), "refining groups")
                if len(g_sv) == 1:
                    self.phylogroups.append(set(g_sv))
                    # stream checkpoint
                    if workdir and partial_path:
                        with open(partial_path, 'a', newline='') as _fh:
                            csv.writer(_fh).writerow([f"pt__{g_i:05d}_001", next(iter(g_sv))])
                    if workdir and marker_path and ((g_i + 1) % 100 == 0):
                        open(marker_path, 'w').write(str(g_i))
                    continue
                g_sv = sorted(g_sv)
                n = len(g_sv)
                g_sv_dist_mat = np.zeros((n, n), dtype=np.float64)
                for i in range(n):
                    for j in range(i + 1, n):
                        d = self.sv_pair_distance(g_sv[i], g_sv[j], no_dl=no_dl)
                        g_sv_dist_mat[i, j] = d
                        g_sv_dist_mat[j, i] = d
                    log_every(i, n, max(1, self._log_interval // 10), f"pairwise PD (group {g_i})")
                g_sv_clusters = AgglomerativeClustering(
                    n_clusters=None, distance_threshold=pd_threshold,
                    metric='precomputed', linkage='average'
                ).fit_predict(g_sv_dist_mat)
                g_phylotype_svs = defaultdict(set)
                for sv, cl in zip(g_sv, g_sv_clusters):
                    g_phylotype_svs[cl].add(sv)
                self.phylogroups += [svs for svs in g_phylotype_svs.values()]
                # stream checkpoint
                if workdir and partial_path:
                    with open(partial_path, 'a', newline='') as _fh:
                        w = csv.writer(_fh)
                        local_counter = 0
                        for svs in g_phylotype_svs.values():
                            local_counter += 1
                            for sv in svs:
                                w.writerow([f"pt__{g_i:05d}_{local_counter:03d}", sv])
                if workdir and marker_path and ((g_i + 1) % 100 == 0):
                    open(marker_path, 'w').write(str(g_i))
        logging.info("Final phylotypes: {} (max size={})".format(
            len(self.phylogroups), max(len(p) for p in self.phylogroups)))
        _log_mem("after refine")
        return self.phylogroups

    # ---------------------- outputs: phylotypes long ----------------------

    def to_long(self):
        pg_size = sorted(
            [(pg_i, len(pg)) for pg_i, pg in enumerate(self.phylogroups)],
            key=lambda pgc: -1 * pgc[1]
        )
        pg_sv_long = [
            ('pt__{:05d}'.format(pg_i + 1), sv)
            for pg_i, (pg_idx, _pg_size) in enumerate(pg_size)
            for sv in self.phylogroups[pg_idx]
        ]
        return pg_sv_long

    def to_csv(self, out_h):
        pg_long = self.to_long()
        writer = csv.writer(out_h)
        writer.writerow(['phylotype', 'sv'])
        writer.writerows(pg_long)


def main():
    ap = argparse.ArgumentParser(
        description="""Given a JPLACE file of placed features on a phylogenetic tree,
        generate phylotypes with checkpointing and verbose logging."""
    )
    ap.add_argument('--jplace', '-J', type=argparse.FileType('r'), required=True,
                    help='JPLACE file, as created by pplacer or epa-ng')
    ap.add_argument('--out', '-O', type=argparse.FileType('wt'), required=True,
                    help='Output CSV (long format): phylotype,sv')
    ap.add_argument('--lwr-overlap', '-L', default=0.01, type=float,
                    help='Minimum like-weight ratio overlap to pregroup SVs (default: 0.01)')
    ap.add_argument('--threshold_pd', '-T', default=0.1, type=float,
                    help='Phylogenetic distance threshold for clustering (default: 0.1)')
    ap.add_argument('--no-distal-length', '-ndl', action='store_true',
                    help='Ignore distal length to nodes')
    ap.add_argument('--workdir', '-W', default=None,
                    help='Directory to store checkpoints (pregroups.csv, phylogroups_partial.csv)')
    ap.add_argument('--resume', action='store_true',
                    help='Resume from checkpoints in --workdir if available')

    # Logging controls
    ap.add_argument('--log-file', '-LF', type=str,
                    help='Write verbose DEBUG logs to this file')
    ap.add_argument('--log-level', '-LL', type=str, default='INFO',
                    help='Console log level: DEBUG, INFO, WARNING, ERROR (default: INFO)')
    ap.add_argument('--log-interval', '-LI', type=int, default=1000,
                    help='Log progress every N items/steps inside heavy loops (default: 1000)')

    # Parallelism (Linux fork)
    ap.add_argument('--n-jobs', '-NJ', type=int, default=1,
                    help='Parallel workers for the refine stage (Linux fork only). Default: 1')

    args = ap.parse_args()

    # Setup logging
    setup_logging(args.log_file, args.log_level)

    # Run
    jplace = Jplace(args.jplace, log_interval=args.log_interval)
    # checkpoint config
    jplace._workdir = args.workdir
    jplace._resume = bool(args.resume)
    jplace.group_features(args.lwr_overlap, args.threshold_pd,
                          no_dl=args.no_distal_length, n_jobs=args.n_jobs)

    logging.info("Outputting phylotypes long CSV")
    jplace.to_csv(args.out)
    _flush_close_and_report(args.out, "phylotypes CSV")

    # Ensure persistent cache is flushed/closed
    try:
        jplace.close()
    except Exception:
        pass
    logging.info("DONE!")


if __name__ == "__main__":
    main()