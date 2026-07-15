# Phylotypes

Group features (e.g. 16s rRNA gene variable region sequence variants) placed on a common phylogenetic tree by phylogenetic distance.

This is particularly useful when attempting a metanalysis of 16s rRNA gene variable region amplicons, in which the amplicons were made with different primers, etc.

### Input:
`--jplace <file>` A jplace file (deduplicated) as made by [pplacer](https://matsen.fhcrc.org/pplacer/) or [epa-ng](https://github.com/Pbdas/epa-ng).

### Output:
`--out <file>` Path where the output should be placed in `CSV` format. This will be a header-containing CSV file in long-format with two columns: phylogroup and feature-id.

### Parameters:
`--lwr-overlap <0.0 - 1.0>` Minimum like-weight-ratio overlap for two placed features (e.g. sequence variants) to be pre-grouped. Default of 0.1 should be fine for most use cases.

`--threshold_pd <float>` Phylogenetic distance at which to cluster features. Default of 1.0 corresponds roughly to somewhere between species-level and genus-level grouping for 16s rRNA V4 region amplicons.

`--distance {legacy,kr}` Pairwise distance metric. `legacy` (default) is an exact, vectorized reimplementation of the original LCA-weighted-average metric and reproduces prior phylotypes. `kr` is the true tree-Wasserstein (Kantorovich-Rubinstein) distance, a proper metric -- opt-in.

**Note:** `--threshold_pd`'s default of `1.0` is calibrated for `--distance legacy`. The `kr` metric is on a different (smaller-valued) scale, so the same threshold means a different cut. If you use `--distance kr`, re-calibrate `--threshold_pd` on your own data; the legacy default will over- or under-split. (No default `kr` threshold is provided yet -- this is a TODO pending empirical calibration on a real dataset.)

`--no-distal-length` Ignore distal length to nodes when computing distances. Default is `False` (distal length is included).

`--incremental` Use an incremental seed→apply→expand→reconcile clustering path instead of the default batch path. Scales to larger inputs with lower peak memory at the cost of approximate (single/centroid-like) linkage near phylotype boundaries.

`--seed-size <int>` Number of (most specific) placements used to seed the initial phylotype pool when `--incremental` is set. Default: `200`.

`--expand-batch-size <int>` Maximum number of orphaned placements clustered together per expand pass when `--incremental` is set. Default: `200`.

`--device <str>` PyTorch device for tensor computations, e.g. `cpu` or `cuda`. Default: `cpu`.

`--max-pregroup-size <int>` Maximum SVs in a single pregroup after LCA-based re-clustering; larger pregroups are split back to their pre-merge groups to bound memory use. Default: `5000`.

---

## `add_phylotypes`

Assign a new set of placed sequence variants into an existing set of phylotypes.

Given a previous JPLACE file and its phylotype assignments (produced by `phylotypes`), and a new JPLACE of sequence variants placed on the **same** reference tree, each new SV is assigned to the nearest existing phylotype.

### Input:
`--previous_jp / -P <file>` Previous JPLACE file (same reference tree as `--new_jp`).

`--previous_phylotypes / -p <file>` CSV file with two columns (`phylotype`, `sv`) representing the existing phylotype assignments.

`--new_jp / -N <file>` New JPLACE file containing the sequence variants to be assigned.

### Output:
`--out / -O <file>` Output CSV (`phylotype`, `sv`) placing the new SVs into the existing phylotypes. SVs that share no tree-edge overlap with any existing phylotype are reported as orphans in the log and omitted from the output.

### Parameters:
`--no-distal-length` Ignore distal length when computing distances. Default: `False`.

`--device <str>` PyTorch device. Default: `cpu`.

`--distance {legacy,kr}` Pairwise distance metric. Default: `legacy`.

---

## `phylotype_taxonomy`

Assign a consensus species (or best-effort taxon) to each phylotype using per-SV taxonomic annotations.

### Input:
`--phylotypes / -p <file>` CSV with `phylotype` and `sv` columns (output of `phylotypes`).

`--taxonomy / -t <file>` Per-SV taxonomic assignments in long format (as produced by [MaLiAmPi](https://github.com/jgolob/maliampi)), with columns `sv`, `rank`, `want_rank`, and `tax_name`.

### Output:
`--out / -O <file>` CSV with `phylotype` and `species` columns. Phylotypes without a species-level assignment are annotated at the best available rank with a ` spp.` suffix.

---

## Release notes

### v2.0.1

- `add_phylotypes` now shares the same JPLACE loader and distance metric as `phylotypes`, eliminating a divergent implementation that failed on standard SEPP/epa-ng edge-numbered trees.
- New `--incremental` clustering path for large datasets.
- GPU support via `--device cuda` (or any valid PyTorch device string).
- **KR-distance threshold caveat:** the `--distance kr` metric is on a different (smaller-valued) scale than `legacy`. The default `--threshold_pd 1.0` is calibrated for `legacy` only. If using `--distance kr`, re-calibrate the threshold on your own data.
