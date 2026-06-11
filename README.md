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
