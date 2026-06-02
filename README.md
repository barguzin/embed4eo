# embed4eo

Weakly supervised downscaling experiments for Accra. The current pipeline uses
fine-grid Earth observation embeddings and optional World Settlement Footprint
(WSF) features to allocate coarse GHSL built-up surface totals onto a finer grid.

The neural baselines are trained with coarse GHSL cell totals only. ESA
WorldCover 2020 is used as the default out-of-pipeline categorical validation
layer. GHSL 10 m remains available as an optional GHSL-family proxy comparison,
but it is no longer the primary validation target.

## Data Sources

- **AEF / Google Earth observation embeddings**: downloaded from Source
  Cooperative and cropped to the Accra AOI. These are the main fine-grid
  predictor features.
- **GHSL BUILT-S coarse training target**: a 2019 1 km built-surface raster
  derived in `scripts/00_fit_GHSL_2019.R` by fitting a per-pixel linear trend to
  100 m GHSL epochs from 1975-2020, predicting 2019, then aggregating to 1 km.
- **WSF 2019**: used as a binary settlement support mask, as derived local
  density/distance features, and as the WSF-uniform baseline.
- **ESA WorldCover 2020**: the default final validation layer. Class `50` is
  treated as built-up; evaluation is categorical/binary agreement and spatial
  plausibility assessment, not continuous built-surface error validation.
- **GHSL 10 m reference**: cropped separately for optional fine-scale comparison.
  In this project this reference is the separate GHSL 2018 10 m product/pipeline,
  so report it as a GHSL-family proxy rather than literal same-year ground truth.

## Environment

Most scripts assume the local geospatial Python/R environment used during
development. On this machine that environment is named `diss`, for example:

```bash
mamba run -n diss python scripts/09_evaluate_against_ghsl10m.py --help
```

If the environment is already activated, the commands below can be run with
plain `python`.

## Pipeline Overview

1. Download/prep AEF embeddings, WSF, and GHSL layers.
2. Reduce the embedding raster with PCA.
3. Create a fine-grid raster of coarse GHSL cell IDs and a lookup table of
   coarse-cell targets.
4. Run baselines:
   - WSF-uniform mass allocation.
   - embeddings-only neural allocator.
   - embeddings + WSF neural allocator.
5. Create visual diagnostics.
6. Evaluate predictions quantitatively against ESA WorldCover 2020.

## Data Download And Preprocessing

### AEF Embeddings

Download AEF tiles intersecting the Accra bounding box:

```bash
bash scripts/00_run_accra_aef.sh --outdir ~/data/aef_accra_2019 --years 2019
```

The download script filters the Source Cooperative AEF annual index to the Accra
AOI and can download both VRT and TIFF assets.

### WSF 2019

Download the WSF 2019 tiles used by the preprocessing script:

```bash
bash scripts/00_get_WSF.sh
```

### ESA WorldCover 2020

Download the ESA WorldCover tile intersecting the Accra AOI:

```bash
bash scripts/00_get_ESA.sh
```

This saves the raw 2020 WorldCover tile under:

```text
~/data/ESA_cover_2020/
```

### GHSL 2019 Coarse Training Layer

Build the trend-derived 2019 GHSL BUILT-S layer:

```bash
Rscript scripts/00_fit_GHSL_2019.R
```

This writes:

- `~/data/GHSL_BUILD/GHS_BUILT_S_E2019_R2023A_54009_100_trend_100m.tif`
- `~/data/GHSL_BUILD/GHS_BUILT_S_E2019_R2023A_54009_1000_sum_trend.tif`

### Crop And Align Layers

Once the source data and AOI vector are available, crop and align the layers to
the embedding grid:

```bash
Rscript scripts/01_crop_prep_layers.R
```

Main outputs:

- `~/data/aef_accra_2019/mosaic_accra_2019.tiff`
- `~/data/WSF_Data/cropped_wsf.tif`
- `~/data/WSF_Data/cropped_wsf_features.tif`
- `~/data/ESA_cover_2020/cropped_esa_worldcover_2020.tif`
- `~/data/ESA_cover_2020/cropped_esa_built_2020.tif`
- `~/data/GHSL_BUILD/cropped_ghsl.tif`
- `~/data/GHSL_BUILD/cropped_ghsl_raw_10m.tif`

## PCA On Embeddings

Fit IncrementalPCA on sampled valid embedding pixels and write an 8-band PCA
raster:

```bash
mamba run -n diss python scripts/02_pca_embeddings.py \
  --input ~/data/aef_accra_2019/mosaic_accra_2019.tiff \
  --output ~/data/aef_accra_2019/mosaic_accra_2019_pca8.tif \
  --model ~/data/aef_accra_2019/mosaic_accra_2019_pca8.joblib \
  --report ~/data/aef_accra_2019/mosaic_accra_2019_pca8_report.json \
  --n-components 8 \
  --sample-pixels 250000 \
  --window-size 512 \
  --batch-size 20000
```

## Fine-To-Coarse GHSL Cell IDs

Create a fine-grid cell-ID raster and a lookup table of GHSL coarse-cell totals.
The neural models use `ghsl_value_adj`, which adjusts coarse-cell mass by the AOI
overlap fraction.

```bash
mamba run -n diss python scripts/03_make_cell_ids.py \
  --coarse ~/data/GHSL_BUILD/cropped_ghsl.tif \
  --template ~/data/aef_accra_2019/mosaic_accra_2019_pca8.tif \
  --aoi ~/data/aux/bbox_accra_dissolve.gpkg \
  --out-raster ~/data/GHSL_BUILD/cropped_ghsl_cell_ids.tif \
  --out-lookup ~/data/GHSL_BUILD/cropped_ghsl_cell_lookup.csv
```

## Baseline 0: WSF-Uniform

This mass-preserving baseline allocates each coarse GHSL value uniformly across
WSF-positive fine pixels inside that coarse cell. If a cell has no WSF-positive
fine pixels, it falls back to uniform allocation across valid fine pixels.

```bash
mamba run -n diss python scripts/04_baseline_wsf_uniform.py \
  --wsf ~/data/WSF_Data/cropped_wsf.tif \
  --cell-ids ~/data/GHSL_BUILD/cropped_ghsl_cell_ids.tif \
  --lookup ~/data/GHSL_BUILD/cropped_ghsl_cell_lookup.csv \
  --value-column ghsl_value_adj \
  --output ~/data/outputs/wsf_uniform_baseline.tif \
  --report ~/data/outputs/wsf_uniform_baseline_report.json \
  --fallback ~/data/outputs/wsf_uniform_baseline_fallback_cells.tif
```

## Baseline 1: Embeddings-Only Neural Allocator

This model uses only PCA-reduced embedding bands. It predicts a positive fine
surface, aggregates predictions back to GHSL coarse cells during training, and
renormalizes each cell after inference to preserve coarse mass exactly.

```bash
mamba run -n diss python scripts/06_train_embed_only.py \
  --pca ~/data/aef_accra_2019/mosaic_accra_2019_pca8.tif \
  --cell-ids ~/data/GHSL_BUILD/cropped_ghsl_cell_ids.tif \
  --lookup ~/data/GHSL_BUILD/cropped_ghsl_cell_lookup.csv \
  --value-column ghsl_value_adj \
  --pred-out ~/data/outputs/embed_only_raw.tif \
  --pred-norm-out ~/data/outputs/embed_only_norm.tif \
  --report ~/data/outputs/embed_only_report.json \
  --loss-plot ~/data/outputs/embed_only_loss.png \
  --model-out ~/data/outputs/embed_only_model.pt \
  --epochs 500 \
  --lr 1e-3 \
  --tv-weight 1e-5 \
  --hidden 32 \
  --depth 4
```

## Baseline 2: Embeddings + WSF Neural Allocator

This model concatenates the embedding PCA bands with WSF-derived features:
binary WSF support, local WSF density at 5x5 and 11x11 windows, and distance to
nearest WSF-built pixel. The loss adds a penalty for predicted mass outside WSF
support.

```bash
mamba run -n diss python scripts/07_train_embed_wsf.py \
  --pca ~/data/aef_accra_2019/mosaic_accra_2019_pca8.tif \
  --wsf-features ~/data/WSF_Data/cropped_wsf_features.tif \
  --cell-ids ~/data/GHSL_BUILD/cropped_ghsl_cell_ids.tif \
  --lookup ~/data/GHSL_BUILD/cropped_ghsl_cell_lookup.csv \
  --value-column ghsl_value_adj \
  --pred-out ~/data/outputs/embed_wsf_raw.tif \
  --pred-norm-out ~/data/outputs/embed_wsf_norm.tif \
  --report ~/data/outputs/embed_wsf_report.json \
  --loss-plot ~/data/outputs/embed_wsf_loss.png \
  --model-out ~/data/outputs/embed_wsf_model.pt \
  --epochs 500 \
  --lr 1e-3 \
  --tv-weight 5e-5 \
  --wsf-weight 0.1 \
  --hidden 32 \
  --depth 4 \
  --wsf-band 1
```

## Visual Diagnostics

Create a four-panel plot for the WSF-uniform baseline:

```bash
mamba run -n diss python scripts/05_plot_four_panel.py \
  --ghsl ~/data/GHSL_BUILD/cropped_ghsl.tif \
  --wsf ~/data/WSF_Data/cropped_wsf.tif \
  --baseline ~/data/outputs/wsf_uniform_baseline.tif \
  --fallback ~/data/outputs/wsf_uniform_baseline_fallback_cells.tif \
  --output ~/data/outputs/wsf_uniform_four_panel.png \
  --agg-factor 10 \
  --title "WSF-uniform baseline"
```

Compare all baselines against the external GHSL 10 m reference visually:

```bash
mamba run -n diss python scripts/08_compare_baselines.py \
  --baseline0 ~/data/outputs/wsf_uniform_baseline.tif \
  --baseline1 ~/data/outputs/embed_only_norm.tif \
  --baseline2 ~/data/outputs/embed_wsf_norm.tif \
  --ghsl10m ~/data/GHSL_BUILD/cropped_ghsl_raw_10m.tif \
  --output ~/data/outputs/baseline_comparison.png \
  --agg-factor 10 \
  --title "Baseline comparison with external GHSL 10 m reference" \
  --ghsl ~/data/GHSL_BUILD/cropped_ghsl.tif \
  --wsf ~/data/WSF_Data/cropped_wsf.tif \
  --context-output ~/data/outputs/baseline_context.png
```

## Quantitative Evaluation

Run categorical evaluation against ESA WorldCover 2020:

```bash
mamba run -n diss python scripts/09_evaluate_against_esa_worldcover.py \
  --predictions ~/data/outputs/wsf_uniform_baseline.tif \
                ~/data/outputs/embed_only_norm.tif \
                ~/data/outputs/embed_wsf_norm.tif \
  --names wsf_uniform embed_only embed_wsf \
  --esa-worldcover ~/data/ESA_cover_2020/cropped_esa_worldcover_2020.tif \
  --wsf ~/data/WSF_Data/cropped_wsf.tif \
  --cell-ids ~/data/GHSL_BUILD/cropped_ghsl_cell_ids.tif \
  --built-class 50 \
  --output-csv ~/data/outputs/evaluation_esa_metrics.csv \
  --output-json ~/data/outputs/evaluation_esa_metrics.json \
  --output-fig ~/data/outputs/evaluation_esa_summary.png \
  --output-map-dir ~/data/outputs/evaluation_esa_maps
```

ESA WorldCover is used as an out-of-pipeline categorical land-cover proxy, not
as continuous built-up-surface ground truth. Because ESA WorldCover provides
class labels rather than built-surface area, the evaluation reports categorical
agreement, mass concentration within ESA built-up cells, top-k overlap, hard
non-built allocation flags, and class-conditioned diagnostics. These metrics
should be interpreted as spatial plausibility diagnostics rather than same-unit
error estimates.

The ESA evaluator also reports absolute predicted mass in hard non-built
classes, class-specific leakage into water, wetland, mangroves, bare/sparse
vegetation, cropland, and grassland, top-k lift over ESA built-up prevalence,
and top-k predicted-mass overlap with ESA built-up. If `--output-map-dir` is
provided, it writes per-model rasters showing predicted mass placed in hard
non-built classes and in water/wetland/mangrove classes.

For top-k diagnostics, `esa_built_prevalence` gives the background share of
valid pixels classified as ESA built-up, while `topk_esa_built_lift` reports
`P(ESA built-up | top-k predicted) / P(ESA built-up)`. Leakage by land-cover
class is available in the `hard_nonbuilt_by_class` metric group.

The legacy GHSL 10 m evaluator remains available for optional proxy comparison:

```bash
mamba run -n diss python scripts/09_evaluate_against_ghsl10m.py \
  --predictions ~/data/outputs/wsf_uniform_baseline.tif \
                ~/data/outputs/embed_only_norm.tif \
                ~/data/outputs/embed_wsf_norm.tif \
  --names wsf_uniform embed_only embed_wsf \
  --reference ~/data/GHSL_BUILD/cropped_ghsl_raw_10m.tif \
  --wsf ~/data/WSF_Data/cropped_wsf.tif \
  --cell-ids ~/data/GHSL_BUILD/cropped_ghsl_cell_ids.tif \
  --output-csv ~/data/outputs/evaluation_metrics.csv \
  --output-json ~/data/outputs/evaluation_metrics.json \
  --output-fig ~/data/outputs/evaluation_summary.png
```
