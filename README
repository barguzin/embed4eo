# Setting up and running

## Downloading Data 

ssdfsdfsd

## Preprocessing

### Cropping the layers 

Once the data is downloaded you have the study_area vector for the bounding box, you can run *01_crop_prep_layers.R*. 

### PCA on embeddings 

Run the PCA analysis on embeddings by calling the corresponding Python script

```bash
python 02_pca_embeddings.py \
  --input ~/data/aef_accra_2019/mosaic_accra_2019.tiff \
  --output ~/data/aef_accra_2019/mosaic_accra_2019_pca8.tif \
  --model ~/data/aef_accra_2019/mosaic_accra_2019_pca8.joblib \
  --report ~/data/aef_accra_2019/mosaic_accra_2019_pca8_report.json \
  --n-components 8 \
  --sample-pixels 250000 \
  --window-size 512 \
  --batch-size 20000
```

### Build the fine-to-coarse GHSL cell-ID raster

We will need per-cell GHSL totals for allocation tasks. To generate these, simply run: 

```bash
python 03_make_cell_ids.py \
  --coarse ~/data/GHSL_BUILD/cropped_ghsl.tif \
  --template ~/data/aef_accra_2019/mosaic_accra_2019_pca8.tif \
  --out-raster ~/data/GHSL_BUILD/cropped_ghsl_cell_ids.tif \
  --out-lookup ~/data/GHSL_BUILD/cropped_ghsl_cell_lookup.csv
```

## Baseline 0: binary WSF

```bash
python 04_baseline_wsf_uniform.py \
  --wsf ~/data/WSF_Data/cropped_wsf.tif \
  --cell-ids ~/data/GHSL_BUILD/cropped_ghsl_cell_ids.tif \
  --lookup ~/data/GHSL_BUILD/cropped_ghsl_cell_lookup.csv \
  --output ~/data/outputs/wsf_uniform_baseline.tif \
  --report ~/data/outputs/wsf_uniform_baseline_report.json \
  --fallback ~/data/outputs/wsf_uniform_baseline_fallback_cells.tif \
```

## Visual Diagnostics - 1

asdasdas

```bash
python 05_plot_four_panel.py \
  --ghsl ~/data/GHSL_BUILD/cropped_ghsl.tif \
  --wsf ~/data/WSF_Data/cropped_wsf.tif \
  --baseline ~/data/outputs/wsf_uniform_baseline.tif \
  --fallback ~/data/outputs/wsf_uniform_baseline_fallback_cells.tif \
  --output ~/data/outputs/wsf_uniform_four_panel.png \
  --agg-factor 10 \
  --title "WSF-uniform baseline"
```

## Baseline 1: Embeddings and No WSF

asdasdas

```bash
python 06_train_embed_only.py \
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

## Baseline 2: Embeddings + WSF

sasdfasd

```bash
python 07_train_embed_wsf.py \
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

## Visualize All Baselines 

asdas

```bash
python 08_compare_baselines.py \
  --baseline0 ~/data/outputs/wsf_uniform_baseline.tif \
  --baseline1 ~/data/outputs/embed_only_norm.tif \
  --baseline2 ~/data/outputs/embed_wsf_norm.tif \
  --ghsl10m ~/data/GHSL_BUILD/cropped_ghsl_raw_10m.tif \
  --output ~/data/outputs/baseline_comparison.png \
  --agg-factor 10 \
  --title "Baseline comparison with GHSL 10 m reference" \
  --ghsl ~/data/GHSL_BUILD/cropped_ghsl.tif \
  --wsf ~/data/WSF_Data/cropped_wsf.tif \
  --context-output ~/data/outputs/baseline_context.png
```