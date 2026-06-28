# InvasiveLens

**Classifying invasive vs. native plant species in Portugal from a single photograph.**

Ajimoti Rofiat Mobolaji (129556) · Instituto Superior de Agronomia, ULisboa · MSc Green Data Science 2025–2026 · Practical Machine Learning

InvasiveLens takes a plant photo, identifies the species with a fine-tuned deep learning model, and
turns the prediction into a **risk level and a recommended action** — and it abstains when it isn't
confident enough to decide. The whole project runs from **one notebook** that downloads its own data,
trains and evaluates the models, and prints every result.

---

## Quick start (Google Colab)

1. Open `invasivelens_analysis.ipynb` in Colab.
2. `Runtime` → `Change runtime type` → **T4 GPU** → Save (recommended — much faster, uses the full dataset).
3. `Runtime` → **Run all**.

That's it. The notebook downloads the native-species images (a shared Google Drive zip) and the
invasive-species images (from GBIF) by itself — **no manual setup, file uploads, or Drive mounting.**

---

## What it does

- Downloads and organises the dataset automatically.
- Trains three models: a **from-scratch CNN baseline**, **EfficientNetV2-S**, and **ResNet50**
  (the last two fine-tuned from ImageNet).
- Evaluates them with **stratified k-fold cross-validation** (accuracy + macro-F1).
- Collapses predictions into the task that matters: **invasive vs. native**.
- Confirms the best model with **McNemar's test**.
- Explains predictions with **Grad-CAM** heatmaps.
- Runs a **decision engine** that maps confidence → risk level → action.

---

## Species

| Species | Status | Source |
|---|---|---|
| *Acacia dealbata* | invasive | GBIF |
| *Arundo donax* | invasive | GBIF |
| *Cortaderia selloana* | invasive | GBIF |
| *Phragmites australis* | native look-alike | curated set + GBIF |
| *Ammophila arenaria* | native look-alike | curated set |

Invasive images are pulled from **GBIF** (iNaturalist + Pl@ntNet observations in Portugal),
~300 per species; the native look-alikes come from a curated image set. ~1,600 images in total.

---

## Results

Best model: **ResNet50** (fine-tuned).

| Model | Accuracy | Macro-F1 |
|---|---|---|
| Baseline CNN | 0.43 | 0.41 |
| EfficientNetV2-S | 0.63 | 0.62 |
| **ResNet50** | **0.68** | **0.67** |

- **Invasive vs. native:** accuracy **0.79**, invasive-class F1 **0.83** (recall 0.85 — it rarely misses an invasive plant).
- ResNet50 beats EfficientNet significantly (McNemar's test, p ≈ 0.03).
- Easiest species: *Acacia dealbata* (0.95 recall). Hardest: *Phragmites* ↔ *Arundo* — the tall reed-like look-alikes.

> These figures are from a **CPU run** (120 images/species). A **GPU run** uses the full dataset,
> 5 folds and 224 px images, and gives stronger, more stable numbers.

---

## Decision engine

| Prediction | Risk level | Action |
|---|---|---|
| Invasive, confidence ≥ 90% | CRITICAL | Immediate removal |
| Invasive, confidence 70–90% | HIGH | Containment within 30 days |
| Confidence < 70% | MEDIUM | Hold back — schedule field verification |
| Native | NATIVE | No action |

The confidence threshold is deliberate: when the model is unsure, it asks for a human check instead
of making a high-stakes guess.

---

## Repository contents

```
invasivelens_analysis.ipynb   # the main deliverable — runs the whole project end to end
REPORT.md / REPORT.pdf        # written project report
src/                          # supporting library: CLI, REST API, training, data utilities
config.py                     # species pairs and settings
```

---

## Method notes

- **Transfer learning** is used because a few hundred images per class is too little to train a deep
  network from scratch; ImageNet features transfer well to plant photos.
- **Cross-validation is stratified, not geographic.** Geographic splitting would need GPS
  coordinates, which the GBIF images don't carry. The notebook switches to geographic grouping
  automatically if coordinates are ever added.
- Training uses class-weighted cross-entropy, light augmentation, the Adam optimiser, and a fixed
  random seed for repeatability. Full settings are in the report's Appendix.

---

## Limitations

- Small dataset, five species only.
- GBIF labels are citizen-science observations and can contain mistakes.
- Not yet tested on real field photos across seasons and conditions.
- The reported numbers are from a reduced CPU run; a GPU run is recommended for final figures.

---

## Credits

Training images from **GBIF** (iNaturalist, Pl@ntNet); pre-trained weights from **torchvision**
(ImageNet). Built with PyTorch and scikit-learn. See `REPORT.md` for full references.
