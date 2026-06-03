# Malaria Diagnosis CNN — Full Project Execution Plan

**Due**: 12 June 2026 at 11:59 pm | **Group**: 5 members | **Total marks**: 35

---

## Context

This project builds five CNN models for binary classification of malaria cell images (Parasitised vs Uninfected) using the NIH dataset. The deliverables are five Jupyter notebooks, a ~10-page academic PDF report, and a 10–12 minute group oral defence. Notebooks run on Google Colab (via the VS Code Colab extension) for free GPU access. The dataset is downloaded reproducibly via `kagglehub` inside each notebook.

---

## Environment Notes

- **Runtime**: Local execution via VS Code with the `.venv` Python environment as the notebook kernel. Notebooks are in `.ipynb` format (Google Colab-compatible) but run locally.
- **Kernel**: Select `.venv` as the kernel in VS Code (Python: Select Interpreter → `.\.venv\Scripts\python.exe`).
- **GPU**: Use whatever GPU is available locally. Verify with `tf.config.list_physical_devices('GPU')`. If no local GPU, pretrained models (VGG16, ResNet50, MobileNetV2) will be slow — use Google Colab cloud as a fallback for those three notebooks only.
- **Persistence**: All files save to the local project directory — no Drive mounting needed.
- **Dataset**: `kagglehub` downloads to the local system cache (`~/.cache/kagglehub/` or `C:\Users\Luqma\.cache\kagglehub\`). The `archive (11).zip` is an offline fallback if Kaggle credentials aren't configured.

---

## Phase 0: One-Time Setup (3 June — 1–2 hours) ✅ COMPLETE

### Step 0.1 — Install dependencies into `.venv`

Run once in VS Code terminal (PowerShell):

```powershell
.\.venv\Scripts\Activate.ps1
pip install tensorflow numpy pandas matplotlib seaborn scikit-learn Pillow tqdm kagglehub ipykernel
```

Register the venv as a Jupyter kernel so VS Code sees it:
```powershell
python -m ipykernel install --user --name malaria_cnn --display-name "Python (malaria_cnn)"
```

### Step 0.2 — Create project folder structure

Run once in terminal (or create manually):

```powershell
mkdir notebooks, figures, checkpoints, report
```

Full structure:
```
Group14_Formative2/
├── utils.py          ← shared helpers (Step 0.4)
├── notebooks/        ← all 5 .ipynb files
├── figures/          ← all plot PNGs (P1–P5)
├── checkpoints/      ← best model .h5 weights
├── report/           ← report drafts and final PDF
└── data/             ← optional, if using zip fallback
```

### Step 0.3 — Download the dataset via kagglehub

```python
import kagglehub, os

path = kagglehub.dataset_download("iarunava/cell-images-for-detecting-malaria")
print("Downloaded to:", path)

# Dynamically find the folder containing Parasitized/ and Uninfected/
DATA_DIR = None
for root, dirs, _ in os.walk(path):
    if 'Parasitized' in dirs and 'Uninfected' in dirs:
        DATA_DIR = root
        break

print("DATA_DIR:", DATA_DIR)
# Verify counts
print(len(os.listdir(f'{DATA_DIR}/Parasitized')))  # expect 13779
print(len(os.listdir(f'{DATA_DIR}/Uninfected')))   # expect 13779
```

> `kagglehub` caches to `C:\Users\Luqma\.cache\kagglehub\` on Windows. The `os.walk` search makes `DATA_DIR` robust to any directory structure. The zip file fallback: extract `archive (11).zip` to `./data/` and set `DATA_DIR` to whichever subdirectory contains `Parasitized/` and `Uninfected/`.

### Step 0.4 — Create `utils.py` in the project root

`utils.py` lives at the project root (`Group14_Formative2/utils.py`). Every notebook imports it with:

```python
import sys, os
sys.path.insert(0, os.path.abspath('..'))   # from notebooks/ subfolder
from utils import load_dataset, evaluate_model, plot_learning_curves, \
                  plot_confusion_matrix, plot_roc_curve, build_results_table, \
                  get_callbacks, error_analysis, data_augmentation
```

`utils.py` must contain exactly the following functions (copy from spec verbatim — do not rewrite):

| Function | Purpose |
|----------|---------|
| `load_dataset(image_size, batch_size)` | Returns (train_ds, val_ds, test_ds) with 80/10/10 split |
| `data_augmentation` | Keras Sequential augmentation layer |
| `evaluate_model(model, test_ds)` | Returns dict: accuracy, precision, recall, f1, auc |
| `plot_learning_curves(history, name, save_path)` | Loss + accuracy curves per experiment |
| `plot_confusion_matrix(metrics_dict, class_names, name, save_path)` | Labelled CM with TP/TN/FP/FN annotation |
| `plot_roc_curve(metrics_dict, name, save_path)` | ROC with AUC, fills under curve |
| `build_results_table(experiments_list)` | Returns sorted DataFrame of all experiments |
| `get_callbacks(model_name, exp_num, patience_es, patience_lr)` | EarlyStopping + ReduceLROnPlateau + ModelCheckpoint |
| `error_analysis(model, test_ds, class_names, n_samples)` | Displays misclassified images grid |

**Key constants in `utils.py`** — `DATA_DIR` is passed in from the notebook (since kagglehub returns a session-specific path), not hardcoded:

```python
# utils.py — constants (DATA_DIR is passed as parameter to load_dataset)
IMAGE_SIZE  = (224, 224)
BATCH_SIZE  = 32
SEED        = 42
CLASS_NAMES = ['Parasitized', 'Uninfected']
```

Each notebook calls `load_dataset(data_dir=DATA_DIR, image_size=..., batch_size=...)` passing the dynamically resolved `DATA_DIR` from Step 0.3.

### Step 0.5 — Verify shared pipeline (all 5 members run this)

Each member must confirm they get the same split:
```
Train: ~22046 | Val: ~2756 | Test: ~2756
```

---

## Phase 1: Baseline CNN — collective build (4–5 June, ~2 hours group session)

**File**: `notebooks/baseline_cnn_P1.ipynb`  
**Owner**: P1 leads, all 5 code together

### Notebook cell structure (same for all 5 notebooks)

1. **Header cell** (Markdown): Group name, all 5 member names, model name, date
2. **Seeds & imports**: Set SEED=42 for `os`, `random`, `numpy`, `tensorflow`; verify GPU
3. **Dataset download**: kagglehub download + `os.walk` to find `DATA_DIR`; verify 13779/13779
4. **Import from utils.py**: `sys.path.insert(0, os.path.abspath('..'))` then import all shared functions
5. **Load dataset**: `load_dataset(data_dir=DATA_DIR, image_size=(64,64))` for baseline (others use 224×224)
6. **Build model**: architecture function from spec
7. **Experiment loop** (7 experiments — see table below)
8. **Final results table**: `display(build_results_table(results_log))`
9. **Best model visualisations**: confusion matrix, ROC curve, error analysis for the best F1 experiment

### Baseline CNN experiment variables

| Exp # | Change | Value |
|-------|--------|-------|
| 1 | Baseline config (starting point) | LR=1e-3, dropout=0.5, 20 epochs |
| 2 | Lower learning rate | LR=1e-4 |
| 3 | Higher dropout | dropout=0.6 |
| 4 | More filters | 64/128/256 |
| 5 | L2 regularisation | kernel_regularizer=l2(1e-4) |
| 6 | Larger batch size | batch_size=64 |
| 7 | Early stopping | patience=5 |

### Per-experiment documentation template (Markdown cell before each run)

```markdown
## Experiment [N]: [What changed]
**Hypothesis**: Why do you expect this change to help/hurt?
**Change made**: [Specific parameter or architectural change]
**Result**: Accuracy=X, Precision=X, Recall=X, F1=X, AUC=X
**Interpretation**: [Did it improve? Why? Any signs of overfitting/underfitting?]
```

---

## Phase 2: Individual Model Notebooks (5–9 June)

Each person forks `baseline_cnn_P1.ipynb` and renames it. Structure is identical to Phase 1 but with their own architecture and experiments.

---

### Notebook 2: Advanced CNN — `notebooks/advanced_cnn_P2.ipynb`

**Architecture**: 4 Conv blocks (32→64→128→256 filters), GlobalAveragePooling2D, data augmentation  
**Input size**: (128, 128, 3)

| Exp # | Change | Value |
|-------|--------|-------|
| 1 | No augmentation baseline | use_augmentation=False |
| 2 | Add augmentation | use_augmentation=True |
| 3 | Stronger augmentation | RandomRotation(0.4), stronger zoom |
| 4 | GlobalAveragePooling vs Flatten | Swap pooling |
| 5 | 5th Conv block | Add block with 512 filters |
| 6 | SGD + momentum | SGD(lr=0.01, momentum=0.9) |
| 7 | LR schedule | CosineDecay or ReduceLROnPlateau |

---

### Notebook 3: VGG16 — `notebooks/vgg16_P3.ipynb`

**Architecture**: VGG16 (ImageNet weights, include_top=False) + GAP + Dense(256) + Dense(128) + sigmoid  
**Input size**: (224, 224, 3) — use vgg16.preprocess_input  
**Training**: Two-stage (freeze base → fine-tune last N layers with LR=1e-5)

| Exp # | Change | Value |
|-------|--------|-------|
| 1 | Fully frozen base | freeze_base=True, head=Dense(256) |
| 2 | Fine-tune last 4 layers | fine_tune_from=-4 |
| 3 | Fine-tune last 8 layers | fine_tune_from=-8 |
| 4 | Larger head | Add Dense(512) |
| 5 | Augmentation on input | Prepend data_augmentation |
| 6 | Lower fine-tune LR | LR=1e-6 |
| 7 | L2 on head | kernel_regularizer=l2(1e-4) |

> **GPU note**: VGG16 is large. If no local GPU is detected (`tf.config.list_physical_devices('GPU')` returns empty), open this notebook in Google Colab cloud (Runtime → Change runtime type → T4 GPU) as a fallback. Expected training time ~5–10 min/epoch on T4.

---

### Notebook 4: ResNet50 — `notebooks/resnet50_P4.ipynb`

**Architecture**: ResNet50 (ImageNet) + GAP + Dense(512) + BN + Dropout(0.5) + sigmoid  
**Input size**: (224, 224, 3) — use resnet50.preprocess_input

| Exp # | Change | Value |
|-------|--------|-------|
| 1 | Frozen base | freeze_base=True |
| 2 | Fine-tune last ResNet block | Unfreeze 'conv5_block*' layers |
| 3 | Fine-tune last 2 blocks | Unfreeze 'conv4_block*' + 'conv5_block*' |
| 4 | Different batch size | batch_size=16 vs 64 |
| 5 | Add augmentation | Prepend data_augmentation |
| 6 | SGD instead of Adam | SGD(lr=1e-4, momentum=0.9) |
| 7 | LR warmup + decay | LearningRateSchedule with warmup |

---

### Notebook 5: MobileNetV2 — `notebooks/mobilenetv2_P5.ipynb`

**Architecture**: MobileNetV2 (ImageNet, alpha=1.0) + GAP + Dense(128) + Dropout(0.3) + sigmoid  
**Input size**: (224, 224, 3) — use mobilenet_v2.preprocess_input

| Exp # | Change | Value |
|-------|--------|-------|
| 1 | Fully frozen base | freeze_base=True, alpha=1.0 |
| 2 | Fine-tune top 20 layers | Unfreeze last 20 layers |
| 3 | Reduced width | alpha=0.75 |
| 4 | Smaller model | alpha=0.5 |
| 5 | Add augmentation | Prepend data_augmentation |
| 6 | Longer training + LR schedule | 30 epochs + ReduceLROnPlateau |
| 7 | Swap base to EfficientNetB0 | Use tf.keras.applications.EfficientNetB0 |

---

## Phase 3: Per-Notebook Completion Checklist (after all 7 experiments)

For each notebook, verify before moving to Phase 4:

- [ ] 7 rows in `results_log` with hypothesis + interpretation Markdown cells
- [ ] Learning curve plots saved: `./figures/P{N}_exp{1-7}_curves.png`
- [ ] Best model identified by F1-score
- [ ] Confusion matrix: `./figures/P{N}_best_confusion_matrix.png`
- [ ] ROC curve: `./figures/P{N}_best_roc_curve.png`
- [ ] Error analysis: `./figures/P{N}_error_analysis.png`
- [ ] Best model weights saved: `./checkpoints/{model_name}_exp{N}.h5`
- [ ] Final results table displayed via `build_results_table()`
- [ ] Notebook runs top-to-bottom (Restart & Run All — no errors)

---

## Phase 4: Report Writing (7–10 June)

### Report file: `report/malaria_cnn_report_DRAFT.docx` → `report/malaria_cnn_report_FINAL.pdf`

### Section ownership and word targets

| Section | Owner | Target |
|---------|-------|--------|
| Introduction | P1 + P2 | ~500 words |
| Literature Review | P1 + P2 | ~750 words (≥5 scholarly sources) |
| Methodology (each sub-section) | All 5 individually | ~400–500 words each |
| Results (each sub-section) | All 5 individually | ~400–500 words each |
| Discussion + cross-model comparison | P3 + P4 | ~750 words |
| Conclusion + References | P5 | ~250 words + ≥5 refs |
| Integration + tone unification | P1 (lead integrator) | — |

### Critical report rules (rubric-driven)

- **No bullet-point paragraphs** in any prose section — full academic sentences only
- **Every figure** must be: numbered (Fig. 1, Fig. 2, …), captioned, and explicitly referenced and interpreted in text
- **Recall/Sensitivity** must be discussed as the primary clinical metric (missed positives = untreated patients)
- **Cross-model comparison table**: one table showing all 5 models' best F1, AUC, Recall side by side
- **< 20% AI-generated content**: write your own analysis; AI may help with grammar only
- **Citation style**: pick APA or IEEE and use it consistently throughout

### Suggested sources to cite

- Rajaraman et al. (2018) — Pre-trained CNNs for malaria detection
- Liang et al. (2020) — Deep learning for malaria diagnosis
- Simonyan & Zisserman (2014) — VGGNet
- He et al. (2016) — ResNet
- Howard et al. (2017) — MobileNets
- Pan & Yang (2010) — A survey on transfer learning
- Esteva et al. (2017) — Dermatologist-level classification

### Methodology sub-section template (per member, ~400–500 words)

1. Architecture description (layer-by-layer table or description)
2. Key design choices + justification (cite ≥2 sources)
3. Transfer learning strategy: what was frozen vs fine-tuned
4. Data preprocessing and augmentation decisions
5. Optimiser, loss, regularisation choices
6. Hyperparameter table for all 7 experiments

### Results sub-section template (per member, ~400–500 words)

1. Results table (all 7 experiments): Accuracy, Precision, Recall, F1, AUC, Epochs
2. Interpreted learning curves — do not describe; explain what the pattern means
3. Confusion matrix interpretation (TP/TN/FP/FN with clinical framing)
4. ROC/AUC interpretation
5. Error analysis: what patterns appear in misclassified images?
6. Model ranking (1st–5th) with one-line justification per rank

---

## Phase 5: Review & Presentation Prep (10–11 June)

### Report review tasks
- Each member peer-reviews one other member's section
- P1 merges all sections, adds bridging paragraphs, unifies tone
- Verify consistent citation format throughout
- Export PDF and check all figures render correctly
- Confirm report is ~10 pages (excluding references)

### Presentation structure (10–12 min total, ~1.5 min per person)

Each member covers (in their 1.5 min):
1. Architecture name + one key design decision (20 sec)
2. Best F1, AUC, Recall — is this clinically acceptable? (30 sec)
3. Point to one pattern in learning curves or confusion matrix (30 sec)
4. One misclassification pattern and its likely cause (15 sec)
5. Your rank among the 5 models, justified with one data point (15 sec)

### Likely Q&A — prepare answers for each
- Why did you choose that learning rate?
- What does [this part of the learning curve] indicate?
- Why are your False Negatives higher than False Positives?
- How did you decide when to stop fine-tuning?
- What would you change with more GPU time?
- What does your AUC of X mean clinically?

### Pre-presentation dry run
- Time the full group run: target 7–8 minutes speaking
- Each person can explain their curves, CM, and ROC curve live without notes
- Camera + screen share tested and working

---

## Phase 6: Final Submission (12 June, before 11:59 pm)

### Submission package
1. `report/malaria_cnn_report_FINAL.pdf`
2. `notebooks/baseline_cnn_P1.ipynb`
3. `notebooks/advanced_cnn_P2.ipynb`
4. `notebooks/vgg16_P3.ipynb`
5. `notebooks/resnet50_P4.ipynb`
6. `notebooks/mobilenetv2_P5.ipynb`
7. Peer contribution sheet (signed by all members)

### Final pre-submission checks
- [ ] Restart & Run All on every notebook — zero errors
- [ ] All figures render correctly in PDF
- [ ] All in-text references have a reference list entry
- [ ] Report is under 11 pages
- [ ] Peer contribution sheet signed by all 5 members
- [ ] Upload to Canvas before 11:59 pm on 12 June

---

## Timeline Summary

| Date | Task | Owner |
|------|------|-------|
| 3 June | Setup: extract data, create structure, write utils.py, verify split | All (P1 leads) |
| 4–5 June | Collective baseline CNN build + Experiment 1 | All (P1 leads) |
| 5–9 June | Individual model development (7 experiments each) | P1–P5 independently |
| 7 June | Begin report writing (methodology sub-sections) | All |
| 9 June EOD | All individual report sections submitted to P1 | P1–P5 |
| 10 June | P1 integrates report; cross-model comparison table | P1 + P3/P4 |
| 10–11 June | Report review, PDF export, presentation dry run | All |
| 12 June | Final submission to Canvas before 11:59 pm | Designated submitter |

---

## Rubric Quick Reference

| Criterion | Points | How to maximise |
|-----------|--------|-----------------|
| Problem Framing & Lit Review | 5 | ≥5 sources, critical comparison, model choices linked to literature |
| Model Implementation & Experiments | 5 | Exactly 7 documented experiments with hypothesis + interpretation each |
| Evaluation, Results & Visuals | 5 | All shared functions used, every figure interpreted, cross-model table |
| Code Quality & Documentation | 5 | Markdown before every code cell, functions not copy-paste, seeds set, runs end-to-end |
| Report Quality & Collaboration | **10** | Academic prose, no bullets in paragraphs, <20% AI, integrated narrative |
| Presentation | 5 | Camera on, interprets visuals live, answers Q&A confidently |

> **Report Quality is worth 10/35 points** — the largest single criterion. Invest proportionally.

---

## Key Implementation Notes

1. **DATA_DIR path**: Resolved automatically in each notebook via `kagglehub.dataset_download()` + an `os.walk` search for the folder containing `Parasitized/` and `Uninfected/`. Never hardcode this path — it differs per machine.

2. **image_size for baseline**: The baseline CNN uses `(64, 64)` input (not 224×224) to reduce memory. `load_dataset(image_size=(64,64))` is called only in the baseline notebook. All other notebooks use `IMAGE_SIZE = (224, 224)`.

3. **Pretrained model preprocessing**: Each pretrained model has its own preprocess function — never use the generic `/255.0` normalisation for pretrained models. Use `vgg16.preprocess_input`, `resnet50.preprocess_input`, `mobilenet_v2.preprocess_input` directly inside the model.

4. **Two-stage training** (pretrained models): Always train head-only first (10 epochs, LR=1e-3), then fine-tune with a lower LR (1e-5). Never fine-tune from scratch.

5. **Saving figures**: Use `save_path=f'../figures/P{N}_exp{exp_num}_curves.png'` consistently (relative to `notebooks/`) so the report can reference them by predictable filenames.

6. **results_log**: Initialise `results_log = []` at the top of each notebook. Append after every experiment. Never overwrite previous entries.

7. **Medical context**: In every results discussion, explicitly note that **Recall (Sensitivity)** is the primary metric — a False Negative (missed malaria case) is more dangerous than a False Positive.

8. **GPU on native Windows**: TF ≥ 2.11 has no native Windows GPU support. Use Google Colab (free T4) for VGG16, ResNet50, and MobileNetV2 notebooks. Baseline and Advanced CNN can train on CPU locally.
