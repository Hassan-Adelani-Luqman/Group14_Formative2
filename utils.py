import os
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, confusion_matrix, roc_curve,
)

# ── Shared constants ──────────────────────────────────────────────────────────
IMAGE_SIZE  = (224, 224)
BATCH_SIZE  = 32
SEED        = 42
CLASS_NAMES = ['Parasitized', 'Uninfected']

TRAIN_SPLIT = 0.80
VAL_SPLIT   = 0.10
TEST_SPLIT  = 0.10

# ── Reproducibility ───────────────────────────────────────────────────────────
def set_seeds(seed=SEED):
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)

# ── Augmentation layer (shared; only active during training) ──────────────────
data_augmentation = tf.keras.Sequential([
    tf.keras.layers.RandomFlip('horizontal_and_vertical'),
    tf.keras.layers.RandomRotation(0.2),
    tf.keras.layers.RandomZoom(0.1),
    tf.keras.layers.RandomContrast(0.1),
], name='data_augmentation')

# ── Dataset loader ────────────────────────────────────────────────────────────
def load_dataset(data_dir, image_size=IMAGE_SIZE, batch_size=BATCH_SIZE):
    """
    Returns (train_ds, val_ds, test_ds) using the shared 80/10/10 split.
    data_dir: path to the folder containing Parasitized/ and Uninfected/.
    All members call this function — never split manually.
    """
    full_ds = tf.keras.utils.image_dataset_from_directory(
        data_dir,
        labels='inferred',
        label_mode='binary',
        class_names=CLASS_NAMES,
        image_size=image_size,
        batch_size=None,
        shuffle=True,
        seed=SEED,
    )

    total   = sum(1 for _ in full_ds)
    n_train = int(total * TRAIN_SPLIT)
    n_val   = int(total * VAL_SPLIT)

    train_ds  = full_ds.take(n_train)
    remaining = full_ds.skip(n_train)
    val_ds    = remaining.take(n_val)
    test_ds   = remaining.skip(n_val)

    normalise = lambda img, lbl: (tf.cast(img, tf.float32) / 255.0, lbl)
    AUTOTUNE  = tf.data.AUTOTUNE

    train_ds = (train_ds
                .map(normalise, num_parallel_calls=AUTOTUNE)
                .cache()
                .shuffle(1000, seed=SEED)
                .batch(batch_size)
                .prefetch(AUTOTUNE))

    val_ds = (val_ds
              .map(normalise, num_parallel_calls=AUTOTUNE)
              .cache()
              .batch(batch_size)
              .prefetch(AUTOTUNE))

    test_ds = (test_ds
               .map(normalise, num_parallel_calls=AUTOTUNE)
               .cache()
               .batch(batch_size)
               .prefetch(AUTOTUNE))

    return train_ds, val_ds, test_ds

# ── Metrics ───────────────────────────────────────────────────────────────────
def evaluate_model(model, test_ds):
    """Returns a dict of all required metrics computed on the test set."""
    y_true, y_pred_prob = [], []

    for images, labels in test_ds:
        preds = model.predict(images, verbose=0)
        y_pred_prob.extend(preds.flatten())
        y_true.extend(labels.numpy().flatten())

    y_true      = np.array(y_true)
    y_pred_prob = np.array(y_pred_prob)
    y_pred      = (y_pred_prob >= 0.5).astype(int)

    return {
        'accuracy':  round(accuracy_score(y_true, y_pred),        4),
        'precision': round(precision_score(y_true, y_pred),       4),
        'recall':    round(recall_score(y_true, y_pred),          4),
        'f1':        round(f1_score(y_true, y_pred),              4),
        'auc':       round(roc_auc_score(y_true, y_pred_prob),    4),
        'y_true':    y_true,
        'y_pred':    y_pred,
        'y_prob':    y_pred_prob,
    }

# ── Learning curves ───────────────────────────────────────────────────────────
def plot_learning_curves(history, experiment_name, save_path=None):
    """Plots training vs validation loss and accuracy."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle(f'Learning curves — {experiment_name}', fontsize=14, fontweight='bold')

    axes[0].plot(history.history['loss'],     label='Train loss',     linewidth=2)
    axes[0].plot(history.history['val_loss'], label='Val loss',       linewidth=2, linestyle='--')
    axes[0].set_title('Loss over epochs')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(history.history['accuracy'],     label='Train accuracy', linewidth=2)
    axes[1].plot(history.history['val_accuracy'], label='Val accuracy',   linewidth=2, linestyle='--')
    axes[1].set_title('Accuracy over epochs')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Accuracy')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()

    final_train = history.history['accuracy'][-1]
    final_val   = history.history['val_accuracy'][-1]
    gap         = final_train - final_val
    if gap > 0.05:
        print(f"⚠️  Overfitting detected: train {final_train:.3f} vs val {final_val:.3f} (gap={gap:.3f})")
    elif gap < -0.02:
        print(f"ℹ️  Underfitting detected: train {final_train:.3f} vs val {final_val:.3f}")
    else:
        print(f"✅ Good fit: train {final_train:.3f} vs val {final_val:.3f} (gap={gap:.3f})")

# ── Confusion matrix ──────────────────────────────────────────────────────────
def plot_confusion_matrix(metrics_dict, class_names, experiment_name, save_path=None):
    """Plots a labelled confusion matrix with TP/TN/FP/FN annotation."""
    cm  = confusion_matrix(metrics_dict['y_true'], metrics_dict['y_pred'])
    fig, ax = plt.subplots(figsize=(7, 6))

    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names,
                yticklabels=class_names,
                ax=ax, linewidths=0.5)

    ax.set_title(f'Confusion matrix — {experiment_name}', fontsize=13, fontweight='bold', pad=14)
    ax.set_ylabel('True label', fontsize=11)
    ax.set_xlabel('Predicted label', fontsize=11)

    tn, fp, fn, tp = cm.ravel()
    ax.text(0.5, -0.12,
            f'TP={tp}  TN={tn}  FP={fp}  FN={fn}  |  '
            f'Sensitivity={tp/(tp+fn):.3f}  Specificity={tn/(tn+fp):.3f}',
            ha='center', va='top', transform=ax.transAxes, fontsize=10, color='dimgray')

    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()

# ── ROC curve ─────────────────────────────────────────────────────────────────
def plot_roc_curve(metrics_dict, experiment_name, save_path=None):
    """Plots ROC curve with AUC annotation."""
    fpr, tpr, _ = roc_curve(metrics_dict['y_true'], metrics_dict['y_prob'])
    auc_val      = metrics_dict['auc']

    fig, ax = plt.subplots(figsize=(7, 6))
    ax.plot(fpr, tpr, color='royalblue', lw=2,
            label=f'ROC curve (AUC = {auc_val:.4f})')
    ax.plot([0, 1], [0, 1], 'k--', lw=1, label='Random classifier')
    ax.fill_between(fpr, tpr, alpha=0.08, color='royalblue')

    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('False Positive Rate (1 – Specificity)', fontsize=11)
    ax.set_ylabel('True Positive Rate (Sensitivity)', fontsize=11)
    ax.set_title(f'ROC Curve — {experiment_name}', fontsize=13, fontweight='bold')
    ax.legend(loc='lower right', fontsize=10)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()

# ── Results table ─────────────────────────────────────────────────────────────
def build_results_table(experiments_list):
    """
    experiments_list: list of dicts with keys:
      exp_num, description, accuracy, precision, recall, f1, auc, epochs, notes
    Returns a DataFrame sorted by F1 descending.
    """
    df = pd.DataFrame(experiments_list)
    df = df[['exp_num', 'description', 'accuracy', 'precision',
             'recall', 'f1', 'auc', 'epochs', 'notes']]
    df.columns = ['Exp #', 'Description', 'Accuracy', 'Precision',
                  'Recall', 'F1', 'AUC', 'Epochs', 'Notes']
    return df.sort_values('F1', ascending=False).reset_index(drop=True)

# ── Callbacks ─────────────────────────────────────────────────────────────────
def get_callbacks(model_name, experiment_num, patience_es=10, patience_lr=5):
    """Standard callbacks: EarlyStopping + ReduceLROnPlateau + ModelCheckpoint."""
    os.makedirs('./checkpoints', exist_ok=True)
    return [
        tf.keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=patience_es,
            restore_best_weights=True,
            verbose=1,
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=patience_lr,
            min_lr=1e-7,
            verbose=1,
        ),
        tf.keras.callbacks.ModelCheckpoint(
            filepath=f'./checkpoints/{model_name}_exp{experiment_num}.h5',
            monitor='val_accuracy',
            save_best_only=True,
            verbose=0,
        ),
    ]

# ── Error analysis ────────────────────────────────────────────────────────────
def error_analysis(model, test_ds, class_names, n_samples=12):
    """Finds misclassified images and displays them with true vs predicted labels."""
    misclassified_images = []
    misclassified_labels = []
    misclassified_preds  = []

    for images, labels in test_ds:
        preds        = model.predict(images, verbose=0).flatten()
        pred_classes = (preds >= 0.5).astype(int)
        mask         = pred_classes != labels.numpy().astype(int)

        misclassified_images.extend(images.numpy()[mask])
        misclassified_labels.extend(labels.numpy()[mask])
        misclassified_preds.extend(preds[mask])

        if len(misclassified_images) >= n_samples:
            break

    n   = min(n_samples, len(misclassified_images))
    fig, axes = plt.subplots(3, 4, figsize=(16, 12))
    fig.suptitle('Error Analysis — Misclassified Samples', fontsize=14, fontweight='bold')

    for i, ax in enumerate(axes.flatten()[:n]):
        ax.imshow(misclassified_images[i])
        true_lbl = class_names[int(misclassified_labels[i])]
        pred_lbl = class_names[int(misclassified_preds[i] >= 0.5)]
        conf     = misclassified_preds[i] if pred_lbl == class_names[1] else 1 - misclassified_preds[i]
        ax.set_title(f'True: {true_lbl}\nPred: {pred_lbl} ({conf:.2f})', fontsize=9, color='red')
        ax.axis('off')

    plt.tight_layout()
    os.makedirs('./figures', exist_ok=True)
    plt.savefig('./figures/error_analysis.png', dpi=150, bbox_inches='tight')
    plt.show()
