import numpy as np
import pandas as pd
from sklearn.metrics import (
    f1_score, precision_score, recall_score, accuracy_score,
    balanced_accuracy_score, roc_auc_score, confusion_matrix,
    classification_report, brier_score_loss
)
from sklearn.utils import resample
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
import matplotlib.pyplot as plt


# Optional: import bootstrapped AUC CI from scikit-bootstrap
try:
    from sklearn_bootstrap import bootstrap_auc
    SKLEARN_BOOTSTRAP_AVAILABLE = True
except ImportError:
    SKLEARN_BOOTSTRAP_AVAILABLE = False


def bootstrap_auc_ci(y_true, y_probs, n_bootstraps=1000, ci=0.95, seed=42):
    """Compute AUC confidence interval using bootstrapping."""
    rng = np.random.RandomState(seed)
    aucs = []

    for _ in range(n_bootstraps):
        indices = rng.choice(np.arange(len(y_probs)), size=len(y_probs), replace=True)
        if len(np.unique(y_true[indices])) < 2:
            continue  # skip if one class is missing in the resample
        auc = roc_auc_score(y_true[indices], y_probs[indices])
        aucs.append(auc)

    lower = np.percentile(aucs, (1 - ci) / 2 * 100)
    upper = np.percentile(aucs, (1 + ci) / 2 * 100)
    return lower, upper


def evaluate(
    y_test, y_pred_probs,
    y_train=None, y_train_pred_probs=None,
    threshold=None, step=0.01
):
    y_test = np.array(y_test)
    y_pred_probs = np.array(y_pred_probs)

    # Determine best threshold by maximizing Youden Index (on test set)
    if threshold is None:
        best_threshold = 0.5
        best_j = -1

        for t in np.arange(0.01, 1.0, step):
            y_pred = (y_pred_probs > t).astype(int)
            cm = confusion_matrix(y_test, y_pred)
            tn, fp, fn, tp = cm.ravel()
            sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
            specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
            j_stat = sensitivity + specificity - 1

            if j_stat > best_j:
                best_j = j_stat
                best_threshold = t
    else:
        best_threshold = threshold

    # Final test predictions
    y_pred_best = (np.ravel(y_pred_probs) > best_threshold).astype(int)

    # Test metrics
    precision = precision_score(y_test, y_pred_best, zero_division=0)
    recall = recall_score(y_test, y_pred_best, zero_division=0)
    f1 = f1_score(y_test, y_pred_best, zero_division=0)
    f2 = (5 * precision * recall) / ((4 * precision) + recall) if (4 * precision + recall) != 0 else 0
    acc = accuracy_score(y_test, y_pred_best)
    balanced_acc = balanced_accuracy_score(y_test, y_pred_best)
    auc_roc = roc_auc_score(y_test, y_pred_probs)
    brier = brier_score_loss(y_test, y_pred_probs)

    # AUC CI
    if SKLEARN_BOOTSTRAP_AVAILABLE:
        auc_lower, auc_upper = bootstrap_auc(y_test, y_pred_probs, ci=0.95)
    else:
        auc_lower, auc_upper = bootstrap_auc_ci(y_test, y_pred_probs)

    # Specificity
    cm = confusion_matrix(y_test, y_pred_best)
    tn, fp, fn, tp = cm.ravel()
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0

    # Training AUC
    if y_train is not None and y_train_pred_probs is not None:
        train_auc = roc_auc_score(y_train, y_train_pred_probs)
    else:
        train_auc = None

    report = classification_report(y_test, y_pred_best, digits=4)

    print(f"\n🔹 Threshold Used (Youden's J): {best_threshold:.2f}")
    if train_auc is not None:
        print(f"🧪 AUC-ROC Score (Train): {train_auc:.4f}")
    print(f"✔️ AUC-ROC Score (Test): {auc_roc:.4f} (95% CI: {auc_lower:.4f} - {auc_upper:.4f})")
    print(f"✔️ Accuracy: {acc:.4f}")
    print(f"✔️ Balanced Accuracy: {balanced_acc:.4f}")
    print(f"✔️ Precision: {precision:.4f}")
    print(f"✔️ Recall (Sensitivity): {recall:.4f}")
    print(f"✔️ Specificity: {specificity:.4f}")
    print(f"✔️ F1-score: {f1:.4f}")
    print(f"✔️ F2-Score: {f2:.4f}")
    print(f"✔️ Brier Score: {brier:.4f}")
    print("\nConfusion Matrix:\n", cm)
    print("\nClassification Report:\n", report)

    metrics = {
        "Threshold": best_threshold,
        "AUC-ROC": auc_roc,
        "AUC-ROC 95% CI": (auc_lower, auc_upper),
        "AUC-ROC Train": train_auc,
        "Accuracy": acc,
        "Balanced Accuracy": balanced_acc,
        "Precision": precision,
        "Recall": recall,
        "Specificity": specificity,
        "F1-Score": f1,
        "F2-Score": f2,
        "Brier Score": brier
    }

    return best_threshold, metrics



def nri_score(y_true, old_preds, new_preds):
    df = pd.DataFrame({
        'True Label': y_true,
        'Old Model Prediction': old_preds,
        'New Model Prediction': new_preds
    })

    # Remove rows where old_preds == -1 (missing)
    initial_len = len(df)
    df_clean = df[df['Old Model Prediction'] != -1].copy()
    removed_count = initial_len - len(df_clean)
    print(f"🧹 Removed {removed_count} rows with Old Model Prediction = -1 (missing values).")

    # Tag reclassification
    df_clean['Reclassification'] = 'Unchanged'
    df_clean.loc[(df_clean['Old Model Prediction'] != 1) & (df_clean['New Model Prediction'] == 1), 'Reclassification'] = 'Upgraded'
    df_clean.loc[(df_clean['Old Model Prediction'] == 1) & (df_clean['New Model Prediction'] == 0), 'Reclassification'] = 'Downgraded'

    # Count positives and negatives
    num_events = (df_clean['True Label'] == 1).sum()
    num_nonevents = (df_clean['True Label'] == 0).sum()

    correct_upgrade = ((df_clean['Reclassification'] == 'Upgraded') & (df_clean['True Label'] == 1)).sum()
    incorrect_upgrade = ((df_clean['Reclassification'] == 'Upgraded') & (df_clean['True Label'] == 0)).sum()

    correct_downgrade = ((df_clean['Reclassification'] == 'Downgraded') & (df_clean['True Label'] == 0)).sum()
    incorrect_downgrade = ((df_clean['Reclassification'] == 'Downgraded') & (df_clean['True Label'] == 1)).sum()

    print("\n🔹 Reclassification Summary:")
    print(f"✔️ Correctly Upgraded (0 → 1, actual = 1): {correct_upgrade}")
    print(f"❌ Incorrectly Upgraded (0 → 1, actual = 0): {incorrect_upgrade}")
    print(f"✔️ Correctly Downgraded (1 → 0, actual = 0): {correct_downgrade}")
    print(f"❌ Incorrectly Downgraded (1 → 0, actual = 1): {incorrect_downgrade}")
    print(f"\n📊 Events (actual = 1): {num_events}, Non-events (actual = 0): {num_nonevents}")

    if num_events > 0 and num_nonevents > 0:
        nri = ((correct_upgrade - incorrect_downgrade) / num_events) - \
              ((incorrect_upgrade - correct_downgrade) / num_nonevents)
        print(f"\n✅ Net Reclassification Index (NRI): {nri:.4f}")
    else:
        nri = None
        print("⚠️ Cannot compute NRI — no events or non-events in the filtered data.")

    return nri



def calibrate_and_plot_with_nri(
    estimator,
    method,
    X_calib, y_calib,
    X_test, y_test,
    timi_score_test,
    evaluate_fn,
    nri_score_fn,
    title_prefix=""
):
    """
    Calibrate a classifier and evaluate calibration, threshold metrics, and NRI.

    Parameters:
    - estimator: a fitted sklearn classifier
    - method: 'isotonic' or 'sigmoid'
    - X_calib, y_calib: calibration dataset
    - X_test, y_test: evaluation dataset
    - timi_score_test: TIMI scores for NRI comparison
    - evaluate_fn: function to determine best threshold and metrics
    - nri_score_fn: function to compute NRI
    - title_prefix: optional string to prefix the calibration plot title
    """

    # Ensure categorical dtypes
    for df in [X_calib, X_test]:
        for col in df.select_dtypes(include='object').columns:
            df[col] = df[col].astype('category')

    # Fit calibrator
    calibrator = CalibratedClassifierCV(estimator=estimator, method=method, cv='prefit')
    calibrator.fit(X_calib, y_calib)

    # Probabilities
    y_calib_probs = calibrator.predict_proba(X_calib)[:, 1]
    y_test_probs = calibrator.predict_proba(X_test)[:, 1]

    # Brier Scores
    brier_orig = brier_score_loss(y_test, estimator.predict_proba(X_test)[:, 1])
    brier_calib = brier_score_loss(y_test, y_test_probs)
    # print(f"🔹 Original Brier Score: {brier_orig:.4f}")
    # print(f"✅ {method.capitalize()} Calibrated Brier Score: {brier_calib:.4f}")

    # Calibration plot
    plt.figure(figsize=(7, 6))
    for label, prob in [("Original", estimator.predict_proba(X_test)[:, 1]), (f"{method.capitalize()} Calibrated", y_test_probs)]:
        prob_true, prob_pred = calibration_curve(y_test, prob, n_bins=10, strategy='uniform')
        plt.plot(prob_pred, prob_true, marker='o', label=label)

    plt.plot([0, 1], [0, 1], linestyle='--', color='gray')
    plt.xlabel("Predicted probability")
    plt.ylabel("True probability")
    plt.title(f"{title_prefix} Calibration Curve ({method.capitalize()})")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    # Evaluate threshold using calibration set performance
    best_threshold, metrics = evaluate_fn(y_test, y_test_probs, y_train=y_calib, y_train_pred_probs=y_calib_probs)


    # Binarized predictions
    new_preds = (y_test_probs > best_threshold).astype(int)

    # Convert TIMI scores into binary format (1 if ≥4, 0 if <4, -1 if missing)
    old_preds = timi_score_test.apply(lambda x: 1 if x >= 4 else (0 if pd.notna(x) else -1))

    # Calculate NRI
    nri = nri_score_fn(y_test, old_preds, new_preds)

    return calibrator
