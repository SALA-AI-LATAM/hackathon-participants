"""
train_precipitation.py — Multi-station extreme weather nowcasting pipeline.

Trains RNN, LSTM, and GRU baselines on 4 inland weather stations from
San Cristobal Island, Galapagos. Predicts heavy precipitation at a target
station (El Junco by default) at 3h, 6h, and 12h horizons.

Equivalent to precipitation_nowcasting.ipynb sections 2-9, runnable as a standalone script.

Usage (after R2 download):
    python train_precipitation.py --data-dir ./precipitation-nowcasting
    python train_precipitation.py --data-dir ./precipitation-nowcasting --target-station mira --epochs 30
"""

import argparse
import json
import os
import warnings

import matplotlib
matplotlib.use('Agg')  # non-interactive backend for script execution
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from datetime import timedelta
from tqdm.auto import tqdm

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import (
    precision_recall_curve, average_precision_score, roc_auc_score,
    confusion_matrix, brier_score_loss
)
from sklearn.calibration import calibration_curve
from sklearn.linear_model import LogisticRegression

warnings.filterwarnings('ignore')

# ============================================================================
# Configuration
# ============================================================================

STATION_FILES = {
    'cer': 'weather_stations/CER_consolid_f15.csv',
    'jun': 'weather_stations/JUN_consolid_f15.csv',
    'merc': 'weather_stations/MERC_consolid_f15.csv',
    'mira': 'weather_stations/MIRA_consolid_f15.csv',
}

COLUMN_MAP = {
    'rain_mm': ['Rain_mm_Tot'],
    'temp_c': ['AirTC_Avg'],
    'rh_avg': ['RH_Avg'],
    'rh_max': ['RH_Max'],
    'rh_min': ['RH_Min'],
    'solar_kw': ['SlrkW_Avg'],
    'net_rad_wm2': ['NR_Wm2_Avg'],
    'wind_speed_ms': ['WS_ms_Avg'],
    'wind_dir': ['WindDir'],
    'soil_moisture_1': ['VW_Avg', 'VW'],
    'soil_moisture_2': ['VW_2_Avg', 'VW_2'],
    'soil_moisture_3': ['VW_3_Avg', 'VW_3'],
    'leaf_wetness': ['LWmV_Avg'],
    'leaf_wet_minutes': ['LWMWet_Tot'],
}

HORIZONS = {'3h': 12, '6h': 24, '12h': 48}

SEED = 42


# ============================================================================
# Data Loading
# ============================================================================

def load_station(name, data_dir):
    """Load a station CSV, harmonize columns, set datetime index."""
    path = os.path.join(data_dir, STATION_FILES[name])
    df = pd.read_csv(path, low_memory=False)

    # === Parse timestamp (M/D/YYYY H:MM format) ===
    df['TIMESTAMP'] = pd.to_datetime(df['TIMESTAMP'], format='%m/%d/%Y %H:%M')
    df = df.set_index('TIMESTAMP').sort_index()

    # === Multi-candidate column lookup ===
    rename = {}
    for harmonized, candidates in COLUMN_MAP.items():
        for candidate in candidates:
            if candidate in df.columns:
                rename[candidate] = harmonized
                break

    df = df[list(rename.keys())].rename(columns=rename)

    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')

    return df


def load_all_stations(data_dir):
    """Load all 4 stations and return dict of DataFrames."""
    stations = {}
    for name in tqdm(STATION_FILES, desc="Loading stations"):
        stations[name] = load_station(name, data_dir)
        df = stations[name]
        missing = [c for c in COLUMN_MAP if c not in df.columns]
        print(f"  {name:8s}: {df.index.min().date()} -> {df.index.max().date()} "
              f"({df.shape[0]:,} rows, {len(df.columns)} cols)"
              + (f"  missing={missing}" if missing else ""))
    return stations


# ============================================================================
# Preprocessing
# ============================================================================

def merge_stations(stations):
    """Merge all stations into a wide DataFrame with prefixed columns."""
    prefixed = []
    for name, stn_df in stations.items():
        prefixed.append(stn_df.add_prefix(f'{name}_'))
    df = pd.concat(prefixed, axis=1, sort=True)

    # Drop entirely-NaN columns (missing sensors at some stations)
    all_nan_cols = [c for c in df.columns if df[c].isnull().all()]
    if all_nan_cols:
        print(f"  Dropping {len(all_nan_cols)} entirely-NaN columns: {all_nan_cols}")
        df = df.drop(columns=all_nan_cols)

    return df


def impute(df, station_names):
    """Apply per-station imputation, then global fill."""
    for stn in station_names:
        # Temperature, humidity, radiation, soil moisture: interpolate up to 6h
        for var in ['temp_c', 'rh_avg', 'rh_max', 'rh_min', 'solar_kw',
                    'net_rad_wm2', 'soil_moisture_1', 'soil_moisture_2',
                    'soil_moisture_3']:
            col = f'{stn}_{var}'
            if col in df.columns:
                df[col] = df[col].interpolate(method='time', limit=24)

        # Wind speed: forward-fill then interpolate
        col = f'{stn}_wind_speed_ms'
        if col in df.columns:
            df[col] = df[col].ffill(limit=4)
            df[col] = df[col].interpolate(method='time', limit=8)

        # Wind direction: forward-fill only (circular)
        col = f'{stn}_wind_dir'
        if col in df.columns:
            df[col] = df[col].ffill(limit=8)

        # Leaf wetness: forward-fill
        for var in ['leaf_wetness', 'leaf_wet_minutes']:
            col = f'{stn}_{var}'
            if col in df.columns:
                df[col] = df[col].ffill(limit=8)

        # Precipitation: zero-fill + missing indicator
        rain_col = f'{stn}_rain_mm'
        if rain_col in df.columns:
            df[f'{stn}_rain_missing'] = df[rain_col].isnull().astype(float)
            df[rain_col] = df[rain_col].fillna(0.0)

    # === Global forward/backward fill for remaining short gaps ===
    df = df.ffill(limit=96).bfill(limit=96)

    # === Fill remaining NaN with 0 (long sensor outages) ===
    still_nan = df.isnull().sum().sum()
    if still_nan > 0:
        n_cols = (df.isnull().sum() > 0).sum()
        print(f"  Filling {still_nan:,} remaining NaN across {n_cols} columns with 0")
    df = df.fillna(0.0)
    df = df.copy()  # defragment

    return df


def engineer_features(df, station_names):
    """Add cyclical time features and per-station derived features."""
    # === Cyclical time features (shared) ===
    df['hour_sin'] = np.sin(2 * np.pi * df.index.hour / 24)
    df['hour_cos'] = np.cos(2 * np.pi * df.index.hour / 24)
    df['doy_sin'] = np.sin(2 * np.pi * df.index.dayofyear / 365.25)
    df['doy_cos'] = np.cos(2 * np.pi * df.index.dayofyear / 365.25)

    # === Per-station derived features ===
    for stn in station_names:
        wd_col = f'{stn}_wind_dir'
        ws_col = f'{stn}_wind_speed_ms'
        temp_col = f'{stn}_temp_c'
        rh_col = f'{stn}_rh_avg'

        # Wind vector decomposition
        if wd_col in df.columns and ws_col in df.columns:
            wd_rad = np.deg2rad(df[wd_col])
            df[f'{stn}_wind_x'] = df[ws_col] * np.cos(wd_rad)
            df[f'{stn}_wind_y'] = df[ws_col] * np.sin(wd_rad)

        # Dewpoint (Magnus formula)
        if temp_col in df.columns and rh_col in df.columns:
            T = df[temp_col]
            RH = df[rh_col].clip(lower=1)
            alpha = (17.27 * T) / (237.3 + T) + np.log(RH / 100)
            df[f'{stn}_dewpoint'] = (237.3 * alpha) / (17.27 - alpha)
            df[f'{stn}_dewpoint_depression'] = T - df[f'{stn}_dewpoint']

        # Soil moisture tendency (3h change)
        sm_col = f'{stn}_soil_moisture_1'
        if sm_col in df.columns:
            df[f'{stn}_soil_moist_tend_3h'] = df[sm_col].diff(periods=12)

        # Rolling statistics
        for window, wlabel in [(4, '1h'), (12, '3h'), (24, '6h')]:
            rain_col = f'{stn}_rain_mm'
            if rain_col in df.columns:
                df[f'{stn}_rain_sum_{wlabel}'] = df[rain_col].rolling(window, min_periods=1).sum()
            if temp_col in df.columns:
                df[f'{stn}_temp_mean_{wlabel}'] = df[temp_col].rolling(window, min_periods=1).mean()
                df[f'{stn}_temp_std_{wlabel}'] = df[temp_col].rolling(window, min_periods=1).std()
            if ws_col in df.columns:
                df[f'{stn}_wind_mean_{wlabel}'] = df[ws_col].rolling(window, min_periods=1).mean()
            if rh_col in df.columns:
                df[f'{stn}_rh_mean_{wlabel}'] = df[rh_col].rolling(window, min_periods=1).mean()

    return df


def create_labels(df, station_names, target_station):
    """Create per-station heavy rain labels + temperature anomaly."""
    # === Per-station precipitation labels ===
    for stn in station_names:
        rain_col = f'{stn}_rain_mm'
        for label, steps in HORIZONS.items():
            df[f'rain_future_{label}_{stn}'] = (
                df[rain_col].rolling(steps, min_periods=1).sum().shift(-steps)
            )

    # === Thresholds from training period (target station) ===
    train_mask = df.index < '2023-01-01'
    thresholds = {}
    for label, steps in HORIZONS.items():
        col = f'rain_future_{label}_{target_station}'
        wet = df.loc[train_mask, col]
        wet = wet[wet > 0]
        p95 = wet.quantile(0.95)
        thresholds[label] = max(p95, 2.0)
        print(f"  {label}: p95={p95:.2f}mm, threshold={thresholds[label]:.2f}mm")

    # === Binary labels for all stations ===
    for stn in station_names:
        for label in HORIZONS:
            col = f'rain_future_{label}_{stn}'
            df[f'heavy_rain_{label}_{stn}'] = (df[col] >= thresholds[label]).astype(float)
            df.loc[df[col].isnull(), f'heavy_rain_{label}_{stn}'] = np.nan

    # === Default targets from target station ===
    for label in HORIZONS:
        df[f'heavy_rain_{label}'] = df[f'heavy_rain_{label}_{target_station}']

    # === Temperature anomaly (target station) ===
    temp_col = f'{target_station}_temp_c'
    df['day_of_year'] = df.index.dayofyear
    daily_clim = df.loc[train_mask].groupby('day_of_year')[temp_col].agg(['mean', 'std'])
    daily_clim['mean_smooth'] = daily_clim['mean'].rolling(15, center=True, min_periods=5).mean()
    daily_clim['std_smooth'] = daily_clim['std'].rolling(15, center=True, min_periods=5).mean()
    daily_clim = daily_clim.bfill().ffill()
    df['temp_clim_mean'] = df['day_of_year'].map(daily_clim['mean_smooth'])
    df['temp_clim_std'] = df['day_of_year'].map(daily_clim['std_smooth'])
    df['temp_anomaly'] = (df[temp_col] - df['temp_clim_mean']) / df['temp_clim_std']
    df['temp_extreme'] = (df['temp_anomaly'].abs() > 2).astype(float)

    return df, thresholds


# ============================================================================
# Dataset
# ============================================================================

class WeatherDataset(Dataset):
    """Sliding window dataset for weather time series classification."""

    def __init__(self, df, feature_cols, target_col, lookback=96):
        self.lookback = lookback

        valid_mask = df[feature_cols + [target_col]].notna().all(axis=1)
        clean_df = df.loc[valid_mask].copy()

        self.features = clean_df[feature_cols].values.astype(np.float32)
        self.labels = clean_df[target_col].values.astype(np.float32)
        self.timestamps = clean_df.index

        # === Build valid window indices (no time gaps within window) ===
        self.valid_indices = []
        expected_delta = pd.Timedelta(minutes=15)
        for i in tqdm(range(lookback, len(self.features)),
                      desc=f"Building windows ({target_col})", leave=False):
            window_times = self.timestamps[i - lookback:i + 1]
            diffs = window_times[1:] - window_times[:-1]
            if (diffs == expected_delta).all():
                self.valid_indices.append(i)

        self.valid_indices = np.array(self.valid_indices)
        print(f"  {target_col}: {len(self.valid_indices):,} valid windows "
              f"from {len(self.features):,} rows "
              f"(positive rate: {self.labels[self.valid_indices].mean():.3%})")

    def __len__(self):
        return len(self.valid_indices)

    def __getitem__(self, idx):
        i = self.valid_indices[idx]
        x = self.features[i - self.lookback:i]
        y = self.labels[i]
        return torch.from_numpy(x), torch.tensor(y)


# ============================================================================
# Models
# ============================================================================

class RecurrentClassifier(nn.Module):
    """Shared architecture for RNN/LSTM/GRU binary classifiers."""

    def __init__(self, input_dim, hidden_dim=128, num_layers=2, dropout=0.3,
                 rnn_type='lstm'):
        super().__init__()
        self.rnn_type = rnn_type
        rnn_cls = {'rnn': nn.RNN, 'lstm': nn.LSTM, 'gru': nn.GRU}[rnn_type]
        self.rnn = rnn_cls(
            input_size=input_dim, hidden_size=hidden_dim,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            batch_first=True,
        )
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x):
        output, _ = self.rnn(x)
        last_hidden = output[:, -1, :]
        logit = self.classifier(last_hidden)
        return logit.squeeze(-1)


# ============================================================================
# Training
# ============================================================================

def compute_class_weight(dataset):
    """Compute positive class weight from dataset labels."""
    labels = dataset.labels[dataset.valid_indices]
    pos_rate = labels.mean()
    if pos_rate == 0 or pos_rate == 1:
        return 1.0
    return min((1 - pos_rate) / pos_rate, 20.0)


def train_model(model, train_loader, val_loader, pos_weight, device,
                lr=1e-3, max_epochs=50, patience=10, model_name='Model'):
    """Train with early stopping on validation loss."""
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(pos_weight).to(device))
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=2
    )

    best_val_loss = float('inf')
    best_state = None
    epochs_no_improve = 0
    history = {'train_loss': [], 'val_loss': []}

    pbar = tqdm(range(max_epochs), desc=f"Training {model_name}")
    for epoch in pbar:
        # === Train ===
        model.train()
        train_losses = []
        for x_batch, y_batch in train_loader:
            x_batch, y_batch = x_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            logits = model(x_batch)
            loss = criterion(logits, y_batch)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            train_losses.append(loss.item())

        # === Validate ===
        model.eval()
        val_losses = []
        with torch.no_grad():
            for x_batch, y_batch in val_loader:
                x_batch, y_batch = x_batch.to(device), y_batch.to(device)
                logits = model(x_batch)
                loss = criterion(logits, y_batch)
                val_losses.append(loss.item())

        train_loss = np.mean(train_losses)
        val_loss = np.mean(val_losses)
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        scheduler.step(val_loss)

        pbar.set_postfix(train=f"{train_loss:.4f}", val=f"{val_loss:.4f}",
                         lr=f"{optimizer.param_groups[0]['lr']:.1e}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                print(f"  Early stopping at epoch {epoch + 1}")
                break

    model.load_state_dict(best_state)
    return history


# ============================================================================
# Evaluation
# ============================================================================

def get_predictions(model, loader, device):
    """Get model predictions on a DataLoader."""
    model.eval()
    all_probs, all_labels = [], []
    with torch.no_grad():
        for x_batch, y_batch in loader:
            x_batch = x_batch.to(device)
            logits = model(x_batch)
            probs = torch.sigmoid(logits).cpu().numpy()
            all_probs.append(probs)
            all_labels.append(y_batch.numpy())
    return np.concatenate(all_probs), np.concatenate(all_labels)


def platt_scale(val_probs, val_labels, test_probs):
    """Platt scaling: recalibrate probabilities using validation set."""
    val_logits = np.log(val_probs / (1 - val_probs + 1e-9) + 1e-9).reshape(-1, 1)
    test_logits = np.log(test_probs / (1 - test_probs + 1e-9) + 1e-9).reshape(-1, 1)
    lr = LogisticRegression(C=1.0, solver='lbfgs', max_iter=1000)
    lr.fit(val_logits, val_labels)
    return lr.predict_proba(test_logits)[:, 1]


def evaluate_model(y_true, y_prob, model_name='Model', threshold=None):
    """Compute all evaluation metrics."""
    if threshold is None:
        prec, rec, thresholds = precision_recall_curve(y_true, y_prob)
        f1 = 2 * prec[:-1] * rec[:-1] / (prec[:-1] + rec[:-1] + 1e-9)
        threshold = thresholds[np.argmax(f1)]

    y_pred = (y_prob >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()

    pod = tp / (tp + fn) if (tp + fn) > 0 else 0
    far = fp / (tp + fp) if (tp + fp) > 0 else 0
    csi = tp / (tp + fn + fp) if (tp + fn + fp) > 0 else 0
    f1 = 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) > 0 else 0
    pr_auc = average_precision_score(y_true, y_prob)
    roc_auc = roc_auc_score(y_true, y_prob) if y_true.sum() > 0 else 0
    brier = brier_score_loss(y_true, y_prob)
    clim_rate = y_true.mean()
    brier_clim = clim_rate * (1 - clim_rate)
    bss = 1 - brier / brier_clim if brier_clim > 0 else 0

    return {
        'model': model_name, 'threshold': threshold,
        'PR-AUC': pr_auc, 'ROC-AUC': roc_auc, 'BSS': bss,
        'CSI': csi, 'POD': pod, 'FAR': far, 'F1': f1,
        'TP': int(tp), 'FP': int(fp), 'FN': int(fn), 'TN': int(tn),
        'N_events': int(y_true.sum()),
    }


def save_eval_plots(y_true, y_prob, model_name, output_dir):
    """Save PR curve, reliability diagram, and score distribution to file."""
    fig, axes = plt.subplots(1, 3, figsize=(16, 4))

    # PR Curve
    ax = axes[0]
    prec, rec, _ = precision_recall_curve(y_true, y_prob)
    pr_auc = average_precision_score(y_true, y_prob)
    ax.plot(rec, prec, label=f'{model_name} (AP={pr_auc:.3f})')
    ax.axhline(y_true.mean(), color='gray', linestyle='--', label=f'No-skill ({y_true.mean():.3f})')
    ax.set_xlabel('Recall (POD)')
    ax.set_ylabel('Precision (1-FAR)')
    ax.set_title('Precision-Recall Curve')
    ax.legend()

    # Reliability Diagram
    ax = axes[1]
    prob_true, prob_pred = calibration_curve(y_true, y_prob, n_bins=10, strategy='quantile')
    ax.plot(prob_pred, prob_true, 'o-', label=model_name)
    ax.plot([0, 1], [0, 1], 'k--', label='Perfect calibration')
    ax.set_xlabel('Mean Predicted Probability')
    ax.set_ylabel('Observed Frequency')
    ax.set_title('Reliability Diagram')
    ax.legend()

    # Score Distribution
    ax = axes[2]
    ax.hist(y_prob[y_true == 0], bins=50, alpha=0.5, label='Non-events', density=True)
    ax.hist(y_prob[y_true == 1], bins=50, alpha=0.5, label='Events', density=True)
    ax.set_xlabel('Predicted Probability')
    ax.set_ylabel('Density')
    ax.set_title('Score Distribution')
    ax.legend()

    plt.suptitle(f'{model_name} — Test Set Evaluation', fontsize=14, y=1.02)
    plt.tight_layout()
    path = os.path.join(output_dir, f'eval_{model_name.lower()}.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved evaluation plot -> {path}")


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description='Multi-station precipitation nowcasting')
    parser.add_argument('--data-dir', type=str, default='./precipitation-nowcasting',
                        help='Directory containing weather_stations/ and checkpoints/ subdirs (from R2 download)')
    parser.add_argument('--output-dir', type=str, default='./checkpoints',
                        help='Directory for checkpoints and results')
    parser.add_argument('--target-station', type=str, default='jun',
                        choices=['cer', 'jun', 'merc', 'mira'],
                        help='Target station for prediction (default: jun)')
    parser.add_argument('--lookback', type=int, default=96,
                        help='Lookback window in 15-min steps (default: 96 = 24h)')
    parser.add_argument('--batch-size', type=int, default=256)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--patience', type=int, default=10)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--hidden-dim', type=int, default=128)
    args = parser.parse_args()

    # === Reproducibility ===
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    os.makedirs(args.output_dir, exist_ok=True)
    station_names = list(STATION_FILES.keys())

    # ================================================================
    # Stage 1: Load data
    # ================================================================
    print("\n=== Stage 1: Loading data ===")
    stations = load_all_stations(args.data_dir)

    # ================================================================
    # Stage 2: Merge + impute
    # ================================================================
    print("\n=== Stage 2: Merge & impute ===")
    df = merge_stations(stations)
    print(f"  Wide DataFrame: {df.shape[0]:,} rows x {df.shape[1]} cols")
    df = impute(df, station_names)
    print(f"  After imputation: {df.shape[0]:,} rows x {df.shape[1]} cols")

    # ================================================================
    # Stage 3: Feature engineering
    # ================================================================
    print("\n=== Stage 3: Feature engineering ===")
    df = engineer_features(df, station_names)
    print(f"  Features: {df.shape[1]} columns")

    # ================================================================
    # Stage 4: Labels
    # ================================================================
    print(f"\n=== Stage 4: Labels (target: {args.target_station}) ===")
    df, thresholds = create_labels(df, station_names, args.target_station)

    for label in HORIZONS:
        col = f'heavy_rain_{label}'
        print(f"  {col}: {df[col].mean():.3%} positive ({df[col].sum():.0f} events)")

    # ================================================================
    # Stage 5: Train/val/test split
    # ================================================================
    print("\n=== Stage 5: Split ===")
    TRAIN_END = pd.Timestamp('2023-01-01')
    VAL_END = pd.Timestamp('2024-07-01')

    drop_cols = [c for c in df.columns if c.startswith('rain_future_')]
    drop_cols += ['day_of_year', 'temp_clim_mean', 'temp_clim_std']
    df = df.drop(columns=drop_cols)

    train_df = df[df.index < TRAIN_END].copy()
    val_start = TRAIN_END + timedelta(hours=12)
    val_df = df[(df.index >= val_start) & (df.index < VAL_END)].copy()
    test_start = VAL_END + timedelta(hours=12)
    test_df = df[df.index >= test_start].copy()

    print(f"  Train: {train_df.index.min().date()} -> {train_df.index.max().date()} ({len(train_df):,} rows)")
    print(f"  Val:   {val_df.index.min().date()} -> {val_df.index.max().date()} ({len(val_df):,} rows)")
    print(f"  Test:  {test_df.index.min().date()} -> {test_df.index.max().date()} ({len(test_df):,} rows)")

    # ================================================================
    # Stage 6: Normalize
    # ================================================================
    print("\n=== Stage 6: Normalize ===")
    LABEL_COLS = [f'heavy_rain_{h}' for h in HORIZONS]
    for stn in station_names:
        LABEL_COLS += [f'heavy_rain_{h}_{stn}' for h in HORIZONS]
    LABEL_COLS += ['temp_extreme', 'temp_anomaly']

    FEATURE_COLS = [c for c in df.columns if c not in LABEL_COLS]
    print(f"  Feature columns: {len(FEATURE_COLS)}")

    train_mean = train_df[FEATURE_COLS].mean()
    train_std = train_df[FEATURE_COLS].std().replace(0, 1)

    for split_df in [train_df, val_df, test_df]:
        split_df[FEATURE_COLS] = (split_df[FEATURE_COLS] - train_mean) / train_std

    # ================================================================
    # Stage 7: Create datasets
    # ================================================================
    TARGET_COL = 'heavy_rain_3h'
    print(f"\n=== Stage 7: Create datasets (target={TARGET_COL}, lookback={args.lookback}) ===")

    train_ds = WeatherDataset(train_df, FEATURE_COLS, TARGET_COL, args.lookback)
    val_ds = WeatherDataset(val_df, FEATURE_COLS, TARGET_COL, args.lookback)
    test_ds = WeatherDataset(test_df, FEATURE_COLS, TARGET_COL, args.lookback)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False)

    print(f"  Batches: train={len(train_loader)}, val={len(val_loader)}, test={len(test_loader)}")

    # ================================================================
    # Stage 8: Train models
    # ================================================================
    print("\n=== Stage 8: Train models ===")
    n_features = len(FEATURE_COLS)
    pos_weight = compute_class_weight(train_ds)
    print(f"  Positive class weight: {pos_weight:.1f}x")

    models = {
        'RNN': RecurrentClassifier(n_features, hidden_dim=args.hidden_dim, rnn_type='rnn').to(device),
        'LSTM': RecurrentClassifier(n_features, hidden_dim=args.hidden_dim, rnn_type='lstm').to(device),
        'GRU': RecurrentClassifier(n_features, hidden_dim=args.hidden_dim, rnn_type='gru').to(device),
    }

    histories = {}
    for name, model in models.items():
        print(f"\n{'='*50}")
        print(f"  Training {name} ({sum(p.numel() for p in model.parameters()):,} params)")
        print(f"{'='*50}")
        histories[name] = train_model(
            model, train_loader, val_loader,
            pos_weight=pos_weight, device=device,
            lr=args.lr, max_epochs=args.epochs,
            patience=args.patience, model_name=name,
        )

    # === Save checkpoints ===
    for name, model in models.items():
        ckpt_path = os.path.join(args.output_dir, f"{name.lower()}_heavy_rain_3h.pt")
        torch.save({
            'model_state_dict': model.state_dict(),
            'rnn_type': name.lower(),
            'input_dim': n_features,
            'hidden_dim': args.hidden_dim,
            'num_layers': 2,
            'dropout': 0.3,
            'target': TARGET_COL,
            'target_station': args.target_station,
            'lookback': args.lookback,
            'feature_cols': FEATURE_COLS,
            'train_mean': train_mean.to_dict(),
            'train_std': train_std.to_dict(),
            'thresholds': thresholds,
        }, ckpt_path)
        print(f"  Saved {name} checkpoint -> {ckpt_path}")

    # === Save training curves ===
    fig, axes = plt.subplots(1, 3, figsize=(16, 4))
    for ax, (name, hist) in zip(axes, histories.items()):
        ax.plot(hist['train_loss'], label='Train')
        ax.plot(hist['val_loss'], label='Val')
        ax.set_title(f'{name} — Loss Curve')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Loss')
        ax.legend()
    plt.tight_layout()
    curves_path = os.path.join(args.output_dir, 'training_curves.png')
    plt.savefig(curves_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved training curves -> {curves_path}")

    # ================================================================
    # Stage 9: Evaluate
    # ================================================================
    print("\n=== Stage 9: Evaluate (with Platt scaling) ===")
    results = []
    for name, model in models.items():
        val_probs, val_labels = get_predictions(model, val_loader, device)
        test_probs_raw, test_labels = get_predictions(model, test_loader, device)

        # Platt scaling
        test_probs = platt_scale(val_probs, val_labels, test_probs_raw)
        val_probs_cal = platt_scale(val_probs, val_labels, val_probs)

        # Optimal threshold from calibrated val predictions
        prec, rec, pr_thresholds = precision_recall_curve(val_labels, val_probs_cal)
        f1 = 2 * prec[:-1] * rec[:-1] / (prec[:-1] + rec[:-1] + 1e-9)
        best_thresh = pr_thresholds[np.argmax(f1)]

        metrics = evaluate_model(test_labels, test_probs, model_name=name,
                                 threshold=best_thresh)
        results.append(metrics)

        save_eval_plots(test_labels, test_probs, name, args.output_dir)

    # === Print results table ===
    results_df = pd.DataFrame(results)
    display_cols = ['model', 'PR-AUC', 'ROC-AUC', 'BSS', 'CSI', 'POD', 'FAR', 'F1',
                    'threshold', 'N_events']
    print(f"\n{'='*80}")
    print(f"  RESULTS: Heavy Precipitation Nowcasting (3h) — Multi-Station ({args.target_station} target)")
    print(f"{'='*80}")
    print(results_df[display_cols].to_string(index=False, float_format='%.3f'))

    best_idx = results_df['PR-AUC'].idxmax()
    print(f"\nBest model by PR-AUC: {results_df.loc[best_idx, 'model']} "
          f"(PR-AUC = {results_df.loc[best_idx, 'PR-AUC']:.3f})")

    # === Save results JSON (convert numpy types for serialization) ===
    class NumpyEncoder(json.JSONEncoder):
        def default(self, obj):
            if isinstance(obj, (np.integer,)): return int(obj)
            if isinstance(obj, (np.floating,)): return float(obj)
            if isinstance(obj, np.ndarray): return obj.tolist()
            return super().default(obj)

    results_path = os.path.join(args.output_dir, 'results.json')
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2, cls=NumpyEncoder)
    print(f"\nSaved results -> {results_path}")


if __name__ == '__main__':
    main()
