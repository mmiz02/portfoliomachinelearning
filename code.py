import argparse
import json
import os
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer

from sklearn.metrics import (
    mean_absolute_error, mean_squared_error, r2_score,
    silhouette_score, roc_auc_score
)

from sklearn.linear_model import Ridge, LogisticRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA


def make_ohe():
    """OneHotEncoder compatible across sklearn versions."""
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore", sparse=False)


def ensure_outdir(outdir: str) -> str:
    os.makedirs(outdir, exist_ok=True)
    return outdir


def savefig(path: str):
    plt.tight_layout()
    plt.savefig(path, dpi=200, bbox_inches="tight")
    plt.close()


def build_preprocess(cat_cols, num_cols):
    ohe = make_ohe()
    return ColumnTransformer(
        transformers=[
            ("num", Pipeline(steps=[
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler())
            ]), num_cols),
            ("cat", Pipeline(steps=[
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("onehot", ohe)
            ]), cat_cols),
        ],
        remainder="drop"
    )


def basic_clean(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    df = df[df["price"].notna()]
    df = df[df["price"] > 0]
    # Winsorise price to reduce the impact of extreme outliers (use 1st–99th percentiles).
    p_low, p_high = df["price"].quantile([0.01, 0.99])
    df["price_w"] = df["price"].clip(lower=p_low, upper=p_high)

    if "reviews_per_month" not in df.columns:
        df["reviews_per_month"] = 0
    df["reviews_per_month"] = df["reviews_per_month"].fillna(0)

    if "minimum_nights" in df.columns:
        mn_cap = df["minimum_nights"].quantile(0.99)
        df["minimum_nights_c"] = df["minimum_nights"].clip(upper=mn_cap)
    else:
        df["minimum_nights_c"] = 0

    df["log_price"] = np.log1p(df["price_w"])
    return df


def eda_plots(df: pd.DataFrame, outdir: str):
    pivot = df.pivot_table(
        index="neighbourhood_group",
        columns="room_type",
        values="price_w",
        aggfunc="median"
    )
    pivot.to_csv(
        os.path.join(outdir, "eda_median_price_pivot.csv"), 
        float_format="%.8f", 
        index=True
        )

    pivot.plot(kind="bar", figsize=(10, 5))
    plt.title("Median nightly price by neighbourhood_group and room_type (winsorised)")
    plt.ylabel("Median price ($)")
    savefig(os.path.join(outdir, "eda_median_price_bar.png"))

    tmp = df.copy()
    tmp["price_bin"] = pd.qcut(tmp["price_w"], q=20, duplicates="drop")
    demand_by_bin = tmp.groupby("price_bin", observed=False)["reviews_per_month"].mean()

    plt.figure(figsize=(10, 4))
    plt.plot(range(len(demand_by_bin)), demand_by_bin.values, marker="o")
    plt.title("Average reviews_per_month across price bins (proxy for demand)")
    plt.xlabel("Price bin (low → high)")
    plt.ylabel("Mean reviews_per_month")
    savefig(os.path.join(outdir, "eda_demand_vs_price_bins.png"))


def regression_price(df: pd.DataFrame, outdir: str):
    features = [
        "neighbourhood_group", "room_type",
        "latitude", "longitude",
        "minimum_nights_c", "availability_365",
        "calculated_host_listings_count",
        "number_of_reviews", "reviews_per_month",
    ]
    features = [c for c in features if c in df.columns]

    X = df[features].copy()
    y = df["log_price"].copy()

    cat_cols = [c for c in X.columns if X[c].dtype == "object"]
    num_cols = [c for c in X.columns if c not in cat_cols]
    preprocess = build_preprocess(cat_cols, num_cols)

    models = {
        "Ridge": Ridge(alpha=1.0),
        "RandomForest": RandomForestRegressor(n_estimators=300, random_state=42, n_jobs=-1),
    }

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    results = []
    fitted = {}

    for name, model in models.items():
        pipe = Pipeline(steps=[("prep", preprocess), ("model", model)])
        pipe.fit(X_train, y_train)

        pred = pipe.predict(X_test)
        rmse = float(np.sqrt(mean_squared_error(y_test, pred)))
        mae = float(mean_absolute_error(y_test, pred))
        r2 = float(r2_score(y_test, pred))

        results.append({"model": name, "rmse": rmse, "mae": mae, "r2": r2})
        fitted[name] = pipe

    res_df = pd.DataFrame(results).sort_values("rmse")
    res_df.to_csv(os.path.join(outdir, "regression_results.csv"), index=False, float_format="%.8f")

    best_name = res_df.iloc[0]["model"]
    best_pipe = fitted[best_name]
    prep = best_pipe.named_steps["prep"]
    model = best_pipe.named_steps["model"]

    feature_names = []
    feature_names.extend(num_cols)
    if cat_cols:
        ohe = prep.named_transformers_["cat"].named_steps["onehot"]
        feature_names.extend(list(ohe.get_feature_names_out(cat_cols)))
    feature_names = np.array(feature_names, dtype=object)

    if hasattr(model, "feature_importances_"):
        imp_df = pd.DataFrame({
            "feature": feature_names,
            "importance": model.feature_importances_
        }).sort_values("importance", ascending=False).head(20)

        plt.figure(figsize=(10, 5))
        plt.barh(imp_df["feature"][::-1], imp_df["importance"][::-1])
        plt.title(f"Top 20 drivers of price (model: {best_name})")
        plt.xlabel("Importance")
        savefig(os.path.join(outdir, "regression_feature_importance_top20.png"))
        imp_df.to_csv(os.path.join(outdir, "regression_feature_importance_top20.csv"), index=False, float_format="%.8f")

    return {
        "best_model": best_name,
        "metrics": res_df.to_dict(orient="records"),
        "features_used": features,
    }


def clustering_listings(df: pd.DataFrame, outdir: str):
    cluster_features = [
        "neighbourhood_group", "room_type",
        "latitude", "longitude",
        "price_w",
        "minimum_nights_c",
        "availability_365",
        "calculated_host_listings_count",
        "reviews_per_month"
    ]
    cluster_features = [c for c in cluster_features if c in df.columns]
    Xc = df[cluster_features].copy()

    cat_cols = [c for c in Xc.columns if Xc[c].dtype == "object"]
    num_cols = [c for c in Xc.columns if c not in cat_cols]
    preprocess = build_preprocess(cat_cols, num_cols)

    Xc_mat = preprocess.fit_transform(Xc)

    rng = np.random.RandomState(42)
    sample_size = min(5000, Xc_mat.shape[0])
    sample_idx = rng.choice(np.arange(Xc_mat.shape[0]), size=sample_size, replace=False)
    X_sample = Xc_mat[sample_idx]

    k_candidates = list(range(3, 9))
    sil_scores = []
    for k in k_candidates:
        km = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = km.fit_predict(X_sample)
        sil_scores.append(float(silhouette_score(X_sample, labels)))

    # Save silhouette-by-k so k selection is auditable for the report.
    sil_df = pd.DataFrame({
        "k": k_candidates,
        "silhouette_score": sil_scores
    })
    sil_df.to_csv(
        os.path.join(outdir, "clustering_silhouette_scores.csv"),
        index=False,
        float_format="%.6f"
    )

    best_k = int(k_candidates[int(np.argmax(sil_scores))])

    kmeans = KMeans(n_clusters=best_k, random_state=42, n_init=10)
    df_out = df.copy()
    df_out["cluster"] = kmeans.fit_predict(Xc_mat)

    profile = df_out.groupby("cluster").agg(
        listings=("id", "count") if "id" in df_out.columns else ("price_w", "count"),
        median_price=("price_w", "median"),
        mean_reviews_pm=("reviews_per_month", "mean"),
        median_availability=("availability_365", "median") if "availability_365" in df_out.columns else ("price_w", "median"),
        median_min_nights=("minimum_nights_c", "median"),
    ).reset_index().sort_values("listings", ascending=False)

    profile.to_csv(os.path.join(outdir, "cluster_profiles.csv"), index=False, float_format="%.8f")

    pca = PCA(n_components=2, random_state=42, svd_solver="randomized")
    X2 = pca.fit_transform(Xc_mat)

    plt.figure(figsize=(8, 6))
    plt.scatter(X2[:, 0], X2[:, 1], c=df_out["cluster"], s=8)
    plt.title("Listing clusters (PCA projection)")
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    savefig(os.path.join(outdir, "clustering_pca_scatter.png"))

    return df_out, {
        "k_candidates": k_candidates,
        "silhouette_scores": sil_scores,
        "best_k": best_k,
        "cluster_features_used": cluster_features,
    }


def build_high_demand_label(df: pd.DataFrame, seg_cols=("neighbourhood_group", "room_type")):
    """Define high_demand as above-median reviews_per_month within each neighbourhood_group × room_type segment."""
    df = df.copy()
    df["_seg_key"] = df[list(seg_cols)].astype(str).agg(" | ".join, axis=1)

    df["high_demand"] = 0
    for _, g in df.groupby("_seg_key"):
        med = g["reviews_per_month"].median()
        df.loc[g.index, "high_demand"] = (g["reviews_per_month"] > med).astype(int)

    return df.drop(columns=["_seg_key"])


def fit_demand_model(df: pd.DataFrame, outdir: str):
    """Train LogisticRegression to estimate P(high_demand) using structured listing features."""
    demand_features = [
        "price_w",
        "minimum_nights_c",
        "availability_365",
        "calculated_host_listings_count",
        "number_of_reviews",
        "neighbourhood_group",
        "room_type",
    ]
    demand_features = [c for c in demand_features if c in df.columns]

    X = df[demand_features].copy()
    y = df["high_demand"].copy()

    cat_cols = [c for c in X.columns if X[c].dtype == "object"]
    num_cols = [c for c in X.columns if c not in cat_cols]
    preprocess = build_preprocess(cat_cols, num_cols)

    model = LogisticRegression(max_iter=400)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    pipe = Pipeline(steps=[("prep", preprocess), ("model", model)])
    pipe.fit(X_train, y_train)

    proba = pipe.predict_proba(X_test)[:, 1]
    auc = float(roc_auc_score(y_test, proba))

    with open(os.path.join(outdir, "demand_model_eval.json"), "w", encoding="utf-8") as f:
        json.dump({"roc_auc": auc, "features_used": demand_features}, f, indent=2)

    return pipe, {"roc_auc": auc, "features_used": demand_features}


def recommend_price_ranges(
    df: pd.DataFrame,
    pipe: Pipeline,
    outdir: str,
    seg_cols=("neighbourhood_group", "room_type"),
    price_col="price_w",
    min_segment_size=200,
    booking_prob_tolerance=0.95,
    ev_tolerance=0.95,
):
    """Grid-search price (p10–p90) per segment to propose a price range and compromise price using EV proxy."""
    demand_features = getattr(pipe, "feature_names_in_", None)
    if demand_features is not None:
        demand_features = demand_features.tolist()
    else:
        demand_features = [
            "price_w", "minimum_nights_c", "availability_365",
            "calculated_host_listings_count", "number_of_reviews",
            "neighbourhood_group", "room_type"
        ]
        demand_features = [c for c in demand_features if c in df.columns]

    out_rows = []

    for seg_vals, g in df.groupby(list(seg_cols)):
        # Skip small segments to avoid unstable recommendations (min_segment_size).
        if len(g) < min_segment_size:
            continue

        p10, p90 = g[price_col].quantile([0.10, 0.90])
        if not np.isfinite(p10) or not np.isfinite(p90) or p90 <= p10:
            continue
        # Search candidate prices within typical segment range (p10–p90), avoid extreme tails.
        grid = np.linspace(float(p10), float(p90), 60)
        seg_dict = dict(zip(seg_cols, seg_vals))

        base = {}
        for col in demand_features:
            if col in seg_dict:
                base[col] = seg_dict[col]
            elif col == price_col:
                base[col] = None
            else:
                if col in g.columns and np.issubdtype(g[col].dtype, np.number):
                    base[col] = float(g[col].median())
                elif col in g.columns:
                    base[col] = g[col].mode().iloc[0]
                else:
                    base[col] = 0

        grid_df = pd.DataFrame([{**base, price_col: p} for p in grid])

        p_high = pipe.predict_proba(grid_df)[:, 1]
        ev = grid * p_high

        best_booking_idx = int(np.argmax(p_high))
        best_booking_price = float(grid[best_booking_idx])
        best_P = float(p_high[best_booking_idx])

        best_earnings_idx = int(np.argmax(ev))
        best_earnings_price = float(grid[best_earnings_idx])
        best_earnings_ev = float(ev[best_earnings_idx])

        # Compromise: keep booking probability close to best, then maximise EV = price × P(high_demand).
        mask = p_high >= booking_prob_tolerance * best_P
        if np.any(mask):
            idxs = np.where(mask)[0]
            compromise_idx = int(idxs[np.argmax(ev[mask])])
        else:
            compromise_idx = best_earnings_idx

        compromise_price = float(grid[compromise_idx])
        compromise_P = float(p_high[compromise_idx])
        compromise_ev = float(ev[compromise_idx])

        range_mask = ev >= ev_tolerance * compromise_ev
        if np.any(range_mask):
            good_prices = grid[range_mask]
            range_low = float(good_prices.min())
            range_high = float(good_prices.max())
        else:
            range_low = compromise_price
            range_high = compromise_price

        out_rows.append({
            "neighbourhood_group": seg_dict.get("neighbourhood_group"),
            "room_type": seg_dict.get("room_type"),
            "segment_n": int(len(g)),
            "p10": float(p10),
            "p90": float(p90),
            "best_booking_price": round(best_booking_price, 2),
            "best_booking_prob": round(best_P, 4),
            "best_earnings_price": round(best_earnings_price, 2),
            "best_earnings_ev_proxy": round(best_earnings_ev, 4),
            "recommended_price_compromise": round(compromise_price, 2),
            "compromise_booking_prob": round(compromise_P, 4),
            "compromise_ev_proxy": round(compromise_ev, 4),
            "recommended_range_low": round(range_low, 2),
            "recommended_range_high": round(range_high, 2),
        })

    reco = pd.DataFrame(out_rows).sort_values(["neighbourhood_group", "room_type"])
    reco.to_csv(os.path.join(outdir, "segment_price_recommendations.csv"), index=False, float_format="%.8f")

    if not reco.empty:
        plt.figure(figsize=(10, 5))
        plt.scatter(range(len(reco)), reco["recommended_price_compromise"])
        plt.title("Recommended compromise price by neighbourhood_group × room_type")
        plt.xlabel("Segment index")
        plt.ylabel("Recommended price ($)")
        savefig(os.path.join(outdir, "recommended_prices_scatter.png"))

    return reco

def export_segment_counts(df: pd.DataFrame, outdir: str, seg_cols=("neighbourhood_group", "room_type")):
    """Export counts per neighbourhood_group × room_type for eligibility and exclusions."""
    seg_cols = [c for c in seg_cols if c in df.columns]
    if len(seg_cols) != 2:
        return  # silently skip if expected columns are missing

    counts = (
        df.groupby(seg_cols)
          .size()
          .reset_index(name="segment_n")
          .sort_values(seg_cols)
    )
    counts.to_csv(
        os.path.join(outdir, "segment_counts.csv"),
        index=False,
        float_format="%.0f"
    )

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", type=str, required=True, help="Path to AB_NYC_2019.csv")
    parser.add_argument("--outdir", type=str, default="outputs_track1", help="Output directory")
    parser.add_argument("--min_segment_size", type=int, default=200, help="Min rows per segment")
    parser.add_argument("--booking_prob_tolerance", type=float, default=0.95,
                        help="Compromise keeps P >= tol × best_P")
    parser.add_argument("--ev_tolerance", type=float, default=0.95,
                        help="Range keeps EV >= tol × EV(compromise)")
    args = parser.parse_args()

    outdir = ensure_outdir(args.outdir)

    df = pd.read_csv(args.csv)
    
    # Pipeline overview:
    # 1) basic_clean: filter invalid prices, winsorise price (price_w), create log_price
    # 2) export_segment_counts: segment sizes for eligibility/exclusions (min_segment_size)
    # 3) eda_plots: executive visuals for price levels and demand proxy
    # 4) regression_price: compare Ridge vs RandomForest on log_price, export metrics + top drivers
    # 5) clustering_listings: choose k via silhouette, fit KMeans, export cluster profiles + PCA plot
    # 6) build_high_demand_label + fit_demand_model: demand proxy model P(high_demand)
    # 7) recommend_price_ranges: price grid search per segment, export final recommendation table

    df = basic_clean(df)

    export_segment_counts(df, outdir)

    eda_plots(df, outdir)
    reg_summary = regression_price(df, outdir)
    df, cluster_summary = clustering_listings(df, outdir)

    df = build_high_demand_label(df)
    demand_pipe, demand_summary = fit_demand_model(df, outdir)

    reco = recommend_price_ranges(
        df=df,
        pipe=demand_pipe,
        outdir=outdir,
        min_segment_size=args.min_segment_size,
        booking_prob_tolerance=args.booking_prob_tolerance,
        ev_tolerance=args.ev_tolerance
    )

    run_summary = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "rows_after_cleaning": int(df.shape[0]),
        "regression": reg_summary,
        "clustering": cluster_summary,
        "demand_model": demand_summary,
        "recommendations_rows": int(reco.shape[0]),
        "assumptions": {
            "booking_probability_proxy": "P(high_demand) using reviews_per_month thresholded within each neighbourhood_group × room_type segment",
            "host_earnings_proxy": "price × P(high_demand)"
        }
    }
    with open(os.path.join(outdir, "run_summary.json"), "w", encoding="utf-8") as f:
        json.dump(run_summary, f, indent=2)

    print("\nDONE.")
    print(f"Outputs saved to: {outdir}")
    print(f"Main business output: {os.path.join(outdir, 'segment_price_recommendations.csv')}")


if __name__ == "__main__":
    main()