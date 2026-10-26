"""Re-creates the key EDA plots from data pulled out of PostgreSQL."""
import matplotlib
matplotlib.use("Agg")  # no display needed (works inside Docker)
import matplotlib.pyplot as plt
import seaborn as sns

from config import FEATURES, FIG_DIR, TARGET
from data_loader import load_risk_groups


def main() -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    df = load_risk_groups()

    # Correlation heatmap (pandas skips NULLs pairwise)
    plt.figure(figsize=(9, 7))
    sns.heatmap(df[FEATURES + [TARGET]].corr(), annot=True, fmt=".2f", cmap="coolwarm")
    plt.title("Feature Correlation (cleaned data from PostgreSQL)")
    plt.tight_layout()
    plt.savefig(FIG_DIR / "correlation_heatmap.png", dpi=300)
    plt.close()

    # Diabetes rate by BMI category
    order = ["Underweight", "Normal", "Overweight", "Obese", "Unknown"]
    rates = (df.groupby("bmi_category")[TARGET].mean() * 100).reindex(order).dropna()
    plt.figure(figsize=(7, 5))
    sns.barplot(x=rates.index, y=rates.values, color="steelblue")
    plt.ylabel("Diabetes rate (%)")
    plt.title("Diabetes Rate by WHO BMI Category")
    plt.tight_layout()
    plt.savefig(FIG_DIR / "bmi_risk_groups.png", dpi=300)
    plt.close()

    # Glucose distribution by outcome
    plt.figure(figsize=(7, 5))
    sns.boxplot(data=df, x=TARGET, y="glucose")
    plt.title("Glucose by Outcome")
    plt.tight_layout()
    plt.savefig(FIG_DIR / "glucose_boxplot.png", dpi=300)
    plt.close()
    print(f"Saved figures to {FIG_DIR}")


if __name__ == "__main__":
    main()
