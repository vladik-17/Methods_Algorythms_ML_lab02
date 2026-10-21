"""Лабораторная №2, вариант 12. Пока реализована только часть 1."""

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from preprocessing import GoTPreprocessor


SEED = 12
ROOT = Path(__file__).resolve().parent


def main():
    results = ROOT / "results"
    results.mkdir(exist_ok=True)
    data = pd.read_csv(ROOT / "data" / "game_of_thrones_train.csv")
    external_test = pd.read_csv(ROOT / "data" / "game_of_thrones_test.csv")
    if data["S.No"].duplicated().any() or data["isAlive"].isna().any():
        raise ValueError("Проверьте идентификаторы и целевую переменную.")

    # Стратификация сохраняет приблизительную долю классов в каждой части.
    train, holdout = train_test_split(
        data, test_size=0.30, random_state=SEED, stratify=data["isAlive"]
    )
    val, test = train_test_split(
        holdout, test_size=0.50, random_state=SEED, stratify=holdout["isAlive"]
    )
    processor = GoTPreprocessor().fit(train)
    splits = {"train": train, "val": val, "test": test, "external_test": external_test}
    summary = {"variant": 12, "seed": SEED, "features": processor.feature_names_,
               "top_houses": processor.top_houses_, "splits": {}}

    for name, frame in splits.items():
        features = processor.transform(frame)
        # Идентификаторы и имена - для сопоставления строк, они не входят в X.
        output = frame[["S.No", "name"]].join(features)
        if "isAlive" in frame:
            output["isAlive"] = frame["isAlive"]
        output.to_csv(results / f"prepared_{name}.csv", index=False)
        summary["splits"][name] = {"rows": len(frame), "feature_count": features.shape[1]}
        if "isAlive" in frame:
            summary["splits"][name]["alive_fraction"] = float(frame["isAlive"].mean())

    summary["scaler"] = {
        "columns": processor.scaler_.feature_names_in_.tolist(),
        "mean": processor.scaler_.mean_.tolist(),
        "scale": processor.scaler_.scale_.tolist(),
    }
    (results / "preprocessing_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    missing = pd.DataFrame({
        "train_missing": data.isna().sum(),
        "train_missing_percent": data.isna().mean() * 100,
        "external_test_missing": external_test.isna().sum(),
        "external_test_missing_percent": external_test.isna().mean() * 100,
    })
    missing.to_csv(results / "missing_values.csv", index_label="column")

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    data["isAlive"].value_counts().reindex([0, 1]).plot.bar(ax=axes[0], color=["tomato", "seagreen"])
    axes[0].set(title="Баланс классов", xlabel="isAlive: 0 — погиб, 1 — жив", ylabel="Число персонажей")
    axes[1].hist(data["popularity"], bins=50, label="Исходный признак")
    axes[1].set(title="Популярность до преобразования", xlabel="popularity", ylabel="Число персонажей")
    axes[1].legend()
    axes[2].hist(np.log10(data["popularity"] * 200 + 1), bins=50, label="Вариант 12")
    axes[2].set(title="Популярность после преобразования", xlabel="log10(popularity × 200 + 1)", ylabel="Число персонажей")
    axes[2].legend()
    fig.tight_layout()
    fig.savefig(results / "data_overview.png", dpi=160)
    plt.close(fig)

    print("Часть 1 выполнена. Вариант 12, seed=12.")
    for name, info in summary["splits"].items():
        print(f"{name}: {info['rows']} строк, {info['feature_count']} признаков")
    print("Результаты:", results)


if __name__ == "__main__":
    main()

# Вывод по части 1: пропуски сохранены через отдельные индикаторы, culture
# сгруппирована по материалам задания, house представлена top-10 + other.
# Числовые признаки стандартизируются по train. Качество классификации
# пока не оценивалось; итоговый вывод будет дополнен после остальных частей.
