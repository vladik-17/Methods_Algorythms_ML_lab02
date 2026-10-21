"""Подготовка данных GoT для варианта 12."""

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler


CULTURES_GROUPED = {
    "Old Nations": ["valyrian", "first men", "andal", "andals", "rhoynar"],
    "the North": ["northmen", "northern mountain clans", "crannogmen"],
    "the Iron Islands": ["ironborn", "ironmen"],
    "the Mountain and the Vale": ["valemen", "vale", "vale mountain clans", "sistermen"],
    "the Isles and Rivers": ["riverlands", "rivermen"],
    "the Rock": ["westerman", "westermen", "westerlands"],
    "the Stormlands": ["stormlander", "stormlands"],
    "the Reach": ["reach", "reachmen", "the reach"],
    "Dorne": ["dornish", "dornishmen", "dorne"],
    "Essos Nations": [
        "astapor", "astapori", "braavosi", "braavos", "tyroshi", "lysene", "lyseni",
        "myrish", "pentoshi", "qartheen", "qarth", "dothraki", "lhazarene",
        "lhazareen", "meereen", "meereenese", "norvoshi", "qohor", "summer isles",
        "summer islands", "summer islander", "asshai", "asshai'i", "norvos", "ghiscari",
        "ghiscaricari",
    ],
    "Other Nations": ["ibbenese", "westeros", "free folk", "wildling", "wildlings", "naathi"],
}

BINARY_COLUMNS = ["male", "book1", "book2", "book3", "book4", "book5", "isMarried", "isNoble"]
RELATIVE_COLUMNS = ["isAliveMother", "isAliveFather", "isAliveHeir", "isAliveSpouse"]
NUMERIC_COLUMNS = ["age_value", "numDeadRelations", "popularity_log"]


def clean_data(data):
    """Исправления из вспомогательного ноутбука; исходный DataFrame не меняется."""
    data = data.copy()
    for character_id, name, birth_year in [(1685, "Doreah", 278), (1869, "Rhaego", 299)]:
        mask = (data["S.No"] == character_id) & (data["name"] == name)
        data.loc[mask, ["dateOfBirth", "age"]] = [birth_year, 0]
    # Неизвестные ошибки возраста считаем пропусками, а не отрицательным возрастом.
    data.loc[data["age"] < 0, "age"] = np.nan
    return data


class GoTPreprocessor:
    """fit запоминает правила по train; transform применяет их к любой выборке."""

    def fit(self, train):
        train = clean_data(train)
        # Пропуски сохраняем отдельной категорией, они не конкурируют с домами за top-10.
        counts = train["house"].dropna().value_counts()
        self.top_houses_ = sorted(counts.index, key=lambda house: (-counts[house], house))[:10]
        features = self._make_features(train)
        self.feature_names_ = features.columns.tolist()
        self.scaler_ = StandardScaler().fit(features[NUMERIC_COLUMNS])
        return self

    def transform(self, data):
        if not hasattr(self, "scaler_"):
            raise RuntimeError("Сначала вызовите fit на тренировочной выборке.")
        features = self._make_features(clean_data(data))
        features = features.reindex(columns=self.feature_names_, fill_value=0)
        features[NUMERIC_COLUMNS] = self.scaler_.transform(features[NUMERIC_COLUMNS])
        if not np.isfinite(features.to_numpy()).all():
            raise ValueError("После обработки остались NaN или бесконечные значения.")
        return features

    def _make_features(self, data):
        features = data[BINARY_COLUMNS].astype(float).copy()
        features["age_value"] = data["age"].fillna(0)
        features["age_no_data"] = data["age"].isna().astype(int)
        features["numDeadRelations"] = data["numDeadRelations"].fillna(0)
        features["popularity_log"] = np.log10(data["popularity"].fillna(0) * 200 + 1)

        for column in RELATIVE_COLUMNS:
            features[column] = data[column].fillna(0)
            features[column + "_no_data"] = data[column].isna().astype(int)
        features["has_title"] = data["title"].notna().astype(int)

        mapping = {value: group for group, values in CULTURES_GROUPED.items() for value in values}
        culture = data["culture"].str.strip().str.lower().map(mapping)
        culture = culture.fillna("Other Nations").where(data["culture"].notna(), "culture_no_data")
        # Список групп задан материалами лабы, а не извлечён из val/test.
        for group in [*CULTURES_GROUPED, "culture_no_data"]:
            features["culture_" + group] = (culture == group).astype(int)

        house = data["house"].where(data["house"].isin(self.top_houses_), "other")
        house = house.where(data["house"].notna(), "house_no_data")
        for group in [*self.top_houses_, "other", "house_no_data"]:
            features["house_" + group] = (house == group).astype(int)

        # S.No и name сохраняются отдельно. Имена родственников слишком разрежены.
        # dateOfBirth не используем вместе с age: признаки во многом дублируются.
        return features.astype(float)
