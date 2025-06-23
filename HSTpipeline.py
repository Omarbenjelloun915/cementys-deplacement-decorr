from typing import Tuple
import sys
import pandas as pd
import numpy as np
import statsmodels.api as sm

class HSTPipeline:
    def load_and_clean(self, path: str) -> pd.DataFrame:
        """
        Lit et nettoie le fichier de données.
        """
        df = pd.read_csv(path, parse_dates=["TIMESTAMP"], na_values=["NAN"])
        df.columns = df.columns.str.lower().str.strip()
        df.rename(columns={"timestamp": "ts"}, inplace=True)
        df.sort_values("ts", inplace=True)
        df.reset_index(drop=True, inplace=True)
        df["deplacement"] = df["deplacement"].interpolate(limit_direction="both")
        df.dropna(subset=["temperature", "ensoleillement"], inplace=True)
        return df

    def feature_engineering(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Ajoute les features HST : temporel, cyclique et polynomiaux.
        """
        df["t"] = (df["ts"] - df["ts"].min()).dt.total_seconds() / 86400
        df["hour"] = df["ts"].dt.hour + df["ts"].dt.minute / 60
        df["sin_hour"] = np.sin(2 * np.pi * df["hour"] / 24)
        df["cos_hour"] = np.cos(2 * np.pi * df["hour"] / 24)
        df["temp2"] = df["temperature"] ** 2
        df["ens2"] = df["ensoleillement"] ** 2
        df["t2"] = df["t"] ** 2
        return df

    def fit_hst(self, df: pd.DataFrame) -> Tuple[sm.regression.linear_model.RegressionResultsWrapper, pd.DataFrame, pd.Series]:
        """
        Ajuste le modèle HST (OLS) et retourne le modèle ajusté, la matrice X et la série y.
        """
        X = df[[
            "temperature", "temp2",
            "ensoleillement", "ens2",
            "sin_hour", "cos_hour",
            "t", "t2"
        ]]
        X = sm.add_constant(X)
        y = df["deplacement"]
        model = sm.OLS(y, X).fit()
        print(model.summary())
        return model, X, y

    def predict_and_save(self, df: pd.DataFrame, model: sm.regression.linear_model.RegressionResultsWrapper,
                         X: pd.DataFrame, y: pd.Series, out_csv: str = "hst_results.csv") -> None:
        """
        Prédit les déplacements, calcule le résidu et enregistre le résultat.
        """
        df["pred_deplacement"] = model.predict(X)
        df["residu"] = y - df["pred_deplacement"]
        df_out = df[["ts", "deplacement", "pred_deplacement", "residu"]]
        df_out.to_csv(out_csv, index=False)
        print(f"Résultats enregistrés dans '{out_csv}'")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python hst_pipeline.py chemin/vers/dataset.dat")
        sys.exit(1)

    pipeline = HSTPipeline()
    df = pipeline.load_and_clean(sys.argv[1])
    df = pipeline.feature_engineering(df)
    model, X, y = pipeline.fit_hst(df)
    pipeline.predict_and_save(df, model, X, y)
