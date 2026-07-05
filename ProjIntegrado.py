# ============================================================
# QUANTUM FINANCE - CREDIT SCORING
# Regressão Linear + Árvore de Regressão
# ============================================================

import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder
from sklearn.impute import SimpleImputer

from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor

from sklearn.metrics import (
    r2_score,
    mean_absolute_error,
    root_mean_squared_error
)

import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor

# ============================================================
# Leitura da base
# ============================================================

df = pd.read_csv(
    "Base_ScoreCredito_QuantumFinance(8).csv",
    sep=";",
    decimal=","
)

# ============================================================
# Tratamentos
# ============================================================

df["estado_civil"] = df["estado_civil"].replace("na", "nao_informado")

ordem_escola = [[
    "ensino fundam",
    "ensino medio",
    "graduacao",
    "mestrado",
    "doutorado"
]]

df["vl_imovel_log"] = np.log1p(df["vl_imovel_em_mil"])

X = df.drop(
    columns=[
        "id",
        "SCORE_CREDITO",
        "vl_imovel_em_mil"
    ]
)

y = df["SCORE_CREDITO"]

cat_nominais = [
    "sexo",
    "estado_civil",
    "reg_moradia"
]

cat_ordinais = [
    "escola"
]

numericas = [
    c for c in X.columns
    if c not in cat_nominais + cat_ordinais
]

# ============================================================
# Pré-processamento
# ============================================================

preprocessador = ColumnTransformer([

    (
        "num",
        Pipeline([
            ("imputer", SimpleImputer(strategy="median"))
        ]),
        numericas
    ),

    (
        "ord",
        Pipeline([
            ("encoder", OrdinalEncoder(categories=ordem_escola))
        ]),
        cat_ordinais
    ),

    (
        "cat",
        Pipeline([
            ("encoder", OneHotEncoder(drop="first"))
        ]),
        cat_nominais
    )

])

# ============================================================
# Treino/Teste
# ============================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42
)

# ============================================================
# REGRESSÃO LINEAR
# ============================================================

modelo_lr = Pipeline([

    ("prep", preprocessador),
    ("modelo", LinearRegression())

])

modelo_lr.fit(X_train, y_train)

pred_lr = modelo_lr.predict(X_test)

print("\n========== REGRESSÃO LINEAR ==========")

print("R²   :", round(r2_score(y_test, pred_lr),4))
print("RMSE :", round(root_mean_squared_error(y_test, pred_lr),2))
print("MAE  :", round(mean_absolute_error(y_test, pred_lr),2))

# ============================================================
# SUMMARY (coeficientes e p-valores)
# ============================================================

X_stats = preprocessador.fit_transform(X)

nomes = preprocessador.get_feature_names_out()

X_stats = pd.DataFrame(
    X_stats,
    columns=nomes
)

X_stats = sm.add_constant(X_stats)

modelo_stats = sm.OLS(
    y,
    X_stats
).fit()

print(modelo_stats.summary())

# ============================================================
# VIF
# ============================================================

vif = pd.DataFrame()

vif["Variavel"] = X_stats.columns

vif["VIF"] = [
    variance_inflation_factor(
        X_stats.values,
        i
    )
    for i in range(X_stats.shape[1])
]

print(vif.sort_values("VIF", ascending=False))

# ============================================================
# ÁRVORE DE REGRESSÃO
# ============================================================

modelo_tree = Pipeline([

    ("prep", preprocessador),

    ("modelo",
     DecisionTreeRegressor(
         random_state=42,
         max_depth=6
     ))

])

modelo_tree.fit(X_train, y_train)

pred_tree = modelo_tree.predict(X_test)

print("\n========== ÁRVORE ==========")

print("R²   :", round(r2_score(y_test, pred_tree),4))
print("RMSE :", round(root_mean_squared_error(y_test, pred_tree),2))
print("MAE  :", round(mean_absolute_error(y_test, pred_tree),2))

# ============================================================
# Comparação dos modelos
# ============================================================

resultado = pd.DataFrame({

    "Modelo": [
        "Regressão Linear",
        "Árvore"
    ],

    "R2": [
        r2_score(y_test, pred_lr),
        r2_score(y_test, pred_tree)
    ],

    "RMSE": [
        root_mean_squared_error(y_test, pred_lr),
        root_mean_squared_error(y_test, pred_tree)
    ],

    "MAE": [
        mean_absolute_error(y_test, pred_lr),
        mean_absolute_error(y_test, pred_tree)
    ]

})

print("\n========== COMPARAÇÃO ==========")

print(resultado)

# ============================================================
# Simulador
# ============================================================

novo_cliente = pd.DataFrame({

    "idade":[40],
    "sexo":["M"],
    "estado_civil":["casado"],
    "escola":["graduacao"],
    "Qte_dependentes":[2],
    "tempo_ultimoservico":[36],
    "trabalha":[1],
    "vl_salario_mil":[75],
    "reg_moradia":[2],
    "casa_propria":[1],
    "Qte_cartoes":[2],
    "Qte_carros":[1],
    "vl_imovel_log":[np.log1p(350)]

})

score_previsto = modelo_lr.predict(novo_cliente)

print("\nScore previsto:", round(score_previsto[0],2))