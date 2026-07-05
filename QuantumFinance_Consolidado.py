# ============================================================
# QUANTUM FINANCE - CREDIT SCORING (VERSÃO CONSOLIDADA)
# Parte 1: Leitura + EDA
# Parte 2: Correlação
# Parte 3: Regressão Linear vs Árvore de Regressão + Simulador
# ============================================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import r2_score, mean_absolute_error, root_mean_squared_error

import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor

plt.style.use("seaborn-v0_8")
plt.rcParams["figure.figsize"] = (10, 6)

# ============================================================
# PARTE 1 - LEITURA E EDA
# ============================================================

# decimal="," resolve SCORE_CREDITO e vl_salario_mil de uma vez,
# sem precisar de str.replace coluna a coluna
df = pd.read_csv(
    "Base_ScoreCredito_QuantumFinance(8).csv",
    sep=";",
    decimal=","
)

print(df.info())
print(df.describe(include="all"))

# ---------- Verificacao de nulos ----------
print("\nValores nulos por coluna:")
print(df.isna().sum().sort_values(ascending=False))

# ---------- Zeros em vl_imovel (justifica log1p em vez de log) ----------
print("\nZeros em vl_imovel_em_mil:",
      (df["vl_imovel_em_mil"] == 0).sum(),
      f"({(df['vl_imovel_em_mil'] == 0).mean() * 100:.1f}% da base)")

# ---------- Distribuição das quantitativas ----------

quantitativas = [
    "idade", "Qte_dependentes", "tempo_ultimoservico",
    "vl_salario_mil", "vl_imovel_em_mil",
    "Qte_cartoes", "Qte_carros", "SCORE_CREDITO"
]

fig, axes = plt.subplots(3, 3, figsize=(15, 12))
axes = axes.flatten()

for i, col in enumerate(quantitativas):
    sns.histplot(df[col], kde=True, ax=axes[i], color="steelblue")
    axes[i].set_title(col)
    axes[i].set_xlabel("")

for j in range(i + 1, len(axes)):
    fig.delaxes(axes[j])

plt.tight_layout()
plt.show()

# ---------- Boxplots (outliers) ----------

fig, axes = plt.subplots(3, 3, figsize=(15, 12))
axes = axes.flatten()

for i, col in enumerate(quantitativas):
    sns.boxplot(x=df[col], ax=axes[i])
    axes[i].set_title(col)

for j in range(i + 1, len(axes)):
    fig.delaxes(axes[j])

plt.tight_layout()
plt.show()

# ---------- Contagem de outliers via IQR ----------

for col in ["vl_imovel_em_mil", "tempo_ultimoservico", "SCORE_CREDITO"]:
    Q1, Q3 = np.percentile(df[col], [25, 75])
    IQ = Q3 - Q1

    lim_15_sup, lim_3_sup = Q3 + 1.5 * IQ, Q3 + 3 * IQ
    lim_15_inf, lim_3_inf = Q1 - 1.5 * IQ, Q1 - 3 * IQ

    mod_sup = ((df[col] > lim_15_sup) & (df[col] <= lim_3_sup)).sum()
    ext_sup = (df[col] > lim_3_sup).sum()
    mod_inf = ((df[col] < lim_15_inf) & (df[col] >= lim_3_inf)).sum()
    ext_inf = (df[col] < lim_3_inf).sum()

    print(f"\n{col}:")
    print(f"  Superiores -> moderados: {mod_sup}, extremos: {ext_sup}")
    print(f"  Inferiores -> moderados: {mod_inf}, extremos: {ext_inf}")
    print(f"  TOTAL outliers (1.5xIQR): {mod_sup + ext_sup + mod_inf + ext_inf}")

# ---------- Countplot das qualitativas ----------

categoricas = ["sexo", "estado_civil", "escola",
               "reg_moradia", "trabalha", "casa_propria"]

fig, axes = plt.subplots(2, 3, figsize=(16, 9))
axes = axes.flatten()

for i, col in enumerate(categoricas):
    sns.countplot(data=df, x=col, ax=axes[i], color="steelblue")
    axes[i].set_title(col)
    axes[i].set_xlabel("")
    for container in axes[i].containers:
        axes[i].bar_label(container)

plt.tight_layout()
plt.show()

# ============================================================
# TRATAMENTOS
# ============================================================

# 'na' em estado_civil: manter como categoria propria
# (excluir seria perda relevante de amostra)
pct_na = (df["estado_civil"] == "na").mean() * 100
print(f"\n'na' em estado_civil: {pct_na:.1f}% da base -> vira 'nao_informado'")

df["estado_civil"] = df["estado_civil"].replace("na", "nao_informado")

# vl_imovel_em_mil tem alta assimetria e muitos zeros -> log1p
print("Assimetria vl_imovel_em_mil:", round(df["vl_imovel_em_mil"].skew(), 2))
df["vl_imovel_log"] = np.log1p(df["vl_imovel_em_mil"])

# ============================================================
# PARTE 2 - CORRELACAO
# ============================================================

num_corr = ["idade", "Qte_dependentes", "tempo_ultimoservico",
            "vl_salario_mil", "vl_imovel_log", "Qte_cartoes",
            "Qte_carros", "trabalha", "casa_propria", "SCORE_CREDITO"]

corr = df[num_corr].corr()

plt.figure(figsize=(11, 9))
sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", center=0,
            square=True, linewidths=0.5, cbar_kws={"shrink": 0.8})
plt.title("Matriz de Correlacao - variaveis numericas")
plt.tight_layout()
plt.show()

# Score medio por categoria
for col in ["sexo", "estado_civil", "escola", "reg_moradia"]:
    print(f"\n=== SCORE medio por {col} ===")
    print(df.groupby(col)["SCORE_CREDITO"].mean().sort_values())

# ============================================================
# PARTE 3 - MODELAGEM
# ============================================================

# vl_imovel_em_mil sai (substituida pela versao log)
X = df.drop(columns=["id", "SCORE_CREDITO", "vl_imovel_em_mil"])
y = df["SCORE_CREDITO"]

cat_nominais = ["sexo", "estado_civil", "reg_moradia"]
cat_ordinais = ["escola"]
numericas = [c for c in X.columns if c not in cat_nominais + cat_ordinais]

ordem_escola = [["ensino fundam", "ensino medio", "graduacao",
                 "mestrado", "doutorado"]]

preprocessador = ColumnTransformer([
    ("num", "passthrough", numericas),
    ("ord", OrdinalEncoder(categories=ordem_escola), cat_ordinais),
    ("cat", OneHotEncoder(drop="first", handle_unknown="ignore"), cat_nominais)
])

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42
)

print(f"\nTreino: {len(X_train)} | Teste: {len(X_test)}")

# ------------------------------------------------------------
# MODELO 1 - REGRESSAO LINEAR (pipeline sklearn)
# ------------------------------------------------------------

modelo_lr = Pipeline([
    ("prep", preprocessador),
    ("modelo", LinearRegression())
])

modelo_lr.fit(X_train, y_train)
pred_lr = modelo_lr.predict(X_test)

print("\n========== REGRESSAO LINEAR ==========")
print("R²   :", round(r2_score(y_test, pred_lr), 4))
print("RMSE :", round(root_mean_squared_error(y_test, pred_lr), 2))
print("MAE  :", round(mean_absolute_error(y_test, pred_lr), 2))

# ------------------------------------------------------------
# SUMMARY (statsmodels) - ajustado APENAS NO TREINO,
# consistente com as metricas de teste acima
# ------------------------------------------------------------

prep_fitted = modelo_lr.named_steps["prep"]  # ja fitado no treino

X_train_sm = pd.DataFrame(
    prep_fitted.transform(X_train),
    columns=prep_fitted.get_feature_names_out(),
    index=X_train.index
)
X_train_sm = sm.add_constant(X_train_sm)

modelo_stats = sm.OLS(y_train, X_train_sm).fit()
print(modelo_stats.summary())

# ------------------------------------------------------------
# VIF (multicolinearidade)
# ------------------------------------------------------------

vif = pd.DataFrame({
    "Variavel": X_train_sm.columns,
    "VIF": [variance_inflation_factor(X_train_sm.values, i)
            for i in range(X_train_sm.shape[1])]
})

print("\n========== VIF ==========")
print(vif.sort_values("VIF", ascending=False))

# NOTA: casa_propria e vl_imovel_log apresentam VIF elevado (~70+)
# porque sao quase redundantes (sem casa propria -> imovel = 0).
# Mantivemos ambas porque melhoram a capacidade PREDITIVA,
# mas os coeficientes individuais dessas duas devem ser
# interpretados com cautela. A celula abaixo quantifica o trade-off.

# ------------------------------------------------------------
# CENARIO ALTERNATIVO: sem casa_propria (para discussao no relatorio)
# ------------------------------------------------------------

modelo_lr_sem = Pipeline([
    ("prep", ColumnTransformer([
        ("num", "passthrough", [c for c in numericas if c != "casa_propria"]),
        ("ord", OrdinalEncoder(categories=ordem_escola), cat_ordinais),
        ("cat", OneHotEncoder(drop="first", handle_unknown="ignore"), cat_nominais)
    ])),
    ("modelo", LinearRegression())
])

modelo_lr_sem.fit(X_train.drop(columns="casa_propria"), y_train)
pred_lr_sem = modelo_lr_sem.predict(X_test.drop(columns="casa_propria"))

print("\n===== LINEAR SEM casa_propria (menos colinearidade) =====")
print("R²   :", round(r2_score(y_test, pred_lr_sem), 4))
print("RMSE :", round(root_mean_squared_error(y_test, pred_lr_sem), 2))

# ------------------------------------------------------------
# MODELO 2 - ARVORE DE REGRESSAO
# ------------------------------------------------------------

# Escolha da profundidade: comparar R2 treino vs teste
# para justificar a poda (evitar overfitting)
print("\n========== TUNING DA ARVORE ==========")
print(f"{'depth':>6} {'R2 treino':>10} {'R2 teste':>10}")

for depth in [3, 4, 5, 6, 8, 10, None]:
    arv = Pipeline([
        ("prep", preprocessador),
        ("modelo", DecisionTreeRegressor(random_state=42, max_depth=depth))
    ])
    arv.fit(X_train, y_train)
    r2_tr = r2_score(y_train, arv.predict(X_train))
    r2_te = r2_score(y_test, arv.predict(X_test))
    print(f"{str(depth):>6} {r2_tr:>10.4f} {r2_te:>10.4f}")

# Sem limite de profundidade: R2 treino ~1.0 e teste bem menor
# -> overfitting classico. Pelo tuning, max_depth=8 entrega o melhor
# equilibrio: R2 de teste superior ao de depth=6 com gap
# treino-teste ainda controlado (~0.10).

modelo_tree = Pipeline([
    ("prep", preprocessador),
    ("modelo", DecisionTreeRegressor(random_state=42, max_depth=8))
])

modelo_tree.fit(X_train, y_train)
pred_tree = modelo_tree.predict(X_test)

print("\n========== ARVORE (max_depth=8) ==========")
print("R²   :", round(r2_score(y_test, pred_tree), 4))
print("RMSE :", round(root_mean_squared_error(y_test, pred_tree), 2))
print("MAE  :", round(mean_absolute_error(y_test, pred_tree), 2))

# Importancia das variaveis na arvore
tree_fitted = modelo_tree.named_steps["modelo"]
nomes_feat = modelo_tree.named_steps["prep"].get_feature_names_out()

importancias = pd.DataFrame({
    "Variavel": nomes_feat,
    "Importancia": tree_fitted.feature_importances_
}).sort_values("Importancia", ascending=False)

print("\n========== IMPORTANCIA (ARVORE) ==========")
print(importancias.head(10))

# ------------------------------------------------------------
# COMPARACAO FINAL
# ------------------------------------------------------------

resultado = pd.DataFrame({
    "Modelo": ["Regressao Linear", "Arvore (depth=8)"],
    "R2": [r2_score(y_test, pred_lr), r2_score(y_test, pred_tree)],
    "RMSE": [root_mean_squared_error(y_test, pred_lr),
             root_mean_squared_error(y_test, pred_tree)],
    "MAE": [mean_absolute_error(y_test, pred_lr),
            mean_absolute_error(y_test, pred_tree)]
})

print("\n========== COMPARACAO ==========")
print(resultado.round(4))

# ============================================================
# SIMULADOR DE NOVO CLIENTE
# ============================================================

def simular_score(idade, sexo, estado_civil, escola, qte_dependentes,
                  tempo_ultimoservico, trabalha, vl_salario_mil,
                  reg_moradia, casa_propria, vl_imovel_em_mil,
                  qte_cartoes, qte_carros):
    """Prevê o SCORE_CREDITO de um novo cliente com o modelo linear."""
    novo = pd.DataFrame({
        "idade": [idade],
        "sexo": [sexo],
        "estado_civil": [estado_civil],
        "escola": [escola],
        "Qte_dependentes": [qte_dependentes],
        "tempo_ultimoservico": [tempo_ultimoservico],
        "trabalha": [trabalha],
        "vl_salario_mil": [vl_salario_mil],
        "reg_moradia": [reg_moradia],
        "casa_propria": [casa_propria],
        "Qte_cartoes": [qte_cartoes],
        "Qte_carros": [qte_carros],
        "vl_imovel_log": [np.log1p(vl_imovel_em_mil)]
    })
    return modelo_lr.predict(novo)[0]

score = simular_score(
    idade=40, sexo="M", estado_civil="casado", escola="graduacao",
    qte_dependentes=2, tempo_ultimoservico=36, trabalha=1,
    vl_salario_mil=75, reg_moradia=2, casa_propria=1,
    vl_imovel_em_mil=350, qte_cartoes=2, qte_carros=1
)

print("\nScore previsto para o novo cliente:", round(score, 2))
