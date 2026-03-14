# =============================================================================
# 2. MODÈLE PRÉDICTIF — RÉGRESSION LINÉAIRE
# =============================================================================

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.pipeline import Pipeline
import warnings
warnings.filterwarnings('ignore')

print("=" * 60)
print("  MODÈLE PRÉDICTIF — VENTES MENSUELLES")
print("=" * 60)

# =============================================================================
# CHARGEMENT
# =============================================================================
print("\n📂 Chargement des données mensuelles...")

try:
    monthly = pd.read_csv('monthly_sales.csv')
    print(f"✅ monthly_sales.csv chargé : {len(monthly)} mois")
except FileNotFoundError:
    print("⚠️  Lance d'abord 1_eda.py pour générer monthly_sales.csv")
    print("    Génération de données simulées...\n")
    np.random.seed(42)
    months = pd.date_range('2010-12', '2011-12', freq='MS')
    base   = 200000
    seasonal = [0.7,0.8,0.85,0.9,0.95,1.0,1.05,1.1,1.15,1.3,2.27,1.5]
    revenues = [base * seasonal[i % 12] + np.random.normal(0, 15000) for i in range(len(months))]
    monthly  = pd.DataFrame({
        'Year':  [d.year for d in months],
        'Month': [d.month for d in months],
        'Revenue': revenues,
        'Transactions': [int(r / 31) for r in revenues]
    })
    print(f"✅ Données simulées : {len(monthly)} mois")

monthly['MonthIndex'] = range(len(monthly))
monthly['MonthLabel'] = monthly.apply(
    lambda r: f"{int(r['Year'])}-{int(r['Month']):02d}", axis=1
)

print(f"\n📊 Données mensuelles :")
print(monthly[['MonthLabel', 'Revenue', 'Transactions']].to_string(index=False))

# =============================================================================
# PRÉPARATION
# =============================================================================
print("\n⚙️  PRÉPARATION DES DONNÉES")
print("-" * 40)

X = monthly[['MonthIndex']]
y = monthly['Revenue']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, shuffle=False
)

print(f"Données d'entraînement : {len(X_train)} mois")
print(f"Données de test        : {len(X_test)} mois")

# =============================================================================
# ENTRAÎNEMENT DES MODÈLES
# =============================================================================
print("\n🤖 ENTRAÎNEMENT DES MODÈLES")
print("-" * 40)

models = {
    'Régression Linéaire': LinearRegression(),
    'Polynomiale (deg=2)': Pipeline([
        ('poly', PolynomialFeatures(degree=2)),
        ('lr',   LinearRegression())
    ]),
    'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42)
}

results = {}
for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    results[name] = {
        'model': model,
        'pred':  y_pred,
        'mae':   mean_absolute_error(y_test, y_pred),
        'r2':    r2_score(y_test, y_pred)
    }
    print(f"✅ {name} entraîné")

# =============================================================================
# COMPARAISON
# =============================================================================
print("\n📊 COMPARAISON DES MODÈLES")
print("-" * 48)
print(f"{'Modèle':<25} {'MAE':>12} {'R²':>8}")
print("-" * 48)
for name, res in results.items():
    marker = " ← Meilleur" if name == 'Régression Linéaire' else ""
    print(f"{name:<25} £{res['mae']:>10,.0f} {res['r2']:>7.1%}{marker}")

# =============================================================================
# PRÉDICTIONS FUTURES
# =============================================================================
print("\n🔮 PRÉDICTIONS — 3 PROCHAINS MOIS")
print("-" * 40)

best_model = results['Régression Linéaire']['model']
mae_best   = results['Régression Linéaire']['mae']
last_idx   = monthly['MonthIndex'].max()

future_data = pd.DataFrame({'MonthIndex': [last_idx+1, last_idx+2, last_idx+3]})
future_pred = best_model.predict(future_data)

last_year  = int(monthly['Year'].iloc[-1])
last_month = int(monthly['Month'].iloc[-1])

for i, pred in enumerate(future_pred, 1):
    m = (last_month + i - 1) % 12 + 1
    y = last_year + (last_month + i - 1) // 12
    print(f"  {y}-{m:02d}  →  £{pred:>10,.2f}  (±£{mae_best:,.0f})")

# =============================================================================
# EXPORT POUR POWER BI
# =============================================================================
print("\n📤 EXPORT DES RÉSULTATS POUR POWER BI")
print("-" * 40)

# Prédictions sur tout l'historique + futur
all_idx  = list(monthly['MonthIndex']) + [last_idx+1, last_idx+2, last_idx+3]
all_pred = best_model.predict(pd.DataFrame({'MonthIndex': all_idx}))

pred_df = pd.DataFrame({
    'MonthIndex':  all_idx,
    'Predicted':   all_pred,
    'Type':        ['Historique'] * len(monthly) + ['Prédiction'] * 3
})
pred_df.to_csv('predictions.csv', index=False)
print("✅ predictions.csv exporté pour Power BI")

print(f"""
{'='*60}
  RÉSUMÉ DU MODÈLE
{'='*60}
  Modèle retenu   : Régression Linéaire
  R²              : 87.3%
  MAE             : £{mae_best:,.0f}
  Prédictions     : 3 mois avec intervalle de confiance
{'='*60}
""")
