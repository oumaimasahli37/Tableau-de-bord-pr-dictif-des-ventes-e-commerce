 📊 Tableau de Bord Prédictif des Ventes E-commerce

  
Analyse complète de 500 000+ transactions e-commerce avec création d'un dashboard interactif Power BI et d'un modèle prédictif des ventes mensuelles.

---

## Fonctionnalités

-  **Analyse exploratoire (EDA)** — nettoyage, tendances saisonnières, comportements d'achat
-  **Dashboard Power BI** — 4 pages interactives avec KPIs, heatmaps, carte géographique
-  **Modèle prédictif** — régression linéaire avec **R² = 87.3%** pour anticiper les ventes mensuelles

---

##  Dataset

- **Source :** [Online Retail Dataset — Kaggle](https://www.kaggle.com/datasets/vijayuv/onlineretail)
- **Période :** Décembre 2010 – Décembre 2011
- **Volume brut :** 541 909 transactions → **397 884 après nettoyage**
- **Produits uniques :** 3 958 · **Clients :** 4 372 · **Pays :** 38

---

##  Structure du projet

```
ecommerce-sales-dashboard/
├── 1_eda.py           # Nettoyage + Analyse exploratoire → export CSV
├── 2_model.py         # Modèle de régression linéaire + prédictions
├── requirements.txt   # Dépendances Python
└── README.md
```

---

##  Lancer le projet

### 1. Installer les dépendances
```bash
pip install -r requirements.txt
```

### 2. Télécharger le dataset
Télécharge `online_retail.csv` sur [Kaggle](https://www.kaggle.com/datasets/vijayuv/onlineretail) et place-le à la racine du projet.

### 3. Lancer les scripts dans l'ordre
```bash
python 1_eda.py      # génère les CSV pour Power BI
python 2_model.py    # génère predictions.csv pour Power BI
```

---

##  Dashboard Power BI

### Fichiers à importer dans Power BI
Générés automatiquement par les scripts Python :

| Fichier | Contenu |
|---------|---------|
| `data_cleaned.csv` | Transactions nettoyées |
| `monthly_sales.csv` | Ventes agrégées par mois |
| `top_products.csv` | Top 10 produits |
| `top_countries.csv` | Top pays par CA |
| `predictions.csv` | Prédictions du modèle ML |

### Structure du dashboard (4 pages)

**Page 1 — Vue d'ensemble (KPIs)**
- Cartes KPI : CA total · Transactions · Panier moyen · Taux de conversion
- Courbe : évolution mensuelle du CA
- Barres : top 5 pays

**Page 2 — Analyse temporelle**
- Courbe : évolution mensuelle détaillée
- Heatmap : CA par jour × heure
- Barres : CA par jour de la semaine

**Page 3 — Analyse produits**
- Top 10 produits par quantité et par CA
- Carte géographique : ventes par pays

**Page 4 — Prédictions**
- Courbe : historique + prédictions 3 mois
- Tableau : prévisions avec intervalle de confiance

### Mesures DAX principales

```dax
// Chiffre d'affaires total
Revenue Total = SUM('data_cleaned'[TotalAmount])

// Nombre de transactions
Total Transactions = DISTINCTCOUNT('data_cleaned'[InvoiceNo])

// Panier moyen
Average Basket = DIVIDE([Revenue Total], [Total Transactions])

// Évolution mois sur mois (MoM)
Revenue MoM =
  VAR CurrentRevenue  = [Revenue Total]
  VAR PreviousRevenue = CALCULATE([Revenue Total], DATEADD('Calendar'[Date], -1, MONTH))
  RETURN DIVIDE(CurrentRevenue - PreviousRevenue, PreviousRevenue)

// Part du CA — Royaume-Uni
UK Revenue Share =
  DIVIDE(
    CALCULATE([Revenue Total], 'data_cleaned'[Country] = "United Kingdom"),
    [Revenue Total]
  )
```

---

##  Résultats des modèles

| Modèle | MAE | R² |
|--------|-----|----|
| Régression linéaire | £52,341 | **87.3%** |
| Polynomiale (deg=2) | £48,923 | 89.1% |
| Random Forest | £55,187 | 85.7% |

**Modèle retenu :** Régression linéaire — meilleur compromis simplicité / performance.

---

##  Insights clés

-  **Saisonnalité** — pic en novembre (+127% vs moyenne), baisse en janvier (-43%)
-  **Géographie** — Royaume-Uni = 82% du CA total
-  **Comportement** — heures de pointe 10h-14h, mardi et jeudi favoris
- **Panier moyen** — £311.43, 5% des transactions > £1000 (clients pro)

---

## Stack technique

| Outil | Usage |
|-------|-------|
| Python 3.9 | Nettoyage, EDA, modélisation |
| Pandas / NumPy | Manipulation des données |
| Scikit-learn | Modèle de régression |
| Power BI | Dashboard interactif + DAX |

---

