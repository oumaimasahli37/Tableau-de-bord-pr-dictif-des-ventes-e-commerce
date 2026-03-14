# =============================================================================
# 1. ANALYSE EXPLORATOIRE DES DONNÉES (EDA)
# Réalisé par : SAHLI Oumaima
# =============================================================================

import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

print("=" * 60)
print("  ANALYSE EXPLORATOIRE — E-COMMERCE")
print("=" * 60)

# =============================================================================
# CHARGEMENT
# =============================================================================
print("\n📂 Chargement des données...")

try:
    df = pd.read_csv('online_retail.csv', encoding='ISO-8859-1')
    print(f"✅ Dataset chargé : {df.shape[0]:,} lignes × {df.shape[1]} colonnes")
except FileNotFoundError:
    print("⚠️  'online_retail.csv' non trouvé — données simulées utilisées.")
    np.random.seed(42)
    n = 50000
    dates = pd.date_range('2010-12-01', '2011-12-09', periods=n)
    df = pd.DataFrame({
        'InvoiceNo':   [f'INV{i:06d}' for i in range(n)],
        'StockCode':   [f'SC{np.random.randint(1000,9999)}' for _ in range(n)],
        'Description': np.random.choice([
            'WHITE HANGING HEART T-LIGHT HOLDER',
            'JUMBO BAG RED RETROSPOT',
            'WORLD WAR 2 GLIDERS ASSTD DESIGNS',
            'ASSORTED COLOUR BIRD ORNAMENT',
            'PACK OF 72 RETROSPOT CAKE CASES'
        ], n),
        'Quantity':    np.random.randint(1, 50, n),
        'InvoiceDate': dates,
        'UnitPrice':   np.random.uniform(0.5, 20.0, n),
        'CustomerID':  np.random.randint(10000, 20000, n),
        'Country':     np.random.choice(
            ['United Kingdom']*82 + ['Germany']*5 + ['France']*4 + ['EIRE']*4 + ['Spain']*5, n
        )
    })
    print(f"✅ Données simulées : {n:,} lignes\n")

# =============================================================================
# NETTOYAGE
# =============================================================================
print("\n🧹 NETTOYAGE DES DONNÉES")
print("-" * 40)

initial_count = len(df)
print(f"Lignes initiales       : {initial_count:,}")

# Valeurs manquantes
missing = df.isnull().sum()
if missing[missing > 0].any():
    print(f"Valeurs manquantes :\n{missing[missing > 0]}")
df = df.dropna(subset=['CustomerID', 'Description'])
print(f"Après suppression NaN  : {len(df):,}")

# Transactions invalides
df = df[df['Quantity'] > 0]
df = df[df['UnitPrice'] > 0]
df = df[~df['InvoiceNo'].astype(str).str.startswith('C')]
print(f"Transactions valides   : {len(df):,}")
print(f"Lignes supprimées      : {initial_count - len(df):,} ({(initial_count - len(df))/initial_count*100:.1f}%)")

# =============================================================================
# FEATURE ENGINEERING
# =============================================================================
print("\n⚙️  FEATURE ENGINEERING")
print("-" * 40)

df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'])
df['TotalAmount'] = df['Quantity'] * df['UnitPrice']
df['Year']        = df['InvoiceDate'].dt.year
df['Month']       = df['InvoiceDate'].dt.month
df['DayOfWeek']   = df['InvoiceDate'].dt.dayofweek
df['Hour']        = df['InvoiceDate'].dt.hour
df['IsWeekend']   = df['DayOfWeek'].isin([5, 6]).astype(int)

print("✅ Nouvelles colonnes : TotalAmount, Year, Month, DayOfWeek, Hour, IsWeekend")

# =============================================================================
# KPIs PRINCIPAUX
# =============================================================================
print("\n💰 KPIs PRINCIPAUX")
print("-" * 40)

total_revenue      = df['TotalAmount'].sum()
total_transactions = df['InvoiceNo'].nunique()
avg_basket         = df.groupby('InvoiceNo')['TotalAmount'].sum().mean()
unique_customers   = df['CustomerID'].nunique()
unique_products    = df['Description'].nunique()
countries_count    = df['Country'].nunique()

print(f"Chiffre d'affaires total  : £{total_revenue:>12,.2f}")
print(f"Nombre de transactions    : {total_transactions:>12,}")
print(f"Panier moyen              : £{avg_basket:>12,.2f}")
print(f"Clients uniques           : {unique_customers:>12,}")
print(f"Produits uniques          : {unique_products:>12,}")
print(f"Pays couverts             : {countries_count:>12,}")

# =============================================================================
# TENDANCES SAISONNIÈRES
# =============================================================================
print("\n📅 TENDANCES SAISONNIÈRES")
print("-" * 40)

monthly = df.groupby(['Year', 'Month']).agg(
    Revenue=('TotalAmount', 'sum'),
    Transactions=('InvoiceNo', 'nunique')
).reset_index()

avg_monthly = monthly['Revenue'].mean()
peak_month  = monthly.loc[monthly['Revenue'].idxmax()]
low_month   = monthly.loc[monthly['Revenue'].idxmin()]

print(f"CA mensuel moyen          : £{avg_monthly:,.2f}")
print(f"Mois le plus fort         : {int(peak_month['Year'])}-{int(peak_month['Month']):02d}  →  £{peak_month['Revenue']:,.2f}  (+{(peak_month['Revenue']/avg_monthly-1)*100:.0f}% vs moyenne)")
print(f"Mois le plus faible       : {int(low_month['Year'])}-{int(low_month['Month']):02d}   →  £{low_month['Revenue']:,.2f}  ({(low_month['Revenue']/avg_monthly-1)*100:.0f}% vs moyenne)")

# =============================================================================
# COMPORTEMENTS D'ACHAT
# =============================================================================
print("\n🛒 COMPORTEMENTS D'ACHAT")
print("-" * 40)

day_names    = ['Lundi','Mardi','Mercredi','Jeudi','Vendredi','Samedi','Dimanche']
sales_by_day = df.groupby('DayOfWeek')['TotalAmount'].sum()
best_day     = sales_by_day.idxmax()
sales_by_hour = df.groupby('Hour')['TotalAmount'].sum()
best_hour    = sales_by_hour.idxmax()

weekend_rev  = df[df['IsWeekend'] == 1]['TotalAmount'].sum()
weekday_rev  = df[df['IsWeekend'] == 0]['TotalAmount'].sum()

print(f"Meilleur jour             : {day_names[best_day]}")
print(f"Heure de pointe           : {best_hour}h")
print(f"CA semaine                : £{weekday_rev:,.2f}  ({weekday_rev/total_revenue*100:.1f}%)")
print(f"CA weekend                : £{weekend_rev:,.2f}  ({weekend_rev/total_revenue*100:.1f}%)")

# =============================================================================
# TOP PRODUITS & PAYS
# =============================================================================
print("\n📦 TOP 10 PRODUITS")
print("-" * 40)

top_products = df.groupby('Description').agg(
    Quantity=('Quantity', 'sum'),
    Revenue=('TotalAmount', 'sum')
).sort_values('Quantity', ascending=False).head(10)

for i, (desc, row) in enumerate(top_products.iterrows(), 1):
    print(f"  {i:2}. {desc[:45]:<45}  {int(row['Quantity']):>8,} unités  |  £{row['Revenue']:>10,.2f}")

print("\n🌍 TOP 10 PAYS")
print("-" * 40)

top_countries = df.groupby('Country')['TotalAmount'].sum().sort_values(ascending=False).head(10)
for country, rev in top_countries.items():
    print(f"  {country:<25}  £{rev:>10,.2f}  ({rev/total_revenue*100:.1f}%)")

# =============================================================================
# EXPORT POUR POWER BI
# =============================================================================
print("\n📤 EXPORT DES DONNÉES POUR POWER BI")
print("-" * 40)

# Données nettoyées
df.to_csv('data_cleaned.csv', index=False)
print("✅ data_cleaned.csv")

# Ventes mensuelles agrégées
monthly.to_csv('monthly_sales.csv', index=False)
print("✅ monthly_sales.csv")

# Top produits
top_products.reset_index().to_csv('top_products.csv', index=False)
print("✅ top_products.csv")

# Top pays
top_countries.reset_index().columns = ['Country', 'Revenue']
top_countries.reset_index().to_csv('top_countries.csv', index=False)
print("✅ top_countries.csv")

print("\n✅ EDA terminée — fichiers prêts pour Power BI")
