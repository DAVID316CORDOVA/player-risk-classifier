import matplotlib.pyplot as plt
import numpy as np

plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['axes.facecolor'] = '#f8f9fa'
plt.rcParams['figure.facecolor'] = 'white'

COLORS = {'primary': '#1f77b4', 'secondary': '#ff7f0e', 'success': '#2ca02c', 'danger': '#d62728', 'warning': '#ff9800'}

def set_style(ax):
    ax.set_facecolor('#f8f9fa')
    ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Grafico 1: AUC por Mes
meses = ['Enero', 'Febrero', 'Marzo', 'Abril', 'Mayo', 'Junio']
auc_modelo = [0.9812, 0.9834, 0.9891, 0.9918, 0.9905, 0.9847]
n_scoreados = [650000, 680000, 720000, 752000, 748000, 755000]

ax2 = axes[0, 0].twinx()
line = axes[0, 0].plot(meses, auc_modelo, marker='o', linewidth=2.5, markersize=8,
                      color=COLORS['success'], label='AUC P1', markeredgecolor='black', markeredgewidth=1.5)
axes[0, 0].axhline(y=0.9865, color=COLORS['danger'], linestyle='--', linewidth=2, label='Promedio (0.9865)')

bar = ax2.bar(meses, n_scoreados, color=COLORS['primary'], alpha=0.3, edgecolor='black', linewidth=1, label='Jugadores Scoreados')

axes[0, 0].set_ylabel('AUC', fontsize=11, fontweight='bold', color=COLORS['success'])
ax2.set_ylabel('Jugadores (Miles)', fontsize=11, fontweight='bold')
axes[0, 0].set_title('AUC P1 LightGBM por Mes', fontsize=12, fontweight='bold', pad=15)
axes[0, 0].set_ylim([0.97, 1.00])
ax2.set_ylim([0, 800000])
axes[0, 0].legend(fontsize=9, loc='lower left')
set_style(axes[0, 0])

# Grafico 2: Comparacion de Modelos
modelos = ['LightGBM', 'XGBoost', 'RandomForest', 'LogisticRegression']
aucs = [0.9918, 0.9876, 0.9845, 0.9512]

bars = axes[0, 1].barh(modelos, aucs, color=[COLORS['success'], COLORS['primary'], COLORS['warning'], COLORS['danger']],
                       edgecolor='black', linewidth=1.5)
axes[0, 1].set_xlabel('AUC', fontsize=11, fontweight='bold')
axes[0, 1].set_title('Comparacion de Modelos', fontsize=12, fontweight='bold', pad=15)
axes[0, 1].set_xlim([0.94, 1.00])
for i, (bar, auc) in enumerate(zip(bars, aucs)):
    axes[0, 1].text(auc - 0.002, i, f'{auc:.4f}', va='center', ha='right',
                   fontweight='bold', fontsize=9, color='white')
set_style(axes[0, 1])

# Grafico 3: Distribucion de Scores
score_ranges = ['0-0.25', '0.25-0.50', '0.50-0.75', '0.75-0.95', '>0.95']
distribution = [435000, 178000, 89000, 42000, 8000]  # Aproximado

bars = axes[1, 0].bar(score_ranges, distribution, color=COLORS['primary'], alpha=0.7, edgecolor='black', linewidth=1.5)
axes[1, 0].set_ylabel('Jugadores', fontsize=11, fontweight='bold')
axes[1, 0].set_title('Distribucion de Scores de Riesgo', fontsize=12, fontweight='bold', pad=15)
axes[1, 0].set_ylim([0, 500000])
for i, (bar, dist) in enumerate(zip(bars, distribution)):
    height = bar.get_height()
    axes[1, 0].text(bar.get_x() + bar.get_width()/2., height + 10000,
                   f'{dist//1000}K', ha='center', va='bottom', fontweight='bold', fontsize=9)
set_style(axes[1, 0])

# Grafico 4: Feature Importance Top 5
features = ['Apuestas', 'Transacciones', 'KYC', 'Bonos', 'IP/Device']
importance = [28, 24, 18, 16, 14]

bars = axes[1, 1].barh(features, importance, color=[COLORS['success'], COLORS['primary'], COLORS['warning'],
                                                     COLORS['danger'], COLORS['secondary']],
                       edgecolor='black', linewidth=1.5)
axes[1, 1].set_xlabel('Importancia (%)', fontsize=11, fontweight='bold')
axes[1, 1].set_title('Top 5 Features (LightGBM)', fontsize=12, fontweight='bold', pad=15)
axes[1, 1].set_xlim([0, 30])
for i, (bar, imp) in enumerate(zip(bars, importance)):
    axes[1, 1].text(imp + 0.5, i, f'{imp}%', va='center', fontweight='bold', fontsize=9)
set_style(axes[1, 1])

plt.tight_layout()
plt.savefig('graficos_score_risk.png', dpi=300, bbox_inches='tight', facecolor='white')
print("OK - graficos_score_risk.png")
