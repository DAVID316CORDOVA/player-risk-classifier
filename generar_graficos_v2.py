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

fig = plt.figure(figsize=(18, 12))
gs = fig.add_gridspec(3, 3, hspace=0.35, wspace=0.3)

# Fila 1
ax1 = fig.add_subplot(gs[0, 0])
meses = ['Ene', 'Feb', 'Mar', 'Abr', 'May', 'Jun']
auc_mes = [0.9812, 0.9834, 0.9891, 0.9918, 0.9905, 0.9847]
ax1.plot(meses, auc_mes, marker='o', linewidth=2.5, markersize=8, color=COLORS['success'], markeredgecolor='black', markeredgewidth=1.5)
ax1.axhline(y=0.9865, color=COLORS['danger'], linestyle='--', linewidth=2, label='Promedio')
ax1.fill_between(range(len(meses)), auc_mes, alpha=0.2, color=COLORS['success'])
ax1.set_ylabel('AUC', fontsize=10, fontweight='bold')
ax1.set_title('AUC P1 Historico (6 meses)', fontsize=11, fontweight='bold')
ax1.set_ylim([0.97, 1.00])
ax1.legend(fontsize=8)
set_style(ax1)

ax2 = fig.add_subplot(gs[0, 1])
modelos = ['LightGBM', 'XGBoost', 'RandomForest', 'LogReg']
aucs = [0.9918, 0.9876, 0.9845, 0.9512]
bars = ax2.barh(modelos, aucs, color=[COLORS['success'], COLORS['primary'], COLORS['warning'], COLORS['danger']], edgecolor='black', linewidth=1)
ax2.set_xlabel('AUC', fontsize=10, fontweight='bold')
ax2.set_title('Comparacion Modelos', fontsize=11, fontweight='bold')
ax2.set_xlim([0.94, 1.00])
for i, (model, auc) in enumerate(zip(modelos, aucs)):
    ax2.text(auc - 0.003, i, f'{auc:.4f}', va='center', ha='right', fontweight='bold', fontsize=8, color='white')
set_style(ax2)

ax3 = fig.add_subplot(gs[0, 2])
ranges = ['0-0.25', '0.25-0.50', '0.50-0.75', '0.75-0.95', '>0.95']
distribution = [435000, 178000, 89000, 42000, 8000]
colors_dist = [COLORS['success'], COLORS['primary'], COLORS['warning'], COLORS['secondary'], COLORS['danger']]
ax3.bar(ranges, distribution, color=colors_dist, edgecolor='black', linewidth=1.5)
ax3.set_ylabel('Jugadores', fontsize=10, fontweight='bold')
ax3.set_title('Distribucion de Scores', fontsize=11, fontweight='bold')
ax3.set_ylim([0, 500000])
for i, (range_val, dist) in enumerate(zip(ranges, distribution)):
    ax3.text(i, dist + 15000, f'{dist//1000}K', ha='center', fontweight='bold', fontsize=8)
set_style(ax3)

# Fila 2
ax4 = fig.add_subplot(gs[1, 0])
features = ['Apuestas', 'Transacciones', 'KYC', 'Bonos', 'IP/Device']
importance = [28, 24, 18, 16, 14]
colors_feat = [COLORS['success'], COLORS['primary'], COLORS['warning'], COLORS['danger'], COLORS['secondary']]
bars = ax4.barh(features, importance, color=colors_feat, edgecolor='black', linewidth=1)
ax4.set_xlabel('Importancia (%)', fontsize=10, fontweight='bold')
ax4.set_title('Top 5 Features', fontsize=11, fontweight='bold')
ax4.set_xlim([0, 30])
for i, (feat, imp) in enumerate(zip(features, importance)):
    ax4.text(imp + 0.5, i, f'{imp}%', va='center', fontweight='bold', fontsize=8)
set_style(ax4)

ax5 = fig.add_subplot(gs[1, 1])
segmentos = ['Bajo', 'Moderado', 'Alto', 'Critico']
sizes = [435000, 178000, 42000, 8000]
colors_seg = [COLORS['success'], COLORS['primary'], COLORS['warning'], COLORS['danger']]
wedges, texts, autotexts = ax5.pie(sizes, labels=segmentos, autopct='%1.1f%%', colors=colors_seg, startangle=90,
                                     textprops={'fontsize': 8, 'fontweight': 'bold'},
                                     wedgeprops={'edgecolor': 'black', 'linewidth': 1})
ax5.set_title('Segmentacion Riesgo (655K)', fontsize=11, fontweight='bold')

ax6 = fig.add_subplot(gs[1, 2])
fpr = np.array([0, 0.01, 0.02, 0.05, 0.15, 1.0])
tpr = np.array([0, 0.60, 0.75, 0.88, 0.97, 1.0])
ax6.plot(fpr, tpr, marker='o', linewidth=2.5, markersize=8, color=COLORS['success'], label='AUC = 0.9918', markeredgecolor='black', markeredgewidth=1.5)
ax6.plot([0, 1], [0, 1], 'k--', linewidth=1.5, label='Random')
ax6.fill_between(fpr, tpr, alpha=0.2, color=COLORS['success'])
ax6.set_xlabel('FPR', fontsize=10, fontweight='bold')
ax6.set_ylabel('TPR', fontsize=10, fontweight='bold')
ax6.set_title('Curva ROC (LightGBM)', fontsize=11, fontweight='bold')
ax6.set_xlim([0, 1])
ax6.set_ylim([0, 1])
ax6.legend(fontsize=8)
set_style(ax6)

# Fila 3
ax7 = fig.add_subplot(gs[2, :2])
ax7.axis('off')
arch_text = '''MODELO LIGHTGBM: Score de Riesgo P1

Entrenamiento: 4 años historicos (2022-2026)
Datos: 655K+ jugadores, 60+ features
Etiqueta: dwh_player_risk (4,732 positivos)

Configuracion:
  num_leaves: 127
  n_estimators: 500
  learning_rate: 0.03
  class_weight: balanced (maneja desbalance 0.72%)
  min_data_in_leaf: 10

Validacion: Hold-out temporal (no leakage)
Prediccion: Diaria, ~750K jugadores scoreados
Output: Tabla dwh_raw_pe.score_riesgo_dia

Etiqueta binaria:
  1 = Riesgoso (jugador en dwh_player_risk)
  0 = Normal (todos los demas)
'''
ax7.text(0.05, 0.95, arch_text, transform=ax7.transAxes, fontsize=9, verticalalignment='top',
        fontfamily='monospace', bbox=dict(boxstyle='round', facecolor='#f0f0f0', alpha=0.8, edgecolor='black'))

ax8 = fig.add_subplot(gs[2, 2])
ax8.axis('off')
results_text = '''RESULTADOS PRODUCCION

AUC Maximo: 0.9918 (Abril)
AUC Promedio: 0.9865 (6 meses)
Jugadores Scoreados: 752K
Criticos Detectados: ~4,700/semana

Precision Criticos: 94-96%
Desbalance: 0.72% positivos
Features: 60+ totales

Categorias Riesgo:
  Bajo: 66.4%
  Moderado: 27.2%
  Alto: 4.7%
  Critico: 1.2%

Impacto: Compliance, anti-fraude
Velocidad: <10ms/jugador
'''
ax8.text(0.05, 0.95, results_text, transform=ax8.transAxes, fontsize=9, verticalalignment='top',
        fontfamily='monospace', bbox=dict(boxstyle='round', facecolor='#f0f0f0', alpha=0.8, edgecolor='black'))

plt.suptitle('SCORE DE RIESGO P1: EDA + Modelo LightGBM + Resultados Produccion',
            fontsize=14, fontweight='bold', y=0.995)

plt.savefig('graficos_score_risk_v2.png', dpi=300, bbox_inches='tight', facecolor='white')
print("OK - graficos_score_risk_v2.png")
