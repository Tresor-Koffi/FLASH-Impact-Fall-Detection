import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, FancyBboxPatch
import matplotlib.patches as mpatches

plt.rcParams.update({'font.size': 10})

fig, ax = plt.subplots(figsize=(12, 8))
ax.set_xlim(0, 12)
ax.set_ylim(0, 10)
ax.axis('off')

# Helper to draw block
def draw_block(x, y, width, height, label, color='lightblue', text_size=9):
    rect = FancyBboxPatch((x, y), width, height, boxstyle="round,pad=0.1", 
                          edgecolor='black', facecolor=color, linewidth=1)
    ax.add_patch(rect)
    ax.text(x + width/2, y + height/2, label, ha='center', va='center', fontsize=text_size, wrap=True)

# Input
draw_block(0.5, 7.5, 1.5, 1, "Input:\n$T \\times 33 \\times 3$", color='#e6f7ff')
ax.annotate('', xy=(2.0, 8), xytext=(2.5, 8), arrowprops=dict(arrowstyle='->', lw=1.5))

# HGCN Block (expanded)
draw_block(2.5, 8.2, 2.0, 0.6, "$\\mathbf{H}_{\\text{norm}} \\cdot \\mathbf{X} \\cdot \\mathbf{W}_1$", color='#d1e7ff')
draw_block(2.5, 7.4, 2.0, 0.6, "$\\mathbf{H}_{\\text{norm}} \\cdot \\mathbf{X}^{(1)} \\cdot \\mathbf{W}_2$", color='#d1e7ff')
hgcn_label = plt.text(3.5, 9.0, "Hypergraph Convolution (HGCN)", ha='center', weight='bold')
ax.annotate('', xy=(4.5, 8), xytext=(5.0, 8), arrowprops=dict(arrowstyle='->', lw=1.5))

# Reshape → Mamba → Reshape back
draw_block(5.0, 8.0, 1.2, 0.8, "Reshape\n$T \\times (33\\cdot128)$", color='#d4edda')
draw_block(6.3, 8.0, 1.4, 0.8, "MambaBlock\n(SSM Core)", color='#c3e6cb')
draw_block(7.8, 8.0, 1.2, 0.8, "Reshape Back\n$T \\times 33 \\times 128$", color='#d4edda')
mamba_label = plt.text(6.9, 9.0, "Mamba Temporal Modeling", ha='center', weight='bold')
ax.annotate('', xy=(9.0, 8), xytext=(9.5, 8), arrowprops=dict(arrowstyle='->', lw=1.5))

# Permute
draw_block(9.5, 8.0, 1.2, 0.8, "Permute\n$128 \\times T \\times 33$", color='#fff3cd')

# Multi-scale TCN (parallel)
tcn_y = 6.5
draw_block(9.0, tcn_y + 1.0, 1.8, 0.6, "TCN: k=(9,1)", color='#f8d7da')
draw_block(9.0, tcn_y + 0.2, 1.8, 0.6, "TCN: k=(15,1)", color='#f8d7da')
draw_block(9.0, tcn_y - 0.6, 1.8, 0.6, "TCN: k=(20,1)", color='#f8d7da')
tcn_label = plt.text(9.9, tcn_y + 1.8, "Multi-Scale Temporal Conv", ha='center', weight='bold')

# Arrows from permute to each TCN
for dy in [1.0, 0.2, -0.6]:
    ax.annotate('', xy=(10.1, 8.0), xytext=(9.0, tcn_y + dy + 0.3), 
                arrowprops=dict(arrowstyle='->', lw=1, shrinkA=0, shrinkB=0))

# Concat
draw_block(7.0, tcn_y - 0.6, 1.5, 0.6, "Concat\n$192 \\times T \\times 33$", color='#e2e3e5')
ax.annotate('', xy=(8.5, tcn_y - 0.3), xytext=(7.0, tcn_y - 0.3), arrowprops=dict(arrowstyle='->', lw=1.5))

# Classification Head
draw_block(5.0, tcn_y - 0.6, 1.5, 0.6, "Conv2D(1×1)", color='#e2e3e5')
draw_block(3.2, tcn_y - 0.6, 1.5, 0.6, "Conv2D(1×33)", color='#e2e3e5')
draw_block(1.2, tcn_y - 0.6, 1.5, 0.6, "Dense → Dense", color='#e2e3e5')
class_label = plt.text(3.2, tcn_y + 0.4, "Classification Head", ha='center', weight='bold')

# Final output
draw_block(-0.3, tcn_y - 0.6, 1.2, 0.6, "Impact\nPrediction\n$[T]$", color='#d1ecf1')

# Connect classification to output
ax.annotate('', xy=(1.2, tcn_y - 0.3), xytext=(-0.3, tcn_y - 0.3), arrowprops=dict(arrowstyle='->', lw=1.5))

# Title
plt.title("HyperMamba Architecture (Detailed)", fontsize=14, weight='bold', pad=20)

# Legend (optional)
colors = ['#d1e7ff', '#c3e6cb', '#f8d7da', '#e2e3e5', '#d1ecf1']
labels = ['HGCN', 'Mamba', 'Temporal Conv', 'Classification', 'I/O']
patches = [mpatches.Patch(color=c, label=l) for c, l in zip(colors, labels)]
plt.legend(handles=patches, loc='lower center', bbox_to_anchor=(0.5, -0.05), ncol=5)

plt.tight_layout()
plt.savefig("hyperMamba_architecture_detailed.png", dpi=300, bbox_inches='tight')
plt.show()