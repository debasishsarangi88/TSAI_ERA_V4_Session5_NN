"""
Fixed Network Architecture Diagram Generator - No Overlapping Text
"""

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import FancyBboxPatch, Rectangle
import numpy as np

def create_clean_architecture_diagram():
    """Create a clean visual representation of the OptimizedNet architecture"""
    
    fig, ax = plt.subplots(1, 1, figsize=(20, 16))
    ax.set_xlim(0, 16)
    ax.set_ylim(0, 18)
    ax.axis('off')
    
    # Title
    ax.text(8, 17, 'OptimizedNet Architecture - MNIST Classification', 
            fontsize=24, fontweight='bold', ha='center')
    ax.text(8, 16.5, '18,614 Parameters | 99.4% Accuracy | 17 Epochs', 
            fontsize=18, ha='center', style='italic', color='darkgreen')
    
    # Define colors
    colors = {
        'input': '#E3F2FD',      # Light blue
        'conv': '#E8F5E8',       # Light green
        'bn': '#FFF9C4',         # Light yellow
        'pool': '#FFEBEE',       # Light red
        'dropout': '#F3E5F5',    # Light purple
        'transition': '#FCE4EC', # Light pink
        'gap': '#E0F2F1',        # Light teal
        'output': '#F5F5F5'      # Light gray
    }
    
    # Block 1: Input to 14x14
    y_pos = 14
    
    # Input
    input_rect = Rectangle((1, y_pos), 2, 1, facecolor=colors['input'], 
                          edgecolor='black', linewidth=2)
    ax.add_patch(input_rect)
    ax.text(2, y_pos + 0.5, 'Input\n1×28×28', ha='center', va='center', 
            fontweight='bold', fontsize=12)
    
    # Conv1
    conv1_rect = Rectangle((4, y_pos), 2, 1, facecolor=colors['conv'], 
                          edgecolor='black', linewidth=2)
    ax.add_patch(conv1_rect)
    ax.text(5, y_pos + 0.5, 'Conv2d\n1→8, 3×3', ha='center', va='center', fontsize=11)
    
    # BN1
    bn1_rect = Rectangle((7, y_pos), 2, 1, facecolor=colors['bn'], 
                        edgecolor='black', linewidth=2)
    ax.add_patch(bn1_rect)
    ax.text(8, y_pos + 0.5, 'BatchNorm\n8', ha='center', va='center', fontsize=11)
    
    # Conv2
    conv2_rect = Rectangle((10, y_pos), 2, 1, facecolor=colors['conv'], 
                          edgecolor='black', linewidth=2)
    ax.add_patch(conv2_rect)
    ax.text(11, y_pos + 0.5, 'Conv2d\n8→8, 3×3', ha='center', va='center', fontsize=11)
    
    # Dropout indicator - positioned below the image size text
    ax.text(11, y_pos - 0.5, 'Dropout 0.1', ha='center', va='center', 
            fontsize=10, style='italic', color='red', weight='bold')
    
    # Pool1
    pool1_rect = Rectangle((13, y_pos), 2, 1, facecolor=colors['pool'], 
                          edgecolor='black', linewidth=2)
    ax.add_patch(pool1_rect)
    ax.text(14, y_pos + 0.5, 'MaxPool\n2×2', ha='center', va='center', fontsize=11)
    
    # Block 2: 14x14 to 7x7
    y_pos = 11
    
    # Conv3
    conv3_rect = Rectangle((4, y_pos), 2, 1, facecolor=colors['conv'], 
                          edgecolor='black', linewidth=2)
    ax.add_patch(conv3_rect)
    ax.text(5, y_pos + 0.5, 'Conv2d\n8→16, 3×3', ha='center', va='center', fontsize=11)
    
    # BN3
    bn3_rect = Rectangle((7, y_pos), 2, 1, facecolor=colors['bn'], 
                        edgecolor='black', linewidth=2)
    ax.add_patch(bn3_rect)
    ax.text(8, y_pos + 0.5, 'BatchNorm\n16', ha='center', va='center', fontsize=11)
    
    # Conv4
    conv4_rect = Rectangle((10, y_pos), 2, 1, facecolor=colors['conv'], 
                          edgecolor='black', linewidth=2)
    ax.add_patch(conv4_rect)
    ax.text(11, y_pos + 0.5, 'Conv2d\n16→16, 3×3', ha='center', va='center', fontsize=11)
    
    # Dropout indicator - positioned below the image size text
    ax.text(11, y_pos - 0.5, 'Dropout 0.1', ha='center', va='center', 
            fontsize=10, style='italic', color='red', weight='bold')
    
    # Pool2
    pool2_rect = Rectangle((13, y_pos), 2, 1, facecolor=colors['pool'], 
                          edgecolor='black', linewidth=2)
    ax.add_patch(pool2_rect)
    ax.text(14, y_pos + 0.5, 'MaxPool\n2×2', ha='center', va='center', fontsize=11)
    
    # Block 3: 7x7 to 3x3
    y_pos = 8
    
    # Conv5
    conv5_rect = Rectangle((4, y_pos), 2, 1, facecolor=colors['conv'], 
                          edgecolor='black', linewidth=2)
    ax.add_patch(conv5_rect)
    ax.text(5, y_pos + 0.5, 'Conv2d\n16→32, 3×3', ha='center', va='center', fontsize=11)
    
    # BN5
    bn5_rect = Rectangle((7, y_pos), 2, 1, facecolor=colors['bn'], 
                        edgecolor='black', linewidth=2)
    ax.add_patch(bn5_rect)
    ax.text(8, y_pos + 0.5, 'BatchNorm\n32', ha='center', va='center', fontsize=11)
    
    # Conv6
    conv6_rect = Rectangle((10, y_pos), 2, 1, facecolor=colors['conv'], 
                          edgecolor='black', linewidth=2)
    ax.add_patch(conv6_rect)
    ax.text(11, y_pos + 0.5, 'Conv2d\n32→32, 3×3', ha='center', va='center', fontsize=11)
    
    # Dropout indicator - positioned below the image size text
    ax.text(11, y_pos - 0.5, 'Dropout 0.1', ha='center', va='center', 
            fontsize=10, style='italic', color='red', weight='bold')
    
    # Pool3
    pool3_rect = Rectangle((13, y_pos), 2, 1, facecolor=colors['pool'], 
                          edgecolor='black', linewidth=2)
    ax.add_patch(pool3_rect)
    ax.text(14, y_pos + 0.5, 'MaxPool\n2×2', ha='center', va='center', fontsize=11)
    
    # Transition Layer
    y_pos = 5
    trans_rect = Rectangle((6, y_pos), 2, 1, facecolor=colors['transition'], 
                          edgecolor='black', linewidth=2)
    ax.add_patch(trans_rect)
    ax.text(7, y_pos + 0.5, 'Transition\n32→10, 1×1', ha='center', va='center', 
            fontsize=11, fontweight='bold')
    
    # Global Average Pooling
    y_pos = 3
    gap_rect = Rectangle((6, y_pos), 2, 1, facecolor=colors['gap'], 
                        edgecolor='black', linewidth=2)
    ax.add_patch(gap_rect)
    ax.text(7, y_pos + 0.5, 'GAP\n3×3→1×1', ha='center', va='center', 
            fontsize=11, fontweight='bold')
    
    # Output
    y_pos = 1
    output_rect = Rectangle((6, y_pos), 2, 1, facecolor=colors['output'], 
                           edgecolor='black', linewidth=2)
    ax.add_patch(output_rect)
    ax.text(7, y_pos + 0.5, 'Output\n10 Classes', ha='center', va='center', 
            fontsize=11, fontweight='bold')
    
    # Arrows - Horizontal
    arrow_y = 14.5
    arrow_positions_h = [
        ((3, arrow_y), (4, arrow_y)),    # Input to Conv1
        ((6, arrow_y), (7, arrow_y)),    # Conv1 to BN1
        ((9, arrow_y), (10, arrow_y)),   # BN1 to Conv2
        ((12, arrow_y), (13, arrow_y)),  # Conv2 to Pool1
    ]
    
    arrow_y = 11.5
    arrow_positions_h.extend([
        ((6, arrow_y), (7, arrow_y)),    # Conv3 to BN3
        ((9, arrow_y), (10, arrow_y)),   # BN3 to Conv4
        ((12, arrow_y), (13, arrow_y)),  # Conv4 to Pool2
    ])
    
    arrow_y = 8.5
    arrow_positions_h.extend([
        ((6, arrow_y), (7, arrow_y)),    # Conv5 to BN5
        ((9, arrow_y), (10, arrow_y)),   # BN5 to Conv6
        ((12, arrow_y), (13, arrow_y)),  # Conv6 to Pool3
    ])
    
    # Arrows - Vertical
    arrow_positions_v = [
        ((14, 14), (14, 11)),            # Pool1 to Conv3
        ((14, 11), (14, 8)),             # Pool2 to Conv5
        ((14, 8), (7, 5)),               # Pool3 to Transition
        ((7, 5), (7, 3)),                # Transition to GAP
        ((7, 3), (7, 1)),                # GAP to Output
    ]
    
    # Draw horizontal arrows
    for start, end in arrow_positions_h:
        ax.annotate('', xy=end, xytext=start,
                   arrowprops=dict(arrowstyle='->', lw=2, color='black'))
    
    # Draw vertical arrows
    for start, end in arrow_positions_v:
        ax.annotate('', xy=end, xytext=start,
                   arrowprops=dict(arrowstyle='->', lw=2, color='black'))
    
    # Feature map sizes - positioned clearly with proper spacing
    size_positions = [
        (2, 13.7, '28×28×1'),
        (5, 13.7, '28×28×8'),
        (8, 13.7, '28×28×8'),
        (11, 13.7, '28×28×8'),
        (14, 13.7, '14×14×8'),
        (5, 10.7, '14×14×16'),
        (8, 10.7, '14×14×16'),
        (11, 10.7, '14×14×16'),
        (14, 10.7, '7×7×16'),
        (5, 7.7, '7×7×32'),
        (8, 7.7, '7×7×32'),
        (11, 7.7, '7×7×32'),
        (14, 7.7, '3×3×32'),
        (7, 4.3, '3×3×10'),
        (7, 2.3, '1×1×10'),
        (7, 0.3, '10'),
    ]
    
    for x, y, text in size_positions:
        ax.text(x, y, text, ha='center', va='center', 
                fontsize=10, style='italic', color='blue', weight='bold')
    
    # Legend
    legend_elements = [
        patches.Patch(color=colors['input'], label='Input Layer'),
        patches.Patch(color=colors['conv'], label='Convolutional Layer'),
        patches.Patch(color=colors['bn'], label='Batch Normalization'),
        patches.Patch(color=colors['pool'], label='MaxPooling'),
        patches.Patch(color=colors['transition'], label='Transition Layer (1×1)'),
        patches.Patch(color=colors['gap'], label='Global Average Pooling'),
        patches.Patch(color=colors['output'], label='Output Layer'),
    ]
    
    ax.legend(handles=legend_elements, loc='lower right', bbox_to_anchor=(1, 0.05),
              fontsize=12, frameon=True, fancybox=True, shadow=True)
    
    # Architecture details - positioned clearly
    details_text = """Architecture Details:
• Total Parameters: 18,614 (< 20,000 target)
• Receptive Field: ~23 (sufficient for MNIST)
• Training Epochs: 17 (< 20 target)
• Final Accuracy: 99.40% (≥ 99.4% target)
• Memory Usage: 0.45 MB
• Components: BN + Dropout + GAP + Transition Layers"""
    
    ax.text(0.5, 6, details_text, fontsize=12, va='top',
            bbox=dict(boxstyle="round,pad=0.5", facecolor='lightgray', alpha=0.9),
            fontweight='bold')
    
    # Block labels
    ax.text(0.5, 15, 'Block 1\n(RF: 3)', ha='center', va='center', 
            fontsize=12, fontweight='bold', 
            bbox=dict(boxstyle="round,pad=0.3", facecolor='lightblue', alpha=0.7))
    
    ax.text(0.5, 12, 'Block 2\n(RF: 7)', ha='center', va='center', 
            fontsize=12, fontweight='bold',
            bbox=dict(boxstyle="round,pad=0.3", facecolor='lightgreen', alpha=0.7))
    
    ax.text(0.5, 9, 'Block 3\n(RF: 15)', ha='center', va='center', 
            fontsize=12, fontweight='bold',
            bbox=dict(boxstyle="round,pad=0.3", facecolor='lightyellow', alpha=0.7))
    
    plt.tight_layout()
    plt.savefig('network_architecture_clean.png', dpi=300, bbox_inches='tight')
    plt.savefig('network_architecture_clean.pdf', bbox_inches='tight')
    plt.show()
    
    print("Clean architecture diagram saved as 'network_architecture_clean.png' and 'network_architecture_clean.pdf'")

if __name__ == "__main__":
    create_clean_architecture_diagram()
