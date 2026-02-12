"""Plot pruning comparison results from the terminal output."""
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

hotel_tr = np.array([0.005, 0.009, 0.0162, 0.0292, 0.0527,
                     0.0949, 0.171, 0.3081, 0.555, 1.0])

# WGN results (mean over 10 seeds)
wgn = {
    'f1_homo':   [0, 0.147, 0.315, 0.378, 0.452, 0.671, 0.874, 0.968, 0.988, 0.995],
    'f1_greedy': [0, 0.162, 0.442, 0.566, 0.677, 0.813, 0.930, 0.987, 1.000, 1.000],
    'sn_homo':   [0, 0.118, 0.255, 0.286, 0.327, 0.543, 0.805, 0.942, 0.975, 0.990],
    'sn_greedy': [0, 0.135, 0.494, 0.598, 0.676, 0.802, 0.931, 0.988, 0.999, 1.000],
    'sp_homo':   [1, 0.959, 0.919, 0.946, 0.972, 0.981, 0.991, 0.999, 1.000, 1.000],
    'sp_greedy': [1, 0.952, 0.818, 0.872, 0.920, 0.957, 0.982, 0.997, 1.000, 1.000],
    'n_homo':    [0, 0.5, 1.6, 4.4, 14.4, 39.8, 81.0, 98.3, 99.9, 108.8],
    'n_greedy':  [0, 0.5, 1.1, 1.3, 2.2, 4.4, 5.2, 3.2, 1.2, 1.0],
}

# HCP results (mean over 10 seeds)
hcp = {
    'f1_homo':   [0, 0, 0, 0.047, 0.209, 0.559, 0.830, 0.968, 0.993, 0.998],
    'f1_greedy': [0, 0, 0, 0.069, 0.265, 0.674, 0.886, 0.975, 0.988, 0.994],
    'sn_homo':   [0, 0, 0, 0.028, 0.142, 0.406, 0.719, 0.950, 0.996, 1.000],
    'sn_greedy': [0, 0, 0, 0.048, 0.203, 0.543, 0.833, 0.969, 0.994, 1.000],
    'sp_homo':   [1, 1, 1, 1.000, 0.999, 0.998, 0.998, 0.997, 0.998, 0.999],
    'sp_greedy': [1, 1, 1, 1.000, 0.995, 0.995, 0.990, 0.995, 0.995, 0.997],
    'n_homo':    [0, 0, 0, 0.6, 2.0, 8.8, 20.2, 29.6, 33.8, 33.1],
    'n_greedy':  [0, 0, 0, 0.5, 1.4, 4.6, 9.1, 9.8, 10.8, 11.4],
}


def main():
    fig, axes = plt.subplots(2, 3, figsize=(16, 9))
    fig.suptitle('Homogeneity Pruning vs Greedy Pruning', fontsize=15,
                 fontweight='bold', y=0.98)

    colors = {'homo': '#2166ac', 'greedy': '#b2182b'}
    lw = 2.2
    ms = 7

    for row, (label, data) in enumerate([('WGN', wgn), ('HCP', hcp)]):
        # --- F1 ---
        ax = axes[row, 0]
        ax.semilogx(hotel_tr, data['f1_homo'], 'o-', color=colors['homo'],
                     lw=lw, ms=ms, label='Homogeneity', zorder=3)
        ax.semilogx(hotel_tr, data['f1_greedy'], 's--', color=colors['greedy'],
                     lw=lw, ms=ms, label='Greedy', zorder=3)
        # shade the delta
        ax.fill_between(hotel_tr, data['f1_homo'], data['f1_greedy'],
                         alpha=0.15, color=colors['greedy'], zorder=1)
        ax.set_ylabel('F1 Score', fontsize=12)
        ax.set_title(f'{label} -- F1', fontsize=13, fontweight='bold')
        ax.set_ylim(-0.02, 1.05)
        ax.legend(loc='lower right', fontsize=10)
        ax.grid(True, alpha=0.3)
        ax.set_xlabel('hotel_tr (effect size)', fontsize=11)

        # --- Sensitivity ---
        ax = axes[row, 1]
        ax.semilogx(hotel_tr, data['sn_homo'], 'o-', color=colors['homo'],
                     lw=lw, ms=ms, label='Homogeneity', zorder=3)
        ax.semilogx(hotel_tr, data['sn_greedy'], 's--', color=colors['greedy'],
                     lw=lw, ms=ms, label='Greedy', zorder=3)
        ax.fill_between(hotel_tr, data['sn_homo'], data['sn_greedy'],
                         alpha=0.15, color=colors['greedy'], zorder=1)
        ax.set_ylabel('Sensitivity', fontsize=12)
        ax.set_title(f'{label} -- Sensitivity', fontsize=13, fontweight='bold')
        ax.set_ylim(-0.02, 1.05)
        ax.legend(loc='lower right', fontsize=10)
        ax.grid(True, alpha=0.3)
        ax.set_xlabel('hotel_tr (effect size)', fontsize=11)

        # --- Specificity ---
        ax = axes[row, 2]
        ax.semilogx(hotel_tr, data['sp_homo'], 'o-', color=colors['homo'],
                     lw=lw, ms=ms, label='Homogeneity', zorder=3)
        ax.semilogx(hotel_tr, data['sp_greedy'], 's--', color=colors['greedy'],
                     lw=lw, ms=ms, label='Greedy', zorder=3)
        ax.fill_between(hotel_tr, data['sp_greedy'], data['sp_homo'],
                         alpha=0.15, color=colors['homo'], zorder=1)
        ax.set_ylabel('Specificity', fontsize=12)
        ax.set_title(f'{label} -- Specificity', fontsize=13, fontweight='bold')
        ax.set_ylim(0.78, 1.005)
        ax.legend(loc='lower left', fontsize=10)
        ax.grid(True, alpha=0.3)
        ax.set_xlabel('hotel_tr (effect size)', fontsize=11)

    fig.tight_layout(rect=[0, 0, 1, 0.96])

    # --- second figure: delta F1 and region counts ---
    fig2, axes2 = plt.subplots(1, 2, figsize=(14, 5))
    fig2.suptitle('Greedy - Homogeneity: Differences', fontsize=14,
                  fontweight='bold', y=1.0)

    # delta F1
    ax = axes2[0]
    df1_wgn = np.array(wgn['f1_greedy']) - np.array(wgn['f1_homo'])
    df1_hcp = np.array(hcp['f1_greedy']) - np.array(hcp['f1_homo'])
    ax.semilogx(hotel_tr, df1_wgn, 'o-', color='#d6604d', lw=lw, ms=ms,
                label='WGN')
    ax.semilogx(hotel_tr, df1_hcp, 's-', color='#4393c3', lw=lw, ms=ms,
                label='HCP')
    ax.axhline(0, color='k', lw=0.8, ls='--', alpha=0.5)
    ax.fill_between(hotel_tr, 0, df1_wgn, alpha=0.12, color='#d6604d')
    ax.fill_between(hotel_tr, 0, df1_hcp, alpha=0.12, color='#4393c3')
    ax.set_ylabel('delta F1 (greedy - homo)', fontsize=12)
    ax.set_xlabel('hotel_tr (effect size)', fontsize=11)
    ax.set_title('F1 Improvement from Greedy', fontsize=13, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)

    # region counts
    ax = axes2[1]
    ax.semilogx(hotel_tr, wgn['n_homo'], 'o-', color='#d6604d', lw=lw, ms=ms,
                label='WGN homo')
    ax.semilogx(hotel_tr, wgn['n_greedy'], 'o--', color='#d6604d', lw=1.5,
                ms=ms, alpha=0.6, label='WGN greedy')
    ax.semilogx(hotel_tr, hcp['n_homo'], 's-', color='#4393c3', lw=lw, ms=ms,
                label='HCP homo')
    ax.semilogx(hotel_tr, hcp['n_greedy'], 's--', color='#4393c3', lw=1.5,
                ms=ms, alpha=0.6, label='HCP greedy')
    ax.set_ylabel('# regions selected (mean)', fontsize=12)
    ax.set_xlabel('hotel_tr (effect size)', fontsize=11)
    ax.set_title('Number of Output Regions', fontsize=13, fontweight='bold')
    ax.legend(fontsize=10, ncol=2)
    ax.grid(True, alpha=0.3)

    fig2.tight_layout()

    out1 = '/home/matt/Dropbox/glow/src/glow/benchmark/pruning_comparison.png'
    out2 = '/home/matt/Dropbox/glow/src/glow/benchmark/pruning_delta.png'
    fig.savefig(out1, dpi=150, bbox_inches='tight')
    fig2.savefig(out2, dpi=150, bbox_inches='tight')
    print(f'saved: {out1}')
    print(f'saved: {out2}')
    print('done')


if __name__ == '__main__':
    main()
