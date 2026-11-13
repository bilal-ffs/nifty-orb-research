"""Standalone plots of saved IS outputs; no other data accessed."""
from pathlib import Path
import json
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def main():
    root=Path(__file__).resolve().parent
    c=pd.read_csv(root/'atr_multiplier_comparison.csv')
    folder=root/'charts'; folder.mkdir(exist_ok=True)
    fig,axes=plt.subplots(1,2,figsize=(13,5))
    for mode,g in c.groupby('mode'):
        axes[0].plot(g.multiplier,g.net_expectancy_r,'o-',label=mode)
        axes[1].plot(g.multiplier,g.total_net_pnl,'o-',label=mode)
    axes[0].set(ylabel='Mean net R / trade',xlabel='ATR multiplier',title='IS expectancy: all requested configurations')
    axes[1].set(ylabel='Net INR proxy P&L',xlabel='ATR multiplier',title='IS net P&L, 10% fixed initial-capital notional')
    for ax in axes: ax.axhline(0,color='black',lw=.7); ax.legend(); ax.grid(alpha=.3)
    fig.tight_layout(); fig.savefig(folder/'execution_modes.png',dpi=180); plt.close(fig)
    metrics=json.loads((root/'ml_metrics.json').read_text())
    fig,axes=plt.subplots(1,2,figsize=(13,5))
    for ax,key in zip(axes,['log_loss','brier_score']):
        for basis in ['model','baseline']:
            ax.plot([float(k) for k in metrics],[m[basis][key] for m in metrics.values()],'o-',label=basis)
        ax.set(xlabel='ATR multiplier',ylabel=key,title=f'IS OOF {key}: lower is better'); ax.legend(); ax.grid(alpha=.3)
    fig.tight_layout(); fig.savefig(folder/'ml_probability_scores.png',dpi=180); plt.close(fig)
    q=pd.read_csv(root/'probability_quintiles.csv')
    fig,ax=plt.subplots(figsize=(10,6))
    for multiplier,g in q.groupby('multiplier'):
        ax.plot(g.group,g.mean_net_r,'o-',label=f'{multiplier:g}')
    ax.axhline(0,color='black',lw=.7)
    ax.set(xlabel='OOF probability quintile (low → high)',ylabel='Mean realized net R',title='IS retrospective ranking; no trading filters selected')
    ax.legend(title='ATR multiplier',ncol=2); ax.grid(alpha=.3)
    fig.tight_layout(); fig.savefig(folder/'probability_net_r.png',dpi=180); plt.close(fig)
    report=root/'research_report.md'
    text=report.read_text(encoding='utf-8')
    if '## Saved charts' not in text:
        text+='\n## Saved charts\n\n![Execution mode comparison](charts/execution_modes.png)\n\n![OOF probability scores](charts/ml_probability_scores.png)\n\n![Descriptive net R ranking](charts/probability_net_r.png)\n'
        report.write_text(text,encoding='utf-8')


if __name__=='__main__': main()
