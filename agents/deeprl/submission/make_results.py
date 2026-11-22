"""Run: python make_results.py /path/to/folder/containing/the/four/input/files"""
import csv
import json
import statistics as st
import sys
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

source = Path(sys.argv[1])
out = Path(__file__).resolve().parent
names = ['random', 'greedy', 'dqn']
labels = ['Random', 'Greedy', 'DQN']
colors = ['#8795a5', '#d18b2c', '#3479b7']
data = {}
for name in names:
    rows = [json.loads(line) for line in (source / (name + '.jsonl')).read_text().splitlines() if line.strip()]
    assert len(rows) == 100 and sorted(r['seed'] for r in rows) == list(range(100, 200))
    assert all(r['agent'] == name and r['n_moves'] == len(r['moves']) for r in rows)
    data[name] = sorted(rows, key=lambda r: r['seed'])
training = list(csv.DictReader((source / 'dqn_500k.csv').open()))
steps = np.array([int(r['step']) for r in training])
scores = np.array([float(r['score']) for r in training])
epsilon = np.array([float(r['epsilon']) for r in training])
assert np.all(np.diff(steps)>0)
assert not set(int(r['train_seed']) for r in training) & set(range(100, 200))
means = [st.mean(r['score'] for r in data[n]) for n in names]
fig, ax = plt.subplots(figsize=(8,5), layout='constrained')
bars = ax.bar(labels, means, color=colors, width=.58)
ax.bar_label(bars, labels=[f'{x:,.1f}' for x in means], padding=5)
ax.set_ylim(0, 3550)
ax.set_ylabel('Mean game score')
ax.set_title('Same-seed evaluation: 100 games per agent\nSeeds 100–199; DQN trained for 302,560 moves')
ax.spines[['top','right']].set_visible(False)
fig.savefig(out/'agent_comparison.png',dpi=180)
plt.close(fig)
fig, axes = plt.subplots(2,1,figsize=(9,7),sharex=True,layout='constrained')
axes[0].scatter(steps,scores,s=5,color='#3479b7',alpha=.18,label='Completed training game')
window=50
rolling=np.convolve(scores,np.ones(window)/window,mode='valid')
axes[0].plot(steps[window-1:],rolling,color='#173e65',linewidth=2,label='50-game trailing mean')
axes[0].set_ylabel('Training game score')
axes[0].legend(frameon=False)
axes[0].set_title('DQN training diagnostics — exploration was enabled\nThese are not held-out evaluation results')
axes[1].plot(steps,epsilon,color='#d18b2c',linewidth=2)
axes[1].set_ylabel('Random-move probability (epsilon)')
axes[1].set_xlabel('Training moves')
axes[1].set_ylim(0,1.05)
for ax in axes:
    ax.spines[['top','right']].set_visible(False)
    ax.grid(axis='y',alpha=.15)
fig.savefig(out/'training_curve.png',dpi=180)
plt.close(fig)

comparison = {}
for name in ['dqn', 'ntuple']:
    rows = [json.loads(line) for line in (source/'compare5min'/f'{name}.jsonl').read_text().splitlines() if line.strip()]
    assert len(rows) == 100
    assert sorted(r['seed'] for r in rows) == list(range(100000, 100100))
    assert all(r['agent'] == name and r['n_moves'] == len(r['moves']) for r in rows)
    comparison[name] = rows
budget = json.loads((source/'compare5min'/'budget.json').read_text())
assert budget['completed_budget'] and budget['requested_seconds'] == 300
means = [st.mean(r['score'] for r in comparison[n]) for n in ['dqn','ntuple']]
fig, ax = plt.subplots(figsize=(8,5), layout='constrained')
bars = ax.bar(['DQN','N-tuple'], means, color=['#3479b7','#d18b2c'], width=.55)
ax.bar_label(bars, labels=[f'{v:,.1f}' for v in means], padding=5)
ax.set_ylim(0, max(means)*1.18)
ax.set_ylabel('Mean game score')
ax.set_title('Five-minute training budgets: separate experiment\n100 evaluation games each; seeds 100000–100099')
ax.spines[['top','right']].set_visible(False)
fig.savefig(out/'five_minute_comparison.png', dpi=180)
plt.close(fig)
print('Validated inputs and generated three graphs. Report unchanged.')
