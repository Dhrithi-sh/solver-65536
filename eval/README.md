# eval — P8 (Evaluation)

**Owner:** P8 · **Review buddy:** P7 · **Label:** `P8-eval`

`run.py` is the shared-seed benchmark and the only source of reported numbers.
`replay.py` replays a logged game from its seed and move list.

```bash
python -m eval.run --agent greedy --games 100 --seed0 0
python -m eval.replay results/greedy.jsonl --index 0 --show
```

## TODO
- [ ] Fix the official seed ranges (e.g. dev 0–999, final 10000–10999)
- [ ] Late-game benchmark: generate and save 32,768 boards (+ seeds), run each agent from them,
      report the 32,768 → 65,536 conversion rate
- [ ] Results dashboard (reach rates, score distributions, ms/move)
- [ ] Replay viewer / web demo (see `ui/`)
- [ ] Report and slides (see `report/`)
