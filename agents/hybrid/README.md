# agents/hybrid — P7 (Hybrid)

**Owner:** P7 · **Review buddy:** P8 · **Label:** `P7-hybrid`

The strongest agent we expect to have: P5's n-tuple value function used as the leaf
evaluator in P3's expectimax, made fast enough for deep late-game search.

## TODO
- [ ] N-tuple value function inside expectimax
- [ ] Multithreading (e.g. parallel root moves)
- [ ] C++ hot paths (with P1's bitboard bindings)
- [ ] "65,536 mode": deeper search once the 32,768 tile is on the board
- [ ] Register in `eval/run.py` → `AGENTS`; run the late-game benchmark with P8
