# engine — P1 (Engine)

**Owner:** P1 · **Review buddy:** P2 · **Label:** `P1-engine`

`game.py` is the reference engine: simple, correct and slow. Its public API is
shared by everyone, so don't change it without telling the group:

```python
Game(seed); game.reset(seed); game.step(move) -> (board, reward, done)
game.legal_moves(); game.max_tile(); Game.slide(board, move) -> (new_board, gained)
```

## TODO
- [ ] 64-bit bitboard (16 cells x 4-bit exponents) with row lookup tables for all 4 moves
- [ ] Same public API as `game.py`; `tests/test_engine.py` must pass unchanged
- [ ] 65,536 support: 4-bit cells cap at 2**15 = 32,768, so handle 2**16 explicitly
- [ ] Target ≥ 1M moves/sec; add a micro-benchmark script
- [ ] Python bindings (pybind11 / cffi / Cython) for the C/C++ core
- [ ] Differential test: bitboard vs reference engine on random games
