# shift-detect — 5 scripts

```
digest.py              # 1. data digestion  → Digested
interactions.py        # 2. PO-risk / MMD / R-risk / power-loss / RF·DR·CF
run_benchmark_all.py   # 3. main: digest → interactions → DataFrame
utils.py               # 4. tiny shared helpers (RF makers, folds, batch_xy)
subset_localize.py     # 5. ORIGINAL: CFPerm block LOCO → driving feature subset
datasets/              # dataset-specific only (BiasFinder, ModaSense, FE)
```

## Logic

```
datasets/<name>.digest()
        │
        ▼
   Digested  (X, y, schema.blocks / modality)
        │
        ├─ interactions.window_bundle(...)     # RF/DR/RR/MMD/C2ST/ρ_PO/π
        │         └─ run_benchmark_all.py      # B reps → csv/parquet
        │
        └─ subset_localize                     # AFTER block CFPerm wins
              loco_rank_coords / carrier_rank
              → select_subset(names)
```

## Run

```bash
source ./env_full.sh
python digest.py --dataset modasense_emotion --audio --write
python run_benchmark_all.py --dataset modasense_emotion --B 100 --n-perm 150
# localization (packaged original):
python -c "from interactions import localize_subset; ..."
# or BiasFinder demo:
# python subset_localize.py --demo-sentiment
```

Canonical table: `results/joint_shift_analysis.tex`  
Final manuscript: `results/manuscript_final.tex`  
`_legacy/` = old one-offs / shims (not the surface).
