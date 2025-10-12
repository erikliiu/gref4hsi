# AutoTuneMBES Performance Guide

## Problem: Too Many Iterations! 🐌

**Default configuration runs 787 iterations:**
- Round 1 (coarse search): 2 orders × 3 central_fracs × 3 baselines × 3 tilts × 3 centers = **162 iterations**
- Round 2 (refinement): 5 × 5 × 5 × 5 = **625 iterations**
- **Total: 787 iterations** ⚠️

## Solutions to Speed Up

### Strategy 1: Reduce `max_rounds` ⚡
**Set `max_rounds=1`** to skip the refinement round:

```python
best = tuner.fit(max_rounds=1)  # Only 162 iterations
```

- **Speedup: ~80% faster** (162 vs 787 iterations)
- **Trade-off:** Less fine-tuned parameters, but coarse search is usually good enough

---

### Strategy 2: Fix the Polynomial Order 🎯
**Use `allowed_orders=(3,)`** instead of `(2, 3)`:

```python
tuner = AutoTuneMBES(mb, allowed_orders=(3,), ...)
```

- Round 1: **81 iterations** (was 162)
- Round 2: **625 iterations** (unchanged)
- **Speedup: 50% on round 1**

**Why?** Order 3 (quadratic cross-track) is usually better than order 2 (linear), so just use that.

---

### Strategy 3: Coarser Sampling Grid 🏃
**Increase `row_stride` and `col_stride`**:

```python
tuner = AutoTuneMBES(mb, 
                     row_stride=8,   # was 4
                     col_stride=8,   # was 4
                     ...)
```

- **Same number of iterations**, but each one is **4× faster**
- Samples every 8th pixel instead of every 4th
- **Trade-off:** Slightly less accurate quality metrics

---

### Strategy 4: Combined Fast Mode 🚀
**Combine all strategies** for maximum speed:

```python
tuner_fast = AutoTuneMBES(
    mb_ned,
    allowed_orders=(3,),      # Fix order → 81 iterations
    row_stride=8,             # Coarser sampling → 4× faster per iteration
    col_stride=8,
    weights=ScoreWeights(...),
    log_csv="mbes_tuning_log_fast.csv",
    progress_every=10,
)

best = tuner_fast.fit(max_rounds=1)  # Skip refinement
```

**Result:**
- **Only 81 iterations** (was 787)
- Each iteration is **4× faster**
- **Overall: ~40× speedup!** 🎉

---

## Parameter Summary

| Parameter | Default | Fast Mode | Effect |
|-----------|---------|-----------|--------|
| `max_rounds` | 2 | **1** | Skip refinement (80% fewer iterations) |
| `allowed_orders` | `(2, 3)` | **(3,)** | Half the search space |
| `row_stride` | 4 | **8** | 4× faster per iteration |
| `col_stride` | 4 | **8** | 4× faster per iteration |
| **Total iterations** | **787** | **81** | **~10× fewer** |
| **Time per iteration** | 1.0× | **0.25×** | 4× faster |
| **Overall speedup** | 1× | **~40×** | 🚀 |

---

## When to Use Each Mode

### Use SLOW mode (max_rounds=2, orders=(2,3)) when:
- First time tuning a new dataset
- You have time and want the absolute best parameters
- The data is challenging (lots of artifacts, complex bathymetry)

### Use FAST mode (max_rounds=1, orders=(3,), stride=8) when:
- You've already tuned once and just need a quick refinement
- Prototyping or iterating quickly
- The data is relatively clean
- **You value your sanity and don't want to wait forever** 😅

---

## Advanced: Further Optimization

If you really need even faster:

1. **Manually set good starting values** (skip tuning entirely):
   ```python
   mb = MBESDetrender(tif_path).load().detrend(
       order_x=3,
       smooth_baseline_m=0.8,
       smooth_tilt_m=0.7,
       smooth_center_m=1.0,
       robust=True,
       central_frac=0.8,
   )
   ```

2. **Use even coarser stride**: `row_stride=16, col_stride=16` for huge datasets

3. **Reduce the search grid** by modifying `_coarse_space()` in the source code:
   - Change `[0.4, 0.8, 1.2]` to `[0.6, 1.0]` (2 values instead of 3)
   - This would reduce 81 → 32 iterations

---

## Resume Feature

The autotuner automatically saves progress to CSV:
- Set `resume=True` (default) to continue from where you left off
- Change `log_csv` filename to start fresh
- Useful if you need to interrupt a long tuning run

```python
best = tuner.fit(max_rounds=2, resume=True)  # Won't redo completed runs
```
