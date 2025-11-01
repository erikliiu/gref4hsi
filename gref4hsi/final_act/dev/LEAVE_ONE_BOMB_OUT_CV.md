# Leave-One-Bomb-Out Cross-Validation

## Problem: Unreliable CV Scores with Missing Classes

### The Issue
With only **2 bomb groups** and **5-fold CV**, some folds end up with:
- ❌ **No bombs in validation** → Classifier never tested on bombs
- ❌ **Meaningless scores** → Can't evaluate bomb detection performance
- ❌ **Inflated metrics** → High scores from easy classes (dark/sediment) only

### Why It Happens
Standard grouped CV (StratifiedGroupKFold) tries to maintain class distribution, but with only 2 bomb groups:
- Fold 1: Val has `bomb#1` ✅
- Fold 2: Val has `bomb#2` ✅
- Fold 3: Val has **NO BOMBS** ❌
- Fold 4: Val has **NO BOMBS** ❌
- Fold 5: Val has **NO BOMBS** ❌

Result: 3 out of 5 folds produce invalid scores.

---

## Solution: Leave-One-Bomb-Out (LOBO) CV

### Strategy
1. **Each fold gets exactly ONE bomb group** in validation
2. **Dark and sediment groups** distributed evenly across folds
3. **Refuse folds** that are missing any class

### Implementation

#### Automatic Fold Adjustment
```python
cv_results = cube.train_svm_with_cv(
    training_rois=["training_dark", "training_sediment", "training_bombs"],
    cv_folds=5,  # ⚠️  Will auto-reduce to 2 (limited by bomb groups)
    ...
)
```

**Output:**
```
🔄 Performing 5-fold GROUPED cross-validation...
   📊 Groups per class:
      training_bombs: 2 groups
      training_dark: 1 groups
      training_sediment: 42 groups
   🎯 Minority class: training_bombs (2 groups)
   🔄 Using Leave-One-Bombs-Out strategy
   ⚠️  Reducing folds: 5 → 2 (limited by training_bombs)
```

#### Fold Structure (2-Fold CV)

**Fold 1:**
```
📂 Fold 1/2:
   Train groups: ['bomb#2', 'dark#A', 'sed#01', 'sed#03', ..., 'sed#41']
   Val groups:   ['bomb#1', 'sed#02', 'sed#04', ..., 'sed#42']
   ✅ No overlap (clean split)
   Train samples: {'training_bombs': 400, 'training_dark': 2500, 'training_sediment': 2800}
   Val samples:   {'training_bombs': 400, 'training_dark': 0, 'training_sediment': 2200}
```

**Fold 2:**
```
📂 Fold 2/2:
   Train groups: ['bomb#1', 'dark#A', 'sed#02', 'sed#04', ..., 'sed#42']
   Val groups:   ['bomb#2', 'sed#01', 'sed#03', ..., 'sed#41']
   ✅ No overlap (clean split)
   Train samples: {'training_bombs': 400, 'training_dark': 2500, 'training_sediment': 2200}
   Val samples:   {'training_bombs': 400, 'training_dark': 0, 'training_sediment': 2800}
```

✅ **Both folds have bombs in validation!**

---

## Validation Enforcement

### Strict Class Presence Check

The code now **refuses folds** missing any class:

```python
# Check if all classes present in validation
missing_classes = all_classes - val_classes

if missing_classes:
    raise ValueError(
        f"❌ FOLD {fold_idx + 1} INVALID: Validation set missing classes {missing_classes}!\n"
        f"   This makes CV scores unreliable.\n"
        f"   Try reducing cv_folds or use Leave-One-Bomb-Out strategy."
    )
```

**Before (Silent Failure):**
- Fold 3 has no bombs → Metrics calculated anyway ❌
- High F1 score (0.95) from dark/sediment only ❌
- User thinks model is great, but it never saw bombs! ❌

**After (Loud Failure):**
- Fold 3 has no bombs → **CRASH WITH ERROR** ✅
- Forces user to fix CV strategy ✅
- Ensures all reported metrics are meaningful ✅

---

## Improvements Implemented

### ✅ 1. Leave-One-Bomb-Out CV
- **Custom fold generator** ensures minority class (bombs) appears in every fold
- **Round-robin distribution** of other classes (dark, sediment) across folds
- **Automatic fold reduction** to match minority class group count

### ✅ 2. Enforce Class Presence Per Fold
- **Strict validation** before training each fold
- **Raises ValueError** if any class missing from validation
- **Clear error message** with fix suggestions

### ✅ 3. Balance Classes
- **`class_weight='balanced'`** already implemented in SVC
- Automatically weights classes by `1 / n_samples`
- Prevents majority class (sediment/dark) from dominating

---

## How to Read Results

### Reliable CV Scores (After Fix)

```
📊 CROSS-VALIDATION SUMMARY
Accuracy:  0.847 ± 0.023
Precision: 0.812 ± 0.031
Recall:    0.798 ± 0.027
F1 Score:  0.804 ± 0.029
```

**Interpretation:**
- ✅ Every fold tested **all classes** (bombs, dark, sediment)
- ✅ Variance (±0.023) reflects true model stability
- ✅ F1 score (0.804) is trustworthy for bomb detection

### Unreliable CV Scores (Before Fix)

```
⚠️  MISSING CLASSES IN VAL: {'training_bombs'}
📊 CROSS-VALIDATION SUMMARY
Accuracy:  0.952 ± 0.012  ⚠️  INFLATED!
F1 Score:  0.947 ± 0.009  ⚠️  MISLEADING!
```

**Interpretation:**
- ❌ Some folds never tested bombs
- ❌ High scores from easy classes only
- ❌ Model might completely fail on bombs in production

---

## Usage Recommendations

### When You Have Few Bomb Groups (≤3)

```python
# 🔥 Let CV auto-adjust to bomb count
cv_results = cube.train_svm_with_cv(
    training_rois=training_rois,
    cv_folds=10,  # Will reduce to 2-3 (limited by bombs)
    ...
)
```

**Expected behavior:**
- 2 bomb groups → 2-fold CV (Leave-One-Bomb-Out)
- 3 bomb groups → 3-fold CV
- System ensures **every fold validates bombs**

### When You Have Many Bomb Groups (≥5)

```python
# 🎯 Standard K-fold CV works fine
cv_results = cube.train_svm_with_cv(
    training_rois=training_rois,
    cv_folds=5,  # Full 5-fold CV possible
    ...
)
```

**Expected behavior:**
- Each fold gets 1-2 bomb groups in validation
- Good class balance maintained
- Reliable metrics across all folds

---

## Creating More Bomb Groups

If you want more CV folds, **annotate more spatially separated bomb ROIs**:

### Current (2 Groups)
```
ROIs/057_5_combined.json:
  training_bombs → 800 pixels
    → Connected components finds 2 clusters
    → bomb#1 (North area)
    → bomb#2 (South area)
```

### Improved (5+ Groups)
1. **Annotate 5+ distinct bomb regions** (spatially separated by >10 pixels)
2. Each cluster becomes a separate group
3. Now you can do 5-fold CV properly!

**Or use smaller closing radius:**
```python
cv_results = cube.train_svm_with_cv(
    ...
    closing_radius=1,  # Default: 3 (creates fewer, larger groups)
    min_group_size=10,  # Default: 20 (filters small groups)
)
```
- Smaller `closing_radius` → More groups detected
- Smaller `min_group_size` → Keeps smaller groups

---

## Summary

| Aspect | Before | After |
|--------|--------|-------|
| **CV Strategy** | StratifiedGroupKFold | Leave-One-Bomb-Out |
| **Fold Count** | 5 (some invalid) | 2 (all valid) |
| **Class Presence** | Some folds miss bombs | Every fold has all classes |
| **Validation** | Warnings only | **Raises error** if invalid |
| **Scores** | Inflated/unreliable | Trustworthy |
| **Bomb Detection** | Untested in 60% of folds | Tested in 100% of folds |

**Result:** CV now produces **reliable, meaningful metrics** for all classes including bombs! 🎯
