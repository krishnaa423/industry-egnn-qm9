# Equivariant Molecular Force Model

EGNN-style molecular force learning using QM9-like molecular graphs.

## Result

`src/train.py` contains a compact PyTorch implementation of message passing that
updates scalar node features and coordinate-dependent messages. It is structured
so it can be connected to `datasets.load_dataset("qm9")`, PySCF-derived labels,
or local XYZ files.

The target portfolio result is a training curve for energy/force loss and a
plot of predicted versus reference force components.

## Run

```bash
pip install torch numpy matplotlib datasets
python src/train.py
```

The default run uses synthetic molecular graphs so the code path is testable
without downloading QM9. Replace `synthetic_batch()` with a QM9 loader when
network and dataset access are available.
