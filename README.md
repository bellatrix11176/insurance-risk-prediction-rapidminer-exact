# Insurance Risk Prediction — RapidMiner Exact

This project reproduces a RapidMiner decision tree **exactly** using Python by parsing the exported RapidMiner tree text and computing predictions and confidence values directly from each leaf’s class distribution.

The goal is validation and interpretability: demonstrating that a visually designed model can be translated into deterministic, auditable logic **without changing its behavior**.

Running this project produces a scored output file for new applicants using the exact RapidMiner decision logic.

---

## Project Overview

The original model was built in RapidMiner as part of an applied data analytics assignment. RapidMiner was used to:

- prepare the data,
- construct a decision tree,
- generate predictions and confidence values for new applicants.

To verify conceptual understanding beyond the visual tool, the RapidMiner decision tree was exported as text and reconstructed in Python. Instead of training a new model, this project treats the RapidMiner tree as the **source of truth** and evaluates new records by following the same decision paths and leaf distributions.

---

## What “RapidMiner Exact” Means

This implementation does **not** train a new decision tree.

Instead, it:

- parses the exported RapidMiner decision tree text,
- converts each root-to-leaf path into explicit, ordered rules,
- computes confidence values using RapidMiner’s leaf class counts:

