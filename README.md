# Insurance Risk Prediction — RapidMiner Exact

This project reproduces a RapidMiner decision tree **exactly** using Python by parsing the exported RapidMiner tree text and computing predictions and confidence values directly from each leaf’s class distribution.

The goal is validation and interpretability: demonstrating that a visually designed model can be translated into deterministic, auditable logic **without changing its behavior**.

Running this project produces a scored output file for new applicants using the exact RapidMiner decision logic.

---

## What This Project Does

- Reads an insurance dataset from `data/InsuranceData.xlsx`
- Loads the RapidMiner decision tree export from `rapidminer/rapidminer_tree_exports.txt`
- Converts each root-to-leaf path into explicit rules
- Scores every record in the **New Applicants** sheet by following the same decision paths RapidMiner uses
- Writes a scored CSV output containing predictions and confidence values

---

## What “RapidMiner Exact” Means

This implementation does **not** train a new decision tree.

Instead, it treats the RapidMiner tree export as the **source of truth** and:

- parses the exported RapidMiner decision tree text,
- converts each root-to-leaf path into ordered rules,
- computes confidence values using RapidMiner’s leaf class counts:


MIT License

Copyright (c) 2026 Gina Aulabaugh

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
🌐 **PixelKraze Analytics (Portfolio):** https://pixelkraze.com/?utm_source=github&utm_medium=readme&utm_campaign=portfolio&utm_content=homepage

