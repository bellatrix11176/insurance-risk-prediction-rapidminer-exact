# Insurance Data Analysis (RapidMiner)

This repository contains an applied data analysis project focused on analyzing insurance applicant data using **RapidMiner**, with a parallel **Python implementation** used to reproduce and validate the same analytical logic programmatically.

The project emphasizes decision tree interpretability, process validation, and understanding model behavior rather than production-level optimization.

---

## Project Overview

The analysis was structured around a set of predefined assignment questions (Q1–Q10), each designed to evaluate specific aspects of data preparation, decision tree structure, predictions, and confidence levels.

RapidMiner was used to visually design and execute the full analytical workflow, including:
- data cleaning and preparation
- attribute selection
- decision tree generation
- prediction and confidence evaluation for new applicants

To demonstrate conceptual understanding beyond the visual analytics tool, the same analytical steps were later replicated in Python. This included manually implementing decision logic and generating equivalent outputs to confirm the correctness of the results produced in RapidMiner.

---

## Assignment Questions (Q1–Q10)

The file `Decision_Tree_Questions.txt` documents the full set of assignment questions used to guide the analysis. These questions provide the interpretive framework for understanding the decision tree outputs and predictions.

The questions focus on:
- identifying the most predictive attributes in the decision tree
- interpreting classification paths and predicted insurance categories
- evaluating prediction frequencies for new applicants
- analyzing post-probability confidence levels
- understanding how the model distinguishes between risk categories

This file is included to ensure transparency and traceability between the model outputs and the analytical questions they were designed to answer.

---

## Tools & Methods

- **RapidMiner Studio** – visual workflow design and decision tree modeling  
- **Python** – programmatic replication and validation of decision logic  
- **Excel** – source data for analysis  

---

## Repository Structure

- `data/`  
  Contains the raw insurance dataset used in both RapidMiner and Python analyses.

- `rapidminer/`  
  Contains the exported RapidMiner process (`.rmp`), screenshots of the workflow and outputs, and the assignment questions file.

- `python/` (or `src/`)  
  Contains Python scripts and requirements used to replicate and validate the RapidMiner analysis.

---

## Scope & Focus

This project is intentionally structured as a learning and validation exercise. The emphasis is on:
- understanding decision tree logic
- interpreting model outputs and confidence values
- translating visual analytics workflows into code
- documenting analytical reasoning clearly

The goal is not production deployment, but **demonstrating applied understanding and systems-level thinking** across tools.

---

## Notes

A Python-based implementation reproducing the RapidMiner workflow is included to highlight cross-tool consistency and analytical validation.
