# Active Learning in Question Answering Systems

This repository contains the implementation and experiments from my master thesis: "Active Learning in Question Answering Systems", completed at the Higher School of Economics, Faculty of Computer Science.

## Abstract

The research focuses on applying active learning strategies to improve the efficiency of annotating datasets for question-answering (QA) systems. These systems generate answers from natural language text, which often requires expensive manual annotation. The thesis explores different active learning approaches, emphasizing diversity-based strategies versus uncertainty-based methods, in the context of generative QA models applied to long documents.

**Key findings indicate that the investigated active learning strategies do not outperform random selection in terms of quality improvements for annotated datasets.**

## Main Contributions

**Active Learning Framework for QA:**
Developed a framework for iterative selection and annotation of samples in QA tasks. Compared uncertainty-based and diversity-based strategies.

**Two-Stage QA Model:**

Implemented a two-stage pipeline combining a classifier and a generative model to handle long-document contexts.

**Comprehensive Evaluation:**

Evaluated strategies using metrics like ROUGE and Exact Match. Applied active learning experiments to the Google Natural Questions dataset.

## Repository Contents

**Code:** Scripts for training and evaluating the QA models and active learning strategies.

**Data Preprocessing:** Tools for filtering and preparing the Google Natural Questions dataset.

**Experiments:** Reproducible experiments for active learning strategies.

**Results:** Plots and metrics illustrating the outcomes of active learning experiments.

## Technologies Used

**Models:** DistilBERT, T5-small

**Programming Languages:** Python

**Frameworks:** PyTorch, Hugging Face Transformers

Feel free to explore the repository and reach out with any questions or suggestions!
