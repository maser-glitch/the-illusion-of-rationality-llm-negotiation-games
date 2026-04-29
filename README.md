# The Illusion of Rationality: Tacit Bias and Strategic Dominance in Frontier LLMs

This repository holds the code to reproduce the results of the article **The Illusion of Rationality: Tacit Bias and Strategic Dominance in Frontier LLMs**.

## Prerequisites

Our experiments were performed using **Python 3.11.11**, and we recommend adhering to that version.

After you have a fresh python environment set up, install our requirements with:

```bash
pip install -r requirements.txt
```

To reproduce our results, you must have the **NEGOTIATION ARENA FRAMEWORK** installed in your environment.

**Note:** This code relies on the implementation stored at [ManuelRios18/NegotiationArena](https://github.com/ManuelRios18/NegotiationArena), as this version supports the usage of the LLM providers used in Bancolombia.

## Usage

### Reproducing our Plots
All logs obtained during our simulations are stored in the directory `.logs`. All the code needed to reproduce our plots is stored in the package `metrics`.

The code is arranged so that each file reproduces one plot. To use them, simply execute the following code from the root of this project:

```bash
python -m metrics.NAME_OF_THE_FILE
```

For instance, to get the first Figure of our paper, run:

```bash
python -m metrics.buy_sell_anchoring
```

The result will be stored in `metrics/plots`. Each script is self-contained and concise.

### Reproducing our Results
Each of the scripts in the root of this repository runs one of the experiments performed. You can run them with the command:

```bash
python EXPERIMENT.py
```

For instance, to execute the multi-turn ultimatum game experiments, run:

```bash
python ultimatum.py
```