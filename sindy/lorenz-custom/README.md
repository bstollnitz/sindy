# Discovering equations from data using SINDy

For a detailed explanation of the code in this folder, check out the accompanying blog post: [Discovering equations from data using SINDy](https://bea.stollnitz.com/blog/sindy-lorenz/).


## Setup

If you have access to GitHub Codespaces, click on the "Code" button in this GitHub repo, select the "Codespaces" tab, and then click on "New codespace". Alternatively, you can set up your local machine using the following steps.

Install conda environment:

```
conda env create -f conda.yml
```

Activate conda environment:

```
conda activate sindy
```


## Run

Within VS code, open the following files and press F5 to run them, in this order:
* 1_generate_data.py
* 2_fit.py

The last file shows a matplotlib graph. If you're running locally, you can press F5, and a window opens showing the graph. If you're running on Codespaces, you need to display the graph on a VS Code Interactive Window, which you can do by opening the Command Palette (Ctrl + Shift + P) and choosing "Jupyter: Run Current File in Interactive Window".
* 3_predict.py

