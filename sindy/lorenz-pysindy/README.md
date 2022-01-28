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
* 3_predict.py
