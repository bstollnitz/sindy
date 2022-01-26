# Discovering equations from data using SINDy

For a detailed explanation of the code in this folder, check out the accompanying blog post: [Discovering equations from data using SINDy](https://bea.stollnitz.com/blog/sindy-lorenz/).


## How to run this code on a GitHub Codespace

In the repo's GitHub page, click on the "Code" button, then select the Codespace called "main*". This will open VS code on your browser.

Run the following files by pressing F5 with the file open, in this order:
* 1_generate_data.py
* 2_fit.py
* 3_predict.py


## How to run this code on your local machine

Install conda environment:

```
conda env create -f conda.yml
```

Activate conda environment:

```
conda activate sindy
```

Run the following files by pressing F5 with the file open, in this order:
* 1_generate_data.py
* 2_fit.py
* 3_predict.py
