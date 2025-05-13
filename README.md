# grokking_deep_learning
This repository contains structured notes, annotated code walkthroughs, and hands-on implementations of key concepts spanning linear algebra, deep neural networks, optimization, and real-world applications, all aligned with an interactive and practical approach to learning deep learning from the ground up.


### A typical training process

![My Image](assests/training-process.png)


- Start off with a randomly initialized model that cannot do anything useful.
- Grab some of your data (e.g., audio snippets and corresponding `{yes, no}` labels).
- Tweak the knobs to make the model perform better as assessed on those examples.
- Repeat Steps `2` and `3` until the model is awesome.



### Core components in machine learning;
- The `data` that we learn from.
- A `model` of how to transform the data.
- An `objective function` that quantifies how well (or badly) the model is doing.
- An `algorithm` to adjust the model's parameters to optimize the objective function.


### Kinds of Machine Learning Problems;

#### 1. **Supervised Learning**
`Supervised learning` is a type of machine learning where the goal is to **predict a designated 
unknown label** (output) based on **known input features**, using a dataset that contains 
**example pairs of inputs and their corresponding labels**. The model learns a mapping from 
inputs to outputs by generalizing from this labeled data to make accurate predictions on unseen instances.


![My Image](assests/supervised-learning.png)



a. **Regression**

What makes a problem a regression is actually the form of the target. Say that you are in the market 
for a new home. You might want to estimate the fair market value of a house, given some features 
such as (`square footage`, `no. of bedrooms`, `no. of bathrooms`, `walking distance`). 
The data here might consist of historical home listings and the labels might be the 
observed sales prices. When labels take on arbitrary numerical values (even within some interval),
we call this a `regression problem`. The goal is to produce a model whose predictions closely 
approximate the actual label values.




b. **Classification**




c. **Tagging**




d. **Search**





e. **Recommender Systems**




```
poetry run python main.py
poetry run pytest
```