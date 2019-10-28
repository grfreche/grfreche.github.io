---
layout: default
title: Machine learning
permalink: /machine_learning/
---

On this page, we address various machine learning problems.

# Regression

Imagine a system working like a black box,
that we can feed with some input and observe the corresponding output.
**Regression** and **classification** are two important problems
met in **adaptive filtering** and **supervised machine learning**,
aiming at modeling the behavior of this system and predicting its output.
The main difference between these two problems is the nature of the output:
* in regression, both input and output are numerical values,
scalars or vectors, and we aim at finding the mathematical mapping
which gives the best prediction of the output,
in a sense that we will define in the documents;
* in classification, the output is a **label** belonging to a finite or countable set,
and we aim at determining the correct label associated with the input.

In this part, we will mainly focus on regression, but we will also deal with a simplified classification problem
seen as a special case of regression. Supervised learning algorithms work in two stages:
* a **training** stage, during which the algorithm is fed with examples of input and output
and construct a model mapping them while minimizing some cost function;
* a **test** stage, during which the algorithm is only fed with input
and uses its learned model to estimate the output.

We will insist on this machine learning aspect in our examples. 

## Linear regression

In [this document](/machine_learning/pdfs/LinearRegression.pdf),
we present the simplest model of regression: **linear regression**.
We introduce the model, we derive **least squares** exact solution and **recursive least squares** (RLS) algorithm,
we extend to some variants of the model and we apply these results to **autoregressive signals**.

Code for linear regression:
* [Python script for linear regression](python_RLS)

Code for polynomial regression:
* [Python script for polynomial regression](python_polynomial_regression)

Code for the weighted RLS algorithm applied to speech signals:
* [Python script for weighted RLS applied to speech signals](python_weighted_RLS)
* [An example of wav file to use with script](hello.wav)

## Kalman filters

In [this document](/machine_learning/pdfs/KalmanFilters.pdf),
we introduce a generalization of linear regression: **Kalman filters**,
we derive their update equations and the corresponding algorithm, and we apply them on some examples.

Code for Kalman filters:
* [Python script for Kalman filters](python_kalman)
* [C++ version](cpp_kalman)

## Non-linear regression

In [this document](/machine_learning/pdfs/NonLinearRegression.pdf),
we extend to **non-linear regression**, we talk about the **Newton-Raphson method** and **gradient descent**,
and we apply these results to a simplified version of neural networks: the **single-neuron classifier**.

Code for Single Neuron Classifiers:

* [Python script for SNC affine separation](python_SNC_affine)
* [Python script for SNC quadratic separation](python_SNC_quadratic)
* [Python script for error estimation of SNC affine separation](python_SNC_affine_error)
* [Python script for error estimation of SNC quadratic separation](python_SNC_quadratic_error)

# Multilayer Perceptron

In [this document](/machine_learning/pdfs/MultilayerPerceptron.pdf),
we present the general structure of multilayer perceptrons, 
and derive the **feed-forward** and **back-propagation** equations and algorithms.

Code for Multilayer Perceptrons:

* [C++ code](cpp_mlp)

We use this code to generate **neural activation maps** of multilayer perceptrons trained on the XOR and chessboard problems. 
These maps are displayed in the following pages:

* [Activation maps for the XOR problem](mlp_xor)
* [Activation maps for the chessboard problem](mlp_chessboard)

[jekyll-organization]: https://github.com/jekyll
