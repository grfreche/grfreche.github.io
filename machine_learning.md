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
which gives the best prediction of the ouput,
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

In <a href="https://grfreche.github.io/pdfs/LinearRegression.pdf" class="image fit">this document</a>,
we present the simplest model of regression: **linear regression**.
We introduce the model, we derive **least squares** exact solution and **recursive least squares** (RLS) algorithm,
we extend to some variants of the model and we apply these results to **autoregressive signals**.

Code for linear regression:
* <a href="python_RLS" class="image fit">Python script for linear regression</a>

Code for polynomial regression:
* <a href="python_polynomial_regression" class="image fit">Python script for polynomial regression</a>

Code for the weighted RLS algorithm applied to speech signals:
* <a href="python_weighted_RLS" class="image fit">Python script for weighted RLS applied to speech signal</a>
* <a href="https://grfreche.github.io/sources/linear_regression/hello.wav" class="image fit">An example of wave file to use with script</a>

## Kalman filters

In <a href="https://grfreche.github.io/pdfs/KalmanFilters.pdf" class="image fit">this document</a>,
we introduce a generalization of linear regression: **Kalman filters**,
we derive their update equations and the corresponding algorithm, and we apply them on some examples.

Code for Kalman filters:
* <a href="python_kalman" class="image fit">Python script for Kalman filters</a>
* <a href="cpp_kalman" class="image fit">C++ version</a>

## Non-linear regression

In <a href="https://grfreche.github.io/pdfs/NonLinearRegression.pdf" class="image fit">this document</a>,
we extend to **non-linear regression**, we talk about **Newton-Raphson method** and **gradient descent**,
and we apply these results to a simplified version of neural networks: the **single-neuron classifier**.

Code for Single Neuron Classifiers:
* <a href="python_SNC_affine" class="image fit">Python script for SNC affine separation</a>
* <a href="python_SNC_quadratic" class="image fit">Python script for SNC quadratic separation</a>
* <a href="python_SNC_affine_error" class="image fit">Python script for error estimation of SNC affine separation</a>
* <a href="python_SNC_quadratic_error" class="image fit">Python script for error estimation of SNC quadratic separation</a>

[jekyll-organization]: https://github.com/jekyll
