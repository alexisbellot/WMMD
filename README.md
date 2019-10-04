# A *fair* kernel two sample test

**[Work in progress]** This is a python implementation of the algorithm in the paper ["A *fair* kernel two sample test"](). Hypothesis testing can help decision-making by quantifying distributional differences between two populations from observational data. However, these tests may inherit systematic biases embedded in the data collection mechanism - that leads, for example, to some instances being more likely included in our sample - and reproduce unfair or confounded decisions. We propose a two-sample test that adjusts for differences in marginal distributions of confounding variables. The goal of this project is to provide a test statistic that adjusts for distributional differences that may not be of interest to the researcher. This repository contains synthetic data generation to understand the behaviour of our test in various settings and exisiting tests used in the literature for comparison. 

*Please cite the above paper if this resource is used in any publication*

## Dependencies
The only significant dependency is python 3.6 and standard libraries.

## First steps
To get started, check *Tutorial.ipynb* which will guide you through the test from the beginning. 

If you have questions or comments about anything regarding this work, please do not hesitate to contact [Alexis](https://alexisbellot.github.io/Website/)
