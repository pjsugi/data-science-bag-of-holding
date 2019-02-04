# Table of Contents
[ML - Basics](paper_summaries.md#machine-learning---basics)  
[ML - Specific Techniques](paper_summaries.md#machine-learning---specific-techniques)  
[Network Analysis](paper_summaries.md#network-analysis)  
[Experimentation](paper_summaries.md#experimentation)  
[Deep Learning](paper_summaries.md#deep-learning)  
[Random Stuff](paper_summaries.md#random-stuff)  


## Machine Learning - Basics

- [A few useful things to know about machine learning - Domingos](A%20few%20useful%20things%20to%20know%20about%20machine%20learning.pdf)  
  12 rules of thumb about machine learning

- [Economists (and economics) in Tech Companies](Economists%20in%20Tech%20Companies.pdf)  
  More Ph.D. economists in tech than academia. Economists "have a very particular set of skills" that are valuable to tech companies. Some examples are quantifying causal relationships from experiments (see below for paper on regression discontinuity), designing marketplaces and incentives (supply/demand, auctions, etc.).

- [Leakage in data mining - formulation, detection, and avoidance](Leakage%20in%20data%20mining%20-%20formulation%2C%20detection%2C%20and%20avoidance.pdf)  
  Leakage can easily render a ML model (or Kaggle competition) useless, but it can be hard to catch. However, it should in all cases be possible to catch. If not caught, it can cause you to overestimate the performance of a model that you put into production, potentially costing the company money, give customers a bad experience, and/or cause professional embarrassment. This paper defines the problem, gives examples of different types and sources of leakage, and suggests a few methods to avoid it.

- [Machine Learning - the high-interest credit card of technical debt](Machine%20Learning%20-%20the%20high-interest%20credit%20card%20of%20technical%20debt.pdf)  
  Technical debt in production machine learning systems is remarkably easy to accumulate. This paper identifies many possible risk factors and design (anti-)patterns that lead to tech debt. Some examples are boundary erosion, entanglement, hidden feedback loops, undeclared consumers, data dependencies, and changes in the external world.

- [Rules of ML - best practices for ML engineering (Google)](Rules%20of%20ML%20-%20best%20practices%20for%20ML%20engineering.pdf)  
  - Start simple, make sure both metrics and labels are correct
  - Monitor ML infrastructure, data (distributions, missing values, etc.), and predictions (are offline & online prediction distributions the same?)
  - Avoid leakage and productionization errors
      - Use same code for offline and online pipelines (if possible)
      - Log features/predictions and train future models using logged features
      - Split training/test by time

- [What's your ML test score? A rubric for ML production systems](ml_test_score.pdf)  
  How close is your production system to following best practices? Similar to the [Joel test: 12 steps to better code](https://www.joelonsoftware.com/2000/08/09/the-joel-test-12-steps-to-better-code/)

## Machine Learning - Specific Techniques
- [Collaborative filtering with temporal dynamics](Collaborative%20Filtering%20with%20temporal%20dynamics.pdf) and [The BellKor solution to the Netflix grand prize](The%20BellKor%20Solution%20to%20the%20netflix%20grand%20prize.pdf)  
  The BellKor team won the 1M$ Netflix competition which was to predict what rating a user would give a movie, given only information about past ratings. One of factors that enabled their models to perform well was how they dealt with covariate shift over time. In the case of Netflix, this could arise for a variety of reasons (movies change in popularity over time, a person rates movies differently over time, a product change causes all ratings to increase, etc.). These papers found that dealing with temporal shifts was essential for accurate predictions and showed that 2 methods worked well.

  First, splitting known factors into baseline and temporal effects (e.g. each person has a different baseline avg rating that also changes over time, at different rates for each person). Second, capturing transient effects (such as holidays, oscar nominations, or release of a sequel) by comparing the person's feature value to everybody else's feature values in that time period. This capture the assumption that only the relative effect matters (everyone's rating are affected similarly). The combination of the two method allows them to model both the absolute effects and the relative effects.

- [A Preprocessing Scheme for High-Cardinality Categorical Attributes in Classification and Prediction Problems](A%20Preprocessing%20Scheme%20for%20High-Cardinality%20Categorical%20Attributes%20in%20Classification%20and%20Prediction%20Problems.pdf)  
  Categorical variables must (usually) be transformed before being used by an ML algorithm. Target (mean) encoding is a powerful method which encodes categories by grouping together everyone with the same value for the category, then taking the mean of their labels. This reduces the dimensionality from N categories to a single continuous variable. This has empirically been found to be very useful in many Kaggle competitions. However, it has two major drawbacks. The first is that an algorithm can no longer learn interaction effects, and the second is that categories with small sample sizes (SSS) may be overfit.

  This paper address the SSS problem. An example is a rare category with a single member. They will be given a feature value of exactly their label. Instead, this paper suggests we take an (empiracal) bayesian approach, and estimate the value by combining our prior belief and posterior observation using a blending parameter (all of which are obtained from the data). For a rare category, the prior term will dominate, but for popular categories, the posterior term will dominate. Another great example of using empirical bayes is from a [blog post which uses a baseball example to estimate a player's true batting average](http://varianceexplained.org/r/empirical_bayes_baseball/).

  Note: this paper does not discuss the danger of leakage which can arise since labels are being used to build features. The bayesian approach reduces the impact of leakage, but doesn't eliminate the possibility. One must make sure to set aside a "preprocessing" sample which is separate from the training sample (and obviously the validation/test samples as well).

- [Mining with rarity: a unifying framework](Mining%20with%20rarity%20-%20a%20unifying%20framework.pdf)  
  The problem of rarity refers to both rare classes (class imbalance problem) and rare cases. Rare objects can be more interesting than common objects, but most ML algorithms have difficulty dealing with rarity. Some reasons might be because the evaluation metric is not appropriate (for instance, accuracy, which places more weight on common classes) or because the algorithm uses a greedy search technique (most decision tree based approaches). The authors then suggest methods to address the problem of rarity (sampling methods, cost-sensitive learning (boosting), splitting out rare classes/cases, and more).

## Network Analysis
- [Node2Vec](node2vec-kdd16.pdf)  
  Embedding for graphs. Combines ideas of skip-grams from Word2Vec and random walks from PageRank to extract embeddings. These embeddings are flexible enough (as opposed to previous attempts) to encode representations based on the communities a node belong to (i.e. homophily) and also the structural roles of the node in the network (i.e. structural equivalence). Thus, nodes that are in the same community have similar embeddings and nodes that share the same structural role (e.g. a hub) also have similar embeddings. This is a hard task (prior to Node2Vec) since homophily emphasizes connectivity and structural equivalence does not. Node2Vec can represent both, which is desirable because nodes exhibit both behaviors.

- [Graphs over Time - Densification Laws, Shrinking Diameters and Possible Explanations](Graphs%20over%20Time%20-%20Densification%20Laws%2C%20Shrinking%20Diameters%20and%20Possible%20Explanations.pdf)  
  Graphs change over time. This paper examines why that is the case and how that affects properties of the graph.

- [Inferring User Demographics and Social Strategies in Mobile Social Networks](Inferring%20User%20Demographics%20and%20Social%20Strategies%20in%20Mobile%20Social%20Networks.pdf)  
  Inferring a user's unknown quantities from their behavior in a social network.

- [Predicting poverty and wealth from mobile phone metadata](Predicting%20poverty%20and%20wealth%20from%20mobile%20phone%20metadata%20-%20Blumenstock.pdf) and [Supplementary Materials](Blumenstock.SM.pdf)  
  This paper shows that mobile phone metadata collected from telecoms can be used to accurately estimate a person's income (and by inference other quantities of interest). The supplementary materials detail an interesting method the researchers used to automate their feature engineering process. They defined a set of allowable transformations, then applied all possible transformations to the raw data to extract a set of features.

- [Network diversity and economic development](Network%20Diversity%20and%20Economic%20Development.pdf)  
  Used a telecom dataset to show that larger diversity of relationships correlates to higher economic development.

## Experimentation
- [Why Marketplace Experimentation Is Harder than it Seems- The Role of Test-Control Interference](Why%20Marketplace%20Experimentation%20Is%20Harder%20than%20it%20Seems-%20The%20Role%20of%20Test-Control%20Interference.pdf)  
  When running A/B tests in marketplaces (or networks), one must make sure to make sure the treatment doesn't affect the control group. One way of doing this is propensity matching of markets. For instance Uber/Lyft might apply a treatment to everyone in city A, using city B as control where A & B and as similar as possible (and possibly using more pairs of cities). Or a social network will run an A/B test on two different cliques of users who are more than n-degrees of separation from each other. eBay researchers discuss an example where they ran an email test but did not ensure separation of treatment/control groups for a given auction. This resulted in the outcomes of the treatment group affecting the outcomes of the control group (treatment group received a reminder and were therefore more likely to bid and win an auction, which then meant the control group was less likely to win the auction).

- [A practical guide to regression discontinuity](A%20practical%20guide%20to%20regression%20discontinuity.pdf)  
  Regression discontinuity is a technique used to estimate causal effects from "natural experiments" that arise when a sample group is selected for treatment based on whether their feature value exceeds a threshold. An example would be if a lending company uses a person's credit score to determine what interest rate to offer. If the threshold is a credit score of 740, people with credit scores of 739 and 741 should be similar, but would be offered different interest rates. We can therefore assess the impact of raising interest rates by comparing the samples above and below the threshold.

## Deep Learning
- [Efficient estimation of word representations in vector space](Efficient%20estimation%20of%20word%20representations%20in%20vector%20space.pdf)  
  Original Word2Vec paper. King + Man = Queen + Woman. Showed the power of expressing a categorical variable as a rich vector (word embeddings). Trained the model by using the context of words, assuming that words close to each other are similar to each other.

- [LeNet to ResNet](lec01_cnn_architectures.pdf)  
  Great set of slides covering 17 years of developments in convolutional neural networks.

- [ImageNet Classification with Deep Convolutional Networks (2012)](AlexNet.pdf)  
  Dominated ImageNet 2012 by using a "deep" network that was facilitated by the use of GPUs. Also, the first use of "dropout" as a regularization method. 5 layers.

- [VGGNet (2014)](VGGNet.pdf)  
  Novel idea was to use 3x3 filters as opposed to larger 11x11 or 5x5 filters. This allowed them to stack smaller filters on top of each other compared to a small number of large filters. This resulted in lower training cost and greater performance (deeper is better). 19 layers.

- [ResNet (2015)](ResNet.pdf)  
  Novel idea was to add a parallel "identity shortcut" path which skips multiple layers. The difference between the parallel paths is where the term "residual" comes from. This was in response to the "vanishing gradient" problem. Enables very deep (100s-1000s of layers) networks. 152 layers!

- [Neural ODEs (2019)](Neural%20Ordinary%20Differential%20Equations.pdf)  
  Best paper winner of NeurIPS 2018. The authors define a **continuous** deep neural network and show that out-of-the-box ODE solvers can be used to train it. This is in contrast to previous formulations which used **discrete** number of layers and backpropagation to train them. Continuous networks are the limit of discrete networks as the number of layers increases to infinity, thus it makes sense that ODE solvers can be used to train a continuous network. The advantage of continuous networks is that they have constant memory cost, can adapt evaluation strategy to each input (dynamic network), and can explicitly tradeoff numerical precision for speed.

## Random Stuff
- [Quantifying the human impact of human mobility on malaria](Quantifying%20the%20impact%20of%20human%20mobility%20on%20malaria.pdf)  
  Using "big data" for social good. The researchers use mobile phone location data to quantify the spread of a malarial outbreak. This could be used as an early warning system for the spread of diseases (if the data is available).

- [Predicting SSNs from public data](Predicting%20social%20security%20numbers%20from%20public%20data.pdf)  
  2 main takeaways. First, it's very hard to generate and allocate random numbers. Most public identifiers are not random. Second, one must be very careful when releasing "anonymized" data to the public, since a dedicated researcher may find ways of disambiguating anonymized identities.

- [Social media fingerprints of unemployment](Social%20media%20fingerprint%20of%20unemployment.pdf)  
  Social media data exhaust can be used to estimate economic indicators, in this case unemployment. Some important features were diurnal rhythm, mobility patterns, and communication styles.
