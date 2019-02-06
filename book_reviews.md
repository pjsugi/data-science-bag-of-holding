# The best ML/DS books I've read

As I write this, I realized that this list is rather shallow, which is a reflection of how much I've learned from online classes and blogs relative to formal textbooks.

1. [The Elements of Statistical Learning - Hastie, Tibshirani, and Friedman](https://web.stanford.edu/~hastie/ElemStatLearn/)  
  First machine learning textbook I read and still my go-to resource for details of machine learning algorithms.

2. [The Visual Display of Quantitative Information - Edward Tufte](https://www.edwardtufte.com/tufte/books_vdqi)  
  Can be used equally well as a fascinating coffee table book or as a reference for creating visualizations that maximize impact by clearly displaying information. One of the best features of the book are the examples of great illustrations, spanning the French invasion of Russia in 1812 to metro timetables.

3. [AI Superpowers: China, Silicon Valley, and the New World Order - Kai Fu Lee](https://aisuperpowers.com/)  
  A great book from the former president of Google China to understand how China has caught up so quickly to the US in the field of machine learning, and why it will soon take the lead (driven in large part by the ubiquity of mobile payments and relatively limited privacy concerns).

4. [The Book of Why: The New Science of Cause and Effect - Judea Pearl](http://bayes.cs.ucla.edu/WHY/)  
  Introduction to causal inference meant for a popular audience (but not a light beach read). Causality is important for many reasons. The most important is that Pearl strongly (and I think convincingly) argues that strong AI will not be possible without causal understanding, and that current approaches (including deep learning) will never be able to obtain true intelligence.  
  
    ```"As much as I look into what’s being done with deep learning, I see they’re all stuck there on the level of associations. Curve fitting. That sounds like sacrilege, to say that all the impressive achievements of deep learning amount to just fitting a curve to data. From the point of view of the mathematical hierarchy, no matter how skillfully you manipulate the data and what you read into the data when you manipulate it, it’s still a curve-fitting exercise, albeit complex and nontrivial." - Pearl```

    Pearl uses an analogy of the Ladder of Causality which has 3 rungs: 1) associations/correlations, 2) interventions (given an action, what is the response?), and 3) counterfactuals (if I took an action, what would be the response). Most striking to me was his statement that if one only uses data (most machine learning models, deep learning), you are limited to the first rung. To get past correlations, and to extract causality, one must build a structural understanding of causal relationships by using causal diagrams.

    Another striking point is that in some (realistic) scenarios, adjusting for confounders may bias your estimates. This was surprising to me as the standard advice is that [one should adjust for all possible confounders](https://statmodeling.stat.columbia.edu/2009/07/05/disputes_about/).

5. [Causal Inference - Hernan and Robins, forthcoming](https://www.hsph.harvard.edu/miguel-hernan/causal-inference-book/)  
  Next up on my reading list.
