# Sequence labeling
We have seen that language models estimate the probability of a sequence.
Unlike text classification, we are not predicting any labels so it's an example of unsupervised learning.
This week we will look at supervised learning with sequences.
In many NLP tasks, we want to produce a sequence *conditioned* on another sequence.
Sequence labeling is perhaps the simplest form of a sequence-to-sequence task.
Here our goal is to predict a label to each token in the input sequence.

Let's use **part-of-speech tagging** as a running example.
The input a sequence of tokens (in a sentence),
and we want to tag each token by its grammatical category
such as nouns and verbs.
For example,
```
Language/NOUN is/VERB fun/ADJ ./PUNCT
```
A first thought might be to simply build a word-to-tag dictionary.
However, many words can be assigned multiple tags depending on the context.
In the above example, "fun" can be either a noun (as in "have fun")
or an adjective (as in "a fun evening").
Therefore, the model must take context into consideration.

## A multiclass classification approach
Our second try is to formulate this as a classification problem which we already know how to solve.
Let $x=(x_1, \ldots, x_m) \in\mathcal{X}$ be the input sequence of tokens,
and $y=(y_1, \ldots, y_m)$ be the tag sequence where $y_i\in\mathcal{Y}$.
We want a multiclass classifier $f$ such that:
$$
f(x, i) = y_i \;.
$$
If we choose $f$ to be the multinomial logistic regression model, we have
$$
f(x, i) &= \arg\max_{k\in\mathcal{Y}} p(k\mid \phi(x,i)) \\
&= \arg\max_{k\in\mathcal{Y}} \frac{\exp\left [ w_k\cdot\phi(x,i) \right ]}
{\sum_{k'\in\mathcal{Y}}  \exp\left [ w_{k'}\cdot\phi(x,i) \right ] }
\;.
$$
:eqlabel:`eq_mlr`


Now, what should be the features $\phi(x, i)$?
We can design feature extractors similar to what we have used for text classifcation in :numref:`sec_feature_extractor`.
For example,
$$
\phi_1(x, i) &= \begin{cases}
1 & \text{$x_{i-1}=$"a" and $x_i=$"fun" and $x_{i+1}=$"evening"} \\
0 & \text{otherwise}
\end{cases} \;,
\\
\phi_2(x, i) &= \begin{cases}
1 & \text{$x_{i-1}=$"have" and $x_i=$"fun"} \\
0 & \text{otherwise}
\end{cases} \;.
$$
It's easy to see that this will give us a huge number of features,
specifically, we will have $p\times|\mathcal{Y}|$ features.
In addition, the feature/weight vectors will be very sparse,
e.g. the above two features are only relevant for the word "fun".

To simplify our problem, we need to make a conceptual change
and think of the classifier as a scoring function for the compatibility of an input and a label.
Instead of extracting features from $x$,
we design features for $(x,i,y)$
that suggests how "compatible" are the input and a specific label.
A good model should assign the highest score to the gold label $y_i$:
$w\cdot\phi(x,i,y_i) \ge w\cdot\phi(x,i,y')$ for all $y'\neq y_i$.
Note that now we only need to deal with one weight vector $w$ instead of $|\mathcal{Y}|$ vectors as in :eqref:`eq_mlr`.
Now, if we take the original feature vector $\phi$,
copy it $|\mathcal{Y}|$ times, and concatenate them,
where the $k$-th copy corresponds to $\phi(x,i,k)$,
then these two formulations are equivalent.

We have not gained anything yet except that it's clearner to use only one weight vector
because most computation will be parallel to the binary case.
The advantage comes in feature extraction.
Consider the trigram feature, e.g. $\phi_1$.
We don't want to design a feature for all possible combinations of $x_{i-1}, x_i, x_{i+1}, y$,
because if the tuple doesn't occur in our training data,
we cannot estimate its weight.
Instead, we "read off" the features from our training data.
For example,
$$
\phi_1(x, i, y) &= \begin{cases}
1 & \text{$x_{i-1}=$"<start>" and $x_i=$"language" and $x_{i+1}=$"is" and $y_i=$"NOUN"} \\
0 & \text{otherwise}
\end{cases} \;,
\\
\phi_2(x, i, y) &= \begin{cases}
1 & \text{$x_{i-1}=$"language" and $x_i=$"is" and $x_{i+1}=$"fun" and $y_i=$"VERB"} \\
0 & \text{otherwise}
\end{cases} \;.
$$

Now that we have a feature vector, we can run gradient descent to estimate the parameters of our multinomial logistic regression model.
How well would this work?
Let's consider the word "fun".
With the neighbors, we have better prediction on whether it's a noun or an adjective.
However, we only see so many phrases containing "fun" in our training data.
What if at test time we see "fun book" which never appears in the training set?
However, as long as we know that "book" is a noun, then "fun" is much more likely to be an adjective appearing before a noun.
In general, in addition to neighboring input tokens,
we often need to consider dependence in the output sequence as well,
which is the topic of interest in structured prediction.

## Structrured prediction
In the multiclass classification approach,
we decompose the sequence labeling problem to independent classification problems.
Now, let's directly tackle the prediction problem where
the input is $x\in\mathcal{X}$ and
the output is $y\in\mathcal{Y}$.
Note that now the output space $\mathcal{Y}$ contains sequences instead of categorical labels.
This is called structured prediction because the output is structured,
e.g. a sequence in our case.

### Conditional random fields
We can easily apply the idea of the compatibility score for structured prediciton:
the feature vector now depends on the entire output sequence $y$.
Let's use $\Phi(x,y)$ to denote the global feature vector that depends on $y$.
We extend multinomial logistic regression to structured prediction:
$$
f(x) &= \arg\max_{y\in\mathcal{Y}} p(y\mid x) \\
&= \arg\max_{y\in\mathcal{Y}} \frac{\exp\left [ w\cdot\Phi(x,y) \right ]}
{\sum_{y'\in\mathcal{Y}}  \exp\left [ w\cdot\Phi(x,y') \right ] }
\;.
$$
This model is called a **conditional random field (CRF)**.
The name comes from probabilistic graphical models:
it's a Markov random field (MRF) conditioned on observed variables (input $X$).

The next natural question is how to define $\Phi(x,y)$.
We want it to capture dependencies among the outputs.
So what if we design features depending on the entire sequence of $y$?
There are two problems here.
First, we won't have enough data to estimate the parameters as we probably only see the exact sequence once in our training set.
Second, at inference time we will need to solve the argmax problem, whose complexity grows exponentially with the sequence length.
Therefore, for both learning and inference,
we would like to have *decomposable* feature vectors.

Let's design the global feature vector $\Phi(x,y)$ such that it can be computed from *local* feature vectors that depend on $y_i$ and its neighbors.
For each feature $\Phi_j$, we have
$$
\Phi_j(x,y) = \sum_{i=1}^m \phi_j(x, i, y_i, y_{i-1}) \;.
$$
We can then create features in the same way as we did for tag classification, e.g.
$$
\phi_1(x, i, y_i, y_{i-1}) &= \begin{cases}
1 & \text{$x_{i-1}=$"language" and $x_i=$"is" and $x_{i+1}=$"fun" and $y_i=$"VERB" and $y_{i-1}=$"NOUN"} \\
0 & \text{otherwise}
\end{cases} \;,
$$

### Learning

### Inference


## Neural sequence labeling

## Applications
The input can be images as well, e.g. handwritten digit recognition.
Sometimes a task may not look like a sequence labeling tasks immediately,
but we can formulate it as one.


