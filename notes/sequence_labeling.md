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

Language/`NOUN` &emsp;is/`VERB`&emsp; fun/`ADJ`&emsp; ./`PUNCT`

A first thought might be to simply build a word-to-tag dictionary.
However, many words can be assigned multiple tags depending on the context.
In the above example, "fun" can be either a noun (as in "have fun")
or an adjective (as in "a fun evening").
Therefore, the model must take context into consideration.

## A multiclass classification approach
:label:`sec_tag_classification`
Our second try is to formulate this as a classification problem which we already know how to solve.
Let $x=(x_1, \ldots, x_m) \in\mathcal{X}^m$ be the input sequence of tokens,
and $y=(y_1, \ldots, y_m) \in \mathcal{Y}^m$ be the tag sequence.
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
Note that now we only need to deal with one weight vector $w$ instead of $|\mathcal{Y}|$ vectors.
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
the input is $x\in\mathcal{X}^m$ and
the output is $y\in\mathcal{Y}^m$.
This is called structured prediction because the output is structured,
e.g. a sequence in our case.

### Conditional random fields
We can easily apply the idea of the compatibility score for structured prediciton:
the feature vector now depends on the entire output sequence $y$.
Let's use $\Phi(x,y)$ to denote the global feature vector that depends on $y$.
We extend multinomial logistic regression to structured prediction:
$$
f(x) &= \arg\max_{y\in\mathcal{Y}^m} p(y\mid x) \\
&= \arg\max_{y\in\mathcal{Y}^m} \frac{\exp\left [ w\cdot\Phi(x,y) \right ]}
{\sum_{y'\in\mathcal{Y}^m}  \exp\left [ w\cdot\Phi(x,y') \right ] } \\
&= \arg\max_{y\in\mathcal{Y}^m} \frac{\exp\left [ w\cdot\Phi(x,y) \right ]}
{Z(w)}
\;,
$$
where $Z(w)$ is the normalizer.

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
\Phi_j(x,y) = \sum_{i=1}^{m} \phi_j(x, i, y_i, y_{i-1}) \;.
$$
We can then create features in the same way as we did for tag classification, e.g.
$$
\phi_1(x, i, y_i, y_{i-1}) &= \begin{cases}
1 & \text{$x_{i-1}=$"language" and $x_i=$"is" and $x_{i+1}=$"fun" and $y_i=$"VERB" and $y_{i-1}=$"NOUN"} \\
0 & \text{otherwise}
\end{cases} \;,
$$

Using the local feature vectors, our model can be written as
$$
p(y\mid x) = \frac{1}{Z(w)}\prod_{i=1}^m \psi(y_i, y_{i-1}\mid x, w)
\;,
$$
where
$$
\psi(y_i, y_{i-1}\mid x, w) = \exp\left [
        w \cdot \phi(x,i,y_i,y_{i-1})
    \right ]
\;.
$$
This is one example of **conditional random fields (CRF)**,
specifically a chain-structured CRF.
The name comes from graphical models:
CRF is a Markov random field (MRF) conditioned on observed variables (input $X$).
Using terms in graphical models,
$Z(w)$ is the potential function,
$\psi$ is the potential function or the factor
for each clique $y_i, y_{i-1}$.
Thus we have interaction for adjacent variables on the chain.

### Inference: the Viterbi algorithm
Now, suppose we have a learned model (i.e. known $w$).
To tag a sentence,
we need to find the sequence that maximizes $p(y\mid x; w)$.
This process of finding the argmax output is also called **decoding**.
Note that
$$
\arg\max_{y\in\mathcal{Y}^m} p(y\mid x;w) &= \arg\max_{y\in\mathcal{Y}^m}
    \frac{\exp\left [ w\cdot\Phi(x,y) \right ]}
        {Z(w)} \\
&= \arg\max_{y\in\mathcal{Y}^m} \exp\left [ w\cdot\Phi(x,y) \right ] \\
&= \arg\max_{y\in\mathcal{Y}^m} w\cdot\sum_{i=1}^{m} \phi(x, i, y_i, y_{i-1}) \\
&= \arg\max_{y\in\mathcal{Y}^m} \sum_{i=1}^{m} w\cdot\phi(x, i, y_i, y_{i-1})
\;.
$$

Let's first consider how to compute the max score (without finding the corresponding paths).
Note that the local score of $y_i$ only depends on the previous label $y_{i-1}$.
If we know the score of all sequences of length $i-1$, then it is easy to compute the score of a sequence of length $i$.
Specifically, let $s(j, t)$ be the score of a sequence of length $j$ ending with the tag $y_j=t\in\mathcal{Y}$.
Then
$$
s(j+1, t') = \sum_{t\in\mathcal{Y}} s(j, t) +
    \underbrace{w\cdot\phi(x, j+1, t', t)}_{y_j=t, y_{j+1}=t'} \;.
$$
Thus the maximum score for a sequence of length $j+1$ is
$$
\max_{t'\in\mathcal{Y}} s(j+1, t') \;.
$$
Let's augment a $\texttt{STOP}$ symbol to our sequence,
then the maximum score of all possible sequences in $\mathcal{Y}^m$ is
$$
\max s(m+1, \texttt{STOP}) \;.
$$

We can compute the scores $s(j, t)$ on a lattice as shown in :numref:`fig_viterbi`.
Each column corresponds to a position in the sequence
and each row corresponds to a tag.
A sequence can be represented by a path connecting the nodes.
Each node saves the score of all paths ending at that node, i.e. $s(j, t)$.
It's easy to trace the sequence achieving the maximum score by
saving an additional backpointer at each node:
$$
\text{backpointer}(j, t) = \arg\max_{t'\in\mathcal{Y}} s(j-1, t') + w\cdot\phi(x, j, t, t') \;.
$$

![An example of Viterbi decoding](../plots/sequence/viterbi.pdf)
:label:`fig_viterbi`

This is called the Viterbi algorithm.
It is essentially dynamic programming
and has a polynomial running time of $O(m|\mathcal{Y}|^2)$.

### Learning: the forward-backward algorithm
Same as in logistic regression, we do MLE to find the parameters.
The log-likelihood of a dataset $\{(x^{(i)}, y^{(i)})\}_{i=1}^N$ is
$$
\ell(w) = \sum_{i=1}^N \log p(y^{(i)}\mid x^{(i)})
\;.
$$
In practice, it's important to use regularziation to prevent overfitting,
so let's also add $\ell_2$ regularization to our objective.
$$
\ell(w) = \sum_{i=1}^N \log p(y^{(i)}\mid x^{(i)}) - \lambda \|w\|^2_2
\;,
$$
where $\lambda$ is a hyperparameter controlling the strenght of regularization.

Let's plug in our log-linear model:
$$
\ell(w) &= \sum_{i=1}^N \log 
        \frac{\exp\left [ w\cdot\Phi(x,y) \right ]}
            {Z(w)}
    - \lambda \|w\|^2_2 \\
&= \sum_{i=1}^N
        w\cdot\Phi(x,y) -
        \log Z(w) 
    - \lambda \|w\|^2_2
$$

Take derivative w.r.t. $w_k$
(note that $Z(w) = \sum_{u\in\mathcal{Y}^m} \exp\left [ w\cdot\Phi(x,u) \right ]$):
$$
\frac{\partial}{\partial w_k}\ell(w) &= \sum_{i=1}^N
    \Phi_k(x,y) -
    \frac{\sum_{u\in\mathcal{Y}^m} \exp\left [ w\cdot\Phi(x,u) \right ] \Phi_k(x,u)}
        {Z(w)}
    - 2\lambda w_k \\
&= \sum_{i=1}^N \Phi_k(x,y) - 
   \sum_{i=1}^N \underbrace{ \sum_{u\in\mathcal{Y}^m} p(u\mid x; w) \Phi_k(x, u) }_{\text{feature expectation}}
    - 2\lambda w_k
$$

**Side note:**
If we ignore the regularization term, at the optimal solution (gradient is zero)
we have
$$
\sum_{i=1}^N \Phi_k(x,y) &= \sum_{i=1}^N \sum_{u\in\mathcal{Y}^m} p(u\mid x; w) \Phi_k(x, u) \\
\frac{1}{N}\sum_{i=1}^N \Phi_k(x,y) &= \mathbb{E}\left [ \Phi_k(x, u) \right ]
\;,
$$
which means that the expected feature value taken over $p(y\mid x;w)$
is equal to the average feature value of the observed data.

Let's now focus on computing the feature expectation.
$$
& \sum_{u\in\mathcal{Y}^m} p(u\mid x; w) \Phi_k(x, u) \\
&= \sum_{u\in\mathcal{Y}^m} p(u\mid x; w) \sum_{j=1}^m \phi_k(x, j, u_{j}, u_{j-1}) \\
&= \sum_{i=1}^m \sum_{a,b\in\mathcal{Y}} \sum_{u\in\mathcal{Y}^m\colon\\ u_{j-1}=a, u_j=b} 
    p(u\mid x; w) \phi_k(x, j, u_{j}, u_{j-1}) \\
&= \sum_{i=1}^m \sum_{a,b\in\mathcal{Y}}
    \underbrace{p(u_{j-1}=a, u_j=b \mid x; w)}_{\text{tag bigram marginal probability}} 
    \phi_k(x, j, u_{j}, u_{j-1})
\;.
$$
Note that the two sums and $\phi_k$ are easy to compute.
So our main task is to compute the marginal probability.
We need to sum over all prefix up to $u_{j-1}$
and all suffix after $u_{j}$,
where $u_{j-1}=a$ and $u_j=b$.
$$
p(u_{j-1}=a, u_j=b \mid x; w) = \frac{1}{Z(w)}
    \underbrace{\sum_{u_{1:j-1}\in\mathcal{Y}^{j-1}}}_{\text{prefix}}
    \underbrace{\sum_{u_{j+1:m}\in\mathcal{Y}^{m-j}}}_{\text{suffix}}
    \prod_{i=1}^m \exp\left [ w\cdot\phi_k(x,j,b,a) \right ]
$$

Ignoring the normalizer $Z(w)$ for now,
the sum can be computed in a way very similar to the viterbi algorithm.
Suppose we know the score of all prefix of length $j-1$ ending at the tag $a$,
$\alpha(j-1, a)$,
as well as the score of all suffix of length $m-j$ starting from the tag $b$,
$\beta(m-j, b)$,
we can easily compute
$$
& \sum_{u_{1:j-1}\in\mathcal{Y}^{j-1}} \sum_{u_{j+1:m}\in\mathcal{Y}^{m-j}}
    \prod_{i=1}^m \exp\left [ w\cdot\phi_k(x,j,b,a) \right ] \\
&= \alpha(j-1, a) \times \exp\left [ w\cdot\phi_k(x,j,b,a) \right ] \times \beta(m-j, b)
\;.
$$
How do we compute $\alpha$ and $\beta$?
Note that $\alpha(j, t)$ is exactly $\exp s(j, t)$ which we computed in the Viterbi algorithm when finding the maximum score,
and $\beta(j, t)$ can be computed similarly *backward* on the lattice.
The final missing piece is $Z(w)$, but this is just $\alpha(m+1, \texttt{STOP})$.
Once we have the forward scores and the backward scores,
which can be computed in $O(m|\mathcal{Y}|^2)$,
the gradient can be computed easily.

**Backpropogation.**
Note that the lattice can also be viewed as a computation graph.
In the forward pass, each node computes the forward score by summing outputs from previous nodes and the local score. 
In the backward pass, each node receives gradient from nodes in the subsequent position.
This is basically what forward-backward is doing.
In modern machine learning framworks, this is done through backpropogation (auto differentiation),
so we just need to implement the forward pass.

## Neural sequence labeling
Now, let's think about how to use neural networks for sequence labeling.
The core ideas are pretty much the same,
and we just need to replace handcrafted features with embeddings from neural networks.

### The classification approach: bidirectional RNN
We can easily use neural networks in the classification approach
where each tag is predicted independently as in :numref:`sec_tag_classification`.
The first thought is to use a RNN (:numref:`sec_rnn`) and take the hidden state at each position to be the features used for classification.
However, $h_i$ only summarizes information in $x_{1:i-1}$.
For sequence labeling, ideally we want to use the entire sequence for classification.
The solution is to use a bidirectional RNN (:numref:`fig_birnn`).
One RNN runs forward from $x_1$ to $x_m$ and produces $\overrightarrow{h}_{1:m}$,
and the other runs backward from $x_m$ to $x_1$ and produces $\overleftarrow{h}_{1:m}$.
We then concatenate the two hidden states: $h_i=[\overrightarrow{h}_{1:m}; \overleftarrow{h}_{1:m}]$,
and compute the score $\psi(y_i)=\exp\left [ Wh+b \right ]$.

![Bidirectional RNN](../plots/neural_networks/birnn.pdf)
:label:`fig_birnn`

Compared to feature-based classification, with RNNs it's easier to incorporate more context in the input sentence.
However, we are still prediction the tags independently.
One way to add dependence among output tags is to add the previously predicted tag $y_{i-1}$
as an additional input when computing $\overrightarrow{h}_i$,
which allows us to model $p(y_i\mid x, y_{1:i-1})$.
However, one problem here is that during training, we always condition on the ground truth tags $y_{1:i-1}$,
whereas at inference time we condition on the predicted tags $\hat{y}_{1:i-1}$,
so the distribution of the conditioning variables will change at test time.
This is called the **exposure bias** problem,
which we will encounter again in neural sequence generation.

One solution is to use a CRF in the last layer where
$$
\psi(y_i,y_{i-1}) = \exp\left [
        \underbrace{w_{y_i,y_{i-1}}\cdot h_i}_{\text{tag bigram score}} +
        \underbrace{w_{y_i}\cdot h_i}_{\text{tag unigram score}} +
        \underbrace{b_{y_i,y_{i-1}}}_{\text{bias term}}
    \right ]
\;.
$$
Note that the only difference compared to the classic CRF is that we are now using the hidden states $h_i$ as feature
as opposed to manually defined local feature vector $\phi$.

Training the RNN-CRF model is easy using modern deep learning frameworks
powerd by auto differentiation.
We just need to implement the forward pass to compute $p(y\mid x)$.
At inference time, the sequence with the highest likelihood can be found by Viterbi decoding.

## Applications
**Named entity recognition (NER).**
An important task in information extraction is to extract named entities in the text.
For example, in news articles,
we are interested in people (John), locations (New York City),
organization (NYU), date (September 10th) etc.
In other domains such as biomedical articles,
we might want to extract protein names and cell types.
Here's one example:

$\underbrace{\text{New York City}}_{\text{location}}$,
often abbreviated as $\underbrace{\text{NYC}}_{\text{location}}$,
is the most populous city in the $\underbrace{\text{United States}}_{\text{location}}$, 
with an estimated $\underbrace{\text{2019}}_{\text{year}}$
population of $\underbrace{\text{8,336,817}}_{\text{number}}$.

Note that this is not immediately a sequence labeling problem
since we are extracting spans from the input text.
The standard approach to convert this to a sequence labeling problem is to use the **BIO notation**.
The first token in a span is labeled as `B-<tag>` where `<tag>` is the label of the span.
Other tokens in the span are labeled as `I-<tag>`
and tokens outside any span are labeled as `O`.
For example,

... the/`O` &ensp; most/`O` &ensp;populous/`O` &ensp;city/`O` &ensp;in/`O` &ensp;the/`O`
&ensp;United/`B-location` &ensp;States/`I-location`, ...

The BIO labeling scheme can be used whenever we want to extract spans from the text,
e.g. noun phrase chunking and slot prediction in task-oriented dialogue.

**Chinese word segmentation.**
As we mentioned previously, in many writing systems words are not separated by white spaces and one example is Chinese.
A naive approach is to use a dictionary and greedily separate words that occur in the dictionary.
However, there are often multiple ways to segment a sentence if we only require each token is a valid word.
For example,

研究/study &emsp;  生命/life  &emsp; 的/'s  &emsp; 起源/origin

研究生/graduate student &emsp; 命/life &emsp; 的/'s &emsp; 起源/origin

Therefore, the problem is often solved as a supervised sequence labeling problem.
We can label each character by either `START` (beginning of a word) or `NONSTART`.

Finally, the input doesn't have to be sequence of words.
For example, in handwritten digit recognition,
the input are sequences of images.

## Additional reading
- Sam Wiseman and Alexander M. Rush. [Sequence-to-Sequence Learning as Beam-Search Optimization.](https://arxiv.org/pdf/1606.02960.pdf)
- Mike Collins. [Notes on log-linear models, MEMMs, and CRFs.](http://www.cs.columbia.edu/~mcollins/crf.pdf)
