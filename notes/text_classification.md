# Text classification 
We will start our journey with a simple NLP problem, text classification.
You might have already gone through the core techniques if you have taken an ML course,
however, we hope to provide some new insights from the NLP perspective.

Let's consider the binary sentiment classification task.
Our input is a document (e.g. a review),
and we want to predict whether it is positive or negative.

We will use the IMDB movie review dataset as our running example.
```{.python .input}
from d2l import mxnet as d2l
from mxnet import gluon, np, npx
import gluonnlp as nlp
import os
npx.set_np()
train_dataset = nlp.data.IMDB(root='data/imdb', segment='train')
```

## An intuitive approach
Let's pretend for a second that we don't know anything about ML or NLP.
How should we approach this problem?
Let's take a look at the data to have a better sense of the problem.
```{.python .input}
print('# training examples:', len(train_dataset))
print(train_dataset[0])
print('labels:', set([d[1] for d in train_dataset]))
```

While the review itself can be quite complex, we don't really need to understand everything in it.
To separate positive and negative reviews, we might be able to just look at individual words.
Intuitively, we expect to see more nice words in positive reviews such as "fantastic" and "wonderful".
Let's test our intuition against the data.

Before we start to play with the data, we need to **tokenize** the text,
i.e. separating the sequence of characters into a list of "words".
The definition of words can be language- and task-dependent.
For example, we can use a simple regex to tokenize English text,
however, it doesn't work for Chinese which doesn't have word boundary markers such as spaces.
Similarly, in German there is no spaces in compound nouns, which can get long.
For more information on tokenization, read [E 4.3.1]. 
Fortunately, in most cases we can use existing tools.

Let's tokenize the data and map the ratings to binary labels.
Also, for efficiency we randomly sample a subset.
```{.python .input}
tokenizer = nlp.data.SpacyTokenizer('en')
preprocess = lambda d : (tokenizer(d[0].lower()), 1 if int(d[1]) > 5 else 0)

import random
mini_dataset = [preprocess(d) for d in random.choices(train_dataset, k=1000)]
print(mini_dataset[0][1], mini_dataset[0][0][:20])
```

```{.python .input}
import itertools
pos_tokens = list(itertools.chain.from_iterable([d[0] for d in mini_dataset if d[1] == 1]))
neg_tokens = list(itertools.chain.from_iterable([d[0] for d in mini_dataset if d[1] == 0]))
print(len(pos_tokens), len(neg_tokens))
```

Now we can check the counts of a word in positive and negative examples.
```{.python .input}
token = 'wonderful'
print(pos_tokens.count(token))
print(neg_tokens.count(token))
```

So a simple heuristic approach is to count the frequency of occurence of each word in positive and negative examples,
and classifiy an example as positive if it contains more words of high-frequency in the positive examples.

## Naive Bayes model
Now, let's take a more principled approach,
following the key steps we talked about last time:
design a model, specify a loss function, and minimize the average loss.

Consider a probabilistic model for binary classification:
$$
f_w(x) = \begin{cases} 
1 & \text{if $p_w(y\mid x) > 0.5$} \\
0 & \text{otherwise}
\end{cases} .
$$

How do we parameterize $p_w(y\mid x)$? Recall the Bayes rule from probability:
$$
p(y\mid x) = \frac{p(x\mid y) p(y)}{p(x)}
\propto p(x\mid y) p(y) .
$$

Given that $y$ is binary, it's reasonable to model it as a Bernoulli random variable.
For more than two classes, we can use a categorical distribution.

What about $p(x\mid y)$?
Note that it's the joint probability of all words in the sentence $x$:
$p(x\mid y) = p(x_1, \ldots, x_n \mid y)$ ($n$ is the document length),
where $x_i$ is the token at the $i$-th position in the document.
Directly modeling the joint probability will require a huge number of parameters
(and can only be learned if we have a huge corpus).
Following our intuition,
we can ignore the interaction among all words 
and assume that they are **conditionally independt**.
This is the key assumption in Naive Bayes models:
$$
p(x_1, \ldots, x_n \mid y) = \prod_{i=1}^n p(x_i\mid y) .
$$

Let's think about the data generating process.
To generate a review, we first flip a coin to decide its sentiment,
then roll a $|V|$-sided die for $n$ steps to decide the word at each position
($|V|$ denotes the vocabulary size).
Therefore, $p(x_1, \ldots, x_n \mid y)$ is given by a multinomial distribution.

## Maximum likelihood estimation
Now that we have specified the model,
the next step is to estimate parameters.
For probabilistic models, we can use maximum likelihood estimation (MLE),
corresponding to the negative log-likelihood (NLL) loss.

Our goal is to maximize the log-likelihood of the dataset $D = \{x^{(i)}, y^{(i)}\}_{i=1}^N$:
$$
\log \prod_{i=1}^N p(x^{(i)}, y^{(i)})
= \log \prod_{i=1}^N \prod_{j=1}^{L_i} p(x_j^{(i)}\mid y^{(i)}) p(y^{(i)}),
$$
where $L_i$ is the length of the $i$-th example.
We can plug in the probability mass function (pmf) of Bernoulli and multinomial distribution and carry out the optimization.

In this case, we have a close-form solution (and a very intuitive one).
$$
p(x_j = k \mid y = 1) &= \frac{\text{count of word $k$ in positive documents}}
{\text{total number of tokens in positive documents}},  \\
p(x_j = k \mid y = 0) &= \frac{\text{count of word $k$ in negative documents}}
{\text{total number of tokens in negative documents}}.
$$
So we have re-discovered our intuitive method through the Naive Bayes model!

### Laplace smoothin

## Logistic regression
You might have wondered in Naive Bayes modeling
why did we take the trouble to rewrite $p(y\mid x)$ using the Bayes' rule
instead of directy modeling the conditional distribution.
After all, that's the distribution we use for prediction.
Also, we don't have to make assumptions on how $x$ is generated (e.g. conditional independence).
In fact, models like Naive Bayes that models $p(x\mid y)$ are called **generative models**
and they usually assume a generative story of how the data is generated.
Models that directly model $p(y\mid x)$ are called **discriminative models**.
Both approaches have merits.
However, if you have lots of data, empirically discriminative models may be better since it makes less assumptions about the data distribution.

## Bag-of-words (BoW) representation
We need to decide how to represent the document 
as a vector which easily works with our linear model.

How are we going to represent a piece of text?
If we want the representation the whole sequence it's going to be challenging (an exponentially large set).
Following our intuition, it may be sufficient to just consider individual words for our task.
The BoW representation is a vector $x = (x_1, \ldots, x_d)$ where $x_i \in \mathbb{N}$.
Each dimension corresponds to a unique word in the vocabulary,
thus $d$ is the vocabulary size.
The value $x_i$ is the count of the $i$-th word in the input text. 

**Example.**
To map a sequence of tokens to the BoW vector,
first we need to build the vocabulary.
```{.python .input}
vocab = nlp.Vocab(nlp.data.count_tokens(pos_tokens + neg_tokens))
print(len(vocab))
```
Convert an example to BoW vector representation:
```{.python .input}
# map words to ints
x = np.array(vocab(mini_dataset[0][0]))
# convert to vector of counts
x = npx.one_hot(x, len(vocab)).sum(axis=0)
print('feature vector size:', x.shape)
# show counts
print('top word counts:')
ids = np.argsort(x)[::-1][:10]
print(np.take_along_axis(x, ids, axis=-1))
print([vocab.idx_to_token[int(i)] for i in ids])
```




## Additional readings
- Sida Wang and Christopher D. Manning. [Baselines and Bigrams: Simple, Good Sentiment and Topic Classification.](http://www.sidaw.xyz/pubs/wang2012simple.pdf) ACL 2012.
- Ng, A. and Jordan, M. [On discriminative versus generative classifiers: A comparison of logistic regression and naive Bayes.](https://ai.stanford.edu/~ang/papers/nips01-discriminativegenerative.pdf) NeurIPS 2002. 
