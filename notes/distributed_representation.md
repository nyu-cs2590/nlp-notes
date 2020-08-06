# Distributed word representations
In the last lecture, we used the bag-of-words representation for text classification,
where each word is represented by a single feature,
and a sentence is represented as a collection of words.
More generally, this is a symbolic representation since each feature (symbol) carries complex meaning.
On the other hand, the connectionist argues that the meaning of a concept is *distributed* across multiple units, e.g. a vector in a metric space.
In this lecture, we will study distributed representation of words,
which has been hugely successful when combined with deep learning. 

## Vector-space models
### Document representation
Let's start with a familiar setting:
we have a set of documents (e.g. movie reviews),
now instead of classifying them, we would like to find out which ones are closer.
From the last lecture, we already have a (sparse) vector representation for documents.

Let's load the movie reviews.
```{.python .input}
from mxnet import gluon
import gluonnlp as nlp
import numpy as np
import random
random.seed(1234)


train_dataset = nlp.data.IMDB(root='data/imdb', segment='train')
# remove label
train_dataset = [d[0] for d in train_dataset]
data = [d for d in train_dataset if len(d) < 500]
```

This gives us a document-term matrix $A$ of size $|D| \times |V|$,
where $|D|$ is the number of documents and $|V|$ is the vocabulary size.
Each row is a vector representation of a document.
Each entry $A_{ij}$ is the count of word $i$ in document $j$.

Now, given two vectors $a$ and $b$ in $\mathbb{R}^d$,
we can compute their **Euclidean distance**:
$$
d(a, b) = \|a-b\| = \sqrt{\sum_{i=1}^d (a_i - b_i)^2} .
$$
But there is one problem.
If we repeat each sentence in a document,
the distance between this repetitive document and the original document will be large,
in which case $a_i = 2b_i$.
Ideally, we would want them to have zero distance as they contain the same content.
Therefore, **cosine similarity** is better in this case since it measures the angle between two vectors:
$$
s(a,b) = \frac{a\cdot b}{\|a\|\|b\|} .
$$
It ranges from -1 ot 1 and a larger value means more similar vectors.

Let's see how well it works.
```{.python .input}
from sklearn.metrics import pairwise_distances
from sklearn.feature_extraction.text import CountVectorizer

vectorizer = CountVectorizer()
A = vectorizer.fit_transform(data)
print(A.shape)

dist = pairwise_distances(A[1:, :], A[:1,:], metric='cosine')

print(dist.shape)
# Show the first review
print(data[0] + '\n')

# Get the most similar review
print(data[np.argmax(dist*-1.)] + '\n')
print(np.max(dist*-1.)*-1)
```

One potential problem is that common words that occur in most documents
(e.g. "and", "movie", "watch") may contribute to the similarity scores,
but they are not representative enough for the specific document.
We expect a representative word to occur frequently in a small set of documents
(e.g. review talking about action movies),
but rarely in other documents.
The key idea in **term frequency-inverse document frequency (tfidf)**
is to *reweight* the frequency of each word in a document (term frequency)
by its inverse document frequency (idf):
$$
\text{idf}(w) = \log \frac{\text{count(documents)}}{\text{count(documents containing $w$)}} .
$$
In practice, we might want to use smoothing similar to the Naive Bayes case.
```{.python .input}
from sklearn.feature_extraction.text import TfidfVectorizer

# min_df: ignore words that occur in less than 100 documents
vectorizer = TfidfVectorizer(min_df=5)
A = vectorizer.fit_transform(data)
print(A.shape)

dist = pairwise_distances(A[1:, :], A[:1,:], metric='cosine')
print(dist.shape)
print(data[0] + '\n')

# Get the most similar review
print(data[np.argmax(dist*-1.)] + '\n')
print(np.max(dist*-1.)*-1)
```

### Word representation
Our goal is to represent a word in $\mathbb{R}^d$,
such that words similar in meaming are also closer according to some distance metric.
How should we go about this problem?

*You shall know a word by the company it keeps.* (Firth, 1957)

Now, what counts as a "company"? Words, sentence, or document?
If we consider documents as the context of the word,
then we can simply transpose our document-term matrix $A$
such that each row is a word vector.
Similarly, we can reweight the counts and compute the similarity between any two word vectors.

Simple tfidf word vectors:
```{.python .input}
A = A.T
print(A.shape)

idx_to_vocab = {idx: word for word, idx in vectorizer.vocabulary_.items()}
id_ = vectorizer.vocabulary_["love"]
print(id_)

dist = pairwise_distances(A, metric='cosine')[id_]
sorted_ids = np.argsort(dist)

for i in sorted_ids[:5]:
    print(idx_to_vocab[i], dist[i])
```

Obviously it doesn't work very well since
(a) our dataset is quite small and
(b) document as context for words is probably too coarse.
Here, what "similarity" really means is that two words tend to occur in the same document.
Depending on what kind of similarity we want to capture,
we can design the matrix differently and the same approach still works.
For example, a word-by-word matrix where each entry is the frequency of two words occuring in the same sentence,
a song-by-note matrix where each entry is the frequency of a note in a song,
a person-by-product matrix where each entry is the frequency of a person buying a specific product.

### Latent semantic analysis
You might have already noticed that our matrix $A$ is quite sparse
and the word vectors might live in a subspace.
It would be nice to have a lower-dimensional, dense representation,
which is more efficient to work with.

Recall **singular value decomposition (SVD)** from linear algebra.
Given a $m\times n$ matrix $A$, we want to factorize it into
$$
A_{m\times n} = U_{m\times m}\Sigma_{m\times n}V^T_{n\times n} ,
$$
where $U$ and $V$ are orthonormal matrices (with trailing zero vectors)
and $\Sigma$ is a diagonal matrix.

Let's unpack this equation to understand what it is doing.
Note that
$$
AV = U\Sigma V^TV = U\Sigma 
$$
The columns of $U$ and $V$ corresponds to orthogonal basis in $\mathbb{R}^m$ and $\mathbb{R}^n$ respectively.
Consider $A$ as our term-document matrix where each row is a word vector
and each column is a document vector.
Let's rewrite the matrices in terms of column vectors,
assuming $m > n$ (more words than documents).

$$
\underbrace{
\begin{bmatrix}
d_1 & \ldots & d_n
\end{bmatrix}
}_{\text{document vectors}}
\underbrace{
\begin{bmatrix}
v_1 & \ldots & v_n
\end{bmatrix}
}_{\text{word space basis}}
= 
\underbrace{
\begin{bmatrix}
u_1 & \ldots & u_m
\end{bmatrix}
}_{\text{document space basis}}
\underbrace{
\begin{bmatrix}
 \sigma_1 & & &\\
 & \ddots & &\\
 && \sigma_n & \\
 &&  & 0\\
\end{bmatrix}
}_{\text{singular values}}
\;.
$$

For $v_i$ and $u_i$, we have
$$
\begin{bmatrix}
d_1 & \ldots & d_n
\end{bmatrix}
v_i
=\sigma_i u_i
\;.
$$
Clearly, $u_i$ is a linear combination of document vectors, scaled by $\sigma_i$.
This means that it represents a document cluster,
e.g. good reviews for action movies.
The sigular value $\sigma_i$ represents the significance of this cluster in the dataset.
Similarly, since $u_i^T A = \sigma_i v_i^T$,
we see that $v_i^T$ corresponds to a word cluster.

Now, let's rewrite $A$ in terms of its row vectors, or word vectors:
$$
\begin{bmatrix}
w_1 \\
\vdots \\
w_m
\end{bmatrix}
v_i
=\sigma_i u_i
\;.
$$
Since $w_j^Tv_i = \sigma_i u_{ij}$,
$u_{ij}$ is the projection of $w_j$ on $v_i$, the $i$-th word clusters, scaled by $\sigma_i$.
Thus the entire $j$-th row of $U$ corresponds to the projection of $w_j$
on all word clusters,
i.e. the new word vector in a space spanned by the document clusters. 

In sum, the columns of $U$ corresponds to document clusters
and the rows of $U$ corresponds to the new word vectors.
This is a beautiful decomposition that captures the hidden relation between document and words.
In practice, however, we do not get the meaning of a word/document cluster for free,
a person will have to look at the result and assign it a category (e.g. complaints or critiques for some movie genre in our example).

We have not talked about dimensionality reduction.
But this is really simple given the decomposition.
Note that the sigular values corresponds to the importance of each dimension for the new word vectors,
we can simply take the top $k$ of those,
also called truncated SVD.

TODO: demo

### Summary
Vector space model is a very general framework
for learning the represention of two sets of related objects.

1. Design the row and column for the matrix, e.g. term-document, person-product.
2. [Optional] Reweight the raw counts, e.g. tfidf reweighting.
3. [Optional] Reduce dimensionality by SVD.
4. Compute similarity between vectors using a distance metric, e.g. cosine distance.

## Learning word embeddings
Our goal is to find word vectors such that words similar in meaning are also close to each other according to some distance metric.
Let's think about how for formalize this as a learning problem,
i.e. specifying the learning objective and the model.
The most useful takeaway from the vector space model is that
similar words tend to occur in the same context.
The key idea in learning word embeddings is to design self-supervised learning objectives that would result in vectors of this property (intuitively).

### The skip-gram model
The task is to predict neighbors of a word given the word itself.
Specifically, let $(w_{i-k}, \ldots, w_i, \ldots, w_{i+k})$ be a window around word $w_i$.
We assume that each of the word in the window is generated independently conditioned on $w_i$:
$$
p(w_{i-k}, \ldots, w_{i-1}, w_{i+1}, \ldots, w_{i+k} \mid w_i) =
\prod_{j=i-k}^{i-1}p(w_j\mid w_i) \;.
$$

This is similar to the Naive Bayes model, except that it's now conditioned on a word.
We parameterize the conditional distribution by
$$
p(w_j\mid w_i; f_c, f_w) = \frac{\exp\left [ f_c(w_j)^Tf_w(w_i)\right ]}
{\sum_{k=1}^{|V|}\exp\left [f_c(w_k)^Tf_w(w_i)\right ]} \;,
$$
where $f_c$ is a dictionary that maps a word in the context to a vector,
and $f_w$ maps the center word to a vector.
Note that the same word will have different vector representations
depending on whether it is in the context or at the center.
We can then estimate the parameters/embeddings by MLE using SGD.

The word embeddings are given by $f_w$.
Note that the objective encourages the embedding of a word
to have large dot product with the embedding of it neighboring words
(i.e. a small angle)
Thus if two words tend to occur in similar context,
they will have similar embeddings. 

### The continuous-bag-of-words model (CBOW) 
The task is very similar to that of the skip-gram model. 
Given a window of words, $(w_{i-k}, \ldots, w_i, \ldots, w_{i+k})$,
instead of predicting the context words from the center word,
we can also predict the center word from the context words.

CBOW uses the following model
$$
p(w_i \mid w_{i-k}, \ldots, w_{i-1}, w_{i+1}, \ldots, w_{i+k})
= \frac{\exp\left [
    f_w(w_i)^T \sum_{j=1,j\neq i}^{k} (f_c(w_{i-j}) + f_c(w_{i+j})
    \right ]}
{\sum_{t=1}^{|V|} \exp\left [
    f_w(w_t)^T \sum_{j=1,j\neq i}^{k} (f_c(w_{i-j}) + f_c(w_{i+j})
    \right ]}
\;,
$$
where the context is represented as a sum of embeddings of individual word in the window (hence the name continuous bag of words).
Similar to skip-gram, we use MLE to learn the parameters/embeddings.

### Properties of word embeddings
Recall that we can give physical meanings to each dimension of the word vectors obtained from SVD on the term-document matrix.
What about the skip-gram/CBOW embeddings?
Unfortunately we don't have a good answer (see related work in :numref:`reading`).

Emprically, we have found that word embeddings are useful for finding similar words and solving word analogy problems.

Let's try it out using the GloVe embeddings:
```{.python .input}
glove_6b50d = nlp.embedding.create('glove', source='glove.6B.50d')
vocab = nlp.Vocab(nlp.data.Counter(glove_6b50d.idx_to_token))
vocab.set_embedding(glove_6b50d)
```
Find similar words:
```{.python .input}
from mxnet import nd

def norm_vecs_by_row(x):
    return x / nd.sqrt(nd.sum(x * x, axis=1) + 1E-10).reshape((-1,1))

def get_knn(vocab, k, word):
    word_vec = vocab.embedding[word].reshape((-1, 1))
    vocab_vecs = norm_vecs_by_row(vocab.embedding.idx_to_vec)
    dot_prod = nd.dot(vocab_vecs, word_vec)
    indices = nd.topk(dot_prod.reshape((len(vocab), )), k=k+1, ret_typ='indices')
    indices = [int(i.asscalar()) for i in indices]
    # Remove unknown and input tokens.
    return vocab.to_tokens(indices[1:])

get_knn(vocab, 5, 'movie')
```

Word analogy problems have the form a : b :: c : d,
e.g. man : woman :: king : queen.
We assume that such relations can be obtained through addition in the vector space, e.g.
```
f_w(man) - f_w(woman) \approx f_w(king) - f_w(queen) \;.
```
Thus, given a, b, c, we can find d by finding the word whose embedding is closest to
$f_w(b) - f_w(a) + f_w(c)$,
with some distance metric.
```{.python .input}
def get_top_k_by_analogy(vocab, k, word1, word2, word3):
    word_vecs = vocab.embedding[word1, word2, word3]
    word_diff = (word_vecs[1] - word_vecs[0] + word_vecs[2]).reshape((-1, 1))
    vocab_vecs = norm_vecs_by_row(vocab.embedding.idx_to_vec)
    dot_prod = nd.dot(vocab_vecs, word_diff)
    indices = nd.topk(dot_prod.reshape((len(vocab), )), k=k, ret_typ='indices')
    indices = [int(i.asscalar()) for i in indices]
    return vocab.to_tokens(indices)

get_top_k_by_analogy(vocab, 3, 'man', 'woman', 'son')
```

### Summary
These models are quite similar to the vector space models
in the sense that they connect a word to its context.
Again, this is a general framework where we are free to choose other contexts.
For example, we can learn product embedding by predicting which products are commonly bought together.

## Brown clusters
Both the vector space model and the neural word embeddings
do not provide explicit clusters of words (although we can cluster the obtained vectors).
Now let's take a different approach and think how we can directly index the words.
Much like how we organize books in the library,
e.g. science -> computer science -> artificial intelligence -> natural language processing -> introduction to natural language processing,
if we cluster words hierarchically and form a tree,
we can use the path to each word to represent its meaning.
Similar words will have similar paths (i.e. shared ancestors on the tree).
Here are some example [word clusters on Twitter](http://www.cs.cmu.edu/~ark/TweetNLP/cluster_viewer.html).

How do we hierarchically cluster words (or any other objects)?
Let's start build the clusters bottom up.
We start with a set of words, each in its own cluster,
then we iteratively merge the two closest clusters,
until only one cluster is left (i.e. the root).
Note that this algorithm requires a distance metric.
We measure the distance between two clusters by the expected PMI of pairs of words from different clusters.
Let $C_i$ and $C_j$ be two clusters of words, their similarity is defined by
$$
s(C_i, C_j) = \sum_{w_1\in C_i}\sum_{w_2\in C_j} p(w_1, w_2)
\underbrace{\log\frac{p(w_1, w_2)}{p(w_1)p(w_2)}}_{\text{pointwise mutual information}}
\;,
$$
where $p(w_1, w_2)$ is the probability of the bigram "$w_1 w_2$",
and $p(w_1)$ and $p(w_2)$ are unigram probabilities.
Both can be estimated by counts.

**Exercise:** How would you cluster the words using a top-down approach?

## Evaluation
**Intrinsic evaluation**
measures whether two words closer in meaning are indeed closer in the vector space.
We can use word similarity datset such as [SimLex-999](https://www.aclweb.org/anthology/J15-4004/) with human annotated similarity scores.

**Extrinsic evaluation**
measures the effect of word vectors on downstream tasks.
For example, we can replace the BoW representation with an average of word embeddings for text classification using logistic regression,
and evaluate which embeddings yields better classification accuracy.

## Additional readings
:label:`reading`
- Stanford Encyclopedia of Philosophy. [Connectionism representation.](https://plato.stanford.edu/entries/connectionism/#ConRep)
- Jeffrey Pennington, Richard Socher and Christopher D. Manning. [GloVe: Global Vectors for Word Representation.](https://nlp.stanford.edu/pubs/glove.pdf) EMNLP 2014.
- Omer Levy and Yoav Goldberg. [Neural Word Embedding as Implicit Matrix Factorization.](https://papers.nips.cc/paper/5477-neural-word-embedding-as-implicit-matrix-factorization) NeurIPS 2014.


