# Overview

How do we enable machines to read and speak our languages? 
While we have not solved the problem, tremendous progress has been made in recent years.

- Google Translate supports 109 languages, document translation, and real-time translation.
- Question answering (by search)
- Virtual assistants such as Siri, Alexa, Google Home etc.

## A brief history
### Birth with AI
Early ideas of machine language understanding were developed in building translating machines.

- 1930s: Early proposals in machine translation by Georges Artsrouni and Peter Troyanskii.

The goal of NLP is to make machines understand human language,
which is an integral part of AI since its birth.

- 1950: Alan Turing proposed the famous Turing test as a criteria for intelligence, where an interrogator tries to distinguish between a computer and a real human through text-based conversation.
Turing was optimistic that a machine will pass the Turing test by the end of the century and possess human thinking capability.
We are obviously far from that.
But a more interesting question is, even if a machine passes the Turing test, does it mean it has acquired human intelligence?
One famous objection is the "Chinese room" thought experiment, where a "simulator" operates as if it knows Chinese but obviously does not have any knowlege about language.
- 1954: Aside from the philosophical arguments, there is clear practical motivation for working on NLP.
Machine translation (MT) is one such example. In fact, interests in building MT systems started even before AI.
The Georgetown-IBM experiment is the first public demonstration of a MT system.
It has six grammar rules and a vocabulary of size 250, and was able to translate more than 60 sentences from Russian to English.
The demonstration was widely covered and encouraged subsequent funding in MT.
- 1964: ELIZA was the first chat bot that was able to attempt the Turing test. Unlike the IBM-Georgetown experiment that was meant to showcase the capability of machines,
Elizabuilt was built by Joseph Weizenbaum to demonstrate the superficiality of human-machine communication.
The bot was able to simulate a psychotherapist using simple pattern matching, mostly just slightly rephrasing the patients' utterance.
Nevertheless, many early users were convinced that it has human intelligence.
This is one example of ethical issues arising with AI.
- 1970: Terry Winograd's SHRDLU is another successful demonstration of AI.
It can interact with users in natural language and take actions in "blocks world".
The system demonstrated advanced language understanding skills in this simple world, e.g., grounding, question answering, semantic parsing, coreference resolution, clarification etc.
Unfortunately, subsequent effort in scaling it up to more complex settings failed,
which is a common weakness of earlier AI systems.

In general, researchers and funding agencies were overly optimistic about the prospect of AI during the 50's and 60's:
"Within the very near future---much less than twenty-five years---we shall have the technical capability of substituting machines for any and all human functions in organizations."
Research at that time was largely focused on AI-complete problems.
In hindsight, it is evident that successes in the toy settings were not going to transfer to even slightly more complex scenarios,
but most leading researchers at that time were convinced that AI would be solved very soon.

### Disappointing results
However, progress was much slower than expected. 
Most early systems rely heavily on logical reasoning in "toy" settings (e.g., small sets of objects and words)
and never scaled up to real systems.

In 1966, the ALPAC report concluded that MT was more expensive, less accurate and slower than human translation,
which led to a series funding cut in AI research.
This is known as the first AI winter.
It was hard to find funding for AI research in 70's and research in MT was abandoned for almost a decade.

The failure was due to several limitations:

- Limited computation. The hardware cannot handle the amount of computation required in terms of memory and speed.
- Combinatorial explosion. Most algorithms were based on logical reasoning thus were intractable given exponentially growing search spaces in complex problems.
- Underestimated complexity. Researchers quickly realized the profound difficulty in many problems, for example, word sense disambiguition in machine translation. There may not be a principled solution using logical reasoning.

### The rise of statistical methods
In the late 1980s, there was a rise of empirical/statistical methods due to increased computation power and the lessening of the dominance of Chomskyan theories of linguistics.

- Notable progress in MT from IBM: For the first time, the translation models use only parallel corpora and neglect knowledge of linguistics.
The IBM models form the basis of MT models until the rise of deep learning approaches.
- Statistical methods were widely adopted: perceptron, HMMs, SVMs etc.
- "Every time I fire a linguist, the performance of the speech recognizer goes up."---Frederick Jelinek.

Since 2011, neural networks / deep learning becomes the main driving force in AI.
- In 2015, the [deep learning tsunami](https://www.mitpressjournals.org/doi/pdf/10.1162/COLI_a_00239) hit NLP.
- Powers commercial products: machine translation, chat bots, recommender systems etc.

### Are we there yet?
The language understanding capability of current systems is still quite shallow.

- Chat bot dumb responses.
- Adversarial examples. 
- Gibberish outputs in MT.
- Biases.

### The tension between rationalism and empiricism
Since the beginning of NLP (and AI more broadly), there have been two philosophically different approaches.
Rationalists argue that knowledge is gained through reasoning, thus data is not necessary.
Empiricists claim that knowledge is gained from experience *a posteriori*.
The dominant trend has swung back and forth between the two approaches.
In NLP, the main argument focus on the importance of linguistic knowledge and machine learning (through data).

We are currently in the empiricism era.
However, it is important to keep in mind both perspectives in this course.
The two approaches do not conflict each other and we may find a way to reconcile them.
Experience is needed to test our reasoning and reasoning is needed in turn to generalize our experience.

## Challenges in NLP
Why is language hard? What are prominent features of NLP compared to other fields in AI?

- **Discreteness**: Unlike images and speech, text is symbolic. A single word can carry lots of weight and decides the meaning of a sentence, sometimes in a rather nuanced way.
This makes it difficult to define metrics and transformations for sentences. For example, we can interpolate two images or to blur an image through deterministic mappings, but corresponding operations for sentences cannot be easily formalized.
- **Compositionality**: The discrete symbols can be composed in numerous ways to express different meanings. In addition, Compositionality happens at all levels of language, e.g., words, sentences, paragraphs, documents. Even in toy settings, we cannot expect to see all possible compositions (think the blocks world). This challenges the model to generalize from limited examples.
- **Sparsity**: Language has a long tail distribution (e.g., [Zipf's law](https://en.wikipedia.org/wiki/Zipf%27s_law)). Similar to compositionality, the long tail distribution is observed at many levels, e.g., word frequency, number of speakers per language, number of utterances for a dialogue act etc.
In most cases, it's relatively easy to get the common patterns correct, but it's hard to cover rare linguistic phenomena on the tail.
- **Ambiguity**: The same word / sentence can be interpreted differently depending on the context, e.g., multi-sense word, PP attachment, sarcasm.

We will see these problems coming up again and again in the rest of this course.

## Course overview
We will start with the representation of text, where the goal is to represent strings as a "feature vector" that can be easily processed by machine learning models.
We would like a representation that is both succinct, precise, and flexible.

- Representation of text
    - **Symbolic representation**. Text is represented as a set of objects or concepts, e.g. the bag-of-words representation. The symbolic representation is friendly to structures and logic, however, the concepts often need to be defined for specific domains.
    - **Distributed representation**. Text is represented as a list of components or properties, i.e. dense vectors or embeddings. The distributed representation is continuous and domain-independent, thus it is attractive to modern ML models. However, the components are often not interpretable and the lack of atomic concepts makes it hard to use for logical reasoning.

Many tasks in NLP are concerned with predicting structures, such as sequences, trees, and graphs.

- Predicting structures
    - Modeling: how do we model interactions among substructures? For example, if we are prediction a tree, the label of one node will affect its children.
    - Learning: how do we learn the parameters of the model from labeled / unlabeled data? The common framework is to learn a scoring function such that correct structures are scored higher than the rest. However, we need to do this efficiently given an exponentially large set of possible structures. 
    - Inference: how do we compute the best structure efficiently given the combinatorial output space? This is a search problem and we will often resort to dynamic programming. 

As we can see, the key challenge in answering these questions lies in the fact that we are dealing with an exponentially large output space.

Modern approaches to NLP are largely based on neural networks,
which enabled unified approaches for many applications in NLP that used to be relatively disconnected.

- Neural networks for NLP
    - Encoder-decoder modeling for structured prediction and text generation. The encoder-decoder model (also called sequence-to-sequence model) is extremely general and has been applied to a wide range of tasks. The encoder takes the input (images, tables, text etc.) and produces an embedding (dense vector).
For structured prediction, the simpliest decoder would be a recurrent neural network provided that the output structure can be "linearized", i.e. convert to a sequence, which works surprisingly well even for trees.
    - The paradigm of pre-training and then fine-tuning. The increasing amount of data and compute has proven to be extremely powerful. We can pre-train the model on tons of data by self-supervised learning (i.e. no annotation is needed), and only fine-tune the learned weights on the task we care about and obtain better results than learning from scratch.

Finally, we will briefly touch on discourse and grounding, which go beyond individual sentences.

- Discourse and grounding
    - Discourse: what is the communication goal of the text? Discourse studies the *coherence* of a collection of sentences. Roughly speaking, a piece of text is coherent if we understand what it does after reading it; the text could be telling a story, putting forward an argument, or conversing about a topic.
    - Grounding: how do we relate language to the world? Grounding establishes the connection between text and the perceptible world. At the philosophical level, one may argue that real understanding cannot be achieved by text alone without connecting it with the real world. Regardless of the philosphical arguments, grounding is very useful in task-oriented dialogue and human-robot interaction. More generally, the "world" here doesn't have to be the real world that we can see and feel, it can be simply a set of objects of interet that can be referred to by language. 

## Additional readings 
- Wikipedia. [History of artificial intelligence.](https://en.wikipedia.org/wiki/History_of_artificial_intelligence)
- Kenneth Church. [A Pendulum Swung Too Far.](http://languagelog.ldc.upenn.edu/myl/ldc/swung-too-far.pdf)
