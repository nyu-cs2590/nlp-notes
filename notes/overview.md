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

Language is an integral part of AI since its birth.

- 1950: Alan Turing proposed the Turing test as a criteria for intelligence, which tests if a person can distinguish between a computer and a real human through text-based conversation.
- 1954: The Georgetown-IBM experiment demonstrated rule-based translation from Russian to English and encouraged funding in MT.
- 1964: The first chat bot, ELIZA, was built by Joseph Weizenbaum to simulate a psychotherapist and many early users were convinced that it has human intelligence.
- 1970: Terry Winograd's SHRDLU can interact with users in natural language and take actions in "blocks world".

Researchers and funding agencies were overly optimistic about the prospect of AI during the 50's and 60's:
"Within the very near future---much less than twenty-five years---we shall have the technical capability of substituting machines for any and all human functions in organizations."
Research at that time was largely focused on AI-complete problems.

### Disappointing results
However, progress was much slower than expected. 
Most early systems rely heavily on complex handcrafted rules in "toy" settings and
never scaled up to real systems.

Examples:

- The Georgetown experiment
- ELIZA

Two fundamental limitations:

- Limited computation, e.g., programs were demonstrated with a vocabulary of size 20.
- Combinatorial explosion.

The first AI winter:

- 1966: The ALPAC report concluded that MT was more expensive, less accurate and slower than human translation.
- Hard to find funding for AI research in 70's and research in MT was abandoned for almost a decade.

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

- **Discrete**: Unlike images and speech, text is symbolic. A single word can carry lots of weight and decides the meaning of a sentence.
- **Compositionality**: The discrete symbols can be composed in numerous ways to express different meanings. Text with similar surface forms can mean completely different things.
- **Sparsity**: Language has a long tail distribution (e.g., [Zipf's law](https://en.wikipedia.org/wiki/Zipf%27s_law)). How do we robustly capture rare linguistic phenomena on the long tail?
- **Ambiguity**: The same word / sentence can be interpreted differently depending on the context, e.g., multi-sense word, PP attachment, sarcasm.

## Course overview
We will start with the representation of text, which is inherently discrete and compositional.
We would like a representation that is both succinct, precise, and flexible.

- Representation of text
    - Bag of words
    - Vectors / embeddings
    - Logical forms

Many tasks in NLP are concerned with predicting structures, such as sequences, trees, and graphs.

- Predicting structures
    - Modeling: how do we specify the mapping between the input text and the structure in a mathematical way? 
    - Learning: how do we learn the parameters of the model from labeled / unlabeled data? 
    - Inference: how do we compute the best structure efficiently given the combinatorial output space? 

Modern approaches to NLP are largely based on neural networks,
which enabled unified approaches for many applications in NLP that used to be relatively disconnected.

- Neural networks for NLP
    - Neural sequence modeling for structured prediction and text generation.
    - The paradigm of pre-training and then fine-tuning.

Finally, we will briefly touch on discourse and grounding, which are important topics in communication. 

- Discourse and grounding
    - Discourse: what is the communication goal of the text?
    - Grounding: how do we relate language to the world?

## Additional readings 
- Wikipedia. [History of artificial intelligence.](https://en.wikipedia.org/wiki/History_of_artificial_intelligence)
- Kenneth Church. [A Pendulum Swung Too Far.](http://languagelog.ldc.upenn.edu/myl/ldc/swung-too-far.pdf)
