# Schedule

**Note**: Future schedule is subject to minor change.

#### Assignments
- Assignment 1: Representing and classifying text [hw1.tgz](assignments/hw1.tgz) [hw1.pdf](assignments/hw1.pdf) (Due on Sep 29)
- Assignment 2: Predicting sequences [hw2.tgz](assignments/hw2.tgz) [hw2.pdf](assignments/hw2.pdf) (Due on Oct 13)
- Assignment 3: Hidden Markov models and EM [hw3.tgz](assignments/hw3.tgz) [hw3.pdf](assignments/hw3.pdf) (Due on Nov 8)
- Assignment 4: Constituent parsing [hw4.tgz](assignments/hw4.tgz) [hw4.pdf](assignments/hw4.pdf) (Due on Nov 24)

#### Introduction

- Week 1 (Sep 8). **Overview**: NLP tasks and challenges, basic ML
    - [slides 1](slides/lec01/overview.pdf) [slides 2](slides/lec01/basic_ml.pdf) [notes](notes/overview.html)

#### Representation of text

- Week 2 (Sep 15). **Text classification**: bag-of-words, naive Bayes models, logistic regression
    - [slides](slides/lec02/main.pdf) [board](slides/lec02/board.pdf) [recording](https://nyu.zoom.us/rec/share/H3I606oHp61RYHbbUni0nuEYsrw3PChZ9HSv94LRAS20zxvt_HmK5Tl2Hbvbb2aJ.uQjVIELIo3GqfZfe) [notes](notes/text_classification.html)
- Week 3 (Sep 22). **Distributed representation**: vector space models, Brown clusters, neural word embeddings
    - [slides](slides/lec03/main.pdf) [board](slides/lec03/board.pdf) [recording](https://nyu.zoom.us/rec/share/kT5UNBkHz0cz9slgt1fhXXpx3mwtL2XeoBDejR4Q6sEUek4yFSDRD05h24OR5No.oKxFmakrbLrENoS6) [notes](notes/distributed_representation.html)

#### Predicting sequences

- Week 4 (Sep 29). **Language models**: n-gram LM, neural LM, perplexity
    - [slides](slides/lec04/main.pdf) [board](slides/lec04/board.pdf) [recording](https://nyu.zoom.us/rec/share/rv6x6Z8XlBCIZwvyabCEKR6MjjO3vUvdGdMQkScu3P_tARK87NbNoCUcziC6KaQE.MaqZ-a1G6D5_XPnZ) [notes](notes/language_models.html)
- Week 5 (Oct 6). **Sequence labeling**: log-linear models, decoding, POS tagging
    - [slides](slides/lec05/main-annotated.pdf) [recording](https://nyu.zoom.us/rec/share/yGTpEtxkNk8vzGLJXUPDWS4zppDlnJ8WqnlwLvcrHlltE4XsM2xN_3MGgcdsbhn5.mBnY-Yw2wmoZVpBW 
) [notes](notes/sequence_labeling.html)
- Week 6 (Oct 13). **Hidden Markov models**: HMM, EM
    - [slides](slides/lec06/main-annotated.pdf) [recording](https://nyu.zoom.us/rec/share/4rbRSvK2ZoKsisIEt1NCz58RyQBNogMjuBrLPx29X8pS45ravRFU2fDArbjoNtPe.zZ9CCnrD9Zxl3d2r?startTime=1634159505000) [J&M HMM](https://web.stanford.edu/~jurafsky/slp3/A.pdf), [Collins EM](http://www.cs.columbia.edu/~mcollins/em.pdf)
- Week 7 (Oct 20). Midterm.

#### Predicting trees

- Week 8 (Oct 27). **Context-free parsing**: PCFG, CYK, neural parser
    - [slides](slides/lec07/main-annotated.pdf) [recording](
https://nyu.zoom.us/rec/share/BB4fLJKdctUQT6QBFGqcCBoV4wphOzUZqsIXs4PzRTgAEEJinRMFujgVv0S85-zE.qzc-9cjkTsW3uqNk 
) [Collins PCFG](http://www.cs.columbia.edu/~mcollins/courses/nlp2011/notes/pcfgs.pdf), [Eisner Inside-Outside](http://www.cs.jhu.edu/~jason/465/readings/iobasics.pdf)
- Week 9 (Nov 3). **Semantic parsing**: logical semantics, learning from logical forms / denotations
    - [slides](slides/lec08/main-annotated.pdf) [recording](
https://nyu.zoom.us/rec/share/rzLhe4FKANwfpDcpBVKli4bdQVhSKeC25gzl6hinMB6aZz9D1cQNNQNg44Lwh07A.cCK4pCBRP9NBglDI 
) E Ch12, [Liang 16](https://cs.stanford.edu/~pliang/papers/executable-cacm2016.pdf)

#### Deep learning for NLP

- Week 10 (Nov 10). **Neural sequence modeling**: seq2seq, attention, copy mechanism, text generation
    - [slides](slides/lec09/main-annotated.pdf) [recording](https://nyu.zoom.us/rec/share/XPh9I4GICEbofC0Yj0yzZ1fwvKE14lCFrWUUNXb0u7HfBEGrZJ3mGFFRCe7JQmfv.DBBtuGMBt4jeu6Dn 
) D2L [9.7](https://d2l.ai/chapter_recurrent-modern/seq2seq.html), [9.8](https://d2l.ai/chapter_recurrent-modern/beam-search.html), [10](https://d2l.ai/chapter_attention-mechanisms/index.html)
- Week 11 (Nov 17). **Representation learning**: transformers, contextualized word embedding, pre-training and fine-tuning, autoencoders
    - [slides](slides/lec10/main.pdf) [recording](https://nyu.zoom.us/rec/share/meRSExRkZuKW2ic0aQ7trUCbFMubV1qKGrV6xQVTk2sHoRmPPYzQXfTrNoh26Zpc.XYtx8JjqyZin1OFB 
)
    - [Illustrated transformer](http://jalammar.github.io/illustrated-transformer/#self-attention-in-detail)
    - [Representation Learning: A Review and New Perspectives](https://arxiv.org/abs/1206.5538)
    - [On the Opportunities and Risks of
Foundation Models](https://arxiv.org/pdf/2108.07258.pdf)

#### Beyond text

- Week 12 (Nov 24). **Language grounding**: language+vision/robotics, pragmatics, RL agents 
    - [slides](slides/lec11/main.pdf) [recording 1](https://nyu.zoom.us/rec/share/uv9QPhTLObTkfaF_Kv1UFWJP6DIguLGnOLAv9G6z5MwEe6fFrSCihebqL25yHLrm.EBpbArio8tszJJSs?startTime=1637947633000) [recording 2](https://nyu.zoom.us/rec/share/3o2J3BwzGWAUQ7prN_nRR-XVTPXaoYf97391rlcXh6atAmTkNObrzjNVqHtLvrvr.JvGDurWIB_ST2eeX?startTime=1637948803000)

#### Conclusion

- Week 13 (Dec 1). **Guest lecture by Nasrin Mostafazadeh**: "How far have we come in giving our NLU systems common sense?" 
- Week 14 (Dec 8). **Project presentations**
