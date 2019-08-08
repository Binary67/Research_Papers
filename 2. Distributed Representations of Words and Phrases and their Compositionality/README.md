# Word Embedding
**Word Embedding** is one of the most popular representation of words where the words are mapped into vectors. This technique creates vector representations of a particular word. 

## Why do we need Word Embedding
Neural Network only takes numerical values as inputs. This creates problem in feeding words to any NLP Neural Networks. Hence, **Word Embedding** is very useful as the inputs for NLP Neural Networks. The vector representations of two similar words will be similar as well. The similarities can be calculated in *cosine similarities* or *euclidean distance*

For example, the vector representations of a "CAR" will be closer to vector representations of words such as "TRUCK", "VEHICLE", "DRIVER" and etc. It will be far from vector representaions of words such as "MOON", "PRINCESS" and etc. 

## Word2Vec
**Word2Vec** is one of the method to construct the embedding. There are two methods to create **Word2Vec**, which are SKIP GRAM and COMMON BAG OF WORDS (CBOW). Both of these techniques require Neural Networks. More details can be obtained from the references below.

## Further Reading
1. [Introduction to Word Embedding and Word2Vec](https://towardsdatascience.com/introduction-to-word-embedding-and-word2vec-652d0c2060fa)

2. [Word Embedding Research Paper](https://arxiv.org/pdf/1310.4546.pdf)

3. [Word Embeddings from macheads101](https://www.youtube.com/watch?v=5PL0TmQhItY)
