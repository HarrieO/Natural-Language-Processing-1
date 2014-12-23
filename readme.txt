AUTHORS

Joost van Doorn 10805176 joost.van.doorn@gmail.com
Ties van Rozendaal 10077391 ties@tivaro.nl
Harrie Oosterhuis 10196129 harrieoosterhuis@gmail.com
Carla Groenland 10208429 carla.groenland@gmail.com

CODE DESCRIPTION

-- Topic Model --
The files wordCounts and wordCountsRandom can be used to create topic models with initialization using mean scores or random initialization respectively. 
Gibbs sampling can be applied for various numbers of iterations, where the topic models are written to topicModel(Random)#NUM_ITS.txt. 
The file examineTopicModel contains code that shows how to work with the topic models, which was used to generate Table 4.

-- Information gain / entropy computations --
The code for entropy computation can be found in computeEntropy, wordEntropy and reduceDatapoints. The latter file is used to create informationGain.txt files containing 
the features ranked on information gain. The wordEntropy file contains the code which was used to generate Table 1.
