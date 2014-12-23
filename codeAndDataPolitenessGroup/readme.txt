AUTHORS

Joost van Doorn 10805176 joost.van.doorn@gmail.com
Ties van Rozendaal 10077391 ties@tivaro.nl
Harrie Oosterhuis 10196129 harrieoosterhuis@gmail.com
Carla Groenland 10208429 carla.groenland@gmail.com

CODE DESCRIPTION

-- Data preprocessing / examination --
convert_data is used to create train.csv and test.csv from the Stanford data set.
testAnnotators is used to evaluate the data.


-- Topic Model --
The files wordCounts and wordCountsRandom can be used to create topic models with initialization using mean scores or random initialization respectively. 
Gibbs sampling can be applied for various numbers of iterations, where the topic models are written to topicModel(Random)#NUM_ITS.txt. 
The file examineTopicModel contains code that shows how to work with the topic models, which was used to generate Table 4.

-- Information gain / entropy computations --
The code for entropy computation can be found in computeEntropy, wordEntropy and reduceFeatures. The latter file is used to create informationGain.txt files containing 
the features ranked on information gain. The wordEntropy file contains the code which was used to generate Table 1.

-- Disco ---
Used to convert the data to DOP features, this code uses the BLLIP parsers, you will have to install it yourself and change some of the code for it to work locally. However, the results of each step are saved in sentences and trees in the datasets/preprocessed folder, so you can perform part of this code. Also based on the Disco-dop implementation by Andreas, as referenced in our report.

-- DiscoFeatures --
The code based on data converted to DOP features, used for classification and feature reduction for the DOP based data.
