# Clusterification in time series data - work title

## Abstract
- téma
- toto jsme udělali
- co řešíme (řešíme problem hledání clusterů dat, ne problem nějakeho algoritmu)
- takto
- takto jsme to otestovali
- toto je výsledek
## Introduction
- start with lots of refs
- describe the problem
- our main goal was to
- general -> specific (describe problem as a whole, then why the problems occurs, then why is it a problem for us, technical details, env. variables)
- constribution
- **toto až nakonec až budeme vědět co vlastně fungovalo**
- **here we describe the domain!! - aneb jak ta data vypadají - co je cílem hlavně vysvětlit že chceme cluster anomalii ne jen anomalie**

## Theory

Datapoint
: Datapoint is an observed point with $n$ features.

Regular
: Regular is a datapoint included in the given dataset. Its features are expectable.

Anomaly
: Anomaly is a datapoint, that differs significantly from other observations.

Outlier
: Outlier is an anomaly included in the given dataset. 

Novelty
: Novelty is an anomaly that is not present in the given dataset during learning. Novelties are usually supplied later during evaluation.


## SOTA

The originality of this article can be defined as follows. 
There are many successful algorithms used to analyze times series data, however the selling point here is the characterization of the problem not being simple outlier detection problem, rather a cluster-novelty detection (as defined later). 
The successful cross-cooperation is what makes this solution feasible and interesting. 
This article is also an overview of the methods commonly used, and their drawbacks.

## Data
### Overview
The analyzed data in this article is the time-series data obtained from IoT sensors. 
These sensors are implemented in the smart home environment and produce continuous data which are reported to the server once every $x$ seconds.
In normal circumstances, the data obtained follow a specific distribution with 


- [ ] Todo: tady bych chtěl obrázky a nějaký přehled těch time series dat, co jsou ještě anomalie, co je na vstupu a tak
### Preprocessing
- [ ] TODO: TOTO ASI NE, MOZNA JEN JEDNA VETA


## Methods

The very first task is to thoroughly analyze the domain of the given problem.
The inappropriate choice of the selected solution could lead to undesirable results.
Having the problem already described, we are now able to analyze and establish a learning process. 

Using the data domain knowledge, some constraints usually arise.
As described in the introductory section, we expect the sensors to produce linear-like data, with minor deviations within the *y* axis.
These deviations do not follow any specific pattern and are completely random.
However, the errors report some kind of observable behavior.
This is usually the case when performing cluster analysis.
The main constraint that is crucial for this task is the cluster forming pattern.
The task could become straightforward if we divide it into subordinate tasks.
First of them is to use the knowledge to separate non-anomalies (not yet clusters).
Doing so, the data that is left are anomalies-only where the task of finding anomaly clusters only becomes less challenging. 

The most straightforward solution when trying to find anomalies in above-shown data would be to use some kind of statistical method that would split the data in a certain ratio.
Figure X shows the mean (straight line) of the given data. 

![](https://raw.githubusercontent.com/chazzka/clanekcluster/master/code/figures/mean_great_colored.svg) 
> Figure X - Mean of the given dataset with anomalies.

Although this may look positive on the first glance, several problems arise.
The initial one is with the automated distinction.
When the dataset is polluted with anomalies in close to 1:1 ratio, even for human it is close to impossible to differentiate anomalies and regular observation.
The second problem brings up when anomalies are not present at all, making mean method unusable.
Figure X shows the mean method when used on the dataset polluted by very little anomalies.
Obviously, if the dataset contained no anomalies at all, the result would become even more deficient.

![](https://raw.githubusercontent.com/chazzka/clanekcluster/master/code/figures/mean_wrong_colored.svg) 
> Figure X - Mean of the given dataset with little to zero anomalies.


- [ ] - TODO TOTO ASI NE, PRESUNOUT MOZNA PAK AZ DO CLUSTERINGU NIZE

One could easily argue that there is an option of using pure clustering algorithms (e.g. ([DBScan](doi/10.5555/3001460.3001507)).
This, however, leads to unpleasant outcome.
Such algorithms tend to view the data as a cluster-only data, despite it being irrelevant in cluster regards.
Figure X shows the performance of the DBScan algorithm on previously non-processed data, where different colors represent different clusters.
Even though the algorithm did find some clusters, it would be demanding to differentiate and find the one with anomalies.
Moreover, due to the gap in the measurement, the DBScan incorrectly split the regular observations into two clusters.
This brings up the idea of algorithm cross-cooperation.
Therefore, our proposed solution separates the anomalies first and then tries to find a cluster amongst them.

![](https://raw.githubusercontent.com/chazzka/clanekcluster/master/code/figures/DBScanGap.svg) 
> Figure X - DBScan performance

Traditional approaches for anomaly separation consist of either novelty detection or outlier detection.
Novelty detection is an anomaly detection mechanism, where we search for unusual observations, which are discovered due to their differences from the training data.
Novelty detection is a semi-supervised anomaly-detection technique, whereas outlier detection uses unsupervised methods.
This a crucial distinction, due to a fact that whereas the outlier detection is usually presented with data containing both anomalies and regular observation, it then uses mathematical models that try to make distinction between them, novelty detection on the other hand is usually presented data with little to zero anomalies (the proportion of anomalies in the dataset is called a contamination) and later, when conferred with an anomalous observation, it makes a decision.
This means, that if the dataset contains observations which look like anomalies but are still valid, the performance of unsupervised outlier detection in such case is usually unsatisfactory. 

### Unsupervised methods
The above leads us to consider an anomaly detection algorithms. Outlier detection methods are famous unsupervised methods. 
Unsupervised in this context means, that we do not need any kind of pre-labeled data.
The data are passed to the algorithm as they are.
Note that some preprocessing may be needed, depending on the specific algorithm.

### Outlier detection
- [ ] TODO: na isolation forestu vam ukazeme ze outlier detection urcite neni novelty detection

#### Example: Isolation Forest
Isolation Forest ([1](https://doi.org/10.1016/j.engappai.2022.105730 "article 1"), [2](https://doi.org/10.1016/j.patcog.2023.109334 "article 2")) is an outlier detection, semi-supervised ensemble algorithm. 
This approach is well known to successfully isolate outliers by using recursive partitioning (forming a tree-like structure) to decide whether the analyzed particle is an anomaly or not.
The less partitions required to isolate the more probable it is for a particle to be an anomaly.

The Scikit-Learn platform (scikit-learn.org) offers several implemented, documented and tested machine-learning open-source algorithms.
Its implementation of Isolation Forest has, in time of writing this text, 5 hyperparameters which need to be explicitly chosen and tuned.

Consider a dataset containing no anomalies at all, which we want to use to for the learning.
Figure x shows example dataset with two features, none of the datapoint being an anomaly.

![](https://raw.githubusercontent.com/chazzka/clanekcathedral/master/contamination0.svg)
> Figure X dataset with no anomalies


Note, that one of the parameters of the Isolation Forest is the contamination parameter.
The contamination parameter is to control the proportion of anomalies in the dataset. 
Usually, this has to be known beforehand. 
This parameter has a huge impact on the final result of the detection.
This can be a problem due to a random appearance of the anomalies in our dataset and hence the proportion value varies.
Using this parameter we can, however, deal with datasets already containing some portion of anomalies during learning.
That can easily happen especially during the testing phasis of the development.
Figure X shows the example of running Isolation Forest on the same dataset as above, but with the contamination parameter set to 0.01 (=1% of anomalies) using the Scikit's Isolation Forest *fit_predict* method.


![](https://raw.githubusercontent.com/chazzka/clanekcathedral/master/contamination001.svg)
> Figure X dataset with contamination = 0.01


We can leverage on this knowledge and try to provide new, previously unseen novel datapoints to the algorithm to make it predict its label.
First, we use Scikit Isolation Forest *fit* method, to fit the regular data.
With the data fit, we add a bunch of new, unseen, possibly novelty datapoints.
Figure X shows the result of *predict* method with the new data added.

![](https://raw.githubusercontent.com/chazzka/clanekcathedral/master/contamination001_novelty.svg)

As Figure X shows, the newly provided data (around X=80 and Y=160) are labeled regular.
This is caused by the way the Isolation Forest splits the observation space.
Isolation Forest algorithm does the recursive partitioning to orphan the data in separate nodes. 
The less partition is needed to separate the datapoint, the more anomalous it gets.
The Isolation Forest algorithm creates a new separation space based on the previously seen datapoints, hence there is not much room for the new, possibly novel datapoint, to be marked so.
New datapoints often fall into the same separation space with the previously seen, regular-marked datapoints, marking them regular.
Similar principles goes with other outlier detection algorithms. 
The example shows, that we need some kind of supervised method to make the algorithm learn on the regular data.
This leads us to supervised learning.

### Supervised learning
As shown in the above sections, the unsupervised learning led to undesired outcomes.
If we consider pure supervised learning algorithms on the other hand, we quickly run into the opposite problem.
Let us consider the alpha omega of supervised learning algorithms, the Artificial Neural Network.

#### ANN
Artificial Neural Network is the supervised learning algorithm, where we construct a n-layer network of neurons and by backpropagating we alter their weights so that the inputs lead to the desired outputs, predicting the labels.
This may seem like a perfect method for our problem.
However in our scenario, we have the regular data labeled and ready to provide for the learning but that is not the case for the anomalous data.
This is a huge problem for the ANN algorithm, because it needs to alter its weights based on the previously seen labeled data.
Since we can only provide one of the labels (the regular data), the ANN will fail to find the other label.
For the following example, we use the Scikit's ANN implementation.

##### Example ANN:
In this example, we first get some random, classified arrays of data with respective labels (zero and one).
Then, we filter the data so that we only have one specific label.
Instead of feeding the neural network training set containing both labels, we only feed it with data labeled with **ones**.


```python
from sklearn.neural_network import MLPClassifier
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
import numpy as np
X, y = make_classification(n_samples=100, random_state=1)


onlyOneLabelX = list(filter(lambda x: len(x) != 0, map(lambda X,y: X if y == 1 else [], X,y)))
onlyZeroLabelX = list(filter(lambda x: len(x) != 0, map(lambda X,y: X if y == 0 else [], X,y)))

onlyOneLabely = np.ones(len(onlyOneLabelX))
onlyZeroLabely = np.zeros(len(onlyZeroLabelX))


X_train, X_test, y_train, y_test = train_test_split(onlyOneLabelX, onlyOneLabely, stratify=onlyOneLabely, random_state=1)

clf = MLPClassifier(random_state=1, max_iter=300).fit(X_train, y_train)

labelOnePrediction = clf.predict_proba(X_test[:1])
labelZeroPrediction  = clf.predict_proba(onlyZeroLabelX[:1])
```

The results are shown in Table X. For the *labelOnePrediction*, containing the prediction of the testing set with data labeled **one** are expected, the probability of 0.96727484 for the class labeled **one** and the probability of 0.03272516 for the class labeled **zero**.
However, for the *labelZeroPrediction*, the results are very unsatisfactory,
the probability of 0.97993453 for the class labeled **one** and the probability of 0.02006547 for the class labeled **zero**.

|probability of class: |  1 | 0 |  
|----------------------|--------|-
| test data labeled 1 |  0.96727484 | 0.03272516 | 
| test data labeled 0     |0.97993453 | 0.02006547|   

This means that, for the reasons defined above, the ANN algorithm was not able to detect the class it has not seen before.

### Semi-supervised learning
These finding lead us to the area of datamining, that is in the middle of supervised and unsupervised learning, the semi-supervised learning.
Let us define the semi-supervised learning as follows.
The semi-supervised learning in this context is a type of learning where we only have a part of the data labeled, and are interested in detecting, whether the data observed later fits in our label or not.

#### Novelty detection
Novelty detection is a semi-supervised learning paradigm, where training data is not polluted by outliers and we are interested in detecting whether a new observation is an outlier. 
In this context an outlier is also called a novelty.

Scikit's platform offer us two algorithms for novelty detection.
The OneClass SVM algorithm and the Local Outlier Factor algorithm.
Let us 

#### OneClass SVM
tady vysvětli jak funuguje 

#### Local outlier factor
Tady vysvětli jak funguje

#### setting the right parameters
Oba ty algoritmy maji furu parametrů, které musíme nastavit
Other notable parameters with huge impact on the result are...
This kind of issue is widely known amongst AutoML community.
Some tools have already been implemented that try to deal with the issue of automatic hyperparameter tuning, namely H20 (h2o.ai) or AutoGluon (auto.gluon.ai). 

#### Examples
 obrázky, tabulka, výhody, nevýhody
 

### Finding clusters amongst novelties ??
- tady už stačí asi že prostě to není těžký ukol, vezmeme jen obyčejný DB scan NEBO KNN a bác. oba algoritmy jsou jednoduché ale síla je v jejich kooperaci idk
- možná bychom se tu mohli taky zamyslet nad tím jak funguje ten DB Scan a zkusit trochu potunit aby to dělalo celé clustery


## Results and discussion
- tady můžeme zkusit tabulku kde budeme ukazovat kolik procent novelty to našlo apt možná porovnání s nějakým buď expertem nebo nějakými referenčními olabelovanými daty
- zde napíšeme co se povedlo, jak to neni vubec lehke najit dva či více algoritmů které spolu dobře fungují a velký problem je jejich validace a verifikace, zkus navhrnout nějaké řešení verifikace
## Conclusion

<!--stackedit_data:
eyJoaXN0b3J5IjpbNDEwMTI2NDA2LC0xNTgwMTUyNDczLDExMT
kzNTkxMzQsMzU0Mjc3NDU5LDc2MjAyOTM1NywtMjU1NDUzMTAw
LDIwMzE2MzI4NDcsLTEyMjkxNjY1MTksLTU0MjUyODUwMyw0Nz
Y0NjQxODYsLTE2MjUwMTY0MDAsLTYyNzk1NzUxMCwtMTE1ODE5
MDU3NywtMTA5Mjk1MTI0NSwtMTk2ODQ0LDE3ODU5NTk3ODIsMj
AwODcyNTk3NCwxODg0NTI1NjI2LC01NjYyNjAyOCwtNzcxOTg1
MDQ5XX0=
-->