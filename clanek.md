# Novelty Clusterification in time series data - work title

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
The successful cross-cooperation of algorithms is what makes this solution feasible and interesting. 
This article is also an overview of the methods commonly used for novelty detection versus the outlier detection.

## Data
### Overview
The analyzed data in this article is the time-series data obtained from IoT sensors. 
These sensors are implemented in the smart home environment and produce continuous data which are reported to the server once every $x$ seconds.
In normal circumstances, the data obtained follow a specific distribution with 


- [ ] Todo: tady bych chtěl obrázky a nějaký přehled těch time series dat, co jsou ještě anomalie, co je na vstupu a tak
### Preprocessing
- [ ] TODO: TOTO ASI NE, MOZNA JEN JEDNA VETA
- [ ] novelty se těžko normalizuje protože nevíš kde jsou novelty body, ještě je nemáš

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

 - [ ] - TODO: PŘEGENERAOVAT DO B/W

![](https://raw.githubusercontent.com/chazzka/clanekcathedral/master/code/figures/mean_great_colored.svg) 
> Figure X - Mean of the given dataset with anomalies.

Although this may look positive on the first glance, several problems arise.
The initial one is with the automated distinction.
When the dataset is polluted with anomalies in close to 1:1 ratio, even for human it is close to impossible to differentiate anomalies and regular observation.
The second problem brings up when anomalies are not present at all, making mean method unusable.
Figure X shows the mean method when used on the dataset polluted by very little anomalies.
Obviously, if the dataset contained no anomalies at all, the result would become even more deficient.

 - [ ] - TODO: PŘEGENERAOVAT DO B/W

![](https://raw.githubusercontent.com/chazzka/clanekcathedral/master/code/figures/mean_wrong_colored.svg) 
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

![](https://raw.githubusercontent.com/chazzka/clanekcathedral/master/code/figures/DBScanGap.svg) 
> Figure X - DBScan performance

Traditional approaches for anomaly separation consist of either novelty detection or outlier detection.
Novelty detection is an anomaly detection mechanism, where we search for unusual observations, which are discovered due to their differences from the training data.
Novelty detection is a semi-supervised anomaly-detection technique, whereas outlier detection uses unsupervised methods.
This a crucial distinction, due to a fact that whereas the outlier detection is usually presented with data containing both anomalies and regular observation, it then uses mathematical models that try to make distinction between them, novelty detection on the other hand is usually presented data with little to zero anomalies (the proportion of anomalies in the dataset is called a contamination) and later, when conferred with an anomalous observation, it makes a decision.
This means, that if the dataset contains observations which look like anomalies but are still valid, the performance of unsupervised outlier detection in such case is usually unsatisfactory. 

### Unsupervised methods
The above leads us to consider anomaly detection algorithms.
Outlier detection methods are famous unsupervised methods. 
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

![](https://raw.githubusercontent.com/chazzka/clanekcathedral/master/figures/contamination0.svg)
> Figure X dataset with no anomalies (example 1)


Note, that one of the parameters of the Isolation Forest is the contamination parameter.
The contamination parameter is to control the proportion of anomalies in the dataset. 
Usually, this has to be known beforehand. 
This parameter has a huge impact on the final result of the detection.
This can be a problem due to a random appearance of the anomalies in our dataset and hence the proportion value varies.
Using this parameter, we can, however, deal with datasets already containing some portion of anomalies during learning.
That can easily happen especially during the testing phasis of the development.
Figure X shows the example of running Isolation Forest on the same dataset as above, but with the contamination parameter set to 0.01 (=1% of anomalies) using the Scikit's Isolation Forest *fit_predict* method.


![](https://raw.githubusercontent.com/chazzka/clanekcathedral/master/figures/contamination001.svg)
> Figure X dataset with contamination = 0.01 (example 2)


We can leverage on this knowledge and try to provide new, previously unseen novel datapoints to the algorithm to make it predict its label.
First, we use Scikit Isolation Forest *fit* method, to fit the regular data.
With the data fit, we add a bunch of new, unseen, novelty datapoints.
Figure X shows the result of *predict* method with the new data added.

![](https://raw.githubusercontent.com/chazzka/clanekcathedral/master/figures/contamination001_novelty.svg)
> Figure X. example 3

As Figure X shows, the newly provided data (around X=80 and Y=160) are labeled regular.
This is caused by the way the Isolation Forest splits the observation space.
Isolation Forest algorithm does the recursive partitioning to orphan the data in separate nodes. 
The less partitions are needed to separate the datapoint, the more anomalous it gets.
The Isolation Forest algorithm creates a new separation space based on the previously seen datapoints, hence there is not much room for the new, possibly novel datapoint, to be marked so.
New datapoints often fall into the same separation space with the previously seen, regular-marked datapoints, marking them regular.
Similar principles go with other outlier detection algorithms. 
The example shows that we need some kind of supervised method to make the algorithm learn on the regular data.
This leads us to supervised learning.

### Supervised learning
As shown in the above sections, the unsupervised learning led to undesired outcomes.
If we consider pure supervised learning algorithms on the other hand, we quickly run into the opposite problem.
Let us consider the alpha omega of supervised learning algorithms, the Artificial Neural Network.

#### ANN
Artificial Neural Network is the supervised learning algorithm, where we construct a n-layer network of neurons and by backpropagating we alter their weights so that the inputs lead to the desired outputs, predicting the labels.
This may seem like a perfect method for our problem.
However, in our scenario, we have the regular data labeled and ready to provide for the learning but that is not the case for the anomalous data.
This is a huge problem for the ANN algorithm, because it needs to alter its weights based on the previously seen labeled data.
Since we can only provide one of the labels (the regular data), the ANN will fail to find the other label.
For the following example, we use the Scikit's ANN implementation.

##### Example ANN:
In this example, we first get some random, classified arrays of data with respective labels (REGULAR and OUTLIER).
Then, we filter the data so that we only have one specific label.
Instead of feeding the neural network training set containing both labels, we only feed it with data labeled with **regular**.


```python
from sklearn.neural_network import MLPClassifier
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
import numpy as np
from enum import Enum

X, y = make_classification(n_samples=100)

class Label(Enum):
    REGULAR = 0
    OUTLIER = 1


regularX = X[y==Label.REGULAR.value]

outlierX = X[y==Label.OUTLIER.value]

regularY = np.zeros(len(regularX))

X_train, X_test, y_train, y_test = train_test_split(regularX, regularY)

clf = MLPClassifier().fit(X_train, y_train)

regular_prediction = clf.predict_proba(X_test[:1]))
outlier_prediction = clf.predict_proba(outlierX[:1])
```

The results are shown in Table X. Note, that each prediction was run 30 times, and the average value was taken. For the *regular_prediction *, containing the prediction of the testing set with data labeled **REGULAR** are expected, the average probability of 0.96727484 for the class labeled **REGULAR** and the average probability of 0.03272516 for the class labeled **OUTLIER**.
However, for the *outlier_prediction *, predicting the value given the outlying, never seen before value, the results are very unsatisfactory,
the average probability of 0.97993453 for the class labeled **REGULAR** and the average probability of 0.02006547 for the class labeled **OUTLIER**.

|probability of class: |  1 | 0 |  
|----------------------|--------|-
| REGULAR |  0.96727484 | 0.03272516 | 
| OUTLIER     |0.97993453 | 0.02006547|   

This means that, for the reasons defined above, the ANN algorithm was not able to detect the class it has not seen before.

### Semi-supervised learning
These findings lead us to the area of datamining, that is in the middle of supervised and unsupervised learning, the semi-supervised learning.
Let us define the semi-supervised learning as follows.
The semi-supervised learning in this context is a type of learning where we only have a part of the data labeled, and are interested in detecting, whether the data observed later fits in our label or not.

#### Novelty detection
Novelty detection is a semi-supervised learning paradigm, where training data is not polluted by outliers, and we are interested in detecting whether a new observation is an outlier. 
In this context an outlier is also called a novelty.
As of P. Oliveri in (https://doi.org/10.1016/j.aca.2017.05.013), such problems can also be called One-class problems.
One-class problems are focused on a single class of interest (the target class), which can be properly defined and sampled, while non-target samples do not constitute a meaningful class and cannot be sampled in a thorough and comprehensive way.

Scikit's platform offers us two algorithms for novelty detection.
The One-class SVM algorithm and the Local Outlier Factor algorithm.
Let us describe them and put them to the test.


#### Support Vector Machines
SVM is a family of algorithms where in general a position of an n-dimensional shape is optimized.
It is usually optimized by its nearest points - the support vectors.
##### Standard supervised SVM
The SVM algorithm classifies the points by drawing a hyperplane in the observed space.
The desired outcome here is to position this hyperplane such that the points of one class are on the one side and the second class on the other. 
The best possible hyperplane position is obtained through optimizing the distances between points and the hyperplane (also called a margin).
By providing labeled points, SVM is able to solve this optimization problem.
That is, SVM is a supervised learning algorithm.

##### One-class SVM
One-class SVM described in this article is as of Tax and Duin in (https://link.springer.com/content/pdf/10.1023/B:MACH.0000008084.60811.49.pdf).
One-class SVM is an unsupervised algorithm that learns a decision function for novelty detection: classifying new data as similar or different to the training set.
The above-mentioned One-class SVM definition obtains a spherically shaped boundary around the complete target set.
The sphere is characterized by center a and radius R > 0.
In SVM, by minimizing $R^2$, the volume of the sphere is minimized.
##### Novelty One-class SVM
The demand here is that the sphere contains all training objects - in our case regular labeled.
The One-class SVM learns a decision function so that the input unlabeled data can be classified as a similar or different in comparison with the dataset on which the model is trained.
This way, the One-class classification is possible, and this method is semi-supervised.

Let us return to the previous experiment with new, novel, previously unseen datapoints.
Again, we first perform the fitting operation on the regular datapoints.
Then we get different, similar dataset composed of regular datapoints and novelty datapoints.
Note that Scikit's OneClass SVM implementation - as opposed to Isolation Forest - requires data to be normalized between 0 and 1.
For this, we use Scikit's StandardScaler which can perform data-relative scaling.
However useful this feature is, it also has its downsides.
- [ ] TODO: downsides of scaling


Figure X shows the result of an above-defined experiment with the following settings:
- nu = 0.02
- kernel = rbf

The figure shows that the algorithm successfully marked the novelty data.
Notice that it also shows some of the *regular* data marked as novelty (note, that the testing dataset provided for evaluation phase is different than the training one). 
This phenomenon is called the false positive findings and will be examined later during evaluation of experiments.

> Figure x (Example 4)


- [ ] TODO: s linearnim to vubec nefungovalo, mozona obrazek a rict proc?


#### Local outlier factor
Local Outlier Factor is a neighborhood-based algorithm.
It first calculates the reachability matrix by calculating the reachability distances as in the k-nearest neighborhood. 
For each new datapoint a new reachability distance is calculated. 
If the distance is higher than some threshold, the datapoint is an outlier.
This is perfect for novelty detection, since we can calculate the average distance on the regular datapoints and observe its value on the later provided datapoints.

Again, let us put the Local Outlier Factor to the test. 
Note, that this algorithm does not need the input to be scaled in any form. 
Due to the distances' calculation, it is even undesirable.
For the following experiment, we use the Local Outlier Factor with the following settings:
- novelty: True
Note, that if the novelty parameter is set to True, we cannot use the *fit_predict* method, which is of online outlier detection, anymore.
Instead, we have to first use the *fit* method to fit the matrix on the regular dataset and the use the *predict* to evaluate new datapoints.
Figure x shows the results of the above defined dataset with novelty data added.

> Figure example 5

As we can see, the algorithm was successful in isolating all of the novelty datapoints.

Because the Local Outlier Factor algorithm calculates the distance metric, with our model trained, we can elaborate on that and provide more novel datapoints to observe the distances calculated.

> Figure example 7

#### Setting the right parameters
The last example of above section shows the Local Outlier Factor algorithm to be somewhat useful, however it showed a lot of errors especially considering the false positive finding.
The *n_neighbors* parameter of the Local Outlier Factor algorithm is useful to control the number of neighbors to be taken in the query.
Considering our example, the algorithm clearly lacks sufficient amount of neighbours, 

Oba ty algoritmy maji furu parametrů, které musíme nastavit
Other notable parameters with huge impact on the result are...
This kind of issue is widely known amongst AutoML community.
Some tools have already been implemented that try to deal with the issue of automatic hyperparameter tuning, namely H20 (h2o.ai) or AutoGluon (auto.gluon.ai). 

#### Experiments
For the following experiments, the data described in the introductory section were used.
In the experiment, we observe following properties:

The proportion of true positive novelties
: This is the proportion of the data labeled "novelty" compared to the actual novelty filtered dataset.

The proportion of detected regulars
: This is the proportion of the data labeled "regular" compared to the actual regular filtered dataset.

The proportion of false positive novelties
: This is the proportion of the data labeled "novelty" compared to the regular filtered dataset.
 
Note that the dataset is split in 70:30 ratio for training and testing dataset, to avoid the algorithm to be evaluated on the same datapoints it was learned on.

https://www.researchgate.net/figure/Contingency-table-True-Positive-False-Positive-False-Negative-and-True-Negatives-are_fig5_280535795

tabulka pro kazdy algoritmus zvlast

obrázky, tabulka, výhody, nevýhody


### Finding clusters amongst novelties ??
- tady už stačí asi že prostě to není těžký ukol, vezmeme jen obyčejný DB scan NEBO KNN a bác. oba algoritmy jsou jednoduché ale síla je v jejich kooperaci idk


## Results and discussion
- tady už bude celková tabulka a obrázky i s DB scanem
- tady můžeme zkusit tabulku kde budeme ukazovat kolik procent novelty to našlo apt možná porovnání s nějakým buď expertem nebo nějakými referenčními olabelovanými daty
- zde napíšeme co se povedlo, jak to neni vubec lehke najit dva či více algoritmů které spolu dobře fungují a velký problem je jejich validace a verifikace, zkus navhrnout nějaké řešení verifikace
## Conclusion


## References
https://matplotlib.org/stable/users/project/citing.html
<!--stackedit_data:
eyJoaXN0b3J5IjpbMjA3MDE4MDkzMCwtNDgzNzY2NDYyLDE4Mz
MyMjk2OTQsLTg4MzQ0NjA4NiwtMTc4MTAzMjM5NCwxMjY4MjIw
MDU5LC0zODU0NjA1MTYsLTE1NDg5NjA4MjQsLTUxNjYxNDgxMy
wtMTM3MzI3MjA0NywyMDgwNjY2MTM1LC0xNTcyODM1MzAxLDE5
NDU1ODAzOTcsLTE4NjEyMzM5NzksMTc3MTYxNjU3NCwtOTEyND
Y4NTY0LDEyMjI2MzIwNzAsLTkxNzU4NDM0NywtNDU5MjA5NTQ0
LDk1NjgwNjM0Nl19
-->