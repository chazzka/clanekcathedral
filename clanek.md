# Finding novelty datapoints in time series data - work title

## Abstract
Detection of anomalies is an area of data mining that has shown much interest recently amongst companies working with IoT.
These companies implement many IoT sensors to control some other mechanical or electromechanical devices such as fridges, microwaves, and lights and security devices such as gates, doors, electrical fences, and cameras.
The secondary effect of such devices is the production of large amounts of data.
Such data can be analyzed and mined for interesting patterns or - in the case of this article - anomalous behavior.
We first investigate the problem of the given time series data.
Then, we examine the difference between two types of anomalies - outlying anomalies and novelty anomalies.
We highlight the importance of such distinction and leverage this knowledge to find the algorithms that can be used purely for novelty detection.
We test these algorithms on the real-world scenarios of datasets obtained from the IoT sensors.
Lastly, we compare these algorithms and provide examples and possible usages by companies working with time series data.

## Introduction
Recent advances in automatization brought an intensified deployment of various IoT sensors.
With the sensors producing big data, there is a massive concern for algorithms that analyze them.
Applications ranging from medical data (https://www.sciencedirect.com/science/article/pii/S016926072300411X),
aeronautics (https://www.sciencedirect.com/science/article/pii/S1877050922007207),
to Industry 4.0 (https://www.sciencedirect.com/science/article/pii/S2199853122010253) are to be seen evaluating such data.
The data mining field has been branching out lately to address specific needs.
IoT data mining can be used to find common patterns in data through pattern mining (https://www.sciencedirect.com/science/article/pii/S0952197622004705).
This branch focuses on analyzing previously non-labeled data and mining an interesting pattern, such as variables that tend to report symbiotic behavior.
Pattern mining plays a significant role in the automated understanding of human interactions to provide recommendations.
Another topic is clustering, where the applications are set to find clusters with similar characteristics in data.
Clustering is a famous technique even in time series data because observed data usually form clusters in a particular time.
An example of this can be Haskey's et al. Clustering of periodic multichannel timeseries data.(https://www.sciencedirect.com/science/article/pii/S0010465514000885)
Lastly, one of the most famous techniques regarding time-series data mining is finding anomalies.
Wang et al. focused on active probing for IoT anomaly detection in (https://www.sciencedirect.com/science/article/pii/S235286482300113X).
Gao et al. identify malicious traffic in IoT security applications.
Although anomaly and outlier detection are common terms, novelty detection as proposed in this article, is not a well-known keyword, and we believe this should change.
That is why we propose this comparative study of outlier and novelty detection, where we focus on introducing the concept of novelty detection terminology in particular.
Our main goal was to make a comparative study of already known novelty detection algorithms to make these terms notable in the community and to help engineering applications make the right choice when performing anomaly detection tasks.
The comparative study is done on a real-world scenario of IoT time-series data from smart home environment sensors.

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
Many successful algorithms are used to analyze time series data; however, the point here is to characterize the problem not as a simple outlier detection problem but as a novelty detection (as defined later). 
This article is also an overview of the methods commonly used for novelty detection versus outlier detection.

## Data
### Overview
The analyzed data in this article is the 2D time-series data obtained from IoT sensors. 
These sensors are implemented in the smart home environment and produce continuous data, reported to the server once every $x$ seconds.
Figure X shows an example of the sensors reporting the data for 24 hours.

![](https://raw.githubusercontent.com/chazzka/clanekcathedral/master/code/figures/data_overview.svg)

As Figure X shows, the data follow a regular pattern around $Y = 100$,
however, around $X= 55$, the sensors started producing anomalous behavior.
Since the sensors can be from a security field, such a blackout can lead to detrimental outcomes.
This event must be marked and reported so a company can react immediately.
Note that the values of these anomalous datapoints are utterly random since it is the abnormal erroneous behavior.
This also makes any normalizing impossible since we do not know the location of these points beforehand.

## Methods


The first task is to analyze the domain of the given problem thoroughly.
The inappropriate choice of the selected solution could lead to undesirable results.
Having the problem already described, we are now able to analyze and establish a learning process. 

With the data domain knowledge, some constraints usually arise.
As the introductory section describes, we expect the sensors to produce linear-like data, with minor deviations within the *y* axis.
These deviations do not follow any specific pattern and are entirely random.
However, the errors report some kind of observable behavior.
This is usually the case when performing cluster analysis.
The primary constraint that is crucial for this task is the cluster-forming pattern.
The task could become straightforward if we divide it into subordinate tasks.
The first is to use the knowledge to separate non-anomalies (not yet clusters).
Doing so, the data left are anomalies-only, where finding anomaly clusters becomes less challenging. 

The most straightforward solution when finding anomalies in the above-shown data would be to use a statistical method to split the data in a specific ratio.
Figure X shows the mean (straight line) of the given data. 

![](https://raw.githubusercontent.com/chazzka/clanekcathedral/master/code/figures/mean1.svg) 
> Figure X - Mean of the given dataset with anomalies.

Although this may look positive at first glance, several problems arise.
The initial one is with the automated distinction.
When the dataset is polluted with anomalies in close to a 1:1 ratio, even for humans, it is nearly impossible to differentiate anomalies and regular observation.
The second problem arises when anomalies are not present at all, making the mean method unusable.
Figure X shows the mean method used on the dataset polluted by very few anomalies.
The result would become even more deficient if the dataset contained no anomalies.

![](https://raw.githubusercontent.com/chazzka/clanekcathedral/master/code/figures/mean2.svg) 
> Figure X - Mean of the given dataset with little to zero anomalies.


Traditional approaches for anomaly separation consist of either novelty detection or outlier detection.
Novelty detection is an anomaly detection mechanism where we search for unusual observations, which are discovered due to their differences from the training data.
Novelty detection is a semi-supervised anomaly-detection technique, whereas outlier detection uses unsupervised methods.
This is a crucial distinction due to the fact that whereas outlier detection is usually presented with data containing both anomalies and regular observation, it then uses mathematical models that try to make a distinction between them. On the other hand, novelty detection is usually presented with little to zero anomalies (the proportion of anomalies in the dataset is called contamination), and later, when conferred with an anomalous observation, it makes a decision.
This means that if the dataset contains observations that look like anomalies but are still valid, the performance of unsupervised outlier detection in such case is usually unsatisfactory. 

### Unsupervised methods
The above leads us to consider anomaly detection algorithms.
Outlier detection methods are famous unsupervised methods. 
Unsupervised in this context means we do not need any pre-labeled data.
The data are passed to the algorithm as they are.
Note that some preprocessing may be needed, depending on the specific algorithm.

### Outlier detection
With the following example we will show that outlier detection is certainly not a novelty detection.

#### Example: Isolation Forest

Isolation Forest ([1](https://doi.org/10.1016/j.engappai.2022.105730 "article 1"), [2](https://doi.org/10.1016/j.patcog.2023.109334 "article 2")) is an outlier detection, semi-supervised ensemble algorithm. 
This approach is well known to successfully isolate outliers by using recursive partitioning (forming a tree-like structure) to decide whether the analyzed particle is an anomaly.
The fewer partitions required to isolate, the more probable it is for a particle to be an anomaly.

The Scikit-Learn platform (scikit-learn.org) offers several implemented, documented, and tested machine-learning open-source algorithms.
At the time of writing this text, its implementation of Isolation Forest has five hyperparameters that need to be explicitly chosen and tuned.

Consider a dataset containing no anomalies, which we want to use for the learning.
Figure x shows an example dataset with two features, none of the datapoints being an anomaly.

![](https://raw.githubusercontent.com/chazzka/clanekcathedral/master/code/figures/example1.svg)
> Figure X dataset with no anomalies (example 1)


Note that one of the parameters of the Isolation Forest is the contamination parameter.
The contamination parameter is to control the proportion of anomalies in the dataset. 
Usually, this has to be known beforehand. 
This parameter has a significant impact on the final result of the detection.
This can be a problem due to the random appearance of the anomalies in our dataset; hence, the proportion value varies.
Using this parameter, we can, however, deal with datasets already containing some portion of anomalies during learning.
That can easily happen, especially during the testing phases of the development.
Figure X shows the example of running an Isolation Forest on the same dataset as above but with the contamination parameter set to 0.01 (=1% of anomalies) using Scikit's Isolation Forest *fit_predict* method.


![](https://raw.githubusercontent.com/chazzka/clanekcathedral/master/code/figures/example2.svg)
> Figure X dataset with contamination = 0.01 (example 2)


We can leverage this knowledge and provide new, previously unseen novel datapoints to the algorithm to make it predict its label.
First, we use the Scikit Isolation Forest *fit* method to fit the regular data.
With the data fit, we add a bunch of new, unseen, novelty datapoints.
Figure X shows the result of the *predict* method with the new data added.

![](https://raw.githubusercontent.com/chazzka/clanekcathedral/master/code/figures/example3.svg)
> Figure X. example 3

Figure X shows that the newly provided data (around X=80 and Y=160) are labeled regular.
This is caused by the way the Isolation Forest splits the observation space.
Isolation Forest algorithm does the recursive partitioning to orphan the data in separate nodes. 
The fewer partitions needed to separate the datapoint, the more anomalous it gets.
The Isolation Forest algorithm creates a new separation space based on the previously seen datapoints; hence, there is little room for the new, possibly novel datapoint, to be marked so.
New datapoints often fall into the same separation space as the previously seen, regular-marked datapoints, marking them regular.
Similar principles go with other outlier detection algorithms. 
The example shows that we need some supervised method to make the algorithm learn on the regular data.
This leads us to supervised learning.

### Supervised learning
As shown in the above sections, the unsupervised learning led to undesired outcomes.
If we consider pure supervised learning algorithms, on the other hand, we quickly run into the opposite problem.
Let us consider the alpha omega of supervised learning algorithms, the Artificial Neural Network.
#### ANN
An Artificial Neural Network is a supervised learning algorithm where we construct a n-layer network of neurons. 
By backpropagating, we alter their weights so that the inputs lead to the desired outputs, predicting the labels.
This may seem like a perfect method for our problem.
However, in our scenario, we have the regular data labeled and ready to provide for the learning, but that is not the case for the anomalous data.
This is a huge problem for the ANN algorithm because it needs to alter its weights based on the previously seen labeled data.
Since we can only provide one of the labels (the regular data), the ANN will fail to find the other label.
For the following example, we use Scikit's ANN implementation.

##### Example ANN:
In this example, we first get random, classified data arrays with respective labels (REGULAR and OUTLIER).
Then, we filter the data only to have one specific label.
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

The results are shown in Table X. Note that each prediction was run 30 times, and the average value was taken. For the *regular_prediction *, containing the prediction of the testing set with data labeled **REGULAR** are expected, the average probability of 0.96727484 for the class labeled **REGULAR** and the average probability of 0.03272516 for the class labeled **OUTLIER**.
However, for the *outlier_prediction *, predicting the value given the outlying, never-seen-before value, the results are very unsatisfactory,
the average probability of 0.97993453 for the class labeled **REGULAR** and the average probability of 0.02006547 for the class labeled **OUTLIER**.

|probability of class: |  1 | 0 |  
|----------------------|--------|-
| REGULAR |  0.96727484 | 0.03272516 | 
| OUTLIER     |0.97993453 | 0.02006547|   

This means that, for the above reasons, the ANN algorithm could not detect the class it had not seen before.

### Semi-supervised learning
These findings lead us to the area of datamining, which is in the middle of supervised and unsupervised learning the semi-supervised learning.
Let us define the semi-supervised learning as follows.
Semi-supervised learning in this context is a type of learning where we only have a part of the data labeled and are interested in detecting whether the data observed later fits in our label or not.

#### Novelty detection
Novelty detection is a semi-supervised learning paradigm where training data is not polluted by outliers, and we are interested in detecting whether a new observation is an outlier. 
In this context, an outlier is also called a novelty.
As of P. Oliveri in (https://doi.org/10.1016/j.aca.2017.05.013), such problems can also be called One-class problems.
One-class problems are focused on a single class of interest (the target class), which can be properly defined and sampled. In contrast, non-target samples do not constitute a meaningful class and cannot be sampled in a thorough and comprehensive way.

Scikit's platform offers us two algorithms for novelty detection.
The One-class SVM algorithm and the Local Outlier Factor algorithm.
Let us describe them and put them to the test.


#### Support Vector Machines
SVM is a family of algorithms where, in general, a position of an n-dimensional shape is optimized.
It is usually optimized by its nearest points - the support vectors.
##### Standard supervised SVM
The SVM algorithm classifies the points by drawing a hyperplane in the observed space.
The desired outcome here is to position this hyperplane such that the points of one class are on one side and the second class on the other. 
The best possible hyperplane position is obtained by optimizing the distances between points and the hyperplane (also called a margin).
By providing labeled points, SVM can solve this optimization problem.
That is, SVM is a supervised learning algorithm.
##### One-class SVM
One-class SVM described in this article is as of Tax and Duin in (https://link.springer.com/content/pdf/10.1023/B:MACH.0000008084.60811.49.pdf).
One-class SVM is an unsupervised algorithm that learns a decision function for novelty detection: classifying new data as similar or different to the training set.
The above-mentioned One-class SVM definition obtains a spherically shaped boundary around the complete target set.
The sphere is characterized by center a and radius R > 0.
In SVM, by minimizing $R^2$, the volume of the sphere is minimized.
##### Novelty One-class SVM

The demand here is that the sphere contains all training objects - in our case, regular-labeled.
The One-class SVM learns a decision function so that the input unlabeled data can be classified as similar or different compared to the dataset on which the model is trained.
This way, the One-class classification is possible, and this method is semi-supervised.

Let us return to the previous experiment with new, novel, previously unseen datapoints.
Again, we first perform the fitting operation on the regular datapoints.
Then, we get different but similar datasets composed of regular and novelty datapoints.
Note that Scikit's OneClass SVM implementation - as opposed to Isolation Forest - requires data to be normalized between 0 and 1.
For this, we use Scikit's StandardScaler, which can perform data-relative scaling.
However useful this feature is, it also has its downsides.

One of the downsides is that when two datapoints are far away, they appear closer after scaling.
Moreover, when dealing with novelty data, scaling is not possible since the novelty data are not present yet.

Figure X shows the result of an above-defined experiment with the following settings:
- nu = 0.02
- kernel = rbf

The figure shows that the algorithm successfully marked the novelty data.
Notice that it also shows some of the *regular* data marked as novelty (note, that the testing dataset provided for evaluation phase is different than the training one). 
This phenomenon is called the false positive findings and will be examined later during evaluation of experiments.
![](https://raw.githubusercontent.com/chazzka/clanekcathedral/master/code/figures/example4.svg)
> Figure x (Example 4)


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
![](https://raw.githubusercontent.com/chazzka/clanekcathedral/master/code/figures/example5.svg)
> Figure example 5

As we can see, the algorithm was successful in isolating all of the novelty datapoints.

Because the Local Outlier Factor algorithm calculates the distance metric, with our model trained, we can elaborate on that and provide more novel datapoints to observe the distances calculated.
Figure x shows the algorithm when performed on a mesh of datapoints.

![](https://raw.githubusercontent.com/chazzka/clanekcathedral/master/code/figures/example6.svg)
> Figure example 6


Figure X shows one more crucial observation. 
The area marked by LOF algorithm is cut in the right side.
Since we fit the data on both Time and Observed values, the algorithm treats all the Time values as part of the neighborhood.
This is an undesired behavior in the sense of time series datamining, since the Time variable value is constantly rising. 

#### Time variable preprocessing
In the above sections we described a problem with constantly rising time variable value.
This problem is usually fixed by dividing the dataset into equal intervals.
Since the measurements in our environment are done several times throughout the day, we chose to divide the dataset into 24-hour intervals and remap the time according to this new interval.
This has several positive outcomes.
First, the varying time is no more relevant, since the intervals have the same time span.
Second, the training area gets more condensed, hence more accurate fitting is possible.
Figure X shows the experiment containing novelty datapoints and also some other datapoints from measurement with later time values, both normalized in the 24 hours interval.

![](https://raw.githubusercontent.com/chazzka/clanekcathedral/master/code/figures/example7.svg)
> example 7

As figure shows, the new regular measurements get assigned the correct REGULAR label.
 
#### Setting the right parameters
All of the algorithms introduced in this article suffer from the need to optimize the hyperparameters.
This optimization can be a tedious process, due to not only the number of hyperparameters, but also their interconnectivity.
This kind of issue is widely known amongst AutoML community.
Some tools have already been implemented that try to deal with the issue of automatic hyperparameter tuning, namely H20 (h2o.ai) or AutoGluon (auto.gluon.ai). 

## Experiments
For the following experiments, the data described in the introductory section were used.
Since we use the semi-supervised learning paradigm, we extracted several time intervals with zero to no anomalies to perform fitting on.
Different time intervals were then used to perform evaluation on.
The hyperparameters for each algorithm were set experimentally using autoconfiguration mechanisms.
Each of the values in the table is an average of 30 runs to obtain statistically significant results.

### Hyperparameter settings
For the experiments, following settings were chosen after fine tuning:
#### One-class SVM settings

For the experiments, the following hyperparameters were used for One-class SVM algorithm. 
For additional information, refer to Scikit's documentation: https://scikit-learn.org/stable/modules/generated/sklearn.svm.OneClassSVM.html

| Hyperparameter | value |
|---|---|---|
| kernel |rbf|
| gamma |$\frac{1}{\|features\| \cdot variance(x)}$|
| tol |$1\cdot10^{-3}$|
| nu |0.02|
| shrinking|True|
|cache_size|200|
|max_iter| -1|

#### Local Outlier Factor settings
For the experiments, the following hyperparameters were used for Local Outlier Factor algorithm. 
For additional information, refer to Scikit's documentation: https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.LocalOutlierFactor.html

| Hyperparameter | value |
|---|---|---|
| n_neighbors |20|
| algorithm |auto|
| leaf_size |30|
| metric |minkowski (p=2)|
| novelty|True|

### Experiment 1

Experiment with code number 1 contains 10 visually distinctive novelty datapoints at Y axis values around 185.


Figure x shows the side-by-side visual representation of one of the 30 runs.

One-class SVM            |  LOF
:-------------------------:|:-------------------------:
![](https://raw.githubusercontent.com/chazzka/clanekcathedral/master/code/figures/svm_experiment1.svg)  |  ![](https://raw.githubusercontent.com/chazzka/clanekcathedral/master/code/figures/lof_experiment1.svg)
>Figure x. Experiment 1 side by side comparison. One representative run.

Table X shows the average of regular/novelty datapoints marked by Local Outlier Factor compared to One-class SVM.

|  | One-class SVM | LOF |
|---|---|---|
| Regular |191|199|
| Novelty |19|11|

### Experiment 2
Experiment labeled 2 contains visually distinctive novelty datapoints stretched in Y interval of aprox. $[160,220]$.
|  | One-class SVM | LOF |
|---|---|---|
| Regular |188|200|
| Novelty |24|12|

Figure x shows the side-by-side visual representation of one of the 30 runs.

One-class SVM            |  LOF
:-------------------------:|:-------------------------:
![](https://raw.githubusercontent.com/chazzka/clanekcathedral/master/code/figures/svm_experiment2.svg)  |  ![](https://raw.githubusercontent.com/chazzka/clanekcathedral/master/code/figures/lof_experiment2.svg)
>Figure x. Experiment 2 side by side comparison. One representative run.

Table X shows the average of regular/novelty datapoints marked by Local Outlier Factor compared to One-class SVM.

### Experiment 3
Experiment with code number 3 contains two visually distinctive novelty clusters situated far away from each other.

Figure x shows the side-by-side visual representation of one of the 30 runs.

One-class SVM            |  LOF
:-------------------------:|:-------------------------:
![](https://raw.githubusercontent.com/chazzka/clanekcathedral/master/code/figures/svm_experiment3.svg)  |  ![](https://raw.githubusercontent.com/chazzka/clanekcathedral/master/code/figures/lof_experiment3.svg)
>Figure x. Experiment 3 side by side comparison. One representative run.

Table X shows the average of regular/novelty datapoints marked by Local Outlier Factor compared to One-class SVM.

|  | One-class SVM | LOF |
|---|---|---|
| Regular |189|198|
| Novelty |23|14|

### Experiment 4
Experiment with code number 4 contains one condensed cluster with many overlapping datapoints around X=200 and Y=200.

Figure x shows the side-by-side visual representation of one of the 30 runs.

One-class SVM            |  LOF
:-------------------------:|:-------------------------:
![](https://raw.githubusercontent.com/chazzka/clanekcathedral/master/code/figures/svm_experiment4.svg)  |  ![](https://raw.githubusercontent.com/chazzka/clanekcathedral/master/code/figures/lof_experiment4.svg)
>Figure x. Experiment 4 side by side comparison. One representative run.

Table X shows the average of regular/novelty datapoints marked by Local Outlier Factor compared to One-class SVM.

|  | One-class SVM | LOF |
|---|---|---|
| Regular |191|200|
| Novelty |34|25|

### Experiment 5
Experiment with code number 5 contains one condensed cluster with many overlapping datapoints around X=200 and Y=200.

Figure x shows the side-by-side visual representation of one of the 30 runs.

One-class SVM            |  LOF
:-------------------------:|:-------------------------:
![](https://raw.githubusercontent.com/chazzka/clanekcathedral/master/code/figures/svm_experiment5.svg)  |  ![](https://raw.githubusercontent.com/chazzka/clanekcathedral/master/code/figures/lof_experiment5.svg)
>Figure x. Experiment 5 side by side comparison. One representative run.

Table X shows the average of regular/novelty datapoints marked by Local Outlier Factor compared to One-class SVM.

|  | One-class SVM | LOF |
|---|---|---|
| Regular |197|199|
| Novelty |7|5|

### Experiment 6
The last experiment numbered 6 contains large number of visually anomalous datapoints. 
This is ideal for observing behavior for datasets with regular and novelty datapoints in close to 1:1 ratio.


Figure x shows the side-by-side visual representation of one of the 30 runs.

One-class SVM            |  LOF
:-------------------------:|:-------------------------:
![](https://raw.githubusercontent.com/chazzka/clanekcathedral/master/code/figures/svm_experiment6.svg)  |  ![](https://raw.githubusercontent.com/chazzka/clanekcathedral/master/code/figures/lof_experiment6.svg)
>Figure x. Experiment 6 side by side comparison. One representative run.

Table X shows the average of regular/novelty datapoints marked by Local Outlier Factor compared to One-class SVM.

|  | One-class SVM | LOF |
|---|---|---|
| Regular |191|199|
| Novelty |59|51|


## Results and discussion
We examined six different experiments analyzed by two of the most famous algorithms capable of novelty detection.
Both algorithms performed reasonably well in our scenarios and can be used in production environment.
Several takeouts can be extracted from the experiments.
Both algorithms have to be trained on the regular dataset containing zero to no anomalies.
This could be a problem in a production environment where anomalies are usual.
However, as shown in the article, such training is the only option when trying to label the previously unseen - novelty - datapoints.
Furthermore, without tedious hyperparameters tweaking, the One-class SVM algorithm showed more false positive findings as opposed to the Local Outlier Factor.
Both algorithms are available in the Scikit's python environment and are open-source and ready to use by anyone.

## Conclusion
In the article, the algorithms for labeling previously unseen, novelty datapoints in time series data were examined.
We discussed the important difference between outlier and novelty detection, leaning towards the latter.
We explained the novelty detection paradigm, and introduced the algorithms used in this area.
We analyzed two algorithms for novelty detection, the OneClass SVM and the Local Outlier Factor. 
From the production environment, several experiments were designed and performed analysis on.
Both algorithms showed satisfactory results and thus can compete in such environments.


## References
https://scikit-learn.org/stable/about.html
https://matplotlib.org/stable/users/project/citing.html
<!--stackedit_data:
eyJoaXN0b3J5IjpbLTE1NDExNDQ1MjAsNzAxNDEyNjEwLC03OT
YwOTYyODQsMTk3Nzk5NTM5Nyw5Mjg0ODMwMzIsLTE2OTgyNjc2
NTUsLTE2NDI5OTE3MjcsMTAyMDAxNjg5NiwtNDYyNTkwMTAwLC
04NDQ0NjM3MzksLTE1NzU5ODU5ODksMTk0MDY5NTY1MiwyNzUy
NzEwMDQsLTExNzM1MzI4OTcsMTA1NTc1NzQ2NywxNzExMDQzNT
gsMTYwMzE5MzI4OCwyNzQzNzg4NjYsNzE1ODE1MTM5LDM3OTA0
Nzg1MV19
-->