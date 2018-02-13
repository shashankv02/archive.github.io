---
title: Logistic Regression. 
desc:  Logistic regression is used for classification problems. Classifying emails as spam or not spam, predicting whether the weather will be sunny, cloudy or stormy etc are classification problems
topic: ml
mathjax: true
---

Logistic regression is used for classification problems. Classifying emails as spam or not spam, predicting whether the weather will be sunny, cloudy or stormy etc are classification problems. In these problems we want to predict a discrete valued output from a possible set of values. 

In this article we will choose a hypothesis function that is suitable for classification problems. Then we will choose a cost function that tells us the error between the output predicted by the hypothesis and actual value. We will use this cost function and our training examples to arrive at the parameters for our hypothesis that best fit the training examples using a gradient descent method. Once modelled, we can use the hypothesis to classify any new example.

### Hypothesis

Consider a binary classification problem where we want to classify an email as spam or not spam depending on some features of the email like content, title, sender of the email etc. Consider we have training set of ten thousand emails which are already labbelled as spam or not-spam.
 
We want a hypothesis function h(x) to output a value closer to 1 if the email is spam else a value closer to 0 where x represents the features of the email. In other words, if we consider spam as 1 and not-spam as 0, the hypothesis fucntion gives probability of email being spam. If we chose our decision boundary as h(x)=0.5, we consider h(x) >= 0.5 being the email being spam and h(x) < 0.5 being not-spam. 

One such function which fits our requirements is sigmoid fucntion.

$$ sig(z) = 1/(1+e^{-z}) $$

 <img src="{{ site.baseurl }}/assets/img/ml/logistic_regression_sigmoid.png"> 

The sigmoid function lies between 0 and 1 and is greater than 0.5 when z>0 and less than 0.5 when z<0.

Say we have n features $$ x_1, x_2, .. x_n $$ 

Let $$\theta_0, \theta_1, \theta_2, .. \theta_n $$ be the parameters which we learn from the training examples to construct our hypothesis. 

$$ g(x) = \theta_0 + \theta_1x_1 + \theta_2x_2 + ... + \theta_nx_n $$

$$ \left(\text{Here } x \text{ in }g(x) \text{ implies } g(x_0, x_1, x_2 .. x_n)\right)$$

Let $$ \theta = \begin{bmatrix} \theta_0 \\ \theta_1 \\ \vdots \\ \theta_n  \end{bmatrix} $$ be the parameter vector  and $$ x =  \begin{bmatrix} 1 \\ x_1 \\ \vdots \\ x_n  \end{bmatrix} $$ be the feature vector for any training example. 

$$ g(x) = \theta^Tx $$


We can model our hypothesis as follows.

$$ h(x) = sig\left(g(x)\right) $$  


$$ h(x) = sig(\theta^T.x) $$  

$$ h(x) = 1/(1+e^{-\theta^Tx}) $$ 

Before we can start using this hypothesis to classify emails we need to find the parameters $$\theta_0, \theta_1, .., \theta_n$$. We have many sets of values of $$x$$ available from the training set and we know for all the spam emails from the training set h(x) is 1 and for the rest h(x) is 0. Using this data we can find the parameter vector $$\theta$$ such that the hypothesis best fits the training data. To understand what best fitting the training data means, lets look at decision boundary in the following section.


## Decision boundary

For the email classification problem we have chosen our decision boundary to be h(x) = 0.5. It is the boundary that seperates our classes namely spam and not-spam. Based on which side of the boundary the given example lies, we classify it as one of the classes. 

Since h(x) is a sigmoid function, h(x) = 0.5 when $$ \theta^Tx = 0 $$ i.e., h(x) predicts email as spam whenever $$ \theta^Tx >= 0 $$ and not-spam whenever $$ \theta^Tx < 0 $$

$$ \theta^Tx = 0 $$ is the decision boundary. The boundary can be of any shape depending on the order of $$\theta^Tx$$.


For example, consider we have two features and let $$ \theta = \begin{bmatrix} 5 \\ -2 \\ -1 \end{bmatrix} $$.

$$ \text{h(x) = 1 when } \theta^Tx >= 0  $$

$$           -5 + (-2)x_1 + (-1)x_2 >= 0 $$
  
$$		   2x_1 + x_2 <= 5 $$


The decision boundary is the line $$ 2x_1 + x_2 = 5 $$ in this example. Anything below the line falls under h(x) = 1 class i.e., spam class and anything above falls under non-spam class. 

<img src="{{ site.baseurl }}/assets/img/ml/logistic_regression_1.gif"> 

Understand that the decision boundary is a property of hypothesis for given parameters $$\theta $$. It does not depend on the training examples. It's the inherent property of the hypothesis and it's not necessary to calculate it to solve the problem at hand and is only needed to understand the intuition behind the logistic regression hypothesis. 

The decision boundary in this example is a line because we modelled $$ \theta^T.x $$ as a linear function. But depending on the data set we can model it as any higher order polynomial resulting in decision boundary which can be of any shape. 

## Cost function

The mean squared error cost function used for linear regression results in non-convex cost function for logistic regression because of the non-linear sigmoid hypothesis function. Optimizing non-convex cost function with gradient descent results in gradient descent stabilizing at local minima instead of the global minimum which is undesired. 

The following cost function fits perfectly for any training example of logistic regression and is convex and can be derived using principle of maximum likelihood estimation. 

$$ cost(h(x), y) = -y\log h(x) - (1-y)\log (1-h(x)) $$

y is the actual class value ( 0 or 1 ). The cost function estimates the error between the value predicted by hypothesis and actual value.

Let's see how this cost function makes sense for logistic regression.

$$ \text{when y = 1, } cost(h(x), y) = -\log h(x) $$

<img src="{{ site.baseurl }}/assets/img/ml/logistic_regression_2.png"> 

From the plot of $$ -\log h(x) $$, as h(x) approaches 1, cost approaches 0 and as h(x) approaches 0, h(x) approaches infinity which are desirable properties when y=1. Let's look at y=0 case now.


$$ \text{when y = 0, } cost(h(x), y) = -\log (1-h(x)) $$

<img src="{{ site.baseurl }}/assets/img/ml/logistic_regression_3.png"> 

In the case of y=0, as h(x) approaches 0, cost approaches 0 and as h(x) approaches 1, cost approaches infinity.  


Using the above cost function for single training example, we can write combined cost function for logistic regression as follows

$$ J(\theta) = \frac{1}{m} \left[ \sum_{i=\text{1 to m}} cost(h(x^{(i)}), y^{(i)}) \right] $$
 
where $$ m $$ is the number of training examples.

$$ J(\theta) = \frac{1}{m}\sum_{i=\text{1 to m}} \left[ -y^{(i)}\log h(x^{(i)}) - (1-y^{(i)})\log (1-h(x^{(i)}))\right]  $$

Now that we have the cost function, we can minimize this as function of $$ \theta $$ to find the parameter vector $$ \theta $$ that gives us the hypothesis with minimum cost. Using this hypothesis our model will be able predict the outcome for any features in future i.e., whether the new email is spam or not-spam. We can use any optimization alogirithm like Gradient descent, BFGS, L-BFGS, Conjugate gradient etc for this. Let's look at gradient descent which is the simplest of these. 

## Gradient descent

Gradient descent is a iterative algorithm to find the minumum of a function. Read the [wikipedia](https://en.wikipedia.org/wiki/Gradient_descent) article or any other resource to understand the intuition behind gradient descent. 

Gradient descent update can be written as 

$$ 
\text{Repeat } \theta_j = \theta_j  - \alpha\frac{\partial}{\partial\theta_j}J(\theta) \text{ simulataneoulsy updating all } \theta_j $$ 

We adjust the value of each parameter $$ \theta_j $$ at learning rate $$\alpha$$  until we reach the optimum. Gradient descent converges i.e., stops changing when it reaches the optimum.

Solving the above equation, we arrive at 

$$ 
\text{Repeat } \theta_j = \theta_j  - \alpha \sum_{i=\text{1 to m}} \left(h(x^{(i)})-y^{(i)}\right)x_j^{(i)} $$

Observe that we use all the training examples in every update. Once we arrive at the optimum values of parameter vector $$\theta$$, we can plug the values into out hypothesis. 

Great! We now can use the hypothesis to classify any new email as spam or not-spam.

## Multi class classification

We have looked at classifying the emails as spam or not-spam so far which is a binary classification problem. What if we wanted to classify data into multiple classes? Eg: Tagging email as Primary, Updates, Social, Forums etc just like how gmail does.

### One-vs-all strategy

In One-vs-all or One-vs-rest strategy we train single classifer per class, with samples of that class as positive samples and all other samples as negatives.

Eg: Say we have labelled data of email which are tagged as Social, Updates and Forums. 

We train classifer $$ h^{social}(x) $$ with emails tagged as social as positive i.e., h(x) = 1 and all other emails as negative. 

Similarly we train another classifier $$ h^{updates}(x) $$ with emails tagged as updates as positive and rest as negative and train $$ h^{forums}(x) $$ on similar lines. 

Once we have the three classifiers established, when a new email comes, we run the three classifiers on the features of the email and we tag the email as one of the classes depending on which classifier outputs the highest probability. 