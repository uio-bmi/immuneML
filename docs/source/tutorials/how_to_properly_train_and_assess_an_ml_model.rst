How to properly train and assess an ML model
============================================

.. meta::

   :twitter:card: summary
   :twitter:site: @immuneml
   :twitter:title: immuneML: training and assessing an ML model
   :twitter:image: https://docs.immuneml.uio.no/_images/receptor_classification_overview.png


The powerful pattern discovery abilities of machine learning methods make them well suited to learn signals that characterize receptors reactive to a
given antigen or signals that characterize immune repertoires of donors being positive for a given disease. We refer to this as learning a classifier
for receptors or repertoires. Learning a machine learning classifier amounts to optimizing the parameters of a given model, which is achieved by a
training procedure specific to a particular machine learning model, as well as by selecting appropriate values for what is referred to as
hyper-parameters: tunable characteristics of a model that the training procedure for the method is not able to optimize by itself.
immuneML supports the automatic optimization of such hyper-parameters through the well established approach of splitting data into a training set and
validation set, where a set of models based on different hyper-parameter values are trained on the training set, and then assessed on the separate
validation data to see which hyper-parameter performs best and is to be used. immuneML supports doing this based on either a fixed division of the
data into parts used for training and validation or based on a cross-validation approach with configurable number of folds. In immuneML, this is not
restricted to hyper-parameter of the machine learning methods used, but also includes hyper-parameters related to how the immune receptor data is
processed, filtered and encoded, which can be even more important and hard to select in an optimal and unbiased manner.

Since the training and validation data has been used to optimize the parameters and hyper-parameters (respectively), one needs a third portion of the
data set, previously unseen by the classifier, to get an unbiased assessment of a classifier's prediction performance. immuneML supports doing this
based on either a fixed portion of the dataset set aside as test test or based on a nested cross-validation approach with configurable number of folds
for both the inner and outer loop. One will typically set up a single immuneML run to train models, optimize hyper-parameters and get an unbiased
assessment of its performance. The resulting optimized classifier can also afterwards be applied to further datasets.
