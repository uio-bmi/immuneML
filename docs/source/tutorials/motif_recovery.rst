How to discover disease or antigen specificity associated motifs using immuneML
==================================================================================

immuneML provides several different options for recovering motifs associated with immune states or antigen specificity.
Depending on the context, immuneML provides several different reports which can be used for this purpose.


Recovering unknown motifs when training classifiers
---------------------------------------------------

Discovering motifs learned by classifiers
-----------------------------------------

When using the :ref:`KmerFrequency` encoder in combination with
:ref:`LogisticRegression`, :ref:`SVM` or :ref:`RandomForest`, a straightforward way to investigate which
k-mer subsequences were learned to be important is by running the :ref:`Coefficients` report.
This can be applied both to repertoire and receptor classification problems.

Strongly positive coefficients might indicate (partial) disease or antigen specificity-associated motifs.
Consider the following example, where the groundtruth implanted disease signal was 'VLEQ', the largest coefficients
are associated with the subsequences 'VLE' and 'LEQ' which are contained inside 'VLEQ'.
Furthermore, subsequences that partially overlap with the disease signal, such as 'EQD', 'EQV' and 'YVL'
are also associated with relatively large coefficients.
Note that the coefficient size is not only determined by how important a subsequence is for determining an immune state
or antigen specificity, but also other factors, such as
:ref:`the baseline frequency of subsequences in a dataset <Comparing baseline motif frequencies subsets of the dataset>`.

.. image:: ../_static/images/reports/coefficients_logistic_regression.png
   :alt: Coefficients report
   :width: 600


When using the classifier :ref:`DeepRC` for a repertoire classification task, the report :ref:`DeepRCMotifDiscovery` can
be used to investigate the learned patterns in the data.
This report plots the contributions of input sequences and kernels to trained DeepRC model.
These contributions are computed using integrated gradients (IG).




This report plots which positions in the input sequences and the kernels


The report creates plots


    This report produces two figures:
        - inputs_integrated_gradients: Shows the contributions of the characters within the input sequences (test dataset) that was most important for immune status prediction of the repertoire. IG is only applied to sequences of positive class repertoires.
        - kernel_integrated_gradients: Shows the 1D CNN kernels with the highest contribution over all positions and amino acids.



RelevantSequenceExporter?

A strongly positive coefficient




A straightforward way to get insight to the




For repertoire classification tasks,


The repertoire classification method :ref:`DeepRC` comes with


The repertoire classification methods :ref:`DeepRC` and :ref:`TCRdist` both


When using DeepRC: DeepRCMotifRecovery
When using TCRdist: TCRdistMotifDiscovery

When using Kmer encoder and sklearn methods LogisticRegression,


Recovering simulated immune signals
-----------------------------------
when recovering implanted k-mers: MotifSeedRecovery



Comparing baseline motif frequencies subsets of the dataset
-----------------------------------------------------------
baseline frequencies of motifs:
- kmerencoder + featurevaluebarplot / featurevaluedistplot

more complex motifs:
- matches reports


 When using kmer encoder, FeatureValueBarplot or FeatureValueDistplot could be used to visualize this.
 Otherwise MatchedRegexEncoder + Matches report for investigating more complex motifs (regexes)


- when motifs are unknown

- when motifs are known



in relation to discussions with Chakri: it can be useful to investigate the baseline occurrence of motifs.
When using kmer encoder, FeatureValueBarplot or FeatureValueDistplot could be used to visualize this.
Otherwise MatchedRegexEncoder + Matches report for investigating more complex motifs (regexes)

I guess this could be a tutorial if we have the time for it? We'll have a
feature value report in a week or so (Create a report showing the distribution of feature values across classes),
so we can combine matched encoders and reports, tcrdist report and feature value report with e.g. k-mers as possible scenario? :)


sounds like a good idea! I'd be interested in contributing to this.
I think an important part is that there are several different ways to do motif recovery.
What you describe is one scenario, and then for deeprc/tcrdist they have their own reports,
and when implanted signals are known there is the MotifSeedRecovery when working with known implanted signals.


I agree - it could still be one tutorial, but with different subsections: what if we don't know the motifs
or if we do know them, etc. :) You could then take the lead here if you
want and I can help if things are unclear with tcrdist :)


