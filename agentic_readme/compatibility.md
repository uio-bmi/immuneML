# Compatibility: dataset types, encoders, classifiers, reports

This is a snapshot, so check the source before relying on it.

**Where the rules come from in the code:**
- Classifier → encoders: `get_compatible_encoders()` in each classifier, enforced by `MLMethod.check_encoder_compatibility`. The shared sklearn list is in `SklearnMethod.get_compatible_encoders`.
- Encoder → dataset type: the encoder's `build_object` / `dataset_mapping`, and the `**Dataset type:**` section of its docstring.
- Report requirements: `check_prerequisites()` and the type checks inside `_generate()`. If a prerequisite fails, the report is skipped with a "(failed)" result; no error is raised.

## Encoders by dataset type (Seq / Rec / Rep)

| Supported dataset types | Encoders |
|---|---|
| All three | KmerFrequency, OneHot, GeneFrequency, Metadata, Composite, AminoAcidProperty, ESMC, ProtT5, TCRBert |
| Seq and Rep | Word2Vec |
| Seq and Rec | TCRdist, SequenceLength |
| Seq only | Motif, SimilarToPositiveSequence |
| Rep only | EvennessProfile, ShannonDiversity, DeepRC, Distance, CompAIRRDistance, SequenceAbundance, CompAIRRSequenceAbundance, KmerAbundance, MatchedSequences, MatchedReceptors, MatchedRegex |

## Classifiers and their compatible encoders

| Classifier | Encoders it accepts |
|---|---|
| LogisticRegression, SVM, SVC, RandomForestClassifier, GradientBoosting, XGBClassifier, LogRegressionCustomPenalty | the general sklearn list (most feature-vector encoders, including Composite and the protein embeddings) |
| KNN | KmerFrequency, OneHot, Word2Vec, abundance encoders, EvennessProfile, Distance |
| PrecomputedKNN | Distance, CompAIRRDistance |
| TCRdistClassifier | TCRdist |
| ProbabilisticBinaryClassifier | SequenceAbundance, CompAIRRSequenceAbundance, KmerAbundance (binary label) |
| BinaryFeatureClassifier | Motif, SimilarToPositiveSequence |
| DeepRC | DeepRC |
| ReceptorCNN | OneHot, on receptor data |
| KerasSequenceCNN | OneHot with `flatten: False` and `use_positional_info: False`, on sequence data |

## Preprocessing
- Most filters are for **repertoire datasets only**: ChainRepertoireFilter, ClonesPerRepertoireFilter, CountPerSequenceFilter, SubjectRepertoireCollector.
- DuplicateSequenceFilter also accepts sequence datasets.
- In TrainMLModel `settings`, a preprocessing step must keep the number of examples unchanged.

## Report prerequisites (selected)

| Report | Requires |
|---|---|
| Coefficients | linear or tree models: LR, SVM, SVC, RF, GradientBoosting, XGB, LogRegressionCustomPenalty |
| SHAPReport | an sklearn method or XGB |
| Matches | Matched* encoders |
| Motif-specific reports (PositionalMotifFrequencies, …) | Motif encoder |
| BinaryFeaturePrecisionRecall | Motif encoder + BinaryFeatureClassifier |
| KernelSequenceLogo | ReceptorCNN |
| DeepRCMotifDiscovery | DeepRC |
| TCRdistMotifDiscovery | TCRdistClassifier |
| SequenceAssociationLikelihood | ProbabilisticBinaryClassifier |
| DimensionalityReduction, ClusteringVisualization | not available with distance encoders (TCRdist, Distance) |
| Dendrogram | AgglomerativeClustering |
| Repertoire-level data reports (ShannonDiversityOverview, SignificantFeatures, RepertoireClonotypeSummary, …) | RepertoireDataset |
| GLIPH2Exporter | ReceptorDataset |

## Which report categories each instruction accepts

| Instruction | Report categories |
|---|---|
| TrainMLModel `reports:` | TrainMLModelReport only |
| TrainMLModel `assessment`/`selection` → `reports:` | `data` and `data_splits` take DataReports; `encoding` takes EncodingReports; `models` takes MLReports |
| ExploratoryAnalysis | DataReport; EncodingReport when an encoding is given |
| Clustering | ClusteringReport, ClusteringMethodReport, EncodingReport |
| ValidateClustering | ClusteringMethodReport, DataReport, EncodingReport, ClusteringReport (ExternalLabelMetricHeatmap) |
| TrainGenModel | TrainGenModelReport, GenModelReport, DataReport, MLReport |
| ApplyGenModel | DataReport, MLReport |
