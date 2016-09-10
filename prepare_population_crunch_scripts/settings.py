"""
An example of a settings file that is valid to pass into find_parameters
"""
from sklearn import svm
import numpy as np
############################################################
#Preparation (prepare_population)

#CALLSET_NAME_REGEX (string) used to match the name found in the phenotypic files
CALLSET_NAME_REGEX = "(hu[0-9A-F]+|HG[0-9]+|NA[0-9]+)"
#QUALITY (boolean). If true, uses quality_phase numpy files. If false, uses phase numpy files
QUALITY = True
#Number of phases per each callset
#Also used for run_classifier
NUM_PHASES = 2
#Number of callsets
NUM_CALLSETS = 174
#REMOVE_POORLY_SEQ (boolean) If true, all tile positions with at least one -1 (poorly sequenced tile) will
#    be removed before building the classifier
REMOVE_POORLY_SEQ = True
#REMOVE_HOMOGENOUS (boolean) If true, all tile positions that have only one
#    variant in the population that are equal will be removed before building the classifier.
#    WARNING: If all tile variants are equal to -1, that position will also be removed,
#    even though the tiles might be different in reality
REMOVE_HOMOZYGOUS = True

############################################################
#Classifier Parameterization (find_parameters)

C_range = np.logspace(-4,1,11)
wide_C_range = np.logspace(-3,6,10)
wide_nu_range = np.linspace(0.1, 0.3, 5)

#classifiers_to_parameterize: (List of lists), list of support vector machines to paramterize along with the parameters to vary
#   expects the second element to only have 1 key to enable plots
#       if this key equals 'C', the plot will be on a semilogx plot
#   expects the first element to support:
#       base.clone(element)
#       element.fit()
#       element.set_params()
#       grid_search.GridSearchCV(element, ...)
classifiers_to_parameterize = [
    [svm.LinearSVC(penalty='l1', dual=False),dict(C=C_range)],
    [svm.LinearSVC(penalty='l1', dual=False),dict(C=wide_C_range)],
    [svm.LinearSVC(dual=False),dict(C=C_range)],
    [svm.LinearSVC(),dict(C=wide_C_range)],
    [svm.NuSVC(kernel='rbf'),dict(nu=wide_nu_range)],
    [svm.NuSVC(kernel='linear'),dict(nu=wide_nu_range)],
    #[svm.SVC(kernel='rbf'),dict(C=wide_C_range)],
    #[svm.SVC(kernel='linear'),dict(C=wide_C_range)],
]

############################################################
#Classification and Prediction for one classifier (run_classifier)

#Also uses NUM_PHASES!

#classifier: (support vector machine), SVM to fit and predict with
#   classifier should support:
#       clf.fit()
#       clf.coef_
#       clf.decision_function()
#       (if CROSS_VALIDATE==True): cross_validation.cross_val_score(clf)
#       clf.predict()
classifier = svm.LinearSVC(penalty='l1', dual=False, C=0.05)
#CROSS_VALIDATE (Boolean) If true, will perform cross-validation using leave-one-out
#   information will be written to a log file and variable file
CROSS_VALIDATE = True
#PLOT (Boolean): If true, will plot coef_ of classifier
PLOT = True
