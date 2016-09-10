#!/usr/bin/env python
"""
Purpose: Load numpy files and tune parameters as directed

Does not use parallelization since it analyzes all the population at once

Inputs:
  Required
    'input' : Collection output by prepare-population
        [antigen_type]_training.npy: the training data set, reduced if specified, and scaled by StandardScalar
        [antigen_type]_training_labels.npy: the training labels for the training data set for [antigen-type] antigen
        [antigen_type]_test.npy: the test data set, reduced if specified, and scaled by Standard Scalar
        [antigen_type]_where_well_sequenced.npy: a boolean array indicating which columns (tiles)
            were kept from the original population because they were well sequenced.
            If settings.REMOVE_POORLY_SEQ==False, the array will be entirely true
        [antigen_type]_where_homozygous.npy: a boolean array indicating which columns (tiles)
            were kept from the population after removing poorly sequenced columns.
            If settings.REMOVE_HOMOZYGOUS==False, the array will be entirely true
        python_variables.py: python file defining training_names and test_names

    'settings' : Python file
    'num-retries' : (integer) Specifies the number of retries we should use to access collections
    'antigen-type' : (string) Specifies the antigen to use


Outputs:

"""
import matplotlib as mpl # must be imported first to ensure matplotlib visualization works
mpl.use('Agg')

import matplotlib.pyplot as plt
import helping_functions as fns
from sklearn import cross_validation, preprocessing, grid_search, base
import arvados      # Import the Arvados sdk module
import re           # used for error checking
import imp
import os
import time
import numpy as np

########################################################################################################################
# Read constants
NUM_RETRIES = int(arvados.getjobparam('num-retries'))
assert NUM_RETRIES > 0, "'num-retries' must be strictly positive"

antigen_type = str(arvados.getjobparam('antigen-type'))
########################################################################################################################
#Set-up collection and logging file to write out to
out = arvados.collection.Collection(num_retries=NUM_RETRIES)
time_logging_fh = out.open('time_log.txt', 'w')
info_fh = out.open('log.txt', 'w')
########################################################################################################################
# Load settings
t0 = time.time()
settings = imp.load_source('settings', arvados.get_job_param_mount('settings'))
t1 = time.time()
time_logging_fh.write('Loading settings %fs\n' %(t1-t0))
########################################################################################################################
#Parallelize based on settings
def one_task_per_classifier(num_classifiers_to_parameterize, if_sequence=0, and_end_task=True):
    if if_sequence != arvados.current_task()['sequence']:
        return
    api_client = arvados.api('v1')
    for i in range(num_classifiers_to_parameterize):
        new_task_attrs = {
            'job_uuid': arvados.current_job()['uuid'],
            'created_by_job_task_uuid': arvados.current_task()['uuid'],
            'sequence': if_sequence + 1,
            'parameters': {
                'classifier_index':i,
                'time_to_wait':i*560
            }
        }
        api_client.job_tasks().create(body=new_task_attrs).execute()
    if and_end_task:
        api_client.job_tasks().update(uuid=arvados.current_task()['uuid'],
                                      body={'success':True}
                                      ).execute()
        exit(0)

one_task_per_classifier(len(settings.classifiers_to_parameterize))
########################################################################################################################
#
print "Reading in population and training labels"
t0 = time.time()
input_collection_reader = arvados.CollectionReader(arvados.getjobparam('input'))
with input_collection_reader.open('%s_training.npy' % (antigen_type), 'r') as f:
    population = np.load(f)
with input_collection_reader.open('%s_training_labels.npy' % (antigen_type), 'r') as f:
    training_labels = np.load(f)
t1 = time.time()
time_logging_fh.write('Loading population and labels took %fs\n' % (t1-t0))

########################################################################################################################

print "Starting to parameterize classifier"

info_fh.write("Parameterizing antigen %s\n" % (antigen_type))

num_total = len(training_labels)
num_neg, foo, num_pos = np.bincount(training_labels+np.ones(training_labels.shape, dtype=np.int32))
info_fh.write("%s-antigen inputs: %i %s+ (%f%%), %i %s- (%f%%)\n" % (antigen_type, num_pos, antigen_type, num_pos/float(num_total), num_neg, antigen_type, num_neg/float(num_total)))

#GEt actual classifier
classifier_index = int(arvados.current_task()['parameters'].get('classifier_index'))
classifier, classifier_varying_params = settings.classifiers_to_parameterize[classifier_index]

classifier_to_print = str(classifier).replace('\n', '')
classifier_to_print = classifier_to_print.replace(' ', '')

assert len(classifier_varying_params) == 1, "Only vary one thing at a time for plotting and error checking"

key = classifier_varying_params.keys()[0]
valid_param_vals = []
print "Checking for errors for classifier %s"  % (classifier_to_print)
print "Params: %s" % (classifier_varying_params)
for param_val in classifier_varying_params[key]:
    try:
        alt_classifier = base.clone(classifier)
        alt_classifier = alt_classifier.set_params(**{key:param_val})
        alt_classifier.fit(population, training_labels)
        valid_param_vals.append(param_val)
    except ValueError as e:
        info_fh.write("The parameter %s:%s for SVM %s for antigen %s errored: '%s'\n" % (key, param_val, classifier_to_print, antigen_type, e))
if len(valid_param_vals) > 0:
    try:
        print "Running grid for classifier %s"  % (classifier_to_print)
        print "Params: %s" % (classifier_varying_params)
        cv = cross_validation.LeaveOneOut(len(training_labels))
        grid = grid_search.GridSearchCV(classifier, {key:valid_param_vals}, cv=cv,, refit=False, verbose=3)
        t0 = time.time()
        grid.fit(population, training_labels)
        t1 = time.time()
        time_logging_fh.write('Fitting grid with SVM %s for antigen %s took %f seconds\n' % (classifier_to_print, antigen_type, t1-t0))
        print "The best parameters for antigen %s, SVM %s are:"  % (antigen_type, classifier_to_print)
        print "%s with a score of %f" % (grid.best_params_, grid.best_score_)
        print "Grid scores:"
        for thing in grid.grid_scores_:
            print thing
        info_fh.write("The best parameters for SVM %s for antigen %s are %s with a score of %f\n" % (classifier_to_print, antigen_type, grid.best_params_, grid.best_score_))
        info_fh.write("Grid scores:\n")
        info_fh.write("%s\n" % (grid.grid_scores_))
        plt.figure()
        plt.errorbar(
            [point[0][key] for point in grid.grid_scores_],
            [point[1] for point in grid.grid_scores_],
            yerr=[np.std(point[2]) for point in grid.grid_scores_],
            fmt='o'
        )
        plt.title('%s antigen prediction mean cross-validation scores for %s' % (antigen_type, classifier_to_print))
        plt.xlabel('%s' % (key))
        plt.ylabel('Cross-validation scores using Leave-one-out')
        if key == 'C':
            plt.semilogx()
        with out.open('%s-antigen-%s-classifier.png' %(antigen_type, classifier_to_print), 'w') as f:
            plt.savefig(f)
    except ValueError as e:
        info_fh.write("Fitting the grid for SVM %s for antigen %s errored: '%s'\n" % (classifier_to_print, antigen_type, e))

time_logging_fh.close()
info_fh.close()
# Commit the output to keep
task_output = out.save_new(create_collection_record=False)
arvados.current_task().set_output(task_output)

###########################################################################################################################################
