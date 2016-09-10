#!/usr/bin/env python
"""
Purpose: Load numpy files and run SVM classifier

Does not use parallelization since it analyzes all the population at once

Inputs:
  Required
    'callset-numpy-cgf-files' : Collection output by concat-numpy-files. Will contain some directory substructure of the following form
        callset-name/
            [quality_]phase0.npy
            [quality_]phase1.npy
            ...
        callset-name/
        ...
    'callset-phenotypes' : Collection with files containing phenotypic information
    'path-lengths' : Collection output by make-path-lengths
        path_integers.npy
        path_lengths.npy
    'settings' : Python file
    'num-retries' : (integer) Specifies the number of retries we should use to access collections
    'antigen-type' : (string) Specifies the antigen to use

Outputs:

"""
import matplotlib as mpl # must be imported first to ensure matplotlib visualization works
mpl.use('Agg')

import matplotlib.pyplot as plt
import helping_functions as fns
from sklearn import cross_validation, preprocessing, dummy
import arvados      # Import the Arvados sdk module
import re           # used for error checking
import imp
import os
import time
import itertools
import numpy as np

def convert_to_tile_int(position, path_lengths):
    trunc_path_lengths = path_lengths[1:]
    i = 0
    while position > trunc_path_lengths[i]:
        i += 1
    path = hex(i).lstrip('0x').rstrip('L').zfill(3)
    step = hex(position-path_lengths[i]).lstrip('0x').rstrip('L').zfill(4)
    return int(path+step, 16)

########################################################################################################################
# Hard-code certain booleans
REMOVE_POORLY_SEQ = True
REMOVE_HOMOZYGOUS = True
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
variable_fh = out.open('classifier_variables.py', 'w')
########################################################################################################################
# Load settings
t0 = time.time()
settings = imp.load_source('settings', arvados.get_job_param_mount('settings'))
t1 = time.time()
time_logging_fh.write('Loading settings %fs\n' %(t1-t0))
########################################################################################################################
#Get path lengths and path integers
cr = arvados.CollectionReader(arvados.getjobparam('path-lengths'), num_retries=NUM_RETRIES)
t0 = time.time()
with cr.open("path_integers.npy", 'r') as f:
    path_integers = np.load(f)
t1 = time.time()
with cr.open("path_lengths.npy", 'r') as f:
    path_lengths = np.load(f)
t2 = time.time()
time_logging_fh.write('Loading path integers took %fs\n' %(t1-t0))
time_logging_fh.write('Loading path lengths took %fs\n' %(t2-t1))

callset_collection_reader = arvados.CollectionReader(arvados.getjobparam('callset-numpy-cgf-files'))
callset_collection_reader.normalize()

#Get callset phenotype CollectionReader
phenotype = arvados.get_job_param_mount('callset-phenotypes')
phenotype_file_paths = []
for root, dirs, files in os.walk(phenotype):
    for f in files:
        phenotype_file_paths.append(root+'/'+f)
#phenotype_collection_reader = arvados.CollectionReader(arvados.getjobparam('callset-phenotypes')) --> json loading raises keep error
#phenotype_collection_reader.normalize()
########################################################################################################################

population, subjects, callset_names = fns.get_population(
    path_lengths,
    settings.NUM_CALLSETS,
    settings.NUM_PHASES,
    phenotype_file_paths,
    callset_collection_reader,
    settings.CALLSET_NAME_REGEX,
    time_logging_fh,
    settings.QUALITY
)

if settings.QUALITY:
    t0 = time.time()
    print "Reducing population"
    #Currently population has a large number of features with a relatively low number of samples.
    population, where_well_sequenced, where_homozygous  = fns.reduce_population(population, REMOVE_POORLY_SEQ, REMOVE_HOMOZYGOUS)
    t1 = time.time()
    time_logging_fh.write('Removing the poorly sequenced population tiles took %f seconds\n' % (t1-t0))

print "Making training data, training labels, and test data"
population, training_labels, test_population, training_names, test_names = fns.make_and_label_training_and_test_data(population, callset_names, subjects, antigen_type, time_logging_fh)
print "Training data shape: %s, number of MB: %f" % (population.shape, population.nbytes/1000000.)
print "Test data shape: %s, number of MB: %f" % (test_population.shape, test_population.nbytes/1000000.)

print "Casting to float64 for scalar"
population = population.astype(np.float_)
test_population = test_population.astype(np.float_)
print "Float Training data shape: %s, number of MB: %f" % (population.shape, population.nbytes/1000000.)
print "Float Test data shape: %s, number of MB: %f" % (test_population.shape, test_population.nbytes/1000000.)

print "Making scalar"
t0 = time.time()
scaler = preprocessing.StandardScaler()
population = scaler.fit_transform(population)
t1 = time.time()
time_logging_fh.write("Scaling training data using preprocessing.StandardScalar().fit_transform took %f seconds\n" % (t1-t0))
print "Scaled, truncated, training population shape is %s, number of MB is %f" % (population.shape, population.nbytes/1000000.)

print "Transforming test data to scalar"
t0 = time.time()
test_population = scaler.transform(test_population)
t1 = time.time()
time_logging_fh.write("Scaling test data using preprocessing.StandardScalar().transform took %f seconds\n" % (t1-t0))
print "Scaled, truncated, test population shape is %s, number of MB is %f" % (test_population.shape, test_population.nbytes/1000000.)

print "Building antigen %s classifier" % (antigen_type)
classifier_to_print = str(settings.classifier).replace('\n', '')
classifier_to_print = classifier_to_print.replace(' ', '')
info_fh.write("Building antigen %s classifier %s\n" % (antigen_type, classifier_to_print))

#Note shape and proportion of inputs
num_total = len(training_names)
num_neg, foo, num_pos = np.bincount(training_labels+np.ones(training_labels.shape, dtype=np.int32))
info_fh.write("%i %s-antigen inputs: %i %s+ (%f%%), %i %s- (%f%%)\n" % (num_total, antigen_type, num_pos, antigen_type, num_pos/float(num_total), num_neg, antigen_type, num_neg/float(num_total)))

#Build Classifier
t0 = time.time()
classifier = settings.classifier.fit(population, training_labels)
t1 = time.time()
time_logging_fh.write('%s-antigen classifier building took %f seconds\n' % (antigen_type, t1-t0))
variable_fh.write("svm_coefficients=%s\n" % (classifier.coef_.tolist()))
variable_fh.write("training_names=%s\n" % (training_names))
variable_fh.write("training_labels=%s\n" % (training_labels.tolist()))
variable_fh.write("training_confidence_scores=%s\n" % (classifier.decision_function(population).tolist()))

if settings.CROSS_VALIDATE:
    #Cross Validate Classifier - better than dummy classifiers according to http://scikit-learn.org/stable/modules/model_evaluation.html#model-evaluation
    print "Cross-validating"
    cv = cross_validation.LeaveOneOut(len(training_names))
    t0 = time.time()
    scores = cross_validation.cross_val_score(classifier, population, training_labels, cv=cv, n_jobs=int(os.environ['CRUNCH_NODE_SLOTS']))
    t1 = time.time()
    time_logging_fh.write('%s-antigen cross validation took %f seconds\n' % (antigen_type, t1-t0))
    info_fh.write("Cross Validation accuracy: %f (+/- %f)\n" % (scores.mean(), scores.std()*2))
    variable_fh.write("cv_scores=%s\n" % (scores.tolist()))

print "Predicting test labels"
#Create predicted test labels
t0 = time.time()
predicted_test_labels = classifier.predict(test_population)
t1 = time.time()
time_logging_fh.write('%s-antigen predition took %f seconds\n' % (antigen_type, t1-t0))
variable_fh.write('predicted_names=%s\n' % (test_names))
variable_fh.write('predicted_labels=%s\n' % (predicted_test_labels.tolist()))

#Note proportions of predicted test labels
num_total = len(test_names)
bincount = np.bincount(predicted_test_labels+np.ones(predicted_test_labels.shape, dtype=np.int32))
if len(bincount) == 1:
    num_neg = num_total
    num_pos = 0
else:
    num_neg, foo, num_pos = bincount
info_fh.write("%i %s-antigen test predictions: %i %s+ (%f%%), %i %s- (%f%%)\n" % (num_total, antigen_type, num_pos, antigen_type, num_pos/float(num_total), num_neg, antigen_type, num_neg/float(num_total)))

#Find predicted confidence scores
t0 = time.time()
predicted_confidence_scores = classifier.decision_function(test_population)
t1 = time.time()
time_logging_fh.write('Calculating %s-antigen confidence scores took %f seconds\n' % (antigen_type, t1-t0))
variable_fh.write('predicted_confidence_scores=%s\n' % (predicted_confidence_scores.tolist()))

#Get information about tile positions
tile_x_values = [[] for phase in range(settings.NUM_PHASES)]
tile_int_values = [[] for phase in range(settings.NUM_PHASES)]

print tile_x_values, tile_int_values
homozygous_locations = np.where(where_homozygous)[0]
well_seq_locations = np.where(where_well_sequenced)[0]
curr_phase = 0
places_to_split = []
for i in range(len(classifier.coef_[0])):
    mid_index = homozygous_locations[i]
    position_in_original_list = well_seq_locations[mid_index]
    phase = position_in_original_list/path_lengths[-1]
    if phase != curr_phase:
        places_to_split.append(i)
        curr_phase = phase
    relative_position_in_phase = position_in_original_list % path_lengths[-1]
    tile_x_values[phase].append(relative_position_in_phase)
    tile_int_values[phase].append(convert_to_tile_int(relative_position_in_phase, path_lengths))
#variable_fh.write('where_homozygous=%s\n' % (where_homozygous.tolist())) --> Currently too long and raises a keep error
#variable_fh.write('where_well_sequenced=%s\n' % (where_well_sequenced.tolist()))
variable_fh.write('tile_x_values=%s\n' % (tile_x_values))
variable_fh.write('tile_int_values=%s\n' % (tile_int_values))
variable_fh.write('places_to_split=%s\n' % (places_to_split))

if settings.PLOT:
    colors = itertools.cycle(['r', 'b', 'g', 'c', 'y', 'm'])
    #Plot coefficients
    plt.figure()
    for thing in classifier.coef_:
        split_by_phase = np.split(thing, places_to_split)
        for phase, (x_values, coefficients, color) in enumerate(zip(tile_x_values, split_by_phase, colors)):
            plt.scatter(x_values, coefficients, color=color, label='Phase %i' % (phase))
    plt.title('%s antigen classifier coefficients' % (antigen_type))
    plt.xlabel('Tile position')
    plt.ylabel('Coefficient value (weight assigned to tile position)')
    plt.legend()
    with out.open('coefficients.png', 'w') as f:
        plt.savefig(f)

time_logging_fh.close()
info_fh.close()
variable_fh.close()
# Commit the output to keep
task_output = out.save_new(create_collection_record=False)
arvados.current_task().set_output(task_output)

###########################################################################################################################################
