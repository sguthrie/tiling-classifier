#!/usr/bin/env python
"""
Purpose: Load numpy files and subjects, do memory intensive work, save group

Does not use parallelization due to high memory use

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
    'num-retries' : (integer) Specifies the number of retries we should use to access collections
    'antigen-type' : (string) Specifies the antigen to use

Outputs:
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
"""
import helping_functions as fns
from sklearn import preprocessing
import arvados      # Import the Arvados sdk module
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
#Load Population
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

#Reduce population by removing poorly sequenced and/or removing homozygous
t0 = time.time()
print "Reducing population"
#Currently population has a large number of features with a relatively low number of samples.
population, where_well_sequenced, where_homozygous  = fns.reduce_population(population, settings.REMOVE_POORLY_SEQ, settings.REMOVE_HOMOZYGOUS)
t1 = time.time()
time_logging_fh.write('Removing the poorly sequenced population tiles took %f seconds\n' % (t1-t0))

#Separate the population based on training and test data sets. Label the data
print "Making training data, training labels, and test data"
population, training_labels, test_population, training_names, test_names = fns.make_and_label_training_and_test_data(population, callset_names, subjects, antigen_type, time_logging_fh)
print "Training data shape: %s, number of MB: %f" % (population.shape, population.nbytes/1000000.)
print "Test data shape: %s, number of MB: %f" % (test_population.shape, test_population.nbytes/1000000.)

#Cast the two datasets into floats to prepare for scalar
print "Casting to float64 for scalar"
population = population.astype(np.float_)
test_population = test_population.astype(np.float_)
print "Float Training data shape: %s, number of MB: %f" % (population.shape, population.nbytes/1000000.)
print "Float Test data shape: %s, number of MB: %f" % (test_population.shape, test_population.nbytes/1000000.)

#Run a standard scalar, fitting to the training data, and transforming both datasets
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

with out.open('%s_training.npy' % (antigen_type), 'w') as f:
    np.save(f, population)

with out.open('%s_test.npy' % (antigen_type), 'w') as f:
    np.save(f, test_population)

with out.open('%s_training_labels.npy' % (antigen_type), 'w') as f:
    np.save(f, training_labels)

with out.open('%s_where_well_sequenced.npy' % (antigen_type), 'w') as f:
    np.save(f, where_well_sequenced)

with out.open('%s_where_homozygous.npy' % (antigen_type), 'w') as f:
    np.save(f, where_homozygous)

with out.open('python_variables.py', 'w') as f:
    f.write('training_names=%s\n' % (training_names))
    f.write('test_names=%s\n' % (test_names))

time_logging_fh.close()

# Commit the output to keep
task_output = out.save_new(create_collection_record=False)
arvados.current_task().set_output(task_output)

###########################################################################################################################################
