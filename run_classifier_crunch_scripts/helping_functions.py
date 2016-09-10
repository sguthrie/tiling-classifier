
from sklearn import svm
import numpy as np
import os
import re
import json
import time

def get_subjects_from_json(subject_handle, subjects):
    """
    Extracts subject information from a json file
    Returns subjects filled with pertinant information
        subjects is a dictionary keyed by huID. Values are a dictionary
    """
    samples = json.load(subject_handle)
    for sample in samples:
        if sample['Sample'] in subjects:
            if 'Bloodtype' in sample:
                sample_name = sample['Sample']
                bloodtype = sample['Bloodtype']
                assert re.match('(^[OAB][+-]$|^AB[+-]$)', bloodtype) != None, "%s is not a recognized Bloodtype (O/A/B/AB +/-)" % (bloodtype)
                assert subjects[sample_name]['a'] == None, "Multiple subjects of the same huID: %s" % (sample_name)
                if 'A' in bloodtype:
                    subjects[sample_name]['a'] = 1
                else:
                    subjects[sample_name]['a'] = -1
                if 'B' in bloodtype:
                    subjects[sample_name]['b'] = 1
                else:
                    subjects[sample_name]['b'] = -1
                if bloodtype.strip('ABO') == '+':
                    subjects[sample_name]['rh'] = 1
                else:
                    subjects[sample_name]['rh'] = -1
    return subjects

def get_population(path_lengths, num_callsets, num_phases, phenotype_file_paths, callset_collection_reader, callset_name_regex, logging_fh, quality):
    """
    Expects path_lengths to be a numpy array with a length of NUM_PATHS + 1.
        First entry should be 0. All entries should be greater than or
        equal to the others (since it indicates the number of tiles contained in
        all paths previous to its index). The last entry indicates the total
        number of tiles.
    Expects phenotype_collection_reader to be a CollectionReader of json files with bloodtype per callset information.
    Expects callset_collection_reader to be a _normalized_ CollectionReader

    Returns population, subjects
        population is a 2D numpy file of shape:
            (number_of_people_in_population, number_of_tiles_to_analyze)
            phases are concatonated together
        subjects is a dictionary keyed by callset name. Values are a dictionary:
            Keys are 'a', 'b', 'rh', and 'index'
            If the subject's bloodtype expresses the A antigen, 'a':1, else 'a':-1
            If the subject's bloodtype expresses the B antigen, 'b':1, else 'b':-1
            If the subject is rh-factor positive, 'rh':1, else 'rh':-1
    """
    print "Getting population before memory blow up"
    callset_names = []
    size = path_lengths[-1]
    population = np.zeros((num_callsets, size*num_phases), dtype=np.int32)
    print "population shape: %s, number of MB: %f" % (population.shape, population.nbytes/1000000.)
    ## Fill population with tile identifiers
    t0 = time.time()
    callset_index= 0
    for s in callset_collection_reader.all_streams():
        if s.name() != '.':
            if re.search(callset_name_regex, s.name()) == None:
                print callset_name_regex, s.name(), "did not match search?"
            callset_names.append((s.name(), re.search(callset_name_regex, s.name()).group(0)))
            phase_index = 0
            for f in s.all_files():
                if f.name().startswith('quality_') and quality:
                    assert f.name() == 'quality_phase%i.npy' % (phase_index), \
                        "Expects 'callset-numpy-cgf-files' to be the output from concat-numpy files. %s is not 'quality_phase%i.npy'" % (f.name(), phase_index)
                    with callset_collection_reader.open(s.name()+'/'+f.name(), 'r') as f_handle:
                        callset = np.load(f_handle)
                        population[callset_index][phase_index*size:(phase_index+1)*size] = callset
                    print s.name(), f.name(), "population number of MB: %f" % (population.nbytes/1000000.)
                    phase_index += 1
                elif not quality and not f.name().startswith('quality_'):
                    assert f.name() == 'phase%i.npy' % (phase_index), \
                        "Expects 'callset-numpy-cgf-files' to be the output from concat-numpy files. %s is not 'phase%i.npy'" % (f.name(), phase_index)
                    with callset_collection_reader.open(s.name()+'/'+f.name(), 'r') as f_handle:
                        callset = np.load(f_handle)
                        population[callset_index][phase_index*size:(phase_index+1)*size] = callset
                    print s.name(), f.name(), "population number of MB: %f" % (population.nbytes/1000000.)
                    phase_index += 1
            callset_index += 1
    t1 = time.time()
    subjects = {callset_name:{'a':None, 'b':None, 'rh':None, 'index':i} for i, (dirname, callset_name) in enumerate(callset_names)}
    ## Fill subjects with phenotypic information if it is available
    #for s in phenotype_collection_reader.all_streams():
    #    for f in s.all_files():
    #        with phenotype_collection_reader.open(s.name()+'/'+f.name(), 'r') as f_handle:
    for path in phenotype_file_paths:
        with open(path, 'r') as f_handle:
            subjects = get_subjects_from_json(f_handle, subjects)
    t2 = time.time()
    print "Subjects done"
    logging_fh.write("Subject generation took %f seconds\n" % (t2-t1))
    logging_fh.write("Population generation takes %f seconds\n" % (t1-t0))
    return population, subjects, callset_names

def reduce_population(filtered_pop, remove_poorly_sequenced, remove_homozygous):
    if remove_poorly_sequenced:
        print "Removing poorly sequenced"
        ### Remove tiles (columns) that contain a -1, indicating it was not well sequenced
        not_seq_indicator = np.amin(filtered_pop, axis=0)
        sequenced_pick = np.greater_equal(not_seq_indicator, np.zeros(not_seq_indicator.shape))
        all_seq_pop = np.zeros((filtered_pop.shape[0], np.sum(sequenced_pick)))
        i = 0
        for column, picked in enumerate(sequenced_pick):
            if picked:
                all_seq_pop[:,i] = filtered_pop[:,column]
                i += 1
        print "population without poorly sequenced shape: %s, number of MB: %f" % (all_seq_pop.shape, all_seq_pop.nbytes/1000000.)
    else:
        sequenced_pick = np.ones(filtered_pop.shape[0], dtype=bool)
        all_seq_pop = filtered_pop
    if remove_homozygous:
        print "Removing homozygous"
        ### Remove tiles (columns) that have the same value over all callsets
        min_indicator = np.amin(all_seq_pop, axis=0)
        max_indicator = np.amax(all_seq_pop, axis=0)
        pick = np.not_equal(min_indicator, max_indicator)
        ret_pop = np.zeros((all_seq_pop.shape[0], np.sum(pick)))
        i = 0
        for column, picked in enumerate(pick):
            if picked:
                ret_pop[:,i] = all_seq_pop[:,column]
                i += 1
        print "population without poorly sequenced and homogenous shape: %s, number of MB: %f" % (ret_pop.shape, ret_pop.nbytes/1000000.)
    else:
        pick = np.ones(all_seq_pop.shape[1], dtype=bool)
        ret_pop = all_seq_pop
    return ret_pop, sequenced_pick, pick

def make_and_label_training_and_test_data(population, callset_names, subjects, keyname, time_logging_fh):
    t0 = time.time()
    #keyname currently represents antigen_type
    training = []
    training_names = []
    training_labels = []
    test = []
    test_names = []
    for i, (dirname, callset_name) in enumerate(callset_names):
        if subjects[callset_name][keyname] == None:
            test.append(i)
            test_names.append((dirname, callset_name))
        else:
            training.append(i)
            training_names.append((dirname, callset_name))
            training_labels.append(subjects[callset_name][keyname])
    numpy_training = np.zeros((len(training),population.shape[1]), dtype=np.int32)
    numpy_test = np.zeros((len(test),population.shape[1]), dtype=np.int32)
    for i, index in enumerate(training):
        numpy_training[i,:] = population[index,:]
    for i, index in enumerate(test):
        numpy_test[i,:] = population[index,:]
    t1 = time.time()
    time_logging_fh.write("Making the labels, training sets, and testing sets for the %s factor took %fs\n" % (keyname, t1-t0))
    training_labels = np.array(training_labels)
    return numpy_training, training_labels, numpy_test, training_names, test_names

def get_training_data(population, callset_names, subjects, keyname, time_logging_fh):
    t0 = time.time()
    #keyname currently represents antigen_type
    training = []
    for i, (dirname, callset_name) in enumerate(callset_names):
        if subjects[callset_name][keyname] != None:
            training.append(i)
    numpy_training = np.zeros((len(training),population.shape[1]), dtype=np.int32)
    for i, index in enumerate(training):
        numpy_training[i,:] = population[index,:]
    t1 = time.time()
    time_logging_fh.write("Making the training sets for antigen %s took %fs\n" % (keyname, t1-t0))
    return numpy_training

def make_training_labels(callset_names, subjects, keyname):
    training_names = []
    training_labels = []
    for i, (dirname, callset_name) in enumerate(callset_names):
        if subjects[callset_name][keyname] != None:
            training_names.append((dirname, callset_name))
            training_labels.append(subjects[callset_name][keyname])
    training_labels = np.array(training_labels)
    return training_names, training_labels
