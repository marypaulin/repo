from itertools import combinations
from random import sample

from lib.data_structures.vector import Vector

# Class for storing and performing approximate neighbourhood queries on bitvectors
#  - neighbourhood radius is defined using Hamming distance
# Usage:
# # store sets of bit vectors in the index
# # query for subsets of vectors that are within a maximum hamming distance
# index = SimilarityIndex(distance=1, dimensions=5, tables=5)
# a = Vector('11111')
# b = Vector('01111')
# index.add(b)
# b in index.neighbours(a)
# index.remove(b)
# not b in index.neighbours(a)

class SimilarityIndex:
    # Constructor Arguments:
    #  distance = maximum hamming distance to be considered a neighbour
    #  dimensions = the number of dimensions (elements) in each of the bitvectors
    #  tables = the number of hash tables used to catch nearest neighbours

    # The number of tables should be within [1, K], where K = (dimensions choose distance)
    # As the number of tables increase, the false negative rate of neighbour queries decrease
    # More information on the approximation bounds and design are here:
    # https://en.wikipedia.org/wiki/Locality-sensitive_hashing#Bit_sampling_for_Hamming_distance

    def __init__(self, distance=1, dimensions=1,  tables=1):
        if tables < 1:
            raise Exception("SimilarityIndexError: tables {} must be greater than or equal to {}".format(tables, 1))
        if dimensions < 1:
            raise Exception("SimilarityIndexError: dimensions {} must be greater than or equal to {}".format(dimensions, 1))
        if dimensions < distance:
            raise Exception("SimilarityIndexError: distance {} must be less than or equal to dimension {}".format(distance, dimensions))
        self.dimensions = dimensions
        self.distance = distance
        self.table_count = tables
        self.size = 0
        self.initialized = False # Defer table generation until necessary
        

    # (Private) Creates a bitvector of ones with free_dimensions bits set to 0
    # Any vector would have free_dimensions bits set to 0 so that variatiions in those bits are normalized to 0
    def __normalizer__(self, free_dimensions):
        normalizer = Vector([ int(not j in free_dimensions) for j in range(self.dimensions) ])
        return normalizer
        
    # Creates tuples of (vector, dictionary) pairs where the vector indicates which bits are hashing for the respective table
    def __generate_tables__(self, degrees_of_freedom, tables):
        return tuple((self.__normalizer__(combo), {}) for combo in sample(tuple(combinations(range(self.dimensions), degrees_of_freedom)), tables))

    def initialize(self):
        if self.initialized == False:
            self.tables = self.__generate_tables__(self.distance, self.table_count)
            self.initialized = True

    # Adds a key (vector) into the the index
    def add(self, key):
        if self.initialized == False:
            self.initialize()
        if len(key) != self.dimensions:
            raise Exception("SimilarityIndexError: len(key) {} must be equal to dimension {}".format(len(key), self.dimensions))
        for normalizer, table in self.tables:
            normal_key = key & normalizer
            if not normal_key in table:
                table[normal_key] = set()
            table[normal_key].add(key)
        self.size += 1

    # Removes a key (vector) from the the index
    def remove(self, key):
        if self.initialized == False:
            self.initialize()
        if len(key) != self.dimensions:
            raise Exception("SimilarityIndexError: len(key) {} must be equal to dimension {}".format(len(key), self.dimensions))
        for normalizer, table in self.tables:
            normal_key = key & normalizer
            if normal_key in table and key in table[normal_key]:
                table[normal_key].remove(key)
        self.size -= 1

    # Returns a subset of the vectors that are within a distance from key (vector)
    def neighbours(self, key):
        if self.initialized == False:
            self.initialize()
        if len(key) != self.dimensions:
            raise Exception("SimilarityIndexError: len(key) {} must be equal to dimension {}".format(len(key), self.dimensions))
        return set.union(set(), *(table[key & normalizer] for normalizer, table in self.tables if ((key & normalizer) in table)))

    # Override for x in set operator, returns whether a key (vector) is a member of the index
    def __contains__(self, key):
        if self.initialized == False:
            self.initialize()
        if len(key) != self.dimensions:
            raise Exception("SimilarityIndexError: len(key) {} must be equal to dimension {}".format(len(key), self.dimensions))
        return any(key in table[key & normalizer] for normalizer, table in self.tables if ((key & normalizer) in table))

    # Override for str(x) in set operator, displays the set of vectors in this index
    def __str__(self):
        if self.initialized == False:
            self.initialize()
        return str(set.union(set(), *(bucket for bucket in self.tables[0].values())))

    def __len__(self):
        return self.size
