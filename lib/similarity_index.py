from itertools import combinations
from random import sample

import lib.vector as vect
class SimilarityIndex:
    def __init__(self, dimensions, distance, tables):
        self.dimensions = dimensions
        self.distance = distance
        self.tables = self.__create_tables__(distance, tables)

    def __normalizer__(self, free_dimensions):
        normalizer = [1] * self.dimensions
        for j in free_dimensions:
            normalizer[j] = 0
        return vect.vectorize(normalizer)
        
    def __create_tables__(self, degrees_of_freedom, tables):
        return tuple((self.__normalizer__(combo), {}) for combo in sample(tuple(combinations(range(self.dimensions), degrees_of_freedom)), tables))

    def put(self, key):
        for normalizer, table in self.tables:
            normal_key = key & normalizer
            if not normal_key in table:
                table[normal_key] = set()
            table[normal_key].add(key)

    def remove(self, key):
        for normalizer, table in self.tables:
            normal_key = key & normalizer
            if normal_key in table and key in table[normal_key]:
                table[normal_key].remove(key)

    def neighbours(self, key):
        neighbours = set.union(set(), *(table[key & normalizer] for normalizer, table in self.tables if ((key & normalizer) in table)))
        if key in neighbours:
            neighbours.remove(key)
        return neighbours

    def __str__(self):
        return str(self.tables)
