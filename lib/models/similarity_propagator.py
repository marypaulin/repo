from time import time
from math import floor

from lib.data_structures.result import Result
from lib.data_structures.interval import Interval

class SimilarityPropagator:
    def __init__(self, index, dataset, lamb, cooldown=0):
        self.index = index
        self.dataset = dataset
        self.lamb = lamb
        self.cooldown = cooldown
        self.last_propagation = 0
        # self.dataset = dataset
        # self.lamb = lamb
        # self.similarity_tolerance = 1
        # self.unresolved = SimilarityIndex(self.dataset.height, self.similarity_tolerance, self.dataset.height)

    def track(self, key):
        self.index.add(key)
    
    def untrack(self, key):
        self.index.remove(key)
    
    def tracking(self, key):
        return key in self.index

    def propagate(self, key, value, results):
        if time() > self.last_propagation + self.cooldown:
            self.last_propagation = time()
        else:
            return {}
        
        bound_level = 'high'

        _total, zeros, ones, minority, _majority = self.dataset.label_distribution(key)
        reference_lowerbound = minority / self.dataset.sample_size + self.lamb
        reference_upperbound = min(zeros, ones) / self.dataset.sample_size + self.lamb * key.count()
        disparity = abs(zeros - ones)

        updates = {}

        if bound_level == 'low':
            base_distance = 0.5 * self.index.distance * self.dataset.maximum_group_size
            bounding_interval = Interval(reference_lowerbound - base_distance, reference_upperbound + base_distance)
            
        for neighbour_key in self.index.neighbours(key):
            

            if bound_level == 'medium':
                total, zeros, ones, _minority, _majority = self.dataset.label_distribution(key ^ neighbour_key)
                lowerbound_distance = min(zeros, ones) / self.dataset.sample_size
                # upperbound_distance = total / self.dataset.sample_size
                upperbound_distance =  ( min(disparity, total) + floor(0.5 * max(0, total - disparity)) ) / self.dataset.sample_size

                bounding_interval = Interval(reference_lowerbound - lowerbound_distance, reference_upperbound + upperbound_distance)
            elif bound_level == 'high':

                reference_minus_neighbour = key & ~neighbour_key
                neighbour_minus_reference = neighbour_key & ~key
                # if reference_minus_neighbour in results:
                #     drop = results[reference_minus_neighbour].optimum.upperbound
                # else:
                _total, zeros, ones, minority, _majority = self.dataset.label_distribution(reference_minus_neighbour)
                # This measures the decrease in misclassification when the capture set excludes reference_minus_neighbour
                # Maximize drop in lowerbound to ensure we don't overbound
                lowerbound_drop = min(zeros, ones) / self.dataset.sample_size
                # Minimize the drop in upperbound to ensure we don't overbound
                # upperbound_drop = minority / self.dataset.sample_size
                
                # if neighbour_minus_reference in results:
                #     rise = results[neighbour_minus_reference].optimum.lowerbound
                # else:
                total, zeros, ones, minority, _majority = self.dataset.label_distribution(neighbour_minus_reference)
                # This measure the minimum incease in misclassification when the capture set includes neighbour_minus_reference
                # Minimize rise in lowerbound to ensure we don't overbound
                lowerbound_rise = minority / self.dataset.sample_size
                # Maximize rise in upperbound to ensure we don't overbound
                # upperbound_rise = min(zeros, ones) / self.dataset.sample_size
                # upperbound_distance = total / self.dataset.sample_size
                lowerbound_distance = lowerbound_drop - lowerbound_rise
                upperbound_distance =  ( min(disparity, total) + floor(0.5 * max(0, total - disparity)) ) / self.dataset.sample_size

                localized_lowerbound = reference_lowerbound - lowerbound_distance
                localized_upperbound = reference_upperbound + upperbound_distance # - upperbound_drop + upperbound_rise
                bounding_interval = Interval(localized_lowerbound, localized_upperbound)

            result = results[neighbour_key]
            interval = result.optimum
            if not bounding_interval.overlap(interval):
                raise Exception("SimilarityPropagatorError: Overbounding detected applying {} to {}".format(bounding_interval.value(),interval.value()))
            
            if bounding_interval.lowerbound > interval.lowerbound or bounding_interval.upperbound < interval.upperbound:
                new_interval = bounding_interval.intersection(interval)
                precision_gain = max(bounding_interval.lowerbound - interval.lowerbound, 0) + max(interval.upperbound - bounding_interval.upperbound, 0)
                print("Precision gained {}".format((max(bounding_interval.lowerbound - interval.lowerbound, 0), max(interval.upperbound - bounding_interval.upperbound, 0))))
                new_neighbour = Result(optimizer=result.optimizer, optimum=new_interval)
                updates[neighbour_key] = new_neighbour

        return updates



    # def __similarity_propagation__(self, key, result):
    #     if result.optimizer != None: # The subproblem has a solved optimal subtree
    #         if key in self.unresolved: # Resolved problems don't need further bounding
    #             self.unresolved.remove(key)

    #         _total, zeros, ones, minority, _majority = self.dataset.count(key)
    #         # lowerbound_base = result.optimum.lowerbound + self.lamb * (self.similarity_tolerance-self.__count_leaves__(key))
    #         lowerbound_base = minority / self.dataset.sample_size + self.lamb * (self.similarity_tolerance-self.__count_leaves__(key))

    #         for neighbour_key in self.unresolved.neighbours(key):

    #             self_difference = key & vect.negate(neighbour_key)
    #             neighbour_difference = neighbour_key & vect.negate(key)
    #             if False and self.local_table.get(self_difference) != None:
    #                 drop = self.local_table.get(self_difference).optimum.upperbound
    #             else:
    #                 _total, zeros, ones, minority, _majority = self.dataset.count(self_difference)
    #                 drop = min(zeros, ones) / self.dataset.sample_size
    #             if False and self.local_table.get(neighbour_difference) != None:
    #                 rise = self.local_table.get(neighbour_difference).optimum.lowerbound
    #             else:
    #                 _total, zeros, ones, minority, _majority = self.dataset.count(neighbour_difference)
    #                 rise = minority / self.dataset.sample_size
    #             lowerbound = lowerbound_base - drop + rise

    #             interval = self.local_table.get(neighbour_key).optimum
    #             if lowerbound > interval.upperbound:
    #                 print("Similarity Bound {} appears to be overbounding".format(lowerbound))
    #             elif lowerbound > interval.lowerbound:
    #                 print("Similarity Narrowed {} to {}".format((interval.lowerbound, interval.upperbound), (lowerbound, interval.upperbound)))
    #                 new_neighbour = Result(optimizer=None, optimum=Interval(lowerbound, interval.upperbound))
    #                 self.local_table[neighbour_key] = new_neighbour
    #                 for i in range(self.degree):
    #                     (queue, buffer) = self.outbound_queues[i]
    #                     buffer.append((neighbour_key, new_neighbour))

    #                 self.__similarity_propagation__(neighbour_key, new_neighbour)
    #             else:
    #                 print("Similarity Bound too weak")
    #     else:
    #         # Store all unresolved problem keys in the similarity index
    #         if not key in self.unresolved:
    #             self.unresolved.add(key)

    # def __count_leaves__(self, key, table):
    #     result = table.get(table)
    #     if result.optimizer == None: # Return an upperbound on leaf count instead
    #         return capture.count()
    #     (split, prediction) = self.local_table.get(capture).optimizer
    #     if split == None: # This node is a leaf
    #         return 1
    #     else: # This node splits into subtrees
    #         (left_capture, right_capture) = self.dataset.split(capture, split)
    #         return self.__count_leaves__(left_capture) + self.__count_leaves__(right_capture)
