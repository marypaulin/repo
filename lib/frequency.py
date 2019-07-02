import lib.vector as vect

# Module for adding the summarized training data frequencies
# If the compression rate is high, this should allow us to do faster computations than the bit-vector operations
# since the bit vectors could potentially be very long and sparse

def count(z, indicator=None, label=None):
    indicator = vect.ones(len(z)) if indicator == None else indicator

    # Select the equivalent sets which are indicated, and sum the frequencies matching the label
    return sum(sum(z[i]) if label == None else z[i][label] for i in range(len(z)) if vect.test(indicator, i))


def minority_count(z, indicator=None):
    indicator = vect.ones(len(z)) if indicator == None else indicator

    # Select the equivalent sets which are indicated, and sum the frequencies matching the label
    return sum(min(z[i]) for i in range(len(z)) if vect.test(indicator, i))
