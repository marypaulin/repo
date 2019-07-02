from gmpy2 import mpz, popcount

def vectorize(bits):
    bitstring = '1' + ''.join([str(i) for i in bits])
    return mpz(bitstring , 2)

def devectorize(vector):
    # remove the leading one
    return list(map(int, vector.digits(2)[1:]))

def repeat(element, length):
    return vectorize([element] * length)

def ones(length):
    return repeat(1, length)

def zeros(length):
    return repeat(0, length)

def length(v):
    return len(v) - 1

def negate(v):
    length = v.bit_length()
    return (~v).bit_set(length - 1)

def xor(v1, v2):
    length = v1.bit_length()
    return (v1 ^ v2).bit_set(length - 1)

def count(v):
    # skip the leading 1
    return popcount(v) - 1

def read(v, indicator):
    # Offset by one to skip leading 1
    return 1 if v.bit_test(len(v) - 2 - indicator) else 0

def write(v, indicator, value=None):
    # Offset by one to skip leading 1
    if value == 0:
        return v.bit_clear(len(v) - 2 - indicator)
    elif value == 1:
        return v.bit_set(len(v) - 2 - indicator)
    elif value == None:
        return v.bit_flip(len(v) - 2 - indicator)

def test(v, indicator):
    # Offset by one to skip leading 1
    return v.bit_test(len(v) - 2 - indicator)

def __str__(v):
    return v.digits(2)[-1:0:-1]
