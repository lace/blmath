import numpy as np
import scipy.sparse as sp
import simplejson
from blmath.value import Value


def decode_json(dct):
    if "__value__" in dct.keys():
        return Value.from_json(dct)
    if '__ndarray__' in dct:
        if 'dtype' in dct:
            dtype = np.dtype(dct['dtype'])
        else:
            dtype = np.float64
        return np.array(dct['__ndarray__'], dtype=dtype)
    if '__scipy.sparse.sparsematrix__' in dct:
        if not all(k in dct for k in ['dtype', 'shape', 'data', 'format', 'row', 'col']):
            return dct
        coo = sp.coo_matrix((dct['data'], (dct['row'], dct['col'])), shape=dct['shape'], dtype=np.dtype(dct['dtype']))
        return coo.asformat(dct['format'])
    return dct


def dump(obj, f, *args, **kwargs):
    return simplejson.dump(obj, f, *args, for_json=True, **kwargs)


def load(f, *args, **kwargs):
    return simplejson.load(f, *args, object_hook=decode_json, **kwargs)


def dumps(*args, **kwargs):
    return simplejson.dumps(*args, for_json=True, **kwargs)


def loads(*args, **kwargs):
    return simplejson.loads(*args, object_hook=decode_json, **kwargs)
