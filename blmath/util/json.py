'''
If an object defines for_json(), it will be serialized as what that returns.
    class SerializableThing(object):
        def for_json(self):
            return "|SerializableThing|"

'''

from baiji.serialization import json
from baiji.serialization.json import JSONDecoder

class BlmathJSONDecoder(JSONDecoder):
    SUPPORTED_TYPES = {'value': 'blmath.value.Value'}
    def __init__(self):
        super(BlmathJSONDecoder, self).__init__()
        self.register(self.decode_supported_types)

    def decode_supported_types(self, dct):
        for k in dct.keys():
            if k.startswith('__') and k.endswith('__'):
                type_key = k[2:-2]
                if type_key not in self.SUPPORTED_TYPES:
                    raise TypeError("Unsupported Deserialization Type %s" % type_key)
                if len(dct.keys()) > 1:
                    raise TypeError("Deserialization type %s should be the only key in the dict" % k)
                #decode to support object type
                from baiji.serialization.util.importlib import class_from_str
                cls = class_from_str(self.SUPPORTED_TYPES[type_key])
                if hasattr(cls, 'from_json'):
                    return cls.from_json(dct)
                else:
                    return cls(**dct[k])

def dump(obj, f, *args, **kwargs):
    return json.dump(obj, f, *args, **kwargs)

def load(f, *args, **kwargs):
    kwargs.update(decoder=BlmathJSONDecoder())
    return json.load(f, *args, **kwargs)

def dumps(*args, **kwargs):
    return json.dumps(*args, **kwargs)

def loads(*args, **kwargs):
    kwargs.update(decoder=BlmathJSONDecoder())
    return json.loads(*args, **kwargs)
