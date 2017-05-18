from baiji.serialization import json
from baiji.serialization.json import JSONDecoder

class BlmathJSONDecoder(JSONDecoder):
    def __init__(self):
        super(BlmathJSONDecoder, self).__init__()
        self.register(self.decode_value)

    def decode_value(self, dct):
        from blmath.value import Value
        if "__value__" in dct.keys():
            return Value.from_json(dct)

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
