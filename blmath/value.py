from cached_property import cached_property

class Value(object):
    '''
    Simple value class to encapsulate (value, units) and various conversions
    '''
    def __init__(self, value, units=None):
        '''
        units: A recognized unit, or None for a unitless value like a size or
          ratio.
        '''
        self.value = value
        from blmath import units as unit_conversions
        if units is not None and units not in unit_conversions.all_units:
            raise ValueError("Unknown unit type %s" % units)
        self.units = units

    @property
    def units(self):
        return self._units
    @units.setter
    def units(self, val):
        self._units = val  # pylint: disable=attribute-defined-outside-init

    @property
    def value(self):
        return self._val
    @value.setter
    def value(self, val):
        if val is None:
            raise ValueError("Value can't be None")

        self._val = float(val)  # pylint: disable=attribute-defined-outside-init

    def convert(self, to_units):
        from blmath import units as unit_conversions
        if to_units == self.units:
            return self
        if self.units is None:
            raise ValueError("This value is unitless: can't convert to %s" % to_units)
        return Value(*unit_conversions.convert(self.value, self.units, to_units))

    def convert_to_system_default(self, unit_system):
        from blmath import units
        units_class = units.units_class(self.units)
        to_units = units.default_units(unit_system)[units_class]
        return self.convert(to_units=to_units)

    def round(self, nearest):
        from blmath.numerics import round_to
        return round_to(self, nearest)

    def __float__(self):
        return self.value
    def __int__(self):
        return int(self.value)

    def __getattr__(self, name):
        from blmath import units as unit_conversions
        if name not in unit_conversions.all_units:
            raise AttributeError()
        return self.convert(name)

    def __getitem__(self, key):
        if key == 0:
            return self.value
        elif key == 1:
            return self.units
        else:
            raise KeyError()

    def __mul__(self, other):
        # other must be a scalar. For our present purposes, multpying Value*Value is an error
        if isinstance(other, Value):
            raise ValueError("%s * %s would give you units that we currently don't support" % (self.units, other.units))
        elif hasattr(other, '__iter__'):
            return [self * x for x in other]
        else:
            return Value(self.value*other, self.units)
    def __rmul__(self, other):
        return self.__mul__(other)

    def __add__(self, other):
        if hasattr(other, '__iter__'):
            return [self + x for x in other]
        return Value(self.value + float(self._comparable(other)), self.units)
    def __radd__(self, other):
        return self.__add__(other)

    def __sub__(self, other):
        if hasattr(other, '__iter__'):
            return [self - x for x in other]
        return Value(self.value - float(self._comparable(other)), self.units)
    def __rsub__(self, other):
        if hasattr(other, '__iter__'):
            return [x - self for x in other]
        return Value(float(self._comparable(other)) - self.value, self.units)

    def __div__(self, other):
        if isinstance(other, Value):
            try:
                other_comparable = self._comparable(other)
            except ValueError:
                raise ValueError("%s / %s would give you units that we currently don't support" % (self.units, other.units))
            return Value(self.value / other_comparable.value, None)
        return Value(self.value / other, self.units)
    def __rdiv__(self, other):
        raise ValueError("%s / %s... Wat." % (other, self))

    def __floordiv__(self, other):
        if isinstance(other, Value):
            try:
                other_comparable = self._comparable(other)
            except ValueError:
                raise ValueError("%s // %s would give you units that we currently don't support" % (self.units, other.units))
            return Value(self.value // other_comparable.value, None)
        return Value(self.value // other, self.units)
    def __rfloordiv__(self, other):
        raise ValueError("%s // %s... Wat." % (other, self))

    def __mod__(self, other):
        raise AttributeError("%s %% %s... Wat." % (other, self))
    def __pow__(self, other):
        raise AttributeError("%s ** %s... Wat." % (other, self))

    def __pos__(self):
        return self
    def __neg__(self):
        return Value(-float(self.value), self.units)
    def __abs__(self):
        return Value(abs(self.value), self.units)

    def __str__(self):
        return "%f %s" % (self.value, self.units)
    def __unicode__(self):
        return str(self)
    def __repr__(self):
        return "<Value %f %s>" % (self.value, self.units)

    def __nonzero__(self):
        return bool(self.value)

    def _comparable(self, other):
        '''Make other into something that is sensible to compare to self'''
        if not isinstance(other, Value):
            return other
        return other.convert(self.units)

    def __cmp__(self, other):
        return self.value.__cmp__(float(self._comparable(other)))
    def __eq__(self, other):
        return self.value == float(self._comparable(other))
    def __ne__(self, other):
        return self.value != float(self._comparable(other))
    def __lt__(self, other):
        return self.value < float(self._comparable(other))
    def __gt__(self, other):
        return self.value > float(self._comparable(other))
    def __le__(self, other):
        return self.value <= float(self._comparable(other))
    def __ge__(self, other):
        return self.value >= float(self._comparable(other))

    @cached_property
    def prettified(self):
        from blmath import units as unit_conversions
        return unit_conversions.prettify(self, self.units)

    def for_json(self):
        '''
        Return internal annotated format (__value__)

        '''
        return {'__value__': {'value': self.value, 'units': self.units}}

    @classmethod
    def from_json(cls, data):
        '''
        Accept internal annotated format

        '''
        return cls(**data['__value__'])
