

def input_float(prompt, allow_empty=False):
    '''
    Read a float from the console, showing the given prompt.

    prompt: The prompt message.
    allow_empty: When `True`, allows an empty input. The default is to repeat
      until a valid float is entered.
    '''
    from plumbum.cli import terminal
    if allow_empty:
        return terminal.prompt(prompt, type=float, default=None)
    else:
        return terminal.prompt(prompt, type=float)

def input_value(label, units, allow_empty=False):
    '''
    Read a value from the console, and return an instance of `Value`. The
    units are specified by the caller, but displayed to the user.

    label: A label for the value (included in the prompt)
    units: The units (included in the prompt)
    allow_empty: When `True`, allows an empty input. The default is to repeat
      until a valid float is entered.
    '''
    from blmath.value import Value

    value = input_float(
        prompt='{} ({}): '.format(label, units),
        allow_empty=allow_empty
    )

    if value is None:
        return None

    return Value(value, units)
