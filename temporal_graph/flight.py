import numpy as np

def convert_str2int(in_str: str):
    out = []
    for element in in_str:
        if element.isnumeric():
            out.append(element)
        elif element == "!":
            out.append(-1)
        else:
            out.append(ord(element.upper()) - 44 + 9)
    out = np.array(out, dtype=np.float32)
    return out

def padding_callsign(i):

    i = str(i)

    if len(i) == 0:
        i = "!!!!!!!!"
    while len(i) < 8:
        i += "!"
    
    return i


def padding_typecode(i):

    i = str(i)
    
    if len(i) == 0 or i == "nan":
        i = "!!!!!!!!"
    while len(i) < 8:
        i += "!"
    if len(i) > 8:
        i = "!!!!!!!!"
    
    return i