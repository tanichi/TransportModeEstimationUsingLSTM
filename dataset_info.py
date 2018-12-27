dictionary = {0:'stand', 1:'walk', 2:'train', 3:'cycle', 4:'car'}

def label2name(label):
    return dictionary[int(label)]

def labels():
    return list(dictionary.keys())

def names():
    return list(dictionary.values())

def n_class():
    return len(dictionary)
