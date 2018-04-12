def get_sequences(pkg):
    return pkg['data']

def get_calls(seq):
    return seq['sequence']

def get_packages(doc):
    if "packages" in doc:
        for pkg in doc['packages']:
            yield pkg
    else:
        yield doc

