from __future__ import absolute_import
from .ilidsvidsequence import iLIDSVIDSEQUENCE
from .prid2011sequence import PRID2011SEQUENCE
from .mars import Mars
from .duke import DukeMTMCVidReID


def get_sequence(name, *args, **kwargs):
    __factory = {
        'ilidsvidsequence': iLIDSVIDSEQUENCE,
        'prid2011sequence': PRID2011SEQUENCE,
        'mars': Mars,
        'duke': DukeMTMCVidReID,
    }

    if name not in __factory:
        raise KeyError("Unknown dataset", name)
    return __factory[name](*args, **kwargs)