"""Frozen P0 interface for future P1 combination, no automatic grid search."""
import numpy as np

from p0_core import decode_capacity, soft_bias, validate_scores


def apply_frozen(scores, specification):
    values = validate_scores(scores)
    if specification.get('schema') != 'aictrack1.p0_frozen_decoder.v1':
        raise ValueError('unknown frozen decoder schema')
    if specification['type'] == 'raw':
        return values.argmax(1), {'type': 'raw'}
    if specification['type'] == 'soft':
        if specification.get('iterations') != 60:
            raise ValueError('unregistered iterations')
        temperature, strength = float(specification['temperature']), float(specification['strength'])
        allowed = {(t, s) for t in (.35, .5, .75) for s in (.5, .75, 1.)} | {(.2, .75), (.25, .75)}
        if (temperature, strength) not in allowed:
            raise ValueError('unregistered soft point')
        bias, report = soft_bias(values, temperature, 60)
        return (values + strength * bias).argmax(1), report
    if specification['type'] == 'capacity':
        ratios = tuple(specification['ratios'])
        if ratios not in ((.8, 1.2), (.9, 1.1)) or specification.get('top_ks') != [5, 20]:
            raise ValueError('unregistered capacity point')
        return decode_capacity(values, ratios, (5, 20), min(float(specification.get('time_limit', 3600)), 3600))
    raise ValueError('unsupported frozen decoder')


P7_DEFAULT = dict(schema='aictrack1.p0_frozen_decoder.v1', type='soft',
                  temperature=.5, strength=.75, iterations=60,
                  status='existing P7 default unchanged; new points not frozen')
