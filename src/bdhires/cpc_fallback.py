"""Read-only CPC substitutions for two verified source gaps during inference."""
from collections.abc import Mapping
import numpy as np

KNOWN_CPC_FALLBACKS = {'2004-09-10': '2004-09-09', '2007-02-26': '2007-02-25'}


class CPCFallbackConditions:
    """Replace only CPC channels in a copied daily array; never write the store."""
    def __init__(self, array, donors, channels):
        self.array, self.donors, self.channels = array, donors, channels

    def __getattr__(self, name):
        return getattr(self.array, name)

    def __getitem__(self, key):
        day = key[0] if isinstance(key, tuple) else key
        if not isinstance(day, (int, np.integer)):
            raise ValueError('CPC fallback view requires a single daily index')
        index = int(day)
        if index < 0: index += self.array.shape[0]
        if index not in self.donors:
            return self.array[key]
        values = np.asarray(self.array[index]).copy()
        donor = np.asarray(self.array[self.donors[index]])
        values[self.channels] = donor[self.channels]
        return values[key[1:]] if isinstance(key, tuple) else values


class CPCFallbackStore(Mapping):
    """Array mapping used by both model conditioning and exported CPC inputs."""
    def __init__(self, store, requested_dates=None):
        self.store = store
        self.attrs = store.attrs
        names = list(self.attrs['cond_channels'])
        channels = [names.index(name) for name in ('cpc_precip', 'cpc_valid')]
        times = np.asarray(store['time'][:], dtype='datetime64[ns]').astype('datetime64[D]').astype(str)
        indices = {day: index for index, day in enumerate(times)}
        if len(indices) != len(times): raise ValueError('CPC fallback requires unique daily dates')
        valid = np.asarray(store['valid'][:]) > .5
        donors, self.fallbacks = {}, []
        requested = set(KNOWN_CPC_FALLBACKS if requested_dates is None else requested_dates)
        if requested - set(KNOWN_CPC_FALLBACKS): raise ValueError('unknown CPC fallback date')
        for day in sorted(requested):
            if day not in indices: continue
            values = np.asarray(store['cond'][indices[day]])[channels]
            if np.any(values[1][valid] > 0): continue
            if not np.all(values[:, valid] == 0):
                raise ValueError('CPC gap does not use the native missing placeholder on '+day)
            donor_day = KNOWN_CPC_FALLBACKS[day]
            if donor_day not in indices: raise ValueError('CPC donor date missing: '+donor_day)
            donor = np.asarray(store['cond'][indices[donor_day]])[channels]
            if (not np.isfinite(donor[:, valid]).all() or not np.any(donor[1][valid] > 0)
                    or np.any((donor[0][valid] < 0) | (donor[0][valid] > 1000))
                    or np.any((donor[1][valid] < 0) | (donor[1][valid] > 1))):
                raise ValueError('CPC donor fields unavailable or invalid: '+donor_day)
            donors[indices[day]] = indices[donor_day]
            self.fallbacks.append({'background_date': day, 'cpc_source_date': donor_day})
        self.conditions = CPCFallbackConditions(store['cond'], donors, channels)

    def __getitem__(self, key):
        return self.conditions if key == 'cond' else self.store[key]

    def __iter__(self): return iter(self.store)
    def __len__(self): return len(self.store)
