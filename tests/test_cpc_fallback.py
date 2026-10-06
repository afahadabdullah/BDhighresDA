"""A CPC donor changes only the two CPC channels, never dates or observations."""
from pathlib import Path
import sys
import unittest
import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from bdhires.cpc_fallback import CPCFallbackStore


class Store(dict):
    attrs={'cond_channels':['era5_tcwv','cpc_valid','era5_cape','cpc_precip']}


def fixture():
    times=np.asarray(['2004-09-09','2004-09-10','2007-02-25','2007-02-26'],dtype='datetime64[ns]')
    values=np.ones((4,4,2,2),np.float32)
    values[0,3]=17.;values[2,3]=9.
    values[1,0]=25.;values[1,2]=80.
    values[3,0]=27.;values[3,2]=99.
    values[[1,3],1]=0;values[[1,3],3]=0
    return Store(time=times,cond=values,valid=np.ones((2,2)),target=np.arange(16).reshape(4,2,2))


class CPCFallbackTests(unittest.TestCase):
    def test_both_donors_replace_only_cpc_channels_without_store_mutation(self):
        source=fixture();before=source['cond'].copy()
        view=CPCFallbackStore(source)
        first=view['cond'][1];second=view['cond'][3]
        np.testing.assert_array_equal(first[[1,3]],source['cond'][0][[1,3]])
        np.testing.assert_array_equal(second[[1,3]],source['cond'][2][[1,3]])
        np.testing.assert_array_equal(first[[0,2]],before[1][[0,2]])
        np.testing.assert_array_equal(second[[0,2]],before[3][[0,2]])
        first[:]=999  # the returned replacement is a copy
        np.testing.assert_array_equal(source['cond'],before)
        self.assertIs(view['time'],source['time'])
        self.assertIs(view['target'],source['target'])
        self.assertEqual(view.fallbacks,[
            {'background_date':'2004-09-10','cpc_source_date':'2004-09-09'},
            {'background_date':'2007-02-26','cpc_source_date':'2007-02-25'}])

    def test_cropped_and_exported_cpc_reads_match_model_substitution(self):
        source=fixture();view=CPCFallbackStore(source)
        np.testing.assert_array_equal(view['cond'][1,3,:1,:1],np.full((1,1),17.))
        np.testing.assert_array_equal(view['cond'][-1][3],np.full((2,2),9.))
        np.testing.assert_array_equal(view['cond'][0],source['cond'][0])
        with self.assertRaises(ValueError): view['cond'][:]
        with self.assertRaises(TypeError): view['cond'][1]=np.zeros((4,2,2))

    def test_never_borrows_on_a_date_with_real_cpc_coverage(self):
        source=fixture();source['cond'][1,1]=1;source['cond'][1,3]=42
        view=CPCFallbackStore(source)
        self.assertEqual(len(view.fallbacks),1)
        np.testing.assert_array_equal(view['cond'][1],source['cond'][1])

    def test_invalid_donor_or_non_native_gap_is_rejected(self):
        for field,value in [(1,0),(1,2),(3,-1),(3,np.nan),(3,1001)]:
            with self.subTest(field=field,value=value):
                source=fixture();source['cond'][0,field]=value
                with self.assertRaisesRegex(ValueError,'donor fields'): CPCFallbackStore(source)
        source=fixture();source['cond'][1,3]=5
        with self.assertRaisesRegex(ValueError,'native missing placeholder'): CPCFallbackStore(source)

    def test_missing_or_duplicate_donor_date_and_unknown_gap_are_rejected(self):
        source=fixture();source['time'][0]=np.datetime64('2004-09-08')
        with self.assertRaisesRegex(ValueError,'donor date missing'): CPCFallbackStore(source)
        source=fixture();source['time'][0]=source['time'][1]
        with self.assertRaisesRegex(ValueError,'unique daily dates'): CPCFallbackStore(source)
        with self.assertRaisesRegex(ValueError,'unknown CPC'): CPCFallbackStore(fixture(),['2001-01-01'])


if __name__=='__main__': unittest.main()
