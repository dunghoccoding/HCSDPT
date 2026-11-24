import sqlite3
import unittest

import numpy as np

from portrait_search.database import connect, pack, unpack
from portrait_search.features import FeatureError, color
from portrait_search.search import rank, unit


class FeatureContracts(unittest.TestCase):
    def test_vector_round_trip_and_bad_values(self):
        values = np.array([.1, 2., -3.])
        np.testing.assert_array_equal(unpack(pack(values),3),values.astype('<f4').astype(float))
        with self.assertRaises(ValueError):
            pack([float('nan')])
        with self.assertRaises(ValueError):
            unpack(pack(values),4)

    def test_forehead_rejection_and_histogram_blocks(self):
        rgb = np.full((30,30,3),[200,130,100],dtype=np.uint8)
        masks = {name:np.zeros((30,30),bool) for name in ['cheek_left','cheek_right','forehead']}
        masks['cheek_left'][10:20,0:10] = True
        masks['cheek_right'][10:20,20:30] = True
        masks['forehead'][0:5,10:20] = True
        rgb[masks['forehead']] = [0,0,0]
        raw,matching,metadata = color(rgb,masks)
        self.assertFalse(metadata['forehead_color_used'])
        self.assertAlmostEqual(raw[6:38].sum(),1)
        self.assertAlmostEqual(raw[38:].sum(),1)
        for block in [matching[:6],matching[6:38],matching[38:]]:
            self.assertAlmostEqual(np.linalg.norm(block),np.sqrt(1/3))
        rgb[masks['forehead']] = [200,130,100]
        self.assertTrue(color(rgb,masks)[2]['forehead_color_used'])

    def test_cosine_tie_break_and_zero_error(self):
        from portrait_search.features import DIMS
        values = [np.arange(dim,dtype=float)+1 for dim in DIMS]
        rows = [dict(image_id=i,path=str(i),**{f'f{j}':pack(v) for j,v in enumerate(values,1)},
                     f4_match=pack(values[3])) for i in [4,2]]
        result = rank(values,rows)
        self.assertEqual([r['image_id'] for r in result],[2,4])
        self.assertAlmostEqual(result[0]['score'],1)
        with self.assertRaises(FeatureError):
            unit(np.zeros(7))

    def test_db_foreign_key_and_null_vector_rejection(self):
        db = connect(':memory:')
        with self.assertRaises(sqlite3.IntegrityError):
            db.execute("INSERT INTO normalization_params VALUES ('n','missing','raw','ready','[]','[]',CURRENT_TIMESTAMP)")
        db.rollback()
        db.execute("INSERT INTO images VALUES (1,'a','hash','pixels','other',178,218,NULL,'ok',NULL)")
        db.execute("INSERT INTO feature_sets VALUES ('f','v','{}',CURRENT_TIMESTAMP)")
        db.execute("INSERT INTO normalization_params VALUES ('n','f','raw','ready','[]','[]',CURRENT_TIMESTAMP)")
        with self.assertRaises(sqlite3.IntegrityError):
            db.execute("INSERT INTO image_features(image_id,feature_set_id,normalization_id,status,metadata_json) VALUES (1,'f','n','ok','{}')")
        db.close()


if __name__ == '__main__':
    unittest.main()
