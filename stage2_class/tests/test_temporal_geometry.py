import unittest
import numpy as np

from stage2_class.src.eval.temporal_geometry import (
    annotation_intervals, attention_intervals, clip_support, frame_to_seconds,
    intersection_length, interval_length, merge_intervals, smooth_attention,
)
from inference.inference_any_video import extract_intervals
from stage3_dispatch.src.alert_router import generate_alert


def metadata(fps=30.0, total=1000):
    return dict(src_fps=fps, keep_every_n=max(1,round(fps/15)), src_frame_count=total, clip_len=16, stride=8)


class TemporalGeometryTests(unittest.TestCase):
    def test_integer_decimation_not_fps_ratio(self):
        for fps,n in [(15,1),(24,2),(25,2),(29.97,2),(30,2),(60,4)]:
            m=metadata(fps)
            self.assertEqual(clip_support(8,m),(8*n,24*n))
            self.assertAlmostEqual(frame_to_seconds(8*n,m),8*n/fps)

    def test_eof_cell_clipped_but_no_sample_padding(self):
        self.assertEqual(clip_support(0,metadata(total=31)),(0,31))
        with self.assertRaises(ValueError): clip_support(0,metadata(total=30))

    def test_invalid_metadata_fails(self):
        for field,value in [('src_fps',float('nan')),('keep_every_n',3),('src_frame_count',0),('clip_len',1.5)]:
            m=metadata(); m[field]=value
            with self.assertRaises(ValueError): clip_support(0,m)

    def test_annotation_inclusive_to_half_open(self):
        self.assertEqual(annotation_intervals([(1,1),(9,10),(-1,-1)],10),[(0,1),(8,10)])
        for pair in [(0,2),(-1,2),(2,11),(4,3)]:
            with self.assertRaises(ValueError): annotation_intervals([pair],10)

    def test_union_and_intersection_do_not_double_count(self):
        self.assertEqual(merge_intervals([(0,32),(16,48),(80,96)]),[(0,48),(80,96)])
        self.assertEqual(interval_length([(0,32),(16,48)]),48)
        self.assertEqual(intersection_length([(0,32),(16,48)],[(24,40),(80,90)]),16)

    def test_sparse_gap_not_filled_or_smoothed_across(self):
        starts=[0,80,160]
        scores=[1,0,1]
        np.testing.assert_array_equal(smooth_attention(scores,starts,metadata()),scores)
        self.assertEqual(attention_intervals(scores,starts,metadata()),[(0,32),(320,352)])

    def test_constant_empty_nonfinite_and_duplicates(self):
        self.assertEqual(attention_intervals([1,1],[0,8],metadata()),[])
        self.assertEqual(attention_intervals([],[],metadata()),[])
        with self.assertRaises(ValueError): smooth_attention([float('nan')],[0],metadata())
        with self.assertRaises(ValueError): smooth_attention([1,2],[0,0],metadata())

    def test_renderer_and_router_boundary_conversion(self):
        intervals=extract_intervals([1,0],[0,80],metadata(25))
        self.assertEqual(intervals,[{'start_frame':0,'end_frame':31}])
        alert=generate_alert({'binary_anomaly_detected':True,'fps':25,'anomaly_intervals':intervals})
        self.assertEqual(alert['anomaly_window']['end_sec'],1.28)
        self.assertEqual(alert['priority'],'P3-REVIEW')
        self.assertEqual(generate_alert({'binary_anomaly_detected':False})['dispatch'],[])


if __name__=='__main__': unittest.main()
