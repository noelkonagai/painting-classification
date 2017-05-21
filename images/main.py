from create_features import *

path = './Both/305776.jpg'

img_hsv, h, w, px = pre_process(path)
hsv_df, hue, sat, val = make_dataframe(img_hsv, h, w)
hue_bins = make_bins(hsv_df, 20)
image, image_original, denoised, markers, gradient, labels = water_segmentation(path, hue, 3, 5, 10, 2)
segment_df = make_segment_df(labels)
f1, f2 = f1_2(hsv_df)
f3 = f3(hue_bins, 0.1)
f4_23 = f4_23(hue_bins, px)
f24_35 = f24_35(segment_df, labels, h, w)
f36_50 = f36_50(segment_df, labels, hue, sat, val, h, w)