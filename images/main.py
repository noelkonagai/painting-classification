from create_features import *
import csv

path = './Both/305776.jpg'


def main(path):
	img_hsv, h, w, px = pre_process(path)
	hsv_df, hue, sat, val = make_dataframe(img_hsv, h, w)
	hue_bins = make_bins(hsv_df, 20)
	image, image_original, denoised, markers, gradient, labels = water_segmentation(path, hue, 3, 5, 10, 2)
	segment_df = make_segment_df(labels)
	f1, f2 = get1_2(hsv_df)
	f3 = get3(hue_bins, 0.1)
	f4_23 = get4_23(hue_bins, px)
	f24_f26, f27_f29, f30_f32, f33_f35 = get24_35(segment_df, labels, h, w)
	f36_40, f41_45, f46_50 = get36_50(segment_df, labels, hue, sat, val, h, w)

	features = []

	features.append(f1)
	features.append(f2)
	features.append(f3)

	features = features + f4_23 + f24_f26 + f27_f29 + f30_f32 + f33_f35 + f36_40 + f41_45 + f46_50

	return features

fieldnames = []

for i in range(50):
	temp = 'f' + str(i+1)
	fieldnames.append(temp)

with open('data.csv', 'a') as csvfile:
	writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
	features = main(path)

	string = ''
	
	for i in range(len(features)):
		string += str(features[i]) + ','

	string += "/n"

	print(string)