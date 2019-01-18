#!/usr/bin/python
# -*- coding: ascii -*-
import os, sys
import menpo.io as mio
import matplotlib.pyplot as plt
from glob import glob
from menpodetect import load_dlib_frontal_face_detector
from menpofit.aam import LucasKanadeAAMFitter, WibergInverseCompositional
from menpofit.aam import PatchAAM
from menpo.feature import fast_dsift
from menpo.visualize import print_progress
from menpo.landmark import labeller, face_ibug_68_to_face_ibug_68_trimesh
from pathlib import Path


unitColor = '\033[5;36m\033[5;47m'
endColor = '\033[0;0m\033[0;0m'
count = 45
for i in range(count):
    incre = int(50.0 / count * i)
    sys.stdout.write('\r' + '|%s%s%s%s| %d%%' % (unitColor, '\033[7m' + ' '*incre + ' \033[27m', endColor, ' '*(50-incre), 2*incre),encoding='utf-8') if i != count — 1 else sys.stdout.write('\r' + '|%s%s%s| %d%%' % (unitColor, '\033[7m' + ' '*20 + 'COMPLETE!' + ' '*21 + ' \033[27m’, endColor, 100),encoding='utf-8')
    sys.stdout.flush()
    sleep(0.1)
	path_to_images = './img\\trainset'
	training_images = []
	print print_progress(mio.import_images(path_to_images))
	for img in print_progress(mio.import_images(path_to_images)):
		# convert to greyscale
		if img.n_channels == 3:
			img = img.as_greyscale()
		# crop to landmarks bounding box with an extra 20% padding
		img = img.crop_to_landmarks_proportion(0.2)
		# rescale image if its diagonal is bigger than 400 pixels
		d = img.diagonal()
		if d > 400:
			img = img.rescale(400.0 / d)
		# define a TriMesh which will be useful for Piecewise Affine Warp of HolisticAAM
		labeller(img, 'PTS', face_ibug_68_to_face_ibug_68_trimesh)
		# append to list
		training_images.append(img)


		
	path_to_lfpw = Path('./img\\testset')	
	# Load and convert to grayscale
	image = mio.import_image(path_to_lfpw/'image_0018.png')
	image = image.as_greyscale()

	# Load detector
	detect = load_dlib_frontal_face_detector()
	# Detect face
	bboxes = detect(image)
	print "{} detected faces.".format(len(bboxes))

	# Crop the image for better visualization of the result
	image = image.crop_to_landmarks_proportion(0.3, group='dlib_0')
	bboxes[0] = image.landmarks['dlib_0']

		
	if len(bboxes) > 0:	

		# Fit AAM 
		patch_aam = PatchAAM(training_images, group='PTS', patch_shape=[(15, 15), (23, 23)],diagonal=150, scales=(0.5, 1.0), holistic_features=fast_dsift,max_shape_components=20, max_appearance_components=150,verbose=True)
		#print 
		#print patch_aam
		fitter = LucasKanadeAAMFitter(patch_aam, lk_algorithm_cls=WibergInverseCompositional,n_shape=[5, 20], n_appearance=[30, 150])
		result = fitter.fit_from_bb(image, bboxes[0], max_iters=[15, 5], gt_shape=image.landmarks['PTS'])
		# print result
		#print(result)

		#visualise result
		result.view(render_initial_shape=True)
sys.stdout.write('\n')

	# Visualize
	#plt.subplot(131);
	#image.view()
	#bboxes[0].view(line_width=3, render_markers=False)
	#plt.gca().set_title('Bounding box')
	
	#plt.subplot(132)
	#image.view()
	#result.initial_shape.view(marker_size=4)
	#plt.gca().set_title('Initial shape')
	
	#plt.subplot(133)
	#image.view()
	#result.final_shape.view(marker_size=4, figure_size=(15, 13))
	#plt.gca().set_title('Final shape')