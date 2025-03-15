from ultralytics import YOLO
from wget import download
try:
	download('https://drive.usercontent.google.com/download?id=1fwpJVLt8sL6s13wqLPg4gAPbf8qAo0Xd&export=download&authuser=0&confirm=t&uuid=b1abd8de-dd81-4d36-a64f-5a5788dbc383&at=AEz70l4LfEdeqw1zTnVVP8SD8miI%3A1742060987035','yolo11m-seg.pt')
except:
	print('')
while True:
	file = input('\nfile name: ')

	# Load a model
	model = YOLO("yolo11m-seg.pt")  # load a custom model

	# Predict with the model
	results = model(file)  # predict on an image

	# Access the results
	#for result in results:
	#    xy = result.masks.xy  # mask in polygon format
	#    xyn = result.masks.xyn  # normalized
	#    masks = result.masks.data  # mask in matrix format (num_objects x H x W)
	input()
