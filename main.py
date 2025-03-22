from ultralytics import YOLO
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
