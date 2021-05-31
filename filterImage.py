import os
import shutil

fromFilePath = "C:/Develop/PersonTraking/Market-1501-v15.09.15/bounding_box_train/"

counter = 0
imageCounter = 1
for filename in os.listdir(fromFilePath):
	if counter % 25 == 0:
		shutil.move(fromFilePath + filename, "C:/Develop/PersonTraking/Market-1501-v15.09.15/filtredImages/img_" + str(imageCounter).rjust(4, '0') + ".jpg")
		imageCounter+=1
		print (counter / 25)
	counter +=1
