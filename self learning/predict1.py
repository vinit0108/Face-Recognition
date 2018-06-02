import tensorflow as tf
import numpy as np
import os,glob,cv2
import sys,argparse
import shutil

print ("hi")
#dir_path = os.path.dirname(os.path.realpath(__file__))
#listt = os.listdir(dir_path)
#count = len(listt)
#
count = len(glob.glob('/home/vinit0108/Desktop/final DM1 (copy)/self learning/new/*'))
print(count)
while(count != 0):
	#image_path=sys.argv[1] 
	filename =('/home/vinit0108/Desktop/final DM1 (copy)/self learning/new/%.2d.png'%count)#dir_path #+'/'+image_path
	print(filename)	
	image_size=90
	num_channels=3
	images = []
# Reading 

	image = cv2.imread(filename)
	# Resizing the image to our desired size and preprocessing will be done exactly as done during training
	image = cv2.resize(image, (image_size, image_size),0,0, cv2.INTER_LINEAR)
	images.append(image)
	images = np.array(images, dtype=np.uint8)
	images = images.astype('float32')
	images = np.multiply(images, 1.0/255.0) 
	#The input to the network is of shape [None image_size image_size num_channels]. Hence we reshape.
	x_batch = images.reshape(1, image_size,image_size,num_channels)

	## Let us restore the saved model 
	sess = tf.Session()
	# Step-1: Recreate the network graph. At this step only graph is created.
	saver = tf.train.import_meta_graph('model.meta')
		# Step-2: Now let's load the weights saved using the restore method.
	saver.restore(sess, tf.train.latest_checkpoint('./'))
	
# Accessing the default graph which we have restored
	graph = tf.get_default_graph()
	9rocessed to get the output.
	# In the original network y_pred is the tensor that is the prediction of the network
	y_pred = graph.get_tensor_by_name("y_pred:0")
	
	## Let's feed the images to the input placeholders
	x= graph.get_tensor_by_name("x:0") 
	y_true = graph.get_tensor_by_name("y_true:0")
	y_test_images = np.zeros((1, 10)) 


	### Creating the feed_dict that is required to be fed to calculate y_pred 
	feed_dict_testing = {x: x_batch, y_true: y_test_images}
	result=sess.run(y_pred, feed_dict=feed_dict_testing)
	print(result)

	maxi = np.max(result)
	print(maxi)
	
	res1 = result.ravel()
	res = res1.tolist()
	

	ind = res.index(max(res))
	print(ind)
	if(ind == 0):
		shutil.copy2(filename,'/home/vinit0108/Desktop/final DM1 (copy)/self learning/trainnCif/Airplane')
	elif(ind==1):
		shutil.copy2(filename,'/home/vinit0108/Desktop/final DM1 (copy)/self learning/trainnCif/Automobile')
	elif(ind==2):
		shutil.copy2(filename,'/home/vinit0108/Desktop/final DM1 (copy)/self learning/trainnCif/Bird')
	elif(ind==3):
		shutil.copy2(filename,'/home/vinit0108/Desktop/final DM1 (copy)/self learning/trainnCif/Cat')
	elif(ind==4):
		shutil.copy2(filename,'/home/vinit0108/Desktop/final DM1 (copy)/self learning/trainnCif/Deer')
	elif(ind==5):
		shutil.copy2(filename,'/home/vinit0108/Desktop/final DM1 (copy)/self learning/trainnCif/Dog')
	elif(ind==6):
		shutil.copy2(filename,'/home/vinit0108/Desktop/final DM1 (copy)/self learning/trainnCif/Frog')
	elif(ind==7):
		shutil.copy2(filename,'/home/vinit0108/Desktop/final DM1 (copy)/self learning/trainnCif/Horse')
	elif(ind==8):
		shutil.copy2(filename,'/home/vinit0108/Desktop/final DM1 (copy)/self learning/trainnCif/Ship')
	elif(ind==9):
		shutil.copy2(filename,'/home/vinit0108/Desktop/final DM1 (copy)/self learning/trainnCif/Truck')
	"""elif(ind==10):
		shutil.copy2(filename,'/home/vinit0108/Desktop/final DM1 (copy)/self learning/training_data/AnupamKher')
	elif(ind==11):
		shutil.copy2(filename,'/home/vinit0108/Desktop/final DM1 (copy)/self learning/training_data/Ashwanth')
	elif(ind==12):
		shutil.copy2(filename,'/home/vinit0108/Desktop/final DM1 (copy)/self learning/training_data/Avinash')
	elif(ind==13):
		shutil.copy2(filename,'/home/vinit0108/Desktop/final DM1 (copy)/self learning/training_data/BabuMohan')
	elif(ind==14):
		shutil.copy2(filename,'/home/vinit0108/Desktop/final DM1 (copy)/self learning/training_data/BalaKrishna')
	elif(ind==15):
		shutil.copy2(filename,'/home/vinit0108/Desktop/final DM1 (copy)/self learning/training_data/BhanuPriya')
	elif(ind==16):
		shutil.copy2(filename,'/home/vinit0108/Desktop/final DM1 (copy)/self learning/training_data/Bharathi')
	elif(ind==17):
		shutil.copy2(filename,'/home/vinit0108/Desktop/final DM1 (copy)/self learning/training_data/BomanIrani')
	elif(ind==18):
		shutil.copy2(filename,'/home/vinit0108/Desktop/final DM1 (copy)/self learning/training_data/Brahmanadam')
	elif(ind==19):
		shutil.copy2(filename,'/home/vinit0108/Desktop/final DM1 (copy)/self learning/training_data/Cochinganeefa')
	elif(ind==20):
        	shutil.copy2(filename,'/home/vinit0108/Desktop/final DM1 (copy)/self learning/training_data/Dileep')
	elif(ind==21):
		shutil.copy2(filename,'/home/vinit0108/Desktop/final DM1 (copy)/self learning/training_data/Dwarkish')
	elif(ind==22):
		shutil.copy2(filename,'/home/vinit0108/Desktop/final DM1 (copy)/self learning/training_data/Faridhajalal')
	elif(ind==23):
		shutil.copy2(filename,'/home/vinit0108/Desktop/final DM1 (copy)/self learning/training_data/Hrithik')
	elif(ind==24):
		shutil.copy2(filename,'/home/vinit0108/Desktop/final DM1 (copy)/self learning/training_data/Innocent')
	elif(ind==25):
		shutil.copy2(filename,'/home/vinit0108/Desktop/final DM1 (copy)/self learning/training_data/Jagadeesh')
	elif(ind==26):
		shutil.copy2(filename,'/home/vinit0108/Desktop/final DM1 (copy)/self learning/training_data/JagapathiBabu')
	elif(ind==27):
		shutil.copy2(filename,'/home/vinit0108/Desktop/final DM1 (copy)/self learning/training_data/Jagathy')
	elif(ind==28):
		shutil.copy2(filename,'/home/vinit0108/Desktop/final DM1 (copy)/self learning/training_data/Jamuna')
	elif(ind==29):
		shutil.copy2(filename,'/home/vinit0108/Desktop/final DM1 (copy)/self learning/training_data/JayaBhaduri')
	elif(ind==30):
		shutil.copy2(filename,'/home/vinit0108/Desktop/final DM1 (copy)/self learning/training_data/JayaPrakashReddy')
	elif(ind==31):
		shutil.copy2(filename,'/home/vinit0108/Desktop/final DM1 (copy)/self learning/training_data/Jayaram')
	elif(ind==32):
		shutil.copy2(filename,'/home/vinit0108/Desktop/final DM1 (copy)/self learning/training_data/JosePrakash')
 
	elif(ind==33):
		shutil.copy2(filename,'/home/vinit0108/Desktop/final DM1 (copy)/self learning/training_data/KajalAgarwal')
	elif(ind==34):
		shutil.copy2(filename,'/home/vinit0108/Desktop/final DM1 (copy)/self learning/training_data/Kareena')
	elif(ind==35):
		shutil.copy2(filename,'/home/vinit0108/Desktop/final DM1 (copy)/self learning/training_data/Karuna')
	elif(ind==36):
		shutil.copy2(filename,'/home/vinit0108/Desktop/final DM1 (copy)/self learning/training_data/Katrina')

	elif(ind==37):
		shutil.copy2(filename,'/home/vinit0108/Desktop/final DM1 (copy)/self learning/training_data/Kavya')
	elif(ind==38):
		shutil.copy2(filename,'/home/vinit0108/Desktop/final DM1 (copy)/self learning/training_data/KotaSrinivas')
	elif(ind==39):
		shutil.copy2(filename,'/home/vinit0108/Desktop/final DM1 (copy)/self learning/training_data/KVishwanath')

	elif(ind==40):
		shutil.copy2(filename,'/home/vinit0108/Desktop/final DM1 (copy)/self learning/training_data/Laxmidevi')
	elif(ind==41):
		shutil.copy2(filename,'/home/vinit0108/Desktop/final DM1 (copy)/self learning/training_data/Leelavathi')

	elif(ind==42):
		shutil.copy2(filename,'/home/vinit0108/Desktop/final DM1 (copy)/self learning/training_data/Loknath')
	elif(ind==43):
		shutil.copy2(filename,'/home/vinit0108/Desktop/final DM1 (copy)/self learning/training_data/Madhabi')

	elif(ind==44):
		shutil.copy2(filename,'/home/vinit0108/Desktop/final DM1 (copy)/self learning/training_data/Madhavan')
	elif(ind==45):
		shutil.copy2(filename,'/home/vinit0108/Desktop/final DM1 (copy)/self learning/training_data/Madhu')

	elif(ind==46):
		shutil.copy2(filename,'/home/vinit0108/Desktop/final DM1 (copy)/self learning/training_data/Madhuri')
	elif(ind==47):
		shutil.copy2(filename,'/home/vinit0108/Desktop/final DM1 (copy)/self learning/training_data/Mallikarjunrao')

	elif(ind==48):
		shutil.copy2(filename,'/home/vinit0108/Desktop/final DM1 (copy)/self learning/training_data/Mammootty')
	elif(ind==49):
		shutil.copy2(filename,'/home/vinit0108/Desktop/final DM1 (copy)/self learning/training_data/Mamukkoya')

	elif(ind==50):
		shutil.copy2(filename,'/home/vinit0108/Desktop/final DM1 (copy)/self learning/training_data/Mohanlal')
	elif(ind==51):
		shutil.copy2(filename,'/home/vinit0108/Desktop/final DM1 (copy)/self learning/training_data/MSNarayana')
	elif(ind==52):
		shutil.copy2(filename,'/home/vinit0108/Desktop/final DM1 (copy)/self learning/training_data/Mukesh')
	elif(ind==53):
		shutil.copy2(filename,'/home/vinit0108/Desktop/final DM1 (copy)/self learning/training_data/Nagarjuna')
	elif(ind==54):
		shutil.copy2(filename,'/home/vinit0108/Desktop/final DM1 (copy)/self learning/training_data/Nedumudivenu')

	elif(ind==55):
		shutil.copy2(filename,'/home/vinit0108/Desktop/final DM1 (copy)/self learning/training_data/NTR')


	elif(ind==56):
		shutil.copy2(filename,'/home/vinit0108/Desktop/final DM1 (copy)/self learning/training_data/Paresh')


	elif(ind==57):
		shutil.copy2(filename,'/home/vinit0108/Desktop/final DM1 (copy)/self learning/training_data/Pavithra')

	elif(ind==58):
		shutil.copy2(filename,'/home/vinit0108/Desktop/final DM1 (copy)/self learning/training_data/Prakashraj')


	elif(ind==59):
		shutil.copy2(filename,'/home/vinit0108/Desktop/final DM1 (copy)/self learning/training_data/Prema')

	elif(ind==60):
		shutil.copy2(filename,'/home/vinit0108/Desktop/final DM1 (copy)/self learning/training_data/Premnazir')


	elif(ind==61):
		shutil.copy2(filename,'/home/vinit0108/Desktop/final DM1 (copy)/self learning/training_data/Rajkumar')

	elif(ind==62):
		shutil.copy2(filename,'/home/vinit0108/Desktop/final DM1 (copy)/self learning/training_data/RakheeGulzar')


	elif(ind==63):
		shutil.copy2(filename,'/home/vinit0108/Desktop/final DM1 (copy)/self learning/training_data/Ramanreddy')

	elif(ind==64):
		shutil.copy2(filename,'/home/vinit0108/Desktop/final DM1 (copy)/self learning/training_data/RamaPrabha')

	elif(ind==65):
		shutil.copy2(filename,'/home/vinit0108/Desktop/final DM1 (copy)/self learning/training_data/RaniMukherje')
"""	
	count = count - 1


