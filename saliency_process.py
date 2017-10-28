from pycocotools.coco import COCO
import matplotlib.pyplot as plt
import matplotlib.path as mplpath
from scipy.ndimage.filters import gaussian_filter
from files_filter import filter
import skimage.io as io
import numpy as np
import time, random, copy, sys, os, glob
import scipy.misc

def swap_cols(array, frm, to):
    array[:,[frm,to]] = array[:,[to,frm]]

def calTolSaliency(s,box, poly, total_sal):
    # calculate the total saliency map value in the polygon
    #index = [(x,y) for x, y in np.ndindex(s.shape)]
    
    index = [(int(x+box[1]-1),int(y+box[0]-1)) for x, y in np.ndindex(int(box[3]),int(box[2]))]
    #print("S shape = ",s.shape)
    for i in index :
        if poly.contains_point(i):
            total_sal.append(s[i])
    if len(total_sal) == 0:
        total_sal.append(0)

def darken(s,box,poly,ratio):
    # Make the pixels in poly darker

    #index = [(x,y) for x, y in np.ndindex(s.shape)]
    index = [(int(x+box[1]-1),int(y+box[0]-1)) for x, y in np.ndindex(int(box[3]),int(box[2]))]
    for i in index : 
        if poly.contains_point(i):
            s[i] = s[i]/ratio

def brighten(s, box,poly, ratio):
    # Make the pixels in poly brighter
    ratio = ratio * 1.0
    #index = [(x,y) for x, y in np.ndindex(s.shape)]
    index = [(int(x+box[1]-1),int(y+box[0]-1)) for x, y in np.ndindex(int(box[3]),int(box[2]))]
    for i in index :
        if poly.contains_point(i):
            if s[i] == np.nan:
                s[i] = 0.
            if s[i] * ratio >= 255 :   # since the imread output a uint8 cannot exceed 255
                s[i] = 255
            else:
                s[i] = s[i]*ratio
                #s[i] = 255.

def loadAll(saliencyDir):
    """
    Get all the saliency and images in the dir

    """
    #os.chdir(imageDir)
    I = []
    S = []
    #for filename in glob.glob("*.jpg"):
    #    I.append(io.imread(filename))
    #os.chdir(saliencyDir)
    for filenmae in glob.glob(saliencyDir+"*.jpg"):
        S.append(io.imread(filename))
    return {'saliency':S}

def loadFiles(images_ID):
#images_ID = int(sys.argv[1])
    #print("ID = ", images_ID)
    I = []
    S = []
    for image_ID in images_ID:
        #image_file = "../images/COCO_train2014_{:012}.jpg".format(image_ID)
        #saliency_file = "../../COCO_saliency/COCO_train2014_{:012}.jpg".format(image_ID)
        saliency_file = "/home/yuandy/saliency-salgan-2017/128_32_2down_COCO_prediction/COCO_train2014_{:012}.jpg".format(image_ID)
        #I.append(io.imread(image_file))
        S.append(io.imread(saliency_file))
    #return {'images':I,'saliency':S}
    return {'saliency':S}

def getAnns(c,images_ID):
    anns = []
    for image_ID in images_ID:
        ID = c.getAnnIds(imgIds=image_ID)
        anns.append(c.loadAnns(ID))
    return anns

def getMax(c,anns,S):
    inst_sal = {}

    start_time = time.time()

    for ann in anns:
        if 'segmentation' in ann:
            if type(ann['segmentation']) == list :
                total_sal = []
                box = ann['bbox']
                for seg in ann['segmentation']:
                    poly = np.array(seg).reshape((-1,2))
                    swap_cols(poly,0,1)
                    polyPath = mplpath.Path(poly)
                    calTolSaliency(S, box, polyPath, total_sal)
                inst_sal[(ann['category_id'],ann['id'])] = np.mean(total_sal)

    first_salient_instance = max(inst_sal, key = inst_sal.get) # get the key of the max element
    first_mean = inst_sal[first_salient_instance]
    first_salient_instance_name = c.loadCats(ids=first_salient_instance[0])[0]['name']
    print ("The first salient instance = ",first_salient_instance_name)
    del inst_sal[first_salient_instance] # del the max element in the inst_Sal
    try:
        second_salient_instance = max(inst_sal, key = inst_sal.get) 
        second_mean = inst_sal[second_salient_instance]
        ratio = first_mean / second_mean



    #print("salient test", salient_instance)
        second_salient_instance_name = c.loadCats(ids=second_salient_instance[0])[0]['name']
        print ("The second salient instance = ", second_salient_instance_name)

        end_time = time.time()
        print ("this consume {} seconds".format(end_time - start_time))
        return first_salient_instance, first_mean, second_salient_instance,\
            second_mean, ratio

    except:
        return 0,0,0,0,0

def saliencyExchange(S,inst1, inst2, ratio):
    """
    Change two object saliency by multiply/divide their ratio  
    """
    # darken the original salient instance
    box = inst1['bbox']
    for seg in inst1['segmentation']:
        poly =  np.array(seg).reshape((-1,2)) 
        swap_cols(poly,0,1)
        #print("origin salient ", poly.shape)
        polyPath = mplpath.Path(poly)
        darken(S, box, polyPath, ratio)
    # brighten the new salient instance 
    box =inst2['bbox']
    for seg in inst2['segmentation']:
        poly =  np.array(seg).reshape((-1,2)) 
        swap_cols(poly,0,1)
        #print("new_salient" , poly.shape)
        polyPath = mplpath.Path(poly)
        brighten(S, box,polyPath, ratio)
  

def makeObjMax(S, new_inst, mean):
    """
    Make sure the new_inst is the brightest object in the images
    """
    # Find the max point in the image
    max_points = np.where(S == S.max())
    #print("max point = ", max_points)
    index = np.array(max_points)
    #print("index = ", index, index.shape)
    box = new_inst['bbox']
    for seg in new_inst['segmentation']:
        poly =  np.array(seg).reshape((-1,2)) 
        swap_cols(poly,0,1)
        #print("origin salient ", poly.shape)
        polyPath = mplpath.Path(poly)
        for i in range(index.shape[1]):
            #print("Index = ",i)
            if polyPath.contains_point((index[0,i],index[1,i])):
                pass
     #           return 0
    print("Need to do make sure max.")
    
    for seg in new_inst['segmentation']:
        poly =  np.array(seg).reshape((-1,2)) 
        swap_cols(poly,0,1)
        #print("origin salient ", poly.shape)
        polyPath = mplpath.Path(poly)
        brighten(S, box, polyPath, 1.5)#S.max()/mean)
    S = np.clip(S,0,255)
 

def randomObj(salient_instance,anns):
    ## Make the saliency transfer to other instance
    origin_salient_inst = [x for x in anns if x['id'] == salient_instance[1]][0]
    ## random choose the other object
    try:
        rest_inst = [x for x in anns if x['id'] != salient_instance[1]]
        new_salient_inst = random.choice(rest_inst)
    except:
        print("This images has only one object.")
        new_salient_inst = origin_salient_inst
    new_salient_inst_name = c.loadCats(ids=new_salient_inst['category_id'])[0]['name']
    print("The new instance =",new_salient_inst_name)
    return origin_salient_inst, new_salient_inst, new_salient_inst_name

if __name__ == "__main__":


    # Parameters
    coco_instance_path = "../annotations/instances_train2014.json"
    guided_saliency_path = "128_32_2down_COCO_guided_norm_scale/"
    prefix =  "COCO_train2014_"
    c = COCO(coco_instance_path)

    # Define the area of object that larger than the ratio to whole images.
    # That is, if the threshold = 0.1, only the object larger than 0.1*image
    # area will be consider an object. 
    # And we select the image that contain more than two object.
    threshold = 0.1  
    all_images_ID = filter(c, threshold)

    print("all_images length", len(all_images_ID))

    # Read in the images file by chunks in case out of memory. 
    chuncks = [all_images_ID[x:x+40000] for x in xrange(0, len(all_images_ID), 40000)]

    anns = getAnns(c, all_images_ID)
    count = 0
    
    for images_ID in chuncks:
        print("Loading the chunk data...")
        file_dict = loadFiles(images_ID)
        for a,S,image_ID in zip(anns, file_dict['saliency'], images_ID):
            count += 1
            print("Processing images {}".format(image_ID)) 


            try:
                # Get the max mean instance and second max instance
                [inst1,inst1_mean, inst2, inst2_mean, ratio] = getMax(c,a, S)
                print("inst1=",inst1)
                print("inst2=",inst2)

                if inst1 == inst2 == 0:
                    print("This image has only one object. skip")
                    continue
            

                inst1_ann = [x for x in a if x['id'] == inst1[1]][0]
                inst2_ann = [x for x in a if x['id'] == inst2[1]][0]
                
                inst1_name = c.loadCats(ids=inst1_ann['category_id'])[0]['name']
                inst2_name = c.loadCats(ids=inst2_ann['category_id'])[0]['name']

                S_new = np.copy(S)
                
                saliencyExchange(S_new, inst1_ann, inst2_ann, ratio) # from now, inst2 is the new max obj
                makeObjMax(S_new, inst2_ann, inst2_mean)

                # normalize : min-max normalization
                S_new = (S_new - np.amin(S))*1.0 / (np.amax(S) - np.amin(S)) * 255.

                # apply gaussian filter 
                S_new = gaussian_filter(S_new,sigma=5)

                # Save the images
                scipy.misc.toimage(S_new).save(os.path.join(guided_saliency_path, prefix)+'{:012}.jpg'.format(image_ID))
                print("alreay process {} images".format(count))

            except:
                e = sys.exc_info()[0]
                with open('error_log.txt') as f:
                 f.write('COCO_train2014_{:012}.jpg'.format(image_ID), " cannot process .")
                 f.write(e)
 
