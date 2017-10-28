
from pycocotools.coco import COCO
import time

def getAnns(c,images_ID):
    anns = []
    for image_ID in images_ID:
        ID = c.getAnnIds(imgIds=image_ID)
        anns.append(c.loadAnns(ID))
    return anns
def getObjNum(c, anns, imgId, threshold):
    """
    Get the object that the area the larger than the threshold
    
    input param:
    anns : anns for 1 images
    imgId: 

    output param:
    return if the image contains more than 2 object that are qualified 
    """
    obj_num = 0 
    obj = []
    for ann in anns:
        image_ann = c.loadImgs(imgId)[0]
        #print(image_ann)
        image_area = image_ann['width'] * image_ann['height']
        if 'area' in ann:
            area = ann['area']
            if area >= image_area * threshold:
                obj.append(c.loadCats(ids=ann['category_id'])[0]['name'])


                obj_num += 1
    #print("Image {} has {} object".format(imgId,obj_num))
    #print(obj)
    if obj_num >= 2:
        return True
    else:
        return False


def filter(c,threshold):
    """
    Make for other py file to call function

    input param :
        threshold
    output param :
        return the ids that are qualified in a list
    """
    images_ID = c.getImgIds() # Get all images ID
    anns = getAnns(c, images_ID)

    #threshold = 0.01 # The ratio for object area / image area

    valid_imgId = []

    for a,image_ID in zip(anns,images_ID):
        if getObjNum(c, a, image_ID, threshold):
            valid_imgId.append(image_ID)
    return valid_imgId

if __name__=="__main__":
    c = COCO('../annotations/instances_train2014.json')
    #images_ID = [9,25,30,34,36,49,61,64,71,72,77,78,81,89,92,94]
    #images_ID = [25]

    start_time = time.time()
    images_ID = c.getImgIds() # Get all images ID
    anns = getAnns(c, images_ID)

    threshold = 0.05 # The ratio for object area / image area

    valid_imgId = []

    for a,image_ID in zip(anns,images_ID):
        if getObjNum(c, a, image_ID, threshold):
            valid_imgId.append(image_ID)


    print("The whole dataset contains {} images.".format(len(images_ID)))
    print("The qualified images has {}.".format(len(valid_imgId)))
    end_time = time.time()
    print("This program consume {} seconds".format(end_time-start_time))
                



#print(anns)

#img = c.loadImgs(images_ID)

#print(img)
