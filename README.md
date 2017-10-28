# guided_saliency_process

This preprocess tool is to exchange the salience of two most salient object in
images.  
The dataset are constraint to MSCOCO, since we use their api.
This tool will do 

* Eliminate the objects that is too small(defined by the threshold parameter)
* Filter out the images that contains less than two objects
* Find the most salient and second salient object according to MSCOCO api.
* Change the two object by element wise multiply their mean ratio.
* Detect whether the new most salient object contain max pixel value.
If yes, do nothing, if no, scale the new most salient object.

### Dependencies
#### [MSCOCO](http://cocodataset.org/#download)

Need download COCO API and place cocoapi/PythonAPI/pycocotools on the same
directory.


### Usages
#### Add the parameters
```
coco_instance_path = "../annotations/instances_train2014.json"
original_saliency_path = "my_path/"
guided_saliency_path = "128_32_2down_COCO_guided_norm_scale/"
prefix =  "COCO_train2014_"

```


#### run it
```
python saliency_process.py
```




