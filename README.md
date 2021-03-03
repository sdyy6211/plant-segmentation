# Plant segmentation combined with photogrammetry
## Introduction of the model

This model is a semantic image segmentation model, which assigns label to each pixel of an image to partition different objects into segments. Semantic image segmentation means that there is no distinguishing between different plants, e.g. plant_1 and plant_2. They are all classified as plants as a whole. There exists models which can distinguish different objects of the same category. However, in this case, this function does not provide too much helps.

The whole model is composed of two parts, namely backbone part and classifier part. The backbone part has been pre-trained on a large dataset (may exceed over 1 million of training examples) to have a generalizing ability, while the classifier part is untrained, and should be fine-tuned based on specific tasks. This structure takes the advantage of transfer learning ability of deep learning models, which means that a well trained model can perform well in different similar tasks.

## The aim of this model

This model is used to automatically segment each object within crowdsourced images, which are unstructured data that is considered as difficult to be processed automatically. The practical application of the model is to automatically detect and monitor the changes of historical sites from those unstructured image data overtime. In this case, the aim is to detect and monitor the growth of the plant on the wall of Bothwell castle. 

## Process

### Prepare input images from crowdsourced with the following image as an example
![](https://github.com/sdyy6211/Dissertation_Image_segmentation/blob/20210302/gitpic/original_image.jpg)
|:--:| 
| *An example of training image* |


### Hand labelling the image using an online tool (https://app.labelbox.com/)
![](https://github.com/sdyy6211/Dissertation_Image_segmentation/blob/20210302/gitpic/working.PNG)
|:--:| 
| *The interface of labelling tool* |
### Obtain the label for training examples (overall and local region)

![](https://github.com/sdyy6211/Dissertation_Image_segmentation/blob/20210302/gitpic/label.png)
|:--:| 
| *The label of overall classes for the first model* |

![](https://github.com/sdyy6211/Dissertation_Image_segmentation/blob/20210302/gitpic/segmentated_label.png)
|:--:| 
| *The label of binary classes for the second model to refine prediction* |

This process involves downloading masks from the Labelbox. These codes correspond to the file of *segmentation_data_processing.ipynb*

### Train the two models (the first model with 8 classes and a second model for binary classes) using the original images and labels

Deeplab v3+ with a backbone model of resnet101 is used in this project, implemented by https://github.com/jfzhang95/pytorch-deeplab-xception using PyTorch.

For details of training, the parameters of the backbone model are frozen, and only trained the deeplab head parameters with epoch number of 100 and learning rate of 0.01 for the first model, and epoch number of 100 and same learning rate for the second model. The optimizer is Adam, and the loss function is cross entropy loss with 8 classes for the first model and 2 classes for the second model. A scheduling stepdown of learning rate is applied to both models, which means the learning rate will reduce to its 1/10 every epoch of 50 (for the first model) or 33 (for the second model). This is used for the optimizer to better find the minimum point of the loss function.

### Results of overall segmentation

![](https://github.com/sdyy6211/Dissertation_Image_segmentation/blob/20210302/gitpic/predicted.png)
|:--:| 
| *Segmented objects* |

![](https://github.com/sdyy6211/Dissertation_Image_segmentation/blob/20210302/gitpic/superimpose.png)
|:--:| 
| *comparison of label and segmentation results, with an Intersection over Union (IoU) of 0.82966 for all classes and 0.56250 for plants* |

### Reverse selection of the bounding box of the detected objects

In order to crop the area of plants, and further refine them using the second model,the coordinates of a bounding box of the segmented objects need to be obtained based on its maximum and minimum vertical and horizontal coordinates. This is achieved by using DBSCAN of sklearn (https://scikit-learn.org/stable/modules/generated/sklearn.cluster.DBSCAN.html), which is an unsupervised clustering algorithm that automatically partitions disjoint objects. Therefore, distinguishing objects within a class can be partitioned to be drawn an individual bounding box. Once the coordinates of each object are determined, bounding boxes of each disjoint object can be drawn as shown by the following figure.

![](https://github.com/sdyy6211/Dissertation_Image_segmentation/blob/20210302/gitpic/fig7_annotations_with_rect.png)
|:--:| 
| *bounding box of each object* |

### Refined results

After having the bounding box, the plant can be cropped from the whole image, and feed it into the second model to refine the prediction. The refined prediction is shown as follow.

![](https://github.com/sdyy6211/Dissertation_Image_segmentation/blob/20210302/gitpic/figure3+.png)
|:--:| 
| *selecting interested local area and refining predictions in the area* |

### Combination with photogrammetry

Finally, in order to better alleviate the disturbance of distortion caused by camera angle in measuring the area of plant, the photogrammetry is applied to obtain an all-around view of the plant. The final product is a 3D photogrammetry model with segmented textures as shown below.

![](https://github.com/sdyy6211/Dissertation_Image_segmentation/blob/20210302/gitpic/figure10.png)
|:--:| 
| *Final product of a 3D photogrammetry model with segmented textures* |
