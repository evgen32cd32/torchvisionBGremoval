import torch;
import torch.nn as nn;
import torch.nn.functional as F;
import torch.optim as optim;
import torchvision;
import torchvision.transforms as transforms;

from pycocotools.coco import COCO;

import numpy as np;
from PIL import Image;

import matplotlib.pyplot as plt;

path = './data/';
ann_path = path + 'annotations_train_val2014/instances_train2014.json';
img_path = path + 'train2014/';
ann_path_val = path + 'annotations_train_val2014/instances_val2014.json';
img_path_val = path + 'val2014/';

coco = COCO(ann_path);
coco_val = COCO(ann_path_val);


imgIds = coco.getImgIds();

i = np.random.randint(0,len(imgIds));

imgData = coco.loadImgs(imgIds[i])[0];

img = Image.open(img_path + imgData['file_name']);

imgrgb = img.convert('RGB');

annIds = coco.getAnnIds(imgIds=imgData['id']);

anns = coco.loadAnns(annIds);


plt.imshow(img);
plt.axis('off');
coco.showAnns(anns);
plt.show();


cats = coco.loadCats(coco.getCatIds())

sc = set([cat['supercategory'] for cat in cats]);
d = {s : [] for s in sc};


for c in cats:
	d[c['supercategory']].append(c['name']);

for k in d:
	print(k);
	print(d[k]);


def barycenterWeight(an,w,h):
	box = an['bbox'];
	x = w/2 - (box[0] + box[2]/2);
	y = h/2 - (box[1] + box[3]/2);
	return an['area']/(x*x + y*y)**2;

def bestAnn(coco, imgData):
	annIds = coco.getAnnIds(imgIds=imgData['id']);
	anns = coco.loadAnns(annIds);
	bst = 0;
	bstAnn = None;
	for a in anns:
		if a['iscrowd'] == 1:
			continue;
		scr = barycenterWeight(a,imgData['width'],imgData['height']);
		if scr > bst:
			bst = scr;
			bstAnn = a;
	return bstAnn;

def datasetGenerator(coco):
	ds = [];
	imgIds = coco.getImgIds();
	for i in imgIds:
		imgData = coco.loadImgs(i)[0];
		ann = bestAnn(coco,imgData);
		if ann is not None:
			if (ann['area']/(imgData['width']*imgData['height']) > 0.10):
				ds.append((Image.open(img_path + imgData['file_name']),ann));
		if len(ds) == 10:
			break;
	return ds;


ds = datasetGenerator(coco);

for i in range(len(ds)):
	plt.imshow(ds[i][0]);
	plt.axis('off');
	coco.showAnns([ds[i][1]]);
	plt.show();




for i in range(len(ds)):
	plt.imshow(ds[i][0]);
	plt.axis('off');
	anns = coco.loadAnns(coco.getAnnIds(imgIds=ds[i][1]['image_id']));
	coco.showAnns(anns);
	plt.show();





def imshow(img):
    npimg = img.numpy();
    plt.imshow(np.transpose(npimg, (1, 2, 0)));
    plt.show();

for i in anns[0]:
    k = 0;
    for j in i:
        k += 1;
        if j == 0:
            print(".", end = '');
        else:
            print('X', end = '');
        if k > 201:
            break;
    print('');


t = torch.cat([anns,anns,anns]);

imshow(torchvision.utils.make_grid([t]));

torch.save(model, save_path + 'best')


