# Visual Relationship Detection

### Data Preparation

1. Download VRD Dateset ([image](http://imagenet.stanford.edu/internal/jcjohns/scene_graphs/sg_dataset.zip), [annotation](http://cs.stanford.edu/people/ranjaykrishna/vrd/dataset.zip), [backup](https://drive.google.com/drive/folders/1V8q2i2gHUpSAXTY4Mf6k06WHDVn6MXQ7)) and put it in the path ~/data. Replace ~/data/sg_dataset/sg_test_images/4392556686_44d71ff5a0_o.gif with ~/data/vrd/4392556686_44d71ff5a0_o.jpg

2. Download [VGG16 trained on ImageNet](https://drive.google.com/open?id=0ByuDEGFYmWsbNVF5eExySUtMZmM) and put it in the path ~/data

3. Download the meta data (so_prior.pkl) [[Baidu YUN]](https://pan.baidu.com/s/1qZErdmc) or [[Google Drive]](https://drive.google.com/open?id=1e1agFQ32QYZim-Vj07NyZieJnQaQ7YKa) and put it in ~/data/vrd

4. Download visual genome data (vg.zip) [[Baidu YUN]](https://pan.baidu.com/s/1qZErdmc) or [[Google Drive]](https://drive.google.com/open?id=1QrxXRE4WBPDVN81bYsecCxrlzDkR2zXZ) and put it in ~/data/vg

5. Word2vec representations of the subject and object categories are provided in this project. If you want to use the model for novel categories, please refer to this [blog](http://mccormickml.com/2016/04/12/googles-pretrained-word2vec-model-in-python/).

The folder should be:

    ├── sg_dataset
    │   ├── sg_test_images
    │   ├── sg_train_images
    │   
    ├── VGG_imagenet.npy
    └── vrd
        ├── gt.mat
        ├── obj.txt
        ├── params_emb.pkl
        ├── proposal.pkl
        ├── rel.txt
        ├── so_prior.pkl
        ├── test.pkl
        ├── train.pkl
        └── zeroShot.mat
### Data format

* train.pkl or test.pkl
	* python list
	* each item is a dictionary with the following keys: {'img_path', 'classes', 'boxes', 'ix1', 'ix2', 'rel_classes'}
	  * 'classes' and 'boxes' describe the objects contained in a single image.
	  * 'ix1': subject index.
	  * 'ix2': object index.
	  * 'rel_classes': relationship for a subject-object pair.


* proposal.pkl
	```Python
        >>> proposals.keys()
        ['confs', 'boxes', 'cls']
        >>> proposals['confs'].shape, proposals['boxes'].shape, proposals['cls'].shape
        ((1000,), (1000,), (1000,))
        >>> proposals['confs'][0].shape, proposals['boxes'][0].shape, proposals['cls'][0].shape
        ((9, 1), (9, 4), (9, 1))
        ```

## Citation

Codebase adapted from https://github.com/GriffinLiang/vrd-dsr

	@article{liang2018Visual,
		title={Visual Relationship Detection with Deep Structural Ranking},
		author={Liang, Kongming and Guo, Yuhong and Chang, Hong and Chen, Xilin},
  		booktitle={AAAI Conference on Artificial Intelligence},
  		year={2018}
	}

## License

The source codes and processed data can only be used for none-commercial purpose. 
