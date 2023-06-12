from lib.data_layers.my_vrd_data_layer import VrdDataLayer

ds_name='vrd'
stage='train'

loader = VrdDataLayer(ds_name, stage)

for i in range(10):
    blob, boxes, rel_boxes, SpatialFea, classes, ix1, ix2, rel_labels, rel_so_prior = loader.forward()

    # format and print shapes
    print('blob: ', blob.shape)
    print('boxes: ', boxes.shape)
    print('rel_boxes: ', rel_boxes.shape)
    print('SpatialFea: ', SpatialFea.shape)
    print('classes: ', classes.shape)
    print('ix1: ', ix1.shape)
    print('ix2: ', ix2.shape)
    print('rel_labels: ', rel_labels.shape)
    print('rel_so_prior: ', rel_so_prior.shape)
    print('\n')

