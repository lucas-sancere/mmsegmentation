from .builder import DATASETS
from .custom import CustomDataset

# Register UQ class into DATASETS
@DATASETS.register_module()
class UQDataset(CustomDataset):
    """
    Enter description of the dataset here 
    """
    # Class names of your dataset annotations, i.e., actual names of corresponding label 0, 1, 2 in annotation segmentation maps
    CLASSES = ('background', 'tissue')
    # BGR value of corresponding classes, which are used for visualization
    PALETTE = [0, 255]

    # The formats of image and segmentation map are both .png in this case
    def __init__(self, **kwargs):
        super(UQDataset, self).__init__(
            img_suffix='_image.tif',
            seg_map_suffix='_mask.png',
            reduce_zero_label=False, # reduce_zero_label is False because label 0 is background (first one in CLASSES above)
            **kwargs)