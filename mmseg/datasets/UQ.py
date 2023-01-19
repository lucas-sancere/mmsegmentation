import os.path as osp

import mmcv
from .builder import DATASETS
from .custom import CustomDataset

import PIL
PIL.Image.MAX_IMAGE_PIXELS = 10000000000000 # enable training with huge images as dataset
from PIL import Image
import numpy as np

# Register UQ class into DATASETS
@DATASETS.register_module()
class UQDataset(CustomDataset):
    """
    Enter description of the dataset here 
    """
    # Class names of your dataset annotations, i.e., actual names of corresponding label 0, 1, 2 in annotation segmentation maps
    CLASSES = ('background', 'tissue')
    # BGR value of corresponding classes, which are used for visualization
    PALETTE = [[0,0,0], [255,255,255]]

    def __init__(self, **kwargs):
        super(UQDataset, self).__init__(
          # *** In case of training:
            img_suffix='_image.tif',
          # *** In case of inference:
            # img_suffix='.tif',
            seg_map_suffix='_mask.png',

            reduce_zero_label=False, # reduce_zero_label is False because label 0 is background (first one in CLASSES above)
            **kwargs)


    def results2img(self, results, imgfile_prefix, to_label_id, indices=None):
        """Write the segmentation results to images.

        Args:
            results (list[ndarray]): Testing results of the
                dataset.
            imgfile_prefix (str): The filename prefix of the png files.
                If the prefix is "somepath/xxx",
                the png files will be named "somepath/xxx.png".
            to_label_id (bool): whether convert output to label_id for
                submission.
            indices (list[int], optional): Indices of input results, if not
                set, all the indices of the dataset will be used.
                Default: None.

        Returns:
            list[str: str]: result txt files which contains corresponding
            semantic segmentation images.
        """
        if indices is None:
            indices = list(range(len(self)))

        mmcv.mkdir_or_exist(imgfile_prefix)
        result_files = []
        for result, idx in zip(results, indices):

            filename = self.img_infos[idx]['filename']
            basename = osp.splitext(osp.basename(filename))[0]

            png_filename = osp.join(imgfile_prefix, f'{basename}.png')

            # The  index range of official requirement is from 0 to 150.
            # But the index range of output is from 0 to 149.
            # That is because we set reduce_zero_label=True.
            result = result + 1

            output = Image.fromarray(result.astype(np.uint8))
            output.save(png_filename)
            result_files.append(png_filename)

        return result_files

    def format_results(self,
                       results,
                       imgfile_prefix,
                       to_label_id=True,
                       indices=None):
        """Format the results into dir (standard format for ade20k evaluation).

        Args:
            results (list): Testing results of the dataset.
            imgfile_prefix (str | None): The prefix of images files. It
                includes the file path and the prefix of filename, e.g.,
                "a/b/prefix".
            to_label_id (bool): whether convert output to label_id for
                submission. Default: False
            indices (list[int], optional): Indices of input results, if not
                set, all the indices of the dataset will be used.
                Default: None.

        Returns:
            tuple: (result_files, tmp_dir), result_files is a list containing
               the image paths, tmp_dir is the temporal directory created
                for saving json/png files when img_prefix is not specified.
        """

        if indices is None:
            indices = list(range(len(self)))

        assert isinstance(results, list), 'results must be a list.'
        assert isinstance(indices, list), 'indices must be a list.'

        result_files = self.results2img(results, imgfile_prefix, to_label_id,
                                        indices)
        return result_files
