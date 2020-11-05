from .coco import CocoDataset
from .builder import DATASETS


@DATASETS.register_module()
class ZaloDataset(CocoDataset):

    CLASSES = ('Cam nguoc chieu', 'Cam dung', 'Cam re',
               'Gioi han toc do', 'Cam con lai', 'Nguy hiem',
               'Hieu lenh')