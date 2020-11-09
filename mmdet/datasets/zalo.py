from .coco import CocoDataset
from .builder import DATASETS


@DATASETS.register_module()
class ZaloDataset(CocoDataset):

    CLASSES = ('Cấm ngược chiều', 'Cấm dừng và đỗ', 'Cấm rẽ',
               'Giới hạn tốc độ', 'Cấm còn lại', 'Nguy hiểm',
               'Hiệu lệnh')
    """ 
    CLASSES = ('Cam nguoc chieu', 'Cam dung', 'Cam re',
               'Gioi han toc do', 'Cam con lai', 'Nguy hiem',
               'Hieu lenh')
    """
