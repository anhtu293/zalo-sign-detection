from .coco import CocoDataset
from .builder import DATASETS


@DATASETS.register_module()
class ZaloDataset(CocoDataset):

    CLASSES = ('traffic sign')
    ''' 
    CLASSES = ('Cấm ngược chiều', 'Cấm dừng và đỗ', 'Cấm rẽ',
               'Giới hạn tốc độ', 'Cấm còn lại', 'Nguy hiểm',
               'Hiệu lệnh')
    '''
