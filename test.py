from vocab import Vocabulary
import evaluation

print ('Testing ConVSE on Flickr30K dataset')
evaluation.evalrank('runs/f30k_convse/model_best.pth.tar', data_path='data/', split='test')

print ('Testing ConVSE++ on Flickr30K dataset')
evaluation.evalrank('runs/f30k_convse++/model_best.pth.tar', data_path='data/', split='test')

print ('Testing ConVSE on MS-COCO dataset')
evaluation.evalrank('runs/coco_convse/model_best.pth.tar', data_path='data/', split='test', fold5=True)

print ('Testing ConVSE++ on MS-COCO dataset')
evaluation.evalrank('runs/coco_convse++/model_best.pth.tar', data_path='data/', split='test', fold5=True)
