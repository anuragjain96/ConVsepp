# Cross-modal Retrieval Using Contrastive Learning of Visual-Semantic Embeddings

**In ICPR 2022**

## Dependencies
We recommended to use the following dependencies.

* Python 3.8
* [PyTorch](http://pytorch.org/) (1.4.0)
* [NumPy](http://www.numpy.org/) (>1.12.1)
* [TensorBoard](https://github.com/TeamHG-Memex/tensorboard_logger)


* Punkt Sentence Tokenizer:
```python
import nltk
nltk.download()
> d punkt
```

## Download data and pre-trained models
$DATA_PATH : represents the path where data is stored. $DATA_PATH = data/ <br/>
$RUN_PATH : represents the path where pre-trained models are saved. $RUN_PATH = runs/

For images, please refer to the individual dataset's page.
* For Flickr30K dataset: save the images in $DATA_PATH/f30k/images/ folder.
* For MS-COCO dataset: save train2014/ and val2014/ image folders in $DATA_PATH/coco/images/ folder. Refer to download.sh present in $DATA_PATH/coco/ .

Pre-trained models can be downloaded from: https://drive.google.com/file/d/1CoDtIOtyveVqFKHhsoO9KAWGAR112FMB/view?usp=sharing

## Training new models
Run `train.py`:

```bash
python train.py --data_path "$DATA_PATH" --data_name coco --logger_name 
runs/coco_convse++ --use_restval --lr_update 30 --cnn_type resnet152 --batch_size 256 --loss_fn ConVSE++ --resume runs/coco_baseembed/model_best.pth.tar
```

Arguments used to train pre-trained models:

| Method    | Arguments |
| :-------: | :-------: |
| ConVSE    | `--loss_fn ConVSE` |
| ConVSE++  | `--loss_fn ConVSE++` |
| Flickr30K dataset | `--data_name f30k` |
| MS-COCO dataset   | `--data_name coco --use_restval` |


## Evaluate pre-trained models
```python
python -c "\
from vocab import Vocabulary
import evaluation
evaluation.evalrank("$RUN_PATH/coco_convse++/model_best.pth.tar", data_path="$DATA_PATH", split="test")"
```

To do cross-validation on MSCOCO, pass `fold5=True` with a model trained using 
`--data_name coco`.

For evaluating pre-trained ConVSE and ConVSE++ models on both f30k and coco dataset run `test.py`:
```python
python test.py
```
