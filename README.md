# nmt-pytorch
Pytorch implementation of neural machine translation (seq2seq, transformer)

## Usage
Train
```bash
python train.py de en --data-dir=multi30k --save-dir=./ --epochs=8 --batch=128 --lr=0.0005 --config=configs/transformer.yaml
```
Translate
```bash
python translate.py 'sentence to translated' --model=model.pth
```

## Result
Sample attention visualizations are on images folder