# Dependencies
* [PyTorch](http://pytorch.org/)
* [torchvision](https://github.com/pytorch/vision/)
* [tabulate](https://pypi.python.org/pypi/tabulate/)

# Usage

The code in this repository implements both the RMC and SRMC, with examples on the CIFAR-10 and CIFAR-100 datasets.

## Curve Finding


### Training the endpoints 

To run the curve-finding procedure, you first need to train the two networks that will serve as the end-points of the curve. You can train the endpoints using the following command

```bash
python  train.py --dir=<DIR> \
                 --dataset=<DATASET> \
                 --data_path=<PATH> \
                 --transform=<TRANSFORM> \
                 --model=<MODEL> \
                 --epochs=<EPOCHS> \
                 --lr=<LR_INIT> \
                 --wd=<WD> \
                 --pgd=<PGD>
                 [--use_test]
```

Parameters:

* ```DIR``` &mdash; path to training directory where checkpoints will be stored
* ```DATASET``` &mdash; dataset name [CIFAR10/CIFAR100] 
* ```PATH``` &mdash; path to the data directory
* ```TRANSFORM``` &mdash; type of data transformation [ResNet] 
* ```MODEL``` &mdash; DNN model name: 
    - PreResNet110/PreResNet164
    - WideResNet28x10
* ```EPOCHS``` &mdash; number of training epochs 
* ```LR_INIT``` &mdash; initial learning rate
* ```WD``` &mdash; weight decay 
* ```PGD``` &mdash; type of pgd attack[1/2/inf/msd]

Use the `--use_test` flag if you want to use the test set instead of validation set (formed from the last 5000 training objects) to evaluate performance.

For example, use the following commands to train PreResNet or WideResNet :
```bash
#PreResNet
python train.py --dir=<DIR> --dataset=[CIFAR10 or CIFAR100] --data_path=<PATH>  --model=[PreResNet110 or PreResNet164] --epochs=150  --lr=0.1 --wd=3e-4 --use_test --transform=ResNet --pgd=[1/2/inf/msd]

#WideResNet28x10 
python train.py --dir=<DIR> --dataset=[CIFAR10 or CIFAR100] --data_path=<PATH> --model=WideResNet28x10 --epochs=200 --lr=0.1 --wd=5e-4 --use_test --transform=ResNet --pgd=[1/2/inf/msd]
```

### Training the curves

Once you have two checkpoints to use as the endpoints you can train the curve connecting them using the following comand.

```bash
python  train.py --dir=<DIR> \
                 --dataset=<DATASET> \
                 --data_path=<PATH> \
                 --transform=<TRANSFORM>
                 --model=<MODEL> \
                 --epochs=<EPOCHS> \
                 --lr=<LR_INIT> \
                 --wd=<WD> \
                 --curve=<CURVE>[Bezier|PolyChain] \
                 --num_bends=<N_BENDS> \
                 --init_start=<CKPT1> \ 
                 --init_end=<CKPT2> \
                 --pgd=<PGD>
                 [--fix_start] \
                 [--fix_end] \
                 [--use_test]
```

Parameters:

* ```CURVE``` &mdash; desired curve parametrization [Bezier|PolyChain] 
* ```N_BENDS``` &mdash; number of bends in the curve
* ```CKPT1, CKPT2``` &mdash; paths to the checkpoints to use as the endpoints of the curve

Use the flags `--fix_end --fix_start` if you want to fix the positions of the endpoints; otherwise the endpoints will be updated during training. See the section on [training the endpoints] for the description of the other parameters.

For example, use the following commands to train PreResNet or  WideResNet:
```bash
#PreResNet
python  train.py --dir=<DIR> --dataset=[CIFAR10 or CIFAR100] --use_test --transform=ResNet --data_path=<PATH> --model=PreResNet110 --curve=[Bezier|PolyChain] --num_bends=3  --init_start=<CKPT1> --init_end=<CKPT2> --fix_start --fix_end --epochs=200 --lr=0.03 --wd=3e-4 --pgd=[1/2/inf/msd]

#WideResNet28x10
python3 train.py --dir=<DIR> --dataset=[CIFAR10 or CIFAR100] --use_test --transform=ResNet --data_path=<PATH> --model=WideResNet28x10 --curve=[Bezier|PolyChain] --num_bends=3  --init_start=<CKPT1> --init_end=<CKPT2> --fix_start --fix_end --epochs=200 --lr=0.03 --wd=5e-4 --pgd=[1/2/inf/msd]
```

### Evaluating the curves

To evaluate the found curves, you can use the following command
```bash
python  eval_curve_pgd.py    --dir=<DIR> \
                             --dataset=<DATASET> \
                             --data_path=<PATH> \
                             --transform=<TRANSFORM>
                             --model=<MODEL> \
                             --wd=<WD> \
                             --curve=<CURVE>[Bezier|PolyChain] \
                             --num_bends=<N_BENDS> \
                             --ckpt=<CKPT> \ 
                             --num_points=<NUM_POINTS> \
                             [--use_test]
```
Parameters
* ```CKPT``` &mdash; path to the curves checkpoint saved by `train.py`
* ```NUM_POINTS``` &mdash; number of points along the curve to use for evaluation (default: 61)

See the sections on [training the endpoints] and [training the curves] for the description of other parameters.

`eval_curve.py` outputs the statistics on train and test loss and accuracy along the curve. 

For example, use the following commands to train PreResNet:

```bash
python eval_curve_pgd.py --dir=<DIR> --dataset=[CIFAR10 or CIFAR100]  --data_path=[PATH]  --transform=ResNet     --model=PreResNet110   --curve=Bezier     --num_bends=3         --ckpt=<CKPT>    --num_points=61          --use_test
```



### Selecting the point from  curves

To merge the trained curves into an endpoint at t=T, you can use the following command

```bash
python merge.py --ckpt=<CKPT> --transform=<TRANSFORM> --model=<MODEL> --t=[0.0<=T<=1.0] --dir=<DIR>
```

Parameters

* ```CKPT``` &mdash; path to the curves checkpoint saved by `train.py`
* ```T``` &mdash; where the point you want to get from the curves

See the sections on [training the endpoints] and [training the curves] for the description of other parameters.

For example, use the following commands to select the point from PreResNet curves:

```bash
python merge.py --ckpt=<CKPT> --transform=ResNet --model=PreResNet110 --t=[0.0<=T<=1.0] --dir=<DIR>
```



### Training the endpoints from an existed endpoint merged from curves

To  train the endpoints from an existed endpoint merged from curves,you can use the following command

```bash
python  train_.py    --dir=<DIR> \
                     --dataset=<DATASET> \
                     --data_path=<PATH> \
                     --transform=<TRANSFORM> \
                     --model=<MODEL> \
                     --epochs=<EPOCHS> \
                     --lr=<LR_INIT> \
                     --wd=<WD> \
                     --pgd=<PGD>
                     --origin_model=<CKPT>
                     [--use_test]
```

Parameters

* ```CKPT``` &mdash; path to the existed endpoint merged from curves saved by `merge.py`

See the sections on [training the endpoints] and [training the curves] for the description of other parameters.

For example, use the following commands to train PreResNet:

```bash
python train_.py --dir=<DIR> --dataset=[CIFAR10 or CIFAR100] --data_path=data --model=PreResNet110 --epochs=50 --lr=0.1 --wd=3e-4 --use_test --transform=ResNet --pgd=[1/2/inf/msd] --origin_model=<CKPT>
```







