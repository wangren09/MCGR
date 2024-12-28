# Usage

The code in this repository implements both the RMC ERMC and SRMC .

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
                 --batch_size=<BS> \
                 --lr=<LR_INIT> \
                 --wd=<WD> \
                 --pgd=<PGD>
                 --gpus=<GPUS>
                 [--use_test]
```

Parameters:

* ```DIR``` &mdash; path to training directory where checkpoints will be stored
* ```DATASET``` &mdash; dataset name [CIFAR10/CIFAR100] 
* ```PATH``` &mdash; path to the data directory
* ```TRANSFORM``` &mdash; type of data transformation [VGG/ResNet] 
* ```MODEL``` &mdash; DNN model name:
    - VGG16/VGG16BN/VGG19/VGG19BN 
    - PreResNet110/PreResNet164
    - SimpleViT
* ```EPOCHS``` &mdash; number of training epochs 
* ```LR_INIT``` &mdash; initial learning rate
* ```batch_size``` &mdash; batch size 
* ```WD``` &mdash; weight decay 
* ```PGD``` &mdash; type of pgd attack[1/2/inf/msd]
* ```GPUS```—id of gpus

Use the `--use_test` flag if you want to use the test set instead of validation set (formed from the last 5000 training objects) to evaluate performance.

For example, use the following commands to train PreResNet  , WideResNet or ViT :
```bash
#PreResNet
python train.py --dir=<DIR> --dataset=[CIFAR10/CIFAR100] --data_path=<PATH>  --model=PreResNet110 --epochs=150 --batch_size=128  --lr=0.1 --wd=3e-4 --use_test --transform=ResNet --pgd=[1/2/inf/msd] 

python train.py --dir=<DIR> --dataset=ImageNet100 --data_path=<PATH>  --model=PreResNet110 --epochs=150 --batch_size=32  --lr=0.1 --wd=3e-4 --use_test --transform=ResNet --pgd=[1/2/inf/msd] --gpus=0,1,2,3

#WideResNet
train.py --dir=<DIR>  --dataset=CIFAR10 --data_path=<PATH> --model=WideResNet28x10 --epochs=200 --lr=0.1 --wd=5e-4 --use_test --transform=ResNet --pgd=[1/2/inf/msd]

#ViT
python train.py --dir=<DIR>  --dataset=CIFAR10 --data_path=<PATH> --model=ViT --epochs=150 --lr=0.001 --wd=1e-4 --use_test --transform=ResNet --pgd=[1/2/inf/msd]
```

### Finetuning the endpoints from an existed endpoint

To  finetune the endpoints from an existed endpoint, you can use the following command

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

See the sections on [training the endpoints] for the description of other parameters.

```bash
#PreResNet
python train_.py --dir=<DIR> --dataset=[CIFAR10/CIFAR100] --data_path=<PATH>  --model=PreResNet110 --epochs=50 --batch_size=128  --lr=0.1 --wd=3e-4 --use_test --transform=ResNet --pgd=[1/2/inf/msd] --origin_model=<CKPT>

python train_.py --dir=<DIR> --dataset=ImageNet100 --data_path=<PATH>  --model=PreResNet110 --epochs=50 --batch_size=32  --lr=0.1 --wd=3e-4 --use_test --transform=ResNet --pgd=[1/2/inf/msd] --origin_model=<CKPT>

#WideResNet
train_.py --dir=<DIR>  --dataset=CIFAR10 --data_path=<PATH> --model=WideResNet28x10 --epochs=50 --lr=0.1 --wd=5e-4 --use_test --transform=ResNet --pgd=[1/2/inf/msd] --origin_model=<CKPT>

#ViT
python train_.py --dir=<DIR>  --dataset=CIFAR10 --data_path=<PATH> --model=ViT --epochs=50 --lr=0.001 --wd=1e-4 --use_test --transform=ResNet --pgd=[1/2/inf/msd]  --origin_model=<CKPT>
```

### Training the curves

Once you have two checkpoints to use as the endpoints you can train the curve connecting them using the following command.

Note that msd attacks in the current code contain all three attacks by default. If you need to reduce the number of attack types, manually adjust line 294 in the /attack/att.py file.

```bash
python  train.py --dir=<DIR> \
                 --dataset=<DATASET> \
                 --data_path=<PATH> \
                 --transform=<TRANSFORM>
                 --model=<MODEL> \
                 --epochs=<EPOCHS> \
                 --lr=<LR_INIT> \
                 --wd=<WD> \
                 --batch_size=<BS> \
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

For example, use the following commands to train VGG16 or PreResNet :
```bash
#PreResNet
python  train.py --dir=<DIR> --dataset=[CIFAR10/CIFAR100] --use_test --transform=ResNet --data_path=<PATH> --model=PreResNet110 --curve=[Bezier|PolyChain] --num_bends=3  --init_start=<CKPT1> --init_end=<CKPT2> --fix_start --fix_end --epochs=50 --batch_size=128  --lr=0.03 --wd=3e-4 --pgd=[1/2/inf/msd]

python  train.py --dir=<DIR> --dataset=ImageNet100 --use_test --transform=ResNet --data_path=<PATH> --model=PreResNet110 --curve=[Bezier|PolyChain] --num_bends=3  --init_start=<CKPT1> --init_end=<CKPT2> --fix_start --fix_end --epochs=50 --batch_size=32  --lr=0.01 --wd=3e-4 --pgd=[1/2/inf/msd]

#WideResNet
python  train.py --dir=<DIR>  --dataset=CIFAR10 --use_test --transform=ResNet --data_path=./data --model=WideResNet28x10 --curve=Bezier --num_bends=3 --init_start=<CKPT1> --init_end=<CKPT2>--fix_start --fix_end --epochs=100 --lr=0.03 --wd=5e-4 --pgd=[1/2/inf/msd]

#ViT
python  train.py --dir=<DIR> --dataset=CIFAR10 --use_test --transform=ResNet --data_path=./data --model=ViT --curve=Bezier --num_bends=3  --init_start=<CKPT1> --init_end=<CKPT2> --fix_start --fix_end --epochs=50 --lr=0.001 --wd=3e-4 --pgd=[1/2/inf/msd]
```

### Evaluating the curves

To evaluate the found curves, you can use the following command
```bash
python  eval_curve_pgd.py    --dir=<DIR> \
                             --dataset=<DATASET> \
                             --data_path=<PATH> \
                             --transform=<TRANSFORM>
                             --model=<MODEL> \
                             --batch_size=<BS> \
                             --curve=<CURVE>[Bezier|PolyChain] \
                             --num_bends=<N_BENDS> \
                             --ckpt=<CKPT> \ 
                             --num_points=<NUM_POINTS> \
                             --
```
Parameters
* ```CKPT``` &mdash; path to the curves checkpoint saved by `train.py`
* ```NUM_POINTS``` &mdash; number of points along the curve to use for evaluation (default: 61)

See the sections on [training the endpoints]and [training the curves] for the description of other parameters.

`eval_curve.py` outputs the statistics on train and test loss and accuracy along the curve. 

```bash
python  eval_curve_pgd.py    --dir=<DIR> --dataset=[CIFAR10/CIFAR100/ImageNet100] --data_path=<PATH> --transform=PreResNet --model=[PreResNet110/WideResNet28x10/ViT] --batch_size=128  --curve=Bezier--num_bends=3 --ckpt=<CKPT>  --num_points=61
```



### Merging the curves

To merge the trained curves into an endpoint at t=T, you can use the following command

```bash
python merge.py --ckpt=<CKPT> --t=[0.0<=T<=1.0] --dir=<DIR>
```

Parameters

* ```CKPT``` &mdash; path to the curves checkpoint saved by `train.py`
* ```T``` &mdash; where the point you want to get from the curves

See the sections on [training the endpoints]and [training the curves] for the description of other parameters.

```bash
python merge.py --ckpt=<CKPT>  --t=0.1 --dir=./mergesave/rmc3/0.1
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

See the sections on [training the endpoints] for the description of other parameters.

```bash
#PreResNet
python train_.py --dir=<DIR> --dataset=[CIFAR10/CIFAR100] --data_path=<PATH>  --model=PreResNet110 --epochs=50 --batch_size=128  --lr=0.1 --wd=3e-4 --use_test --transform=ResNet --pgd=[1/2/inf/msd] --origin_model=<CKPT>

python train_.py --dir=<DIR> --dataset=ImageNet100 --data_path=<PATH>  --model=PreResNet110 --epochs=50 --batch_size=32  --lr=0.1 --wd=3e-4 --use_test --transform=ResNet --pgd=[1/2/inf/msd] --origin_model=<CKPT>

#WideResNet
train_.py --dir=<DIR>  --dataset=CIFAR10 --data_path=<PATH> --model=WideResNet28x10 --epochs=50 --lr=0.1 --wd=5e-4 --use_test --transform=ResNet --pgd=[1/2/inf/msd] --origin_model=<CKPT>

#ViT
python train_.py --dir=<DIR>  --dataset=CIFAR10 --data_path=<PATH> --model=ViT --epochs=50 --lr=0.001 --wd=1e-4 --use_test --transform=ResNet --pgd=[1/2/inf/msd]  --origin_model=<CKPT>
```



### Citation

```latex
@article{wang2023robust,
  title={Robust Mode Connectivity-Oriented Adversarial Defense: Enhancing Neural Network Robustness Against Diversified $$\backslash$ell\_p $ Attacks},
  author={Wang, Ren and Li, Yuxuan and Liu, Sijia and Hero, Alfred},
  journal={arXiv preprint arXiv:2303.10225},
  year={2023}
}
```

### Acknowledgement

We would like to express our gratitude for the support provided by NSF grants 2246157 and 2319243, as well as the ORAU Ralph E. Powe Junior Faculty Enhancement Award. Additionally, we are thankful for the computational resources made available through ACCESS and CloudBank.

