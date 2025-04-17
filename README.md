# HiCert

This is the code for HiCert.

## Environment

The code is implemented in Python==3.8, timm==0.9.10, torch==2.0.1.

## Datasets

- [ImageNet](https://image-net.org/download.php) (ILSVRC2012)
- [CIFAR-100](https://www.cs.toronto.edu/~kriz/cifar.html)
- [German Traffic Sign Recognition Benchmark (GTSRB)](https://benchmark.ini.rub.de/)

## Files

├── train_model.py              #Train a vanilla base model 

├── pc_certification.py                  #Generate masked mutants and their prediction results (also the original sample)

├── contextcert_allcase.py    #Check the results of HiCert in all different cases

├── contextcert_all_patchsize.py    #Check the results of HiCert with different patch sizes

├── Patch_attacker_check_inside.py    #The patch attacker

├── Patch_attack_check_inside.py    #Call of the patch attacker for a real patch attack


## Demo

0. You may need to configure the location of datasets and checkpoints.

1. First, train the base vanilla DL models. 

  ```python
`python train_model.py --dataset gtsrb --model vit_base_patch16_224
  ```

  
2. Then, generate and get the inference results of mutants (also the original sample) in the dataset from the DL models.

  ```python
`python pc_certification.py --dataset gtsrb --model vit_base_patch16_224 --patch_size 32
  ```

  

3. Finally, get the final results.

  ```python
`python contextcert_allcase.py --dataset gtsrb --model vit_base_patch16_224 --patch_size 32
  ```



4. Similarly, you may run Patch_attack_check_inside.py to perform an attack.
