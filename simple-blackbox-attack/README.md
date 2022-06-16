This repository uses code from the ICML 2019 paper:

Chuan Guo, Jacob R. Gardner, Yurong You, Andrew G. Wilson, Kilian Q. Weinberger. Simple Black-box Adversarial Attacks.
https://arxiv.org/abs/1905.07121

original code can be found at: https://github.com/cg563/simple-blackbox-attack

Our code uses PyTorch (pytorch >= 0.4.1, torchvision >= 0.2.1) with CUDA 9.0 and Python 3.5. The script run_simba.py contains code to run SimBA and SimBA-DCT with various options.

You can reproduce the attack with:
python run_simba.py --data_root "fake_imagenet" --num_iters 0 --pixel_attack  --freq_dims 224