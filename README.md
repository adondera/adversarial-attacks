## Adversarial attacks

### Imagenet dataset
* The adversarial patch uses samples from the validation set of Imagenet, which contains 50000 images
* Can be downloaded from the official imagenet website under the `ILSVRC2012` validation set
  * The labels in the development kit are of type `ILSVRC2012_ID`, whereas the labels predicted by the pretrained `resnet50` are based on an alphabetical sort of the synset labels. 
  * The `data_util.py` script and the other helper files present here can create the right folder structure to be imported with `torchvision.datasets.ImageFolder`

### Training
* To obtain a patch, just run `Adversarial_Patch_Attack/main.py`.
* Parameters are configured in that file
* We used _Weights and Biases_ to log our results. This requires an account on Weights and Biases.
  * Comment the `wandb` calls inside the code if this is not wanted

### Testing
* Run the `apply_patch.py` script to see the results of a patch being applied to a given image
  * The script runs defenses as well, and shows different grad cam outputs for different perturbed/unperturbed images
* Run the `adv_patch_test.py` script to get different metrics for the attack.