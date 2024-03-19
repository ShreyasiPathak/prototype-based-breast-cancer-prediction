# PIP-Net: Patch-Based Intuitive Prototypes for Interpretable Image Classification
This repository presents the PyTorch code for PIP-Net (Patch-based Intuitive Prototypes Network), published at CVPR 2023.

Paper: ["PIP-Net: Patch-Based Intuitive Prototypes for Interpretable Image Classification"](). 

PIP-Net is an interpretable and intuitive deep learning method for image classification. PIP-Net learns prototypical parts: interpretable concepts visualized as image patches. PIP-Net classifies an image with a sparse scoring sheet where the presence of a prototypical part in an image adds evidence for a class. PIP-Net is globally interpretable since the set of learned prototypes shows the entire reasoning of the model. A smaller local explanation locates the relevant prototypes in a test image. The model can also abstain from a decision for out-of-distribution data by saying “I haven’t seen this before”. The model only uses image-level labels and does not rely on any part annotations. 

### Required Python Packages:
* PyTorch (incl torchvision, following https://pytorch.org/get-started/locally/. Tested with PyTorch 1.13)
* tqdm
* scikit-learn
* pandas
* matplotlib

### Training PIP-Net
PIP-Net can be trained by running `main.py` with arguments. Run `main.py --help` to see all the argument options. Recommended parameters per dataset are present in the `used_arguments.txt` file. 

Check your `--log_dir` to keep track of the training progress. This directory contains `log_epoch_overview.csv` which prints statistics per epoch. File `tqdm.txt` prints updates per iteration and potential errors. File `out.txt` includes all print statements such as additional info. 

Visualizations of prototypes are included in your `--log_dir` / `--dir_for_saving_images`. 

### Data
The code can be applied to any imaging classification data set, structured according to the [Imagefolder format](https://pytorch.org/vision/stable/generated/torchvision.datasets.ImageFolder.html#torchvision.datasets.ImageFolder): 

>root/class1/xxx.png  <br /> root/class1/xxy.png  <br /> root/class2/xyy.png <br /> root/class2/yyy.png

Add or update the paths to your dataset in ``util/data.py``. 

For preparing [CUB-200-2011](http://www.vision.caltech.edu/visipedia/CUB-200-2011.html) with 200 bird species and [Stanford Cars](https://ai.stanford.edu/~jkrause/cars/car_dataset.html) with 196 car types, use the [Instructions of ProtoTree](https://github.com/M-Nauta/ProtoTree/blob/main/README.md#preprocessing-cub).

### Hyperparameter FAQ
* **What is the best number of epochs for my dataset?**
The right number of epochs `--epochs` will depend on the data set size and difficulty of the classification task. Hence, tuning the parameters might require some trial-and-error. As a rule of thumb, we recommend to set the number of epochs such that the number of iterations (i.e., weight updates) during the second training state is around 10,000. Hence, epochs = 10000 / (num_images_in_trainingset / batch_size). Similarly, the number of pretraining epochs `--epochs_pretrain` can be set such that there are 2000 weight updates. 

* **I have CUDA memory issues, what can I do?** PIP-Net is designed to fit onto one GPU. If your GPU has less CUDA memory, you have the following options: 1) reduce your batch size `--batch_size`. Set it as large as possible to still fit in CUDA memory. 2) freeze more layers of the CNN backbone. Rather than optimizing the whole CNN backbone from `--freeze_epochs` onwards, you could keep the first layers frozen during the whole training process. Adapt the code around line 200 in `util/args.py` as indicated in the comments there. Alternatively, set `--freeze_epochs` equal to `--epochs`. 3) Use convnext_tiny_13 instead of convenxt_tiny_26 to make training faster and more efficient. The potential downside is that the latent output grid is less fine-grained and could therefore impact prototype localization, but this will depend on your data and classification task.  



