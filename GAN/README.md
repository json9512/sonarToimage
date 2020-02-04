# Instructions #

1. Unzip the data.zip in the data folder 
2. Make sure the hierachy is <br>
```
  GAN
  |_data
    |_ GAN_data64
    |_ test
    |_ customizing_data_from_cave_dataset.zip
  |_ datasets.py
  |_ models.py
  |_ train.py
  |_ test_generator.py
```
3. Run `python train.py`

## Script Explanation ##

##### GAN directory #####
`datasets.py` = Module for generating the custom Image Loader from the images in the data folder<br>
`models.py` = Module for neural network models<br>
`train.py` = Script for training the model with the data<br>

##### data directory #####
`apply_clahe.py` = Applys Contrast-Limited adaptive histogram equalization to the selected images<br>
`custom_functions.py` = Set of custom functions for manipulating the original underwater cave dataset<br>
`customize_data_for_gan.py` = Customizes the generated data to suit the GAN model<br>
`generate_data.py` = Generates the sonar intensity map from the original underwater cave dataset<br>

##### script_for_fullscan.zip from `sample` folder #####
Make sure both scipt files are under GAN directory with the other script files.<br>
`gen_data_fullscan.py` = Creates GAN generated images from the `data/test` folder<br>
`gen_fullscan.py` = Creates a single image that attaches all the GAN generated images together

**Original Cave Dataset is needed for** `custom_functions.py`, `customize_data_for_gan.py`, **and** `generate_data.py`
<br>
Link: https://cirs.udg.edu/caves-dataset/


