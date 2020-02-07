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
  |_ functions_lib.py
  |_ generate_fullscan.py
  |_ test_generator.py
```
3. Run `python train.py`

## Script Explanation ##

##### GAN directory #####
`datasets.py` = Module for generating the custom Image Loader from the images in the data folder<br>
`models.py` = Module for neural network models<br>
`train.py` = Script for training the model with the data<br>
`function_lib.py` = Custom functions for generating the final image<br>
`generate_fullscan.py` = Generates GAN image tiles for each sonar scan for the angle and outputs the final image<br>
`test_generator.py` = Tests the Generator with the given angle. Will produce GAN generated images.

##### data directory #####
`apply_clahe.py` = Applys Contrast-Limited adaptive histogram equalization to the selected images<br>
`custom_functions.py` = Set of custom functions for manipulating the original underwater cave dataset<br>
`customize_data_for_gan.py` = Customizes the generated data to suit the GAN model<br>
`generate_data.py` = Generates the sonar intensity map from the original underwater cave dataset<br>

**Original Cave Dataset is needed for** `custom_functions.py`, `customize_data_for_gan.py`, **and** `generate_data.py`
<br>
Link: https://cirs.udg.edu/caves-dataset/


