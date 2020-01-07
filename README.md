# sonarToimage
A vacation research scheme project. <br>
Goal is to produce terrains from sonar data using GANs

start date: 28 Nov 2019 
## Data used

https://cirs.udg.edu/caves-dataset/ <br>
Sonar data of underwater cave near Spain. <br><br>
Mallios, A.; Vidal, E.; Campos, R. & Carreras, M. <br>
Underwater caves sonar data set<br>
The International Journal of Robotics Research, 2017, 36, 1247-1251<br>
doi: 10.1177/0278364917732838<br>

### Weekly Log

**Week 1**
- Getting familiar with Machine Learning/ Neural Network/ GANs
- Examining ROS data

**Week 2**
- Tried to make laser scan to point cloud within the ROS functions 
- However, the data seemed to be unorganized when transferring from one form to the other 

**Week 3**
- Trying to create an image directly from the raw laser scan data

**Week 4**
- Customize the created image to sync with the camera information
- Reshape the synced image to ideal size for training

**Week 5**
- Train the model and examine the output
- Recreate the pix2pix model to suit our purpose
