# stitchNsplit

A Python Library To Stitch And Split Images for any dimension, computing grid and windows over the specified dimension

### Split

Split Operation can be performed on two sets of Imagery, <b>Geo Referenced</b> and <b>Non Geo Referenced</b>
The Windows formed for the split operation are adjusted based on the split_size and img_size, whenever 
<code>img_size%split_size != 0</code> is true, this suggests that there will be overlapping windows. 
Overlapping windows are generated only when required. 

- ##### Geo Referenced
    GeoReferenced Imagery have coordinate information stored in them along with a lot of meta associated with it.
This is well considered while splitting geo referenced imagery, assigning correct reference information to the cut images 
thus preserving the over all reference 
    > Geo Reference imagery must be of [tiff](https://en.wikipedia.org/wiki/TIFF) format.

- ##### Non GeoReferenced 
    For Non GeoReferenced the split is straight forward, it gets cropped in to specified dimension

*_Usage_:*
```python
from stitch_n_split.split import SplitGeo
split = SplitGeo(split_size=(124, 267), img_size=(512, 512))
split.perform_directory_split("dir_path")
```
Instead of Splitting the whole directory, single image split operation can also be performed by using split as iterator,
where the function iterates over the windows, An Image split can either be performed by providing the *window number* or the *window*
itself

*_Iterator usage using window_:*
```python
from stitch_n_split.split import SplitGeo
from stitch_n_split.utility import open_image

split = SplitGeo(split_size=(124, 267), img_size=(512, 512))
image = open_image("img_path", is_geo_reference=True)
for win_number, window in split:
    split_image = split.window_split(image, window)
    # perform operation ....
```
*_Iterator usage using window number_:*
```python
from stitch_n_split.split import SplitGeo
from stitch_n_split.utility import open_image

split = SplitGeo(split_size=(124, 267), img_size=(512, 512))
image = open_image("img_path", is_geo_reference=True)
for win_number, window in split:
    split_image = split.window_split(image, win_number)
    # perform operation ....
```

### Stitch 

After a Split Operation is performed, one might require to stitch the images back up, that could be achieved easily by calling
the <code>stitch_image</code> method

While Performing Stitch if the code encounters any overlapping images, it merges them out seamlessly, without
hampering the pixel information 

For any given image it could be stitched to a new image by just providing either win number or the window.

*_Using splitNstitch together_:*
```python
from stitch_n_split.stitch import Stitch
from stitch_n_split.utility import save_image
from stitch_n_split.split import SplitNonGeo
from stitch_n_split.utility import open_image
import numpy as np

split = SplitNonGeo(split_size=(124, 267), img_size=(512, 512, 3))
image = open_image("img_path")
stitched_image = np.zeros((512, 512, 3))

for win_number, window in split:
    split_image = split.window_split(image, win_number)
    # perform operation ....
    stitched_image = Stitch.stitch_image(split_image, stitched_image, window)
save_image("path_to_save", stitched_image)
``` 
> Stitching Of GeoReference is Not Yet Supported

### Potential Use
One Use case where stitch and split is preferred is, while performing Deep Neural Network Inference, while dealing with
imagery for DNN, the imagery available might be quite large in size _`in multiple of thousands`_ and instead of performing prediction 
on images after they have been reduced and then stitching the prediction back up is quite cumbersome, hence on fly stitch and split.

*_using split and stitch model prediction_:*
```python
from stitch_n_split.stitch import Stitch
from stitch_n_split.utility import save_image
from stitch_n_split.split import SplitNonGeo
from stitch_n_split.utility import open_image
import numpy as np

split = SplitNonGeo(split_size=(124, 267), img_size=(512, 512, 3))
for file in files:
    image = open_image(file)
    stitched_prediction = np.zeros(image.shape)
    # image.shape == (512, 512, 3)
    for win_number, window in split:
        split_image = split.window_split(image, win_number)
        # perform inference ....
        prediction = model.predict(split_image)
        stitched_prediction = Stitch.stitch_image(prediction, stitched_prediction, window)
        # keep on stitching the prediction
    save_image("path_to_save", stitched_prediction)
``` 
  
