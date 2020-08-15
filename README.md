# stitchNsplit
![Open Source Love](https://badges.frapsoft.com/os/v1/open-source.svg?v=102)
![GitHub](https://img.shields.io/github/license/cypherics/ShapeMerge)

A Python Library To Stitch And Split Images for any dimension, computing grid and windows over the specified dimension

### Installation

    pip install git+https://github.com/cypherics/stitchNsplit.git#egg=stitchNsplit
    
    
### Split

Split Operation can be performed on two sets of Imagery, <b>Geo Referenced</b> and <b>Non Geo Referenced</b>
The Windows formed for the split operation are adjusted based on the split_size and img_size, whenever 
<code>img_size%split_size != 0</code> is true, this suggests that there will be overlapping windows. 
Overlapping windows are generated only when required. 

<table>
  <tr>
    <td>Original Image</td>
     <td>Images After Split</td>
  </tr>
  <tr>
    <td><img src="https://user-images.githubusercontent.com/24665570/89780629-73256d80-db2f-11ea-9db5-ee50573d8c6d.png" width=600 height=200></td>
    <td><img src="https://user-images.githubusercontent.com/24665570/89780554-483b1980-db2f-11ea-8830-d13c728eadcd.png" width=2000 height=200></td>
  </tr>
 </table>
 
- ##### Geo Referenced
    GeoReferenced Imagery have reference coordinate information stored in them.
This is taken into account while splitting geo referenced imagery, assigning correct reference information to the cut images,
thus preserving the over all reference information 
    > Geo Reference imagery must be of [tiff](https://en.wikipedia.org/wiki/TIFF) format.

- ##### Non GeoReferenced 
    For Non GeoReferenced the split is straight forward, it gets cropped in to specified dimension

*_Split Entire Directory_:*
```python
from stitch_n_split.split.images import SplitGeo
split = SplitGeo(split_size=(124, 267), img_size=(512, 512))
split.perform_directory_split("dir_path")
```
Performing Split over individual images can be done by accessing split as an iterator.

*_Split Iterator using window_:*
```python
from stitch_n_split.split.images import SplitGeo
from stitch_n_split.utility import open_image

split = SplitGeo(split_size=(124, 267), img_size=(512, 512))
image = open_image("img_path", is_geo_reference=True)
for win_number, window in split:
    split_image = split.window_split(image, window)
    # perform operation ....
```
*_Split Iterator using window number_:*
```python
from stitch_n_split.split.images import SplitGeo
from stitch_n_split.utility import open_image

split = SplitGeo(split_size=(124, 267), img_size=(512, 512))
image = open_image("img_path", is_geo_reference=True)
for win_number, window in split:
    split_image = split.window_split(image, win_number)
    # perform operation ....
```

### Stitch 

While Performing Stitch if there are any overlapping window, those windows are merged seamlessly, without
hampering the pixel information and image dimension

Every Split image can be associated to the original image by the *window number* or the *window* itself.

*_Using stitchNsplit together_:*
```python
from stitch_n_split.stitch.images import Stitch
from stitch_n_split.utility import save_image
from stitch_n_split.split.images import SplitNonGeo
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

### stitchNsplit during DeepLearning Inference

While dealing with imagery for DNN, the imagery available might be quite large in size _`in multiple of thousand`_ and performing inference on such large dimension might 
not be a viable option and neither splitting and storing them. *stitchNsplit* can stitch and split the images on the fly, without ever having to store them. 

*_using splitNstitch during Inference_:*
```python
from stitch_n_split.stitch.images import Stitch
from stitch_n_split.utility import save_image
from stitch_n_split.split.images import SplitNonGeo
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
  
### Mesh Computing

![stitchNsplit](https://user-images.githubusercontent.com/24665570/89779619-6e5fba00-db2d-11ea-8705-d8ba781f72ea.gif)

- #### OverLapping Grid
    
    The grid creation process assumes the provided grid size might not be evenly distributed over the mesh size and
    whenever such situation arises, the grid adjusts its position without compromising the grid size, thus generating 
    overlapping grid in the mesh
    
- #### NonOverlapping Grid
    
    No matter what the provided grid size, the goal is to find a grid size which can be evenly distributed over the
    provided mesh size, if the provided sizes presents the possibility of a overlap then the size of the 
    grid is adjusted, to provide non overlapping grid
    
    
<table>
  <tr>
    <td>Mesh with Overlapping Grid</td>
     <td>Mesh with Non Overlapping Grid</td>
  </tr>
  <tr>
    <td><img src="https://user-images.githubusercontent.com/24665570/89773311-49654a00-db21-11ea-9955-f1230d432989.png" width=812 height=350></td>
    <td><img src="https://user-images.githubusercontent.com/24665570/89773649-f8a22100-db21-11ea-8bcc-deeb46939a51.png" width=812 height=350></td>
  </tr>
 </table>
 
 _*mesh size = (10000, 10000)*,  *grid size = (2587, 3000)* were used for above example_

The number of grid generated in both cases are the same, the only difference is, the image in the left doesn't compromises the grid size when it encounters
an overlap, where as the image on the right adjusts its grid size to <code>mesh size // (mesh size / grid size)</code> 
to avoid any overlap


#### Mesh Computing From geo-referenced image
The One mandatory Parameter while computing Mesh is the geo referencing transformation matrix.

- When the size of the mesh and the grid are provided in regular dimension, then the position where the mesh is to be drawn is
extracted from the affine transform and conversion of the dimension to reference coordinate system is done with the help
of pixel resolution present in affine transform

        mesh = mesh_from_geo_transform(
        mesh_size=(w, h),
        transform=transfromation_matrix, 
        grid_size=(w, h)
        )

    _This will generate a *Mesh* of dimension *(w, h)* which will have *Grid* of dimension *(w, h)*, 
which will be bounded within the region *transform * (mesh_size)*_

- When the bounds of mesh are passed, The transformation matrix for the mesh have to be constructed explicitly, the width and
height are computed internally from the given transformation

        transfromation_matrix = get_affine_transform(
        mesh_bounds[0],
        mesh_bounds[-1],
        *get_pixel_resolution(image.transform)
        ) 
        
        mesh = mesh_from_geo_transform(
            grid_size=(w, h),
            transform=transfromation_matrix,
            mesh_bounds=mesh_bounds,
        )

### Output

Grid can can accessed by the extent() call which is a Generator for providing individual grid along with the information associated 
with the grid

    mesh_overlap = mesh_from_geo_transform(mesh_size=(10000, 10000, 3), transform=affine_transform,
    grid_size=(2587, 3000, 3))
    
    for grid in mesh.extent():
        print(grid)
        .....

If the coordinate system available is different than the ones listed [here](#Working-Coordinate-System), then the coordinate must be reprojected before 
mesh computation
    
    transform=geo_transform_to_26190(w, h, arbitrary_image_coordinate_system.bounds,
         arbitrary_image_coordinate_system.crs),

If width and height of the bounds are not known, to calculate it, use

    compute_dimension(arbitrary_image_coordinate_system.bounds, pixel_resolution)

    
### Working Coordinate System
1. EPSG:26910
2. EPSG:26986     
         

### TODO
- [ ] Stitch Geo Referenced Images
