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
    GeoReferenced Imagery have coordinate information stored in them, along with a lot of meta data associated with it.
This is very much considered while splitting geo referenced imagery, assigning correct reference information to the cut images,
thus preserving the over all geo reference information 
    > Geo Reference imagery must be of [tiff](https://en.wikipedia.org/wiki/TIFF) format.

- ##### Non GeoReferenced 
    For Non GeoReferenced the split is straight forward, it gets cropped in to specified dimension

*_Usage_:*
```python
from stitch_n_split.split.images import SplitGeo
split = SplitGeo(split_size=(124, 267), img_size=(512, 512))
split.perform_directory_split("dir_path")
```
Instead of Splitting the whole directory, single image split operation can also be performed by using split as iterator,
where the function iterates over the windows, An Image split can either be performed by providing the *window number* or the *window*
itself

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

After a Split Operation is performed, one might require to stitch the images back up, that could be achieved easily by calling
the <code>stitch_image</code> method

While Performing Stitch if the code encounters an overlapping window, it merges it out seamlessly, without
hampering the pixel information and image dimension

Every Split image can be associated to the original image by looking at either the window number or the window itself.

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

### Potential Use for stitchNsplit

Practical use for stitchNsplit is where an image with overly large dimension is available and the task defined by the user
requires a much smaller dimension, rather than the provided over sized image, for such task, the image must be split and stored in order to accommodate the 
task requirement, which not be the feasible solution to the problem, the better way would be to split and stitch the image
on the fly without ever having to store it.

One use case where stitchNsplit is preferred, is while performing Deep Neural Network Inference. While dealing with
imagery for DNN, the imagery available might be quite large in size _`in multiple of thousand`_ and performing inference on such large dimension might 
not be a viable option and neither splitting and storing the image, as one requires high memory and other requires excellent storage capacity.
The most suitable choice would be to split and stitch the image on the fly. 

*_using split and stitch model prediction_:*
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

The most common requirement in the world of GIS is to compute mesh, the library explores the idea of serving 
mesh in two forms either as a overlapping grid or non overlapping grid. 

![stitchNsplit](https://user-images.githubusercontent.com/24665570/89779619-6e5fba00-db2d-11ea-8705-d8ba781f72ea.gif)

- #### OverLapping Grid
    
    The grid creation process assumes the provided grid size might not be evenly distributed over the mesh size and
    whenever such situation arises, the grid adjusts its position without compromising the grid size, thus generating 
    overlapping grid in the mesh
    
    How To Check if Mesh will generate overlapping grid
    <code>if mesh size % grid size</code> then the returned mesh has overlapped grids
    
- #### NonOverlapping Grid
    
    No matter what the provided grid size, the goal is to find a grid size which can be evenly distributed over the
    provided mesh size, if the provided sizes presents the possibility of a overlap then the size of the 
    grid is adjusted, to provide non overlapping grid
    
    When will my Grid Size Change
    <code>if mesh size % grid size</code> then the grid size will be changed

#### Mesh Computing From the information present in the geo-referenced image

The size of the mesh and the grid are regular image dimension and the position where the mesh is to be drawn is
extracted from the geo-reference image.

    mesh_from_geo_transform(mesh_size=(w, h), grid_geo_transform=image.transform, grid_size=(w, h))

_This will generate a *Mesh* of dimension *(w, h)* which will have *Grid* of dimension *(w, h)*, 
which will be bounded within the region *image.transform* _i.e min_x and max_y_, the term *image.transform*
serves as a starting point for Mesh generation_

The Geo Reference information that needs to be present in image.transform is *min_x*, *min_y* and *pixel_resolution*,
The Geo Reference information for the Mesh will be generated from the mesh size and *min_x*, *min_y* and *pixel_resolution*
provided by the image

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

_In The above Image the mesh is computed over *mesh size = (10000, 10000)* and *grid size = (2587, 3000)*, 
the number of grid produced are the same for both, the only difference is, the non overlapping mesh has adjusted the
grid size from original *(2587, 3000)*, to <code>mesh size // (mesh size / grid size)</code> i.e in this case to *(2500, 2500)*_

*_Usage of Grid_:*

_Computing Mesh with user provided grid size and image geo reference_:
```python
from stitch_n_split.split.mesh import mesh_from_geo_transform
from stitch_n_split.utility import open_image

image = open_image(r"image_used_starting_point_for_compuation.tif", is_geo_reference=True)

# This will return overlapping grid if any
geo_grid_overlap = mesh_from_geo_transform(mesh_size=(10000, 10000, 3), grid_geo_transform=image.transform, grid_bounds=None, grid_size=(2587, 3000, 3))

# if want non overlapping grid set the overlap variable to False
geo_grid_non_overlap = mesh_from_geo_transform(mesh_size=(10000, 10000, 3), grid_geo_transform=image.transform, grid_bounds=None, grid_size=(2587, 3000, 3), overlap=False)

for grid in geo_grid_overlap.extent():
    print(grid)
    # perform operation on grid extent

for grid in geo_grid_non_overlap.extent():
    print(grid)
    # perform operation on grid extent
``` 

### Working Coordinate System
1. EPSG:26910
2. EPSG:26986

If the coordinate system is other the one specified above reproject the coordinate system, it could be done by
```python
from stitch_n_split.utility import open_image
from stitch_n_split.split.mesh import mesh_from_geo_transform, GeoInfo
arbitrary_image_coordinate_system = open_image(r"path_to_image", is_geo_reference=True)
shape = arbitrary_image_coordinate_system.read().shape

mesh = mesh_from_geo_transform(
     grid_size=(128, 128),
     mesh_size=(512, 512, 3),
     grid_geo_transform=GeoInfo.geo_transform_to_26190(shape[2], shape[1], arbitrary_image_coordinate_system.bounds, 
arbitrary_image_coordinate_system.crs),
     overlap=True,
 )
```

### TODO

- [ ] Compute Mesh with just extent i.e without pixel resolution information
- [ ] Stitch Geo Referenced Images
- [ ] Calculate Window Position of the Computed Grid
