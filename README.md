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
from stitch_n_split.spllit.images import SplitGeo
split = SplitGeo(split_size=(124, 267), img_size=(512, 512))
split.perform_directory_split("dir_path")
```
Instead of Splitting the whole directory, single image split operation can also be performed by using split as iterator,
where the function iterates over the windows, An Image split can either be performed by providing the *window number* or the *window*
itself

*_Iterator usage using window_:*
```python
from stitch_n_split.spllit.images import SplitGeo
from stitch_n_split.utility import open_image

split = SplitGeo(split_size=(124, 267), img_size=(512, 512))
image = open_image("img_path", is_geo_reference=True)
for win_number, window in split:
    split_image = split.window_split(image, window)
    # perform operation ....
```
*_Iterator usage using window number_:*
```python
from stitch_n_split.spllit.images import SplitGeo
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

*_Using stitchNsplit together_:*
```python
from stitch_n_split.stitch.images import Stitch
from stitch_n_split.utility import save_image
from stitch_n_split.spllit.images import SplitNonGeo
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
requires a much smaller dimension rather than the over sized image, for such task, the image have to be split and store the image to accomodate the 
task requirement, which is not at all feasible, th smart way would be to split and stitch the image on the fly without ever have to store the image explicitly

One use case where stitchNsplit is the preferred library, is while performing Deep Neural Network Inference. While dealing with
imagery for DNN, the imagery available might be quite large in size _`in multiple of thousand`_ and performing prediction might 
not be a viable option and neither splitting and storing the image, as one requires high memory and other requires excellent storage capacity.
The most suitable choice would be to split and stitch the image on the fly. 

*_using split and stitch model prediction_:*
```python
from stitch_n_split.stitch.images import Stitch
from stitch_n_split.utility import save_image
from stitch_n_split.spllit.images import SplitNonGeo
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
  
### Grid Computing

The most common requirement in the world of GIS is to compute mesh, the library explores the idea of serving 
mesh in two forms either as a overlapping grid or non overlapping grid. 

- #### OverLapped Grid
    
    The grid creation process assumes the provided grid size might not be evenly distributed over the mesh size and
    whenever such situation arises, the grid adjusts its position without compromising the grid size, thus generating 
    overlapping grid in the mesh
    
    How To Check if Mesh will generate overlapping grid
    <code>if mesh size % grid size</code> then the returned mesh has overlapped grids
    
- #### NonOverlapping Grid
    
    No matter what the provided grid size, the goal is to find a grid size which can be evenly distributed over the
    provided mesh size, if the provided sizes present the possibility of overlap then the size of the 
    grid is adjusted to provide non overlapping grid
    
    When will my Grid Size Change
    <code>if mesh size % grid size</code> then change grid size
     
<table>
  <tr>
    <td>Mesh with Overlapping Grid</td>
     <td>Mesh with Non Overlapping Grid</td>
  </tr>
  <tr>
    <td><img src="https://user-images.githubusercontent.com/24665570/89773311-49654a00-db21-11ea-9955-f1230d432989.png" width=812 height=700></td>
    <td><img src="https://user-images.githubusercontent.com/24665570/89773649-f8a22100-db21-11ea-8bcc-deeb46939a51.png" width=812 height=700></td>
  </tr>
 </table>

In Both cases the Mesh is computed over `mesh size = (10000, 10000)` and `grid size = (2587, 3000, 3)`, 
the number of grid produced are same for both, the only difference is the overlapping mesh has adjusted
grid size from original, to <code>mesh size // (mesh size / grid size)</code> i.e in this case to `(2500, 2500)`

*_Usage of Grid_:*

Whenever image geo reference information is available, the the computation of grid could be done in pixel dimension,
the language that could be related to easily, e.g. create a mesh of `10000 pixel x 10000 pixel` which is equivalent to an image that is represented
by `10000 width x 10000 height`

*The required geo reference information is reference coordinates of the image and the pixel resolution for computation
 of mesh and if the pixel resolution information is not provided, use this [link](https://blogs.bing.com/maps/2006/02/25/map-control-zoom-levels-gt-resolution)
 to approximate the pixel resolution based on the zoom level*

_Computing Mesh with user provided grid size and using the image information just as a starting point_:
```python
from stitch_n_split.spllit.grid import GeoGrid
from stitch_n_split.utility import open_image

image = open_image(r"image_used_starting_point_for_compuation.tif", is_geo_reference=True)

# This will return overlapping grid if any
geo_grid_overlap = GeoGrid.mesh_from_geo_transform(mesh_size=(10000, 10000, 3), grid_geo_transform=image.transform, bounds=None, grid_size=(2587, 3000, 3))

# if want non overlapping grid set the overlap variable to False
geo_grid_non_overlap = GeoGrid.mesh_from_geo_transform(mesh_size=(10000, 10000, 3), grid_geo_transform=image.transform, bounds=None, grid_size=(2587, 3000, 3), overlap=False)

for grid in geo_grid_overlap.grid_data:
    grid_extent = grid[0]
    # perform operation on grid extent
``` 
_Computing Mesh with grid size same as input image, the mesh will be computed accordingly_:

```python
from stitch_n_split.spllit.grid import GeoGrid
from stitch_n_split.utility import open_image

image = open_image(r"image_used_starting_point_for_compuation.tif", is_geo_reference=True)

# This will return overlapping grid if any
geo_grid_overlap = GeoGrid.mesh_from_geo_transform(grid_geo_transform=image.transform, bounds=image.bounds, grid_size=(2587, 3000, 3))
``` 

### NOTE 
> THE COORDINATES MUST IN EPSG:26910 FOR THE COMPUTATION WORK EFFECTIVELY 
