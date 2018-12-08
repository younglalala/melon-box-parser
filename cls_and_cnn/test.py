import Augmentor
path='/Users/wywy/Desktop/box-parser-data-set-four/box-parser-data-set-four_0'
p = Augmentor.Pipeline(path)


p.random_distortion(probability=1,grid_height=3,grid_width=13,magnitude=1)    #okokokokokoko
# p.shear(probability=1,max_shear_left=0.1,max_shear_right=0.7)     #wait wait wait wait
p.rotate(probability=0.2,max_left_rotation=1,max_right_rotation=1)    #okokokokokokok

p.sample(10000)
print(p)
# p.zoom(probability=0.5, min_factor=1.1, max_factor=1.5)
#
# p.sample(10000)
# p.process()
# p.sample(100, multi_threaded=False)
# p = Augmentor.Pipeline("/path/to/images")
#
#
# # Point to a directory containing ground truth data.
# # Images with the same file names will be added as ground truth data
# # and augmented in parallel to the original data.
# p.ground_truth("/path/to/ground_truth_images")
# # Add operations to the pipeline as normal:
# p.rotate(probability=1, max_left_rotation=5, max_right_rotation=5)
# p.flip_left_right(probability=0.5)
# p.zoom_random(probability=0.5, percentage_area=0.8)
# p.flip_top_bottom(probability=0.5)
# p.sample(50)
#
#
#
# g = p.keras_generator(batch_size=128)
# images, labels = next(g)
#
#
# p.random_distortion(probability=1, grid_width=4, grid_height=4, magnitude=8)
# p.flip_left_right(probability=0.5)
# p.flip_top_bottom(probability=0.5)
# p.sample(100)
#
# import Augmentor
#
# p = Augmentor.Pipeline("/home/user/augmentor_data_tests")
#
# p.rotate90(probability=0.5)
# p.rotate270(probability=0.5)
# p.flip_left_right(probability=0.8)
# p.flip_top_bottom(probability=0.3)
# p.crop_random(probability=1, percentage_area=0.5)
# p.resize(probability=1.0, width=120, height=120)
