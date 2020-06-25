import os
from PIL import Image

root_dir = '/dataset'

for subset in ['train', 'val']:
    for dirpath, dirnames, filenames in os.walk(os.path.join(root_dir, subset)):
        for filename in os.listdir(dirpath):
            if filename.endswith('.jpg'):
                try:
                    filepath = os.path.join(dirpath, filename)
                    img = Image.open(filepath) # open the image file
                    img.verify() # verify that it is, in fact an image
                except (IOError, SyntaxError) as e:
                    print('Bad file, removing: {}'.format(filename)) # print out the names of corrupt files
                    # os.remove(filepath)
            else:
                print('what is this file: {}'.format(filename))
