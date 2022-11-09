from keras.preprocessing.image import ImageDataGenerator, img_to_array, array_to_img, load_img
datagen = ImageDataGenerator(rotation_range=40,
                            width_shift_range=0.2,
                            height_shift_range=0.2,
                            shear_range=0.2,
                            zoom_range=0.2,
                            horizontal_flip=True,
                            fill_mode='nearest')

import os
import shutil 
import random
import glob

labels = [str(i) for i in range(1, 11)]
labels.extend([chr(i) for i in range(65, 91)])

os.chdir('data/signs')
if '0.jpg' in os.listdirdir('A'):
    for each in labels:        
        for count, filename in enumerate(os.listdir(f'{each}')):
            src = each + '/' + filename
            dst = each + '/' + str(count) + '.jpg'
            os.rename(f'{src}', dst)
                                           
os.chdir('../..')

os.chdir('data/signs')
print('0.jpg' in os.listdir('A'))

os.chdir('../..')

os.listdir()

os.chdir('data/signs')
for each in labels:
    
    os.chdir(f'{each}')
   
    images = os.listdir()
    os.mkdir('preview')
    for image in images:
        if image != 'preview':
            image = load_img(image)
            x = img_to_array(image)
            x = x.reshape((1, ) + x.shape)

            i = 0
            for batch in datagen.flow(x, batch_size=1, save_to_dir='preview', save_prefix=f'{each}', save_format='jpeg'):
                i += 1
                if i > 20:
                    break
    
    os.chdir('..')
os.chdir('../..')

os.chdir('data/signs')
if os.path.isdir('train/1') is False:
    os.mkdir('train')
    os.mkdir('valid')
    os.mkdir('test')
    
    for i in labels:
        shutil.move(f'{i}', 'train')
        os.mkdir(f'valid/{i}')
        os.mkdir(f'test/{i}')
        
        valid_samples = random.sample(os.listdir(f'train/{i}'), 30)
        for j in valid_samples:
            shutil.move(f'train/{i}/{j}', f'valid/{i}')
            
        test_samples = random.sample(os.listdir(f'train/{i}'), 5)
        for k in test_samples:
            shutil.move(f'train/{i}/{k}', f'test/{i}')


os.chdir('../..')

os.listdir()