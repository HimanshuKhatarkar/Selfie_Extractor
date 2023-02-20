# coding: utf-8

# ## Import dependencies
from glob import glob
from keras.preprocessing.image import *
from behind_the_scenes.model import CNN_model

print('Connect your smartphone to this system, '
      'mount your Internal Storage and note the '
      'absolute path of WhatsApp folder')

# define model
choice = input("Would you like save weights for model?")
model = CNN_model(choice)

WA_path = input('Enter absolute path of WhatsApp folder: \n')

WA_img_path = WA_path + '/Media/WhatsApp Images/'
WA_img_path.replace('//', '/')
WA_img_path.replace(' ', '\\ ')  # replace spaces with their escaped versions

notes_path = WA_img_path + 'notes/'
if not os.path.exists(notes_path):
    os.mkdir(notes_path)

print('Created a "notes" folder in your WhatsApp Image folder to keep the notes')


def predict(model, file_path, width, height, depth):
    """
    :param file_path: Path of image file
    :return: predict whether file is a notes image
    """
    img = load_img(file_path, target_size=(width, height, depth))
    x = img_to_array(img) / 255.
    y = model.predict(np.expand_dims(x, axis=0))
    return np.squeeze(y) > 0.5


# get file paths 
files = glob(WA_img_path + '*.*')
# extract notes from WhatsApp Images folder

for count, file_path in enumerate(files):
    # print(file_path)
    file_path = '/'.join(file_path.split('\\'))
    if not count % 3:
        print(str(count) + ' files examined')
    if predict(model, file_path, 124, 124, 3):  # check if the file is one of the notes
        file_name = file_path.split('/')[-1]  # get file name
        os.rename(file_path, notes_path + file_name)  # move the file to 'notes' folder
print(str(count) + ' files examined')

choice = input("Would like to classify notes into Handwritten n Printed?")
if choice.lower() == 'y':
    from behind_the_scenes.model_2 import CNN_model as model2

    ch = input("Would you like save weights for this model?")
    model_2 = model2(ch)
    handwritten_notes_path = notes_path + 'handwritten/'
    printed_notes_path = notes_path + 'printed/'
    if not os.path.exists(handwritten_notes_path):
        os.mkdir(handwritten_notes_path)
    if not os.path.exists(printed_notes_path):
        os.mkdir(printed_notes_path)

    # get file paths
    files = glob(notes_path + '*.*')
    # extract notes from WhatsApp Images folder

    for _count, file_path in enumerate(files):
        # print(file_path)
        file_path = '/'.join(file_path.split('\\'))
        if not _count % 3:
            print(str(_count) + ' files examined')
        file_name = file_path.split('/')[-1]  # get file name
        if predict(model_2, file_path, 128, 128, 3):  # check if the file is one of the notes
            os.rename(file_path, handwritten_notes_path + file_name)  # move the file to 'notes' folder
        else:
            os.rename(file_path, printed_notes_path + file_name)
    print(str(_count) + ' files examined')
