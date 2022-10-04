from google_drive_downloader import GoogleDriveDownloader as gdd
from sys import platform
import gdown
import zipfile


'''--------------'''
'''Check for Data'''
'''--------------'''

if platform != 'linux' and platform != 'linux2':
    raise Exception("The environment is available only on Linux")

print("Downloading environment...")
gdown.download(id="1phfe0bA0bNNvqa_EngWnsgl1-1yWnV3O", output='games/game.zip')
print("Downloading demonstrations...")
gdown.download(id="1kudIIzCJ2J0wPJP4oFpkYnCskV2Im4pZ", output='reward_model/dems/demonstrations.zip')

print("Extracting files...")
with zipfile.ZipFile("games/game.zip", 'r') as zip_ref:
    zip_ref.extractall("games/")
with zipfile.ZipFile("reward_model/dems/demonstrations.zip", 'r') as zip_ref:
    zip_ref.extractall("reward_model/dems/")