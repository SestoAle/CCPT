from sys import platform
import gdown
import zipfile
import argparse


'''--------------'''
'''Check for Data'''
'''--------------'''
parser = argparse.ArgumentParser()
parser.add_argument('-sp', '--speed', help="The speed of the game. There are 2 available environments. One that has time scale x100, and one that has time scale 1", default=100, choices=[100, 1], type=int )

args = parser.parse_args()

if platform != 'linux' and platform != 'linux2':
    raise Exception("The environment is available only on Linux")

print("Downloading environment...")
if args.speed == 100:
    gdown.download(id="186H39V6MzdKlS8dyQXYb38vBJsIGfq2K", output='games/game.zip')
elif args.speed == 1:
    gdown.download(id="1DiFWNYkZ0xH02c5kkELb9WleseSt21Fh", output='games/game.zip')
else:
    raise Exception("Evironment not available")

print("Downloading demonstrations...")
gdown.download(id="1kudIIzCJ2J0wPJP4oFpkYnCskV2Im4pZ", output='reward_model/dems/demonstrations.zip')

print("Extracting files...")
with zipfile.ZipFile("games/game.zip", 'r') as zip_ref:
    zip_ref.extractall("games/")
with zipfile.ZipFile("reward_model/dems/demonstrations.zip", 'r') as zip_ref:
    zip_ref.extractall("reward_model/dems/")
