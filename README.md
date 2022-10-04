# CCPT
Official codebase for *"Automatic Gameplay Testing and Validation with Curiosity-Conditioned Proximal Trajectories"* paper.
Alessandro Sestini, Linus Gisslén, Joakim Bergdahl, Konrad Tollmar, and Andrew D. Bagdanov.

A link to our paper can be found in [arxiv](https://arxiv.org/pdf/2202.10057)

<p align="center">
    <img src="imgs/teasing.png" width="800">
</p>

# Prerequisites
* The code was tested with **Python v3.6**.
* To install all required packages:
```
cd CCPT
pip install -r requirements.txt
```
* To donwload the environment and the demonstrations:
```
python download_data.py
sudo chmod +x games/PlayTesting_Final.x86_64
```
# Instruction
* To train the agent to playtest the environment, use the command:
```
python train_ccpt.py -ga=1 -dm=dem_platest_1 -gn=games/PlayTesting_Final -mn=platest_1
```
where ```-ga``` specifies the goal area we want to test, ```-dm``` the name of the demonstrations to use, ```-gn``` 
the name of the environment and ```-mn``` the name with which we save the results.