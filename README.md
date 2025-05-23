# MineRL using Deep Q Learning from Demonstration
This repo is created for an experiment comparing Deep Q Learning from Demonstration, Behavior Cloning (BC), and Proximal Policy Optimization (PPO) Training with MineRLTreeChop_v0 gym environments.

## MineRL Environment Installation ( 0.4.4 )
Install requirements (Java JDK 8 is required. Mac may require additional steps), and then install MineRL clone this repo locally and run the following command

First, make sure to have all the requirements in our requirements.txt file. 
<pre> pip install -r requirements.txt </pre>

You will likely encounter an issue with the installation of MineRL if you are on macOS or Windows. To circumvent this, follow these instructions:

1. First, follow the installation instructions on the official MineRL docs website https://minerl.readthedocs.io/en/v0.4.4/tutorials/index.html

2. Install the following MixinGradle file that is missing from the repo's cache for Malmo to work: https://drive.google.com/file/d/1z9i21_GQrewE0zIgrpHY5kKMZ5IzDt6U/view?usp=drive_link

3. Clone the mineRL repo locally and checkout their v0.4 branch we use. <pre> git clone https://github.com/minerllabs/minerl.git  </pre>

4. Go into the build.gradle file <pre> cd minerl/Malmo/Minecraft </pre>

5. Then update the build.gradle in the following way:

6. Now you should be able to pip install using this directory and have no issues.


##  File Structure
<pre> <code> 
  MineRL_DQfD/ 
          ├── DQfD/ 
                └── main.py # DQfD file algorithm
          ├── Baseline/ 
                └── BC_plus_script.py # BC algorithm
                └── RL_plus_script.py # PPO algorithm
</code> </pre>

# Full Report on mineRL _DQfD_Report.ipynb

## Result
<div style="text-align: center;">

|Algorithm|Average Reward|
|-|-|
BC|11.79 ± 6.504|
PPO|7.77 ± 10.82|
DQfD(our)|1.64 ± 2.56|

</div>
  
From the result, we can see that the best performing algorithm is the Behavior Cloning (BC) followed by PPO then our implementation of DQfD. The agent trained on BC can perform at an average of 11.79 per episode, around 12 logs, while the dataset was provided as demonstration, finishing at 64 logs for every sequence. Given that the BC is a supervised learning, we expect that the model would be overfit to the expert data eventually. This is also reflected in the BC testing reward in which there is a high fluctuation of the reward in each episode. If the environment that the model get tested in bear no similarity to the dataset the model will eventually.

Another algorithm that seems to perform well enough is PPO. With the mean reward at 7.77 or around 8 logs per episode. If we disregard the low number of mean reward and focus on the max reward that the agent trained by PPO can achieved at 35 is much more higher than the max reward obtain by BC (25) or DQfD (10). 

The last one is our implementation of DQfD which has multiple interesting points. First, the pretrain process of our DQfD is as expected the expert loss decreased steadily as it is expected and the pretrained model is able to obtain a reward that is comparable to our BC agent. However, when we proceed to continue and train our agent with DQN the reward went down even though the TD loss and N-Step loss go down as expected. The final mean reward is at 1.64 which far from being considered successful in this environment. 

Focusing in the pretraining stage of DQfD. While the Expert loss and N-step TD loss gradually decrease, the single step TD loss wasn't in a same trend. This emphasize the complexity and sparseness of the environment since while the agent focus on imitating the expert behaviour. The TD loss just continue to increase. 

Given all these result, there are many aspect that this work can be improved. Firstly, the issue may lies im our DQfD replay buffer implementation since we exclude the expert data completely in the training phase. This might be the root cause that the agent performance get deteriorated when approaching the training phase. Another aspect is in the hyperparameter tuning and training duration. Due to time constraint, we are unable to train every algorithm with the scheduler to find an optimal hyperparameter selection. If we could train the RL-based algorithm with more training episode maybe the performance of these RL-algorithm could surpass the BC eventually. Lastly, the environment itself. Since the issue of sparse reward is affecting the RL-algorithm learning, by introducing a new denser reward term such as distance to the nearest tree although not entirely an observation that provide to the player normally could help improving the agent learning. 

