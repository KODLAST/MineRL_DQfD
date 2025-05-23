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

# Full Report on mineRL_DQfD_Report.ipynb
