LunarLander-v3 deep reinforcement learning for DAT255 project.

Documentation: **https://gymnasium.farama.org/**

To run this code:

Make sure to have a supported version of python and an environment with gymnasium and box2D, pytorch and pygame installed. If you need help with this, ask chatGPT like i did :)

To install run the following commands in console (not powershell):

   < pip install swig
   
   < pip install gymnasium[box2d]
   
   < pip install pygame
   
   < pip install pytorch

   There might be a dependency on other imports, but these can easily be installed in the same way if prompted to do so.

Gymnasium is not fully supported on windows. I encountered a common issue with box2d when running the command << pip install gymnasium[all] >>, but by trying again after running << pip install swig >> it worked.
