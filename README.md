# GalaxyCollider
The culmination of 3 months of blood, sweat, and tears (mostly tears, though) to smash together N number of particles gravitationally. Can also make pretty pictures and videos.

Example run command:
  python3 pm2d.py ./MassivePerturb_Setup.dat leapfrog 5 0.1 0.05 10 512 0.2 OutputPrefix
  
This command will use the massive perturber data file uploaded to this repository to run a particle-mesh gravitational simulation using a leapfrog integration. This runs the code for t=5 with a dt=0.1, softening parameter of 0.05, with gridsize of 10 split into 512 bins. Diagnostic files will be spit out every dt=0.2 with the prefix "OutputPrefix".
