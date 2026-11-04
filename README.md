# Choroid plexus lipid storage paper
This repository contains the code that accompanies the journal article "__" published in __. Specifically we include Jupyter notebooks to reproduce whole slide image processing, and training of deep learning models. The available code is provided as-is, and does not constitute a full-fledged software package for analysis. 
### System Requirments
This code repository was developed on Ubuntu 20. Hardware requirements for running this code includes a CUDA-compatible GPU. This code has been tested with a Nvidia RTX 3090 GPU. 
Software requirments include the installation of  NVIDIA-Docker. All code was tested on Ubuntu 20.04 LTS.
### Running environment
All code was run using a modified docker image. Running a docker container using this image ensures the code uses the correct system and Python packages. To setup the environment and follow the instructions at https://github.com/peterchang77/install and use the "gpu-full" docker image. After starting the docker container, install additional packages using the provided script (docker-additional-packages) before starting the Jupyterhub server.
### CPEC detection improvements (in development)
These files are additions. The original notebooks are unchanged, so the published results remain reproducible.
- `cpec_tools.py`: detection metrics (precision, recall and F1 at IoU ≥ 0.5, swept over score thresholds), an anchor-fit report, an error viewer, a RetinaNet training generator with box-consistent flip/rotation and colour augmentation, and re-encoding of targets for new anchors.
- `CPEC detection evaluation.ipynb`: evaluates an existing model and shows its false positives and missed CPECs. No retraining needed.
- `CPEC detection RetinaNet training v2.ipynb`: retrains with the augmentation, optional new anchors, a learning-rate schedule, and checkpoint selection by F1.
- `tests/test_cpec_tools.py`: tests the tools against a mock box encoder (`python tests/test_cpec_tools.py .`; needs no GPU or TensorFlow).

Place `cpec_tools.py` next to the notebooks, and run them in the same Docker environment as the originals.
