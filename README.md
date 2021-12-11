Food Detection


# How to Quickly set up on Windows
1. Requirements:
    - Windows 10/11 with Python > 3.8
    - Cuda > 10.2
    - Visual Studio 2013-2019
2. Install pycocotools
    - Go to PyPi for pycocotools
    - Download the tar.gz file for pycocotools-2.0.2 (or use the one in the repository)
    - Install manually:
        - `cd pycocotools-2.0.2`
        - `python setup.py build_ext install`

3. Install PyTorch and Torchvision
    - Check your CUDA version
        - `nvcc --version`
    - Go to PyTorch's website for local installation based on CUDA settings
        - Example: `conda install pytorch torchvision torchaudio cudatoolkit=11.3 -c pytorch`

4. Install detectron2
    - ivanpp has a repository with modified patches for windows
    - Execute the following:
        - `git clone https://github.com/ivanpp/detectron2.git`
        - `cd detectron2`
        - `pip install -e .`
    - Check your environment:
        - `python -m detectron2.utils.collect_env`
        - CUDA compiler version should be same ason PyTorch is built for
        
5. Install opencv
    - Return back to root level with `cd ..`
    -  `conda install -c conda-forge opencv`

6. Install pywin32
    - Provides access to Windows API
    - `conda install -c conda-forge pywin32`

7. Install scipy
    - Scientific computing python
    - `conda install -c anaconda scipy`
    
10. Execute tensorboard
    - `tensorboard --logdir=log_dir`