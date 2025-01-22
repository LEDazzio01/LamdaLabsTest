# LamdaLabsTest

Steps to Run This Code on Lambda Labs:

1. Set Up Your Instance:

   Start an instance (e.g., RTX A6000 or A100) using your Lambda Labs credits.

2. Install Required Libraries: Run the following commands on your instance to set up the environment:

    pip install torch torchvision

3. Run the Script: 

E   xecute the script on the instance:

      python train_cifar10.py

4. Monitor GPU Usage: Use nvidia-smi to monitor GPU utilization during training.
