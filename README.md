# 1. Install required tools (if not already installed)
sudo apt install python3-venv python3-full -y

# 2. Create a virtual environment
python3 -m venv ultralytics-env

# 3. Activate the virtual environment
source ultralytics-env/bin/activate

# 4. Now install ultralytics inside the venv
pip install ultralytics
