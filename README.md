# Final project

### Team Member
- Рахарди Сандикха РИМ-130908
- Мухин Виктор Александрович РИМ-130908

### ==============================================================

## How to launch the application

### Step 1: Cloning the repository
First you need to clone the repository using the command:
```bash
git clone https://github.com/kuldii/program_engineer_final_project_2.git
```

### Step 2: Install dependencies
Go to your project directory and install the required dependencies using the command:
```bash
pip install --upgrade pip
```
```bash
pip install -r requirements.txt
```

### Step 3: Make Sure You Already Installed ffmpeg
```bash
# on Ubuntu or Debian
sudo apt update && sudo apt install ffmpeg

# on Arch Linux
sudo pacman -S ffmpeg

# on MacOS using Homebrew (https://brew.sh/)
brew install ffmpeg

# on Windows using Chocolatey (https://chocolatey.org/)
choco install ffmpeg

# on Windows using Scoop (https://scoop.sh/)
scoop install ffmpeg
```
Because we are using openai-whisper that need ffmpeg, or you can read pre-requirement before used this apps
https://github.com/openai/whisper?tab=readme-ov-file

### Step 3: Launch the application
To run the application, enter the following command in the terminal:
```bash
streamlit run main.py
```

### Step 4: Open the application in the browser
Open your web browser and go to the following address:
http://localhost:8501