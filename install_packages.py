import subprocess

# Read packages from requirements.txt
with open("requirements.txt") as f:
    packages = f.readlines()

# Loop through each package
for package in packages:
    package = package.strip()
    if package:  # Make sure the line is not empty
        try:
            print(f"Attempting to install {package}...")
            # Run pip install for each package
            subprocess.check_call(["pip", "install", package])
            print(f"Successfully installed {package}\n")
        except subprocess.CalledProcessError:
            print(f"Failed to install {package}. Skipping...\n")
