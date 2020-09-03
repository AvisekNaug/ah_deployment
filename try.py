import subprocess

r = subprocess.run(['python', 'try2.py'])

print(r.returncode)