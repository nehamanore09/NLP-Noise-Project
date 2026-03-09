@echo off
echo Activating environment...
call nlp_noise_env\Scripts\activate
echo Running 75.5%% accuracy demo...
python improvements.py
pause
s