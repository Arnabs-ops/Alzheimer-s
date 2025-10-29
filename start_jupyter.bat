@echo off
cd /d "C:\Users\Arnab Sahu\OneDrive\Desktop\Alzhemiers"
call .venv\Scripts\activate.bat
jupyter notebook --no-browser --port=8888
