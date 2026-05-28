@echo off
"%PYTHON%" -m pip install git+https://github.com/fhvermei/chemprop_solvation
"%PYTHON%" -m pip install .
IF %ERRORLEVEL% NEQ 0 EXIT /B 1
