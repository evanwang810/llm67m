@echo off
REM Launch the chat tool from anywhere.
REM
REM Copy this one file to a folder on your PATH (C:\Users\evanw\.local\bin is
REM already one) and `llm67m` works from any directory. %~dp0 resolves to the
REM folder holding this script, so the copy still finds the real code as long as
REM it sits next to the repo, which is why the path below is spelled out rather
REM than assumed to be the working directory.
python "%~dp0chat.py" %*
