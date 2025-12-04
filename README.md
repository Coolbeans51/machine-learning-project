# machine-learning-project
The source code folder will contain 2 separate folders, each containing the training data, testing data and the Python program used to perform either the classification or the spam email detection.
## Setup & Execution
- First, download and access the source code folder using the code editor Visual Studio Code.
- Next, create a virtual environment to test the code:
  - Start by clicking the search bar above the code, and select “Show and Run Commands”. Then, click “Python: Select Interpreter”. For environment type, select “Venv”.
  - Next, select the interpreter path, “.venv\Scripts\python.exe”
  - If “requirements.txt” shows up, be sure to check that.
  - It will install everything in requirements.txt into the .venv
- After the venv is finished installing, type “.venv/Scripts/Activate.ps1” into the terminal to activate the virtual machine. 
- Change directories to whichever program you want to test first, either classification or spam email detection
- Then, run the python file in that folder. Compare the results of the programs to the actual results. 
- The classification results are in the files ThekveliPredictions1.txt, ThekveliPredictions2.txt, ThekveliPredictions3.txt and ThekveliPredictions4.txt. Each predictions file corresponds to a test data file (e.g. TestData1.txt to ThekveliPredictions1.txt). 
- The spam email detection results are in the file ThekveliSpam.txt, which corresponds to spam_test.csv.
- After you are finished with testing out all the programs., while in the virtual environment, type “deactivate” in the terminal to exit the virtual environment.
