\# Important Notes for Collaborators



\### ⚠️ Running the Code

\- \*\*ANTICOR 1\*\* takes more than \*\*8 hours\*\* to run.  

\- \*\*ANTICOR 2\*\* should not be run locally. It will be executed on Columbia’s \*\*HPC cluster\*\* instead.  



Please avoid running these versions on local machines unless strictly necessary.



---



\### 🔑 WRDS Connection

To connect to \*\*WRDS\*\* (since the HPCs do not allow direct login), a file named `wrds\_credentials.txt` must be created in your working directory with the following content:



1\. First line: WRDS username  

2\. Second line: WRDS password  



If you wish to use my version of the code, please create this file with your own credentials.  



Do \*\*not\*\* commit or push this file to GitHub, as it contains confidential information.



---



\### 🚫 About Code Modifications

If you choose to modify the code to connect to WRDS using a different method, do \*\*not\*\* push these changes to GitHub.  

This will prevent merge conflicts when the code is deployed on the HPCs.



---



\### 📊 Data Selection Notes

Please refer to the notebook \*\*`Gathering datas.ipynb`\*\*.  

The objective is to create multiple \*\*portfolios\*\* to analyze how the algorithm behaves depending on the \*\*correlation between assets\*\*.  



You will find analyses illustrating these relationships.  

If you have ideas for alternative portfolio compositions that may add value, you can add new cells directly in `Gathering datas.ipynb`.



---



\### 🗣️ Language

Most of the code and documentation are currently written in \*\*French\*\*.  

They will be translated once development is complete.  

You may also use automated translation tools in the meantime if needed.



