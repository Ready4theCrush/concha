# Concha

_A machine learning system for how deciding many things to make each day._

Cafes, grocery stores, restaurants, donut shops, and panaderias face a fundamental
 question every morning: _How many should I make?"_
 
 Concha uses data tracked by the point of sale service,
   combined with local weather conditions to learn demand patterns.
   Then it predicts how much to make of each product to maximize profit.
   
Concha can interface with Square to get the necessary sales history. It takes about 10 minutes to set up.
   
## Try it out

You can run concha entirely on Google Colab (a free deep learning platform).
[Click here to use concha](https://colab.research.google.com/github/Ready4theCrush/concha/blob/master/notebooks/01_run_a_simulation.ipynb) A "notebook" is a bunch of code blocks you can run one by one by clicking the play button in the upper left corner (or by clicking CTRL-ENTER).

If you want to do more than run simulations and use it to predict how much to make or order for each day, you can set it up to run from your Google Drive. Concha will write a file of predictions to your drive for each product that you can open up with Google Sheets.

## Making Predictions from Your Data

The first step is to save the Google Colab notebooks (a kind of Google Drive file that can run Python code) on your own drive. Then you can set up access to 1.) The NOAA weather data, and 2.) Your Square data (to get your sales history.) The "setup_do_once" notebook shows exactly how it works and automates the process. [setup_do_once](https://colab.research.google.com/github/Ready4theCrush/concha/blob/master/notebooks/02_setup_do_once.ipynb)

Once you have setup access to your data and the weather, the model can learn from your sales history and predict the optimal quantity to produce by running code in the [everyday](https://colab.research.google.com/github/Ready4theCrush/concha/blob/master/notebooks/03_setup_do_once.ipynb) notebook (or every few days: predictions go out six days). 
  
### Local Installation 

1. Install [Miniconda](https://docs.conda.io/en/latest/miniconda.html)
(or [Anaconda](https://docs.anaconda.com/anaconda/install/), which is prettier, but uses 500 MB) - it will handle making
all the package versions line up. You will need the "Python 3.x" version, and the 64 bit version (The last 32 bit computer was made in 2002). The default options are fine.

2. [Download Concha](https://github.com/Ready4theCrush/concha/archive/master.zip) to
 somewhere convenient on your computer, and unpack the files to a directory named "concha".
  
3. Open up a [conda prompt](https://docs.anaconda.com/anaconda/install/verify-install/#:~:text=Windows%3A%20Click%20Start%2C%20search%2C,Applications%20%2D%20System%20Tools%20%2D%20terminal.).
 (I realize "conda" and "concha" sound very similiar, sorry!)
 
 Running the following commands will get concha running.
 
4. Navigate to the concha directory:
    ```
    cd [path_to_concha]/concha
    ```
5. Create the conda environment for concha (all the right versions of packages). 
    ```
    conda env create -f environment.yaml
    ```  
6. Activate the new environment.
   ```
   conda activate concha
   ```
7. Install the concha code.
   ```
   python setup.py develop
   ```
8. Make it so Jupyter Lab knows to use the concha environment.
   ```
   ipython kernel install --user --name=concha
   ```
9. Start up Jupyter lab.
    ```
    jupyter lab
    ```
    You should see the files on the left side. Navigate to the
    "notebooks" folder and open up the [run_a_simulation](/notebooks/01_run_a_simulation.ipynb).
10. Check Jupyter Lab is using the concha kernel: it should say "concha" by a little circle in the upper right corner. If it isn't, Set the kernel by going to Kernel" on the menu bar, then pick "Change Kernel..." 
on the bottom. Choose "concha" from the dropdown list and hit "Select".

The next time you want to predict how much product to make, just open the conda prompt, go to the concha
folder and run the command `jupyter lab`. It should open to the right notebook
and be using the right kernel.
   
## Package Layout
The source code is in [/src/concha](/src/concha).
 - [planner.py](/src/concha/planner.py) defines the planner
 - [model.py](/src/concha/model.py) defines the different estimators
 - [product.py](/src/concha/product.py) defines product objects and methods.
The code is documented thoroughly, and you can see many other many other 
settings that can be expirmented with to optimize production planning.

## Note

This project has been set up using PyScaffold 3.2.3 and the [dsproject extension] 0.4.
For details and usage information on PyScaffold see https://pyscaffold.org/.

[conda]: https://docs.conda.io/
[pre-commit]: https://pre-commit.com/
[Jupyter]: https://jupyter.org/
[nbstripout]: https://github.com/kynan/nbstripout
[Google style]: http://google.github.io/styleguide/pyguide.html#38-comments-and-docstrings
[dsproject extension]: https://github.com/pyscaffold/pyscaffoldext-dsproject
