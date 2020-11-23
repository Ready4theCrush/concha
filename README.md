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

If you want to do more than run simulations and use it to predict how much to make/order for each day, you can set it up to run from your Google Drive. Concha will write a file of predictions to your drive that you can open up with Google Sheets.

## Making Predictions from Your Data

The first step is to save the Google Colab notebooks (a kind of Google Drive file that can run Python code) on your own drive. Then you can set up access to 1.) The NOAA weather data, and 2.) Your Square data (to get your sales history.) The "setup_do_once" notebook shows exactly how it works and automates the process. [setup_do_once](https://colab.research.google.com/github/Ready4theCrush/concha/blob/master/notebooks/02_setup_do_once.ipynb)

Once you have setup access to your data and the weather, the model can learn from your sales history and predict the optimal quantity to produce by running code in the [everyday](https://colab.research.google.com/github/Ready4theCrush/concha/blob/master/notebooks/03_setup_do_once.ipynb) notebook (you don't need to run it *every* day - predictions go out six days). 
  
### Local Installation 

`pip install concha`
   
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
