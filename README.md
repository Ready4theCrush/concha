# Concha

_A machine learning system for how deciding many things to make each day._

Cafes, grocery stores, restaurants, donut shops, and panaderias face a fundamental
 question every morning: _How many should I make?"_
 
 Concha uses data tracked by the point of sale service,
   combined with local weather conditions to learn demand patterns.
   Then it predicts how much to make of each product to maximize profit.
   
Concha can interface with Square to get the sales history. It takes about 10 minutes to set up.
   
## Try it out

You can run concha entirely on Google Colab (a free deep learning platform).
[Run a concha simulation in a Colab notebook](https://colab.research.google.com/github/Ready4theCrush/concha/blob/master/notebooks/01_run_a_simulation.ipynb) A Colab "notebook" is a bunch of code blocks you can run one by one by clicking the play button in the upper left corner (or by typing CTRL-ENTER).

If you want to do more than run simulations and use it to predict how much to make/order for each day, you can run it from your Google Drive. Concha will save a file of predictions to your drive that you can open up with Google Sheets.

[This Medium article is a complete guide to setting up Concha and running it on Colab.](https://medium.com/conchaml/machine-learning-for-how-many-donuts-bagels-etc-to-make-each-day-15e41a1eb86f)

## Making Predictions from Your Data

The first step is to save the Google Colab notebooks (a kind of Google Drive file that can run Python code) on your own drive. Then you can set up access to 1.) The NOAA weather data, and 2.) Your Square data (your sales history.) The [setup_do_once](https://colab.research.google.com/github/Ready4theCrush/concha/blob/master/notebooks/02_setup_do_once.ipynb) notebook shows exactly how it works and automates the process.

Once you have setup access to your data and the weather, the model can learn from your sales history and predict the optimal quantity to produce by running code in the [make predictions](https://colab.research.google.com/github/Ready4theCrush/concha/blob/master/notebooks/03_everyday_use.ipynb) notebook. The predictions go out six days (the limit of the weather predictions).
  
## Local Installation 

`pip install concha`
   
## Package Layout
The source code is in [/src/concha](/src/concha).
 - [importers.py](/src/concha/importers.py) defines the Square SDK agent.
 - [planner.py](/src/concha/planner.py) defines the planner.
 - [model.py](/src/concha/model.py) defines the different estimators.
 - [product.py](/src/concha/product.py) defines product objects and methods.
 - [weather.py](/src/concha/weather.py) defines the NOAA agent.
 
The code is documented thoroughly, and you can see many other many other 
settings that can be expirmented with to optimize production planning.

## Usage Guides
These notebooks walk through how to use concha.
 - [01_run_a_simulation](/notebooks/01_run_a_simulation.ipynb)
 - [02_setup_do_once](/notebooks/02_setup_do_once.ipynb)
 - [03_everyday_use](/notebooks/03_everyday_use.ipynb)

## Note

This project has been set up using PyScaffold 3.2.3 and the [dsproject extension] 0.4.
For details and usage information on PyScaffold see https://pyscaffold.org/.

[conda]: https://docs.conda.io/
[pre-commit]: https://pre-commit.com/
[Jupyter]: https://jupyter.org/
[nbstripout]: https://github.com/kynan/nbstripout
[Google style]: http://google.github.io/styleguide/pyguide.html#38-comments-and-docstrings
[dsproject extension]: https://github.com/pyscaffold/pyscaffoldext-dsproject
