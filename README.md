# Concha

_A machine learning system for how deciding many things to make each day._

Cafes, grocery stores, restaurants, donut shops, and panaderias face a fundamental
 question every morning:
  
  **"How many should I make?"**
 
 Concha uses data tracked by the point of sale service (Toast, Square, Clover, etc),
   combined with the day of the week and local weather conditions to learn demand patterns.
   Then it predicts how much to make of each product to maximize profit.
### Why?
If we knew how much people wanted to buy, the _demand_, this problem would be easy.
 But we don't, demand is random: it can't be known exactly. But we do know
 demand can have patterns: Saturdays might be busier than Tuesdays, soup may sell better on cold
 days than hot ones, some salads sell better on weekdays, but only when it's not too cold, etc. 
 We can approximate the conditional probability
  density of demand for the product (given the day of the week and the weather). 
  
To decide
  how much product to actually make to optimize profit, we also need to consider much it costs to make the
  product and how much it sells for. This tells us how much it costs us to make too little
   vs make too much. If a concha sells for $1, and costs $.30 to make, making one too many 
   costs $.30, making one too few costs $1.
   
## Getting the input data
Almost all transactions are processed electronically, and the point of sale provider keeps track
of the transactions. The transactions can be downloaded from the user's POS system dashboard.
Concha can use the transactions downloaded as a .csv file, with columns in the data for:
* Product name
* Time the product was sold
* Quantity sold

The process is different for every POS provider. Here are some guides:
[Square](https://squareup.com/help/us/en/article/5072-summaries-and-reports-from-the-online-dashboard#download-transactions-history
), [Toast](https://central.toasttab.com/s/article/Automated-Nightly-Data-Export-1492723819691)
, [Clover](https://www.clover.com/help/learn-more-about-the-new-sales-report/#export-sales-reports).

Transactions can also be simulated. This is helpful for getting started and
messing around with analysis tools.  

## Installation

1. Install [Miniconda](https://docs.conda.io/en/latest/miniconda.html)
(or [Anaconda](https://docs.anaconda.com/anaconda/install/), which is prettier, but uses 500 MB) - it will handle making
all the package versions line up.

2. [Download Concha](https://github.com/Ready4theCrush/concha/archive/master.zip) to
 somewhere convenient on your computer, and unpack it if it's in a zip file.
  
3. Open up a [conda prompt](https://docs.anaconda.com/anaconda/install/verify-install/#:~:text=Windows%3A%20Click%20Start%2C%20search%2C,Applications%20%2D%20System%20Tools%20%2D%20terminal.).
 (I realize "conda" and "concha" sound very similiar, sorry!) Then navigate to the concha directory:
    ```
    cd [path_to_concha]/concha
    ```
4. Create the conda environment for concha (all the right versions of packages). 
    ```
    conda env create -f environment.yaml
    ```  
5. Activate the new environment with
   ```
   conda activate concha
   ```
6. Install the concha code with:
   ```
   python setup.py develop
   ```
7. Make it so Jupyter Lab knows to use the concha environment:
   ```
   ipython kernel install --user --name=concha
   ```
8. Start up Jupyter lab:
    ```
    jupyter lab
    ```
    You should see the files on the left side. Navigate to the
    "notebooks" folder and open up [predict_production_guide.ipynb](/notebooks/predict_production_guide.ipynb) .
10. Check Jupyter Lab is using the concha kernel: it should say "concha" by a little circle in the upper right corner. If it isn't, Set the kernel by going to Kernel" on the menu bar, then pick "Change Kernel..."

on the bottom. Choose "concha" from the dropdown list and hit "Select".

The next time you want to predict how much product to make, just open the conda prompt, go to the concha
folder and run the command `jupyter lab`. It should open to the right notebook
and be using the right kernel.
 
 ## Usage
 
 In Jupyter Lab, the notebook has boxes called "cells". You can run a cell by putting the cursor
 in it and typing 
 `Control-Enter`. Even if you don't have any transaction data to work with yet, you can simulate
 transactions with the simulator and see how concha works.
 
 All the usage examples below are already in the [predict_production_guide](/notebooks/predict_production_guide.ipynb)
 notebook.
 
 #### Get started
 This sets up a planner, simulates some transactions, then learns from them
  to make predictions.
 1. Create a planner, and specify the price/cost/batch size for the simulated data.
  Let's simulate a fancy cupcake cafe:
    ```python
    from concha import Planner
  
    sim_planner = Planner(planner_name="sim_cafe", batch_size=8, batch_cost=24.0, unit_sale_price=4.75)
    ```
    We assumed cupcakes were made it batches of 8, that they cost $3 to make,
    and sell for $4.75.
2. Simulate some transaction data
    ```python
    sim_planner.simulate_history(
        num_days=180,
        stockout_prob=0.5,
        demand_mean=100,
        demand_std=10,
        num_products=5
   )    
    ```
   This simulates 180 days of sales for 5 products with mean of 100 sales day. 
3. Create a prediction model for each product.
    ```python
    sim_planner.train()
    ```
4. Predict optimal production for the next week.
    ```python
   sim_planner.predict()
    ```

 #### Set up a planner to learn from real data
 1. Create the planner:
    ```python
    from concha import Planner
    
    planner = Planner(planner_name="this_planner", model="ProfitMaximizer")
    ```
    This will create a new folder in the `.../concha/planners` directory called `this_planner`. 
 2. Drop the transaction csv files into the `.../concha/planers/this_planner` directory.
 3. Tell the planner how to read the transaction csv files.
  Open the `.../this_planner/planner_settings.json` 
 file in a text editor. It looks like this:
 
    ```json
    {
        "weather": {
            "noaa_api_key": null,
            "noaa_station_id": null
        },
        "transactions": {
            "time_column": null,
            "product_column": null,
            "quantity_column": null
        },
        "product": {
            "example_product": {
                "batch_size": 1,
                "batch_cost": 1,
                "unit_sale_price": 1.5
            }
        }
    }
    ``` 
    Let's say the columns in the `sales_history.csv` file downloaded from the POS dashboard
    has columns including, "time_of_sale", "item_code", and "num_sold". You would change
    the `transactions` part of the file to:
    ```json
        "transactions": {
            "time_column": "time_of_sale",
            "product_column": "item_code",
            "quantity_column": "num_sold"
        },    
    ```
    Then save the file.
4. Import the transactions, and update the settings file with all the products.
    ```python
    planner.import_transactions()
    planner.update_settings()
    ```
5. Update the `planner_settings.json` file with the batch_size, batch_cost, 
and unit_sale_price for each product. (Each product listed in the transactions
should now show up in the settings file.)
    ```json
    "product": {
        "example_product": {
            "batch_size": 1,
            "batch_cost": 1,
            "unit_sale_price": 1.5
        },
        "concha_small": {
            "batch_size": 10,
            "batch_cost": 3.0,
            "unit_sale_price": 1.0
        },
        "choc_cake": {
            "batch_size": 1,
            "batch_cost": 18.0,
            "unit_sale_price": 27.50
        }
    }
    ```
    You can delete "example_product" if you want, it won't affect anything either way.

6. Train the model
   ```
   planner.train()
   ```
7. Make predictions for the next week
   ```
   planner.predict()
   ```
   The predictions are saved in the file `.../this_planner/forecast_production.csv` 
   
### Make predictions with an existing planner
1. Create the object for the existing planner.
    ```python
    from concha import Planner
  
    planner = Planner(planner_name="this_planner", model="ProfitMaximizer")
    ```
   This tells Concha which folder to look in for settings and training data.

2. Train and predict
    ```python
    planner.train()
    planner.predict()
    ```
### Use the weather
Concha can ask NOAA (National Oceanic and Atmospheric Administration) for the past weather to train the
planner models, and then integrates the weather predictions to make production forecasts.
1. Get a NOAA API key [here](https://www.ncdc.noaa.gov/cdo-web/token).
 You just give them your email and they send you a key.

2. Find the nearest location tracking weather to the store/grocery/cafe.
    - Go to [Climate Data Online Search](https://www.ncdc.noaa.gov/cdo-web/search).
    - For "Select a Dataset" pick "Daily Summaries"
    - For "Enter a Search Term" put in your city/state name. Hit "SEARCH" and a map should come up.
    - On the left side of the map, above the search results and below the "Search" box,
     click on "More Search Options" and check the "Air Temperature" and "Precipitation" boxes.
     Then click "SEARCH" again.
     - When you find the nearest station, click on it and copy the `ID`. It will look like "GHCND:USC00448084"
     
3. Open the settings file for the planner in `.../concha/planners/this_planner/planner_settings.json` and
put in the NOAA API key and station ID.
    ```json
    "weather": {
        "noaa_api_key": "32pretendUEVWnoaa29Skey02",
        "noaa_station_id": "GHCND:USC00448084"
    },
    ```
   Then save the file. Now every time you run `.train()` on your model, Concha will look up the 
   weather for every date when a sale was listed and learn how weather affects demand.
   
### Analyzing Product Performance
Let's take a look at the cupcake shop simulation and look at each product's
performance using the "ProfitMaximizer" deep learning model.
1. Create a planner and simulate the transactions. 
    ```python
    from concha import Planner
  
    sim_planner = Planner(
        planner_name="sim_cafe",
        batch_size=8,
        batch_cost=24.0,
        unit_sale_price=4.75
    )
    
    sim_planner.simulate_history(
        num_days=180,
        stockout_prob=0.5,
        demand_mean=100,
        demand_std=10,
        num_products=5
    )  
    ```
2. We can run a five fold cross validation on the simulated data to see the 
effective performance of each product. 
    ```python
   sim_planner.score_products()
   sim_planner.product_summaries
    ```
   Scoring takes ~5 times as long for each product than training the prediction model
   because `score_mocel()` is creating 5 models with 4/5 of the transaction
   history and then evaluating the performance on the last 1/5. 
   Metrics with the "_avg" suffix are the average *per day*, "_pct" means "percent". 
   
   The summaries show the profit margin is a function of the price and cost,
   but also the model's ability to optimize production. When the model does a
   better job, the profit margin increases.
3. We can also evaluate the performance of different production prediction
 models. Let's compare using:
    - a quantile regressor predicting the 0.1 quantile of the demand distribution.
    - the mean of demand to predict production.
    - the mean of the weekend and weekday demand to predict production.
    - a model trained to maximize profit.
    
    ```python
    sim_planner.demand_quantile = 0.1
    models =['QuantileRegressor', 'Mean', 'MeanWeekPart', 'ProfitMaximizer']
    sim_planner.grid_search(param_grid={"model": models})
    ```
4. Compare model metrics visually:
    - View the loss function outputs during training of the deep learning models:
    ```python
   sim_planner.plot_validation_loss()
    ```
   - Compare the average daily profits for each product and cross validation fold:
   ```python
   sim_planner.plot_profits()
   ```
   - Compare the average daily waste numbers:
   ```python
   sim_planner.plot_wastes()
   ```
   
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
