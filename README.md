## Concha

_A machine learning system for how deciding many things to make each day._

Cafes, grocery stores, restaurants, donut shops, and panaderias face a fundamental
 question every morning: _How many should I make?"_
 
 Concha uses data tracked by the point of sale service (Toast, Square, Clover, etc),
   combined with local weather conditions to learn demand patterns.
   Then it predicts how much to make of each product to maximize profit.
   
### Try it out

You can run concha entirely on Google Colab (a free deep learning platform).
[Click here to use concha](https://colab.research.google.com/github/Ready4theCrush/concha/blob/master/notebooks/concha_examples.ipynb)

If you want to do more than run simulations, you can copy the notebook to your drive, and attach your Google Drive
 to the Colab notebook. It will save everything you do in a `concha_planners` folder so you can come back to it later.
 
 In Jupyter Lab/Colab, the notebook has boxes called "cells". You can run a cell by putting the cursor
 within it and typing 
 `Control-Enter`. When the code is running, an asterisk/spinny thing appears in the upper left of the cell, and then a number
 takes its place when it's done.
 
 ### Usage
 
 This sets up a planner, simulates some transactions, then learns from them
  to make predictions.
 1. Create a planner, and specify the price/cost/batch size for the simulated data.
    Let's simulate a fancy cupcake cafe.
    ```python
    from concha import Planner
  
    sim_planner = Planner(planner_name="sim_cafe", batch_size=8, batch_cost=24.0, unit_sale_price=4.75)
    ```
    This assumes cupcakes are made in batches of 8, that they cost $3 to make,
    and sell for $4.75.
2. Simulate some transaction data.
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
 1. Create the planner.
    ```python
    from concha import Planner
    
    planner = Planner(planner_name="this_planner", model="ProfitMaximizer")
    ```
    This will create a new folder in the home directory called `~/concha_planners/this_planner`. 
 2. Drop the transaction csv files into the `~/concha_planers/this_planner/history` directory.
 3. Tell the planner how to read the transaction csv files.

    Let's say the columns in the `sales_history.csv` file downloaded from the POS dashboard
    has columns including, "time_of_sale", "item_code", and "num_sold". You can import the transactions
    and tell the model which columns are which with:
    ```python
    planner.import_transactions(
        time_column="time_of_sale",
        product_column="item_code",
        quantity_column="num_sold"
    )
    
    #This writes the column names to file so you don't have to do it again.
    planner.update_settings()
    ```
 4. To optimize profit, the model needs to know gross margin of each product. You can see the products
 that were imported with:
    ```python
    # This creates the products
    planner.setup_products()
    
    #This returns a list of the product names
    sim_planner.product()
    ```
    
    You can see the current settings for a product:
    ```python
    planner.product("productname")
    ```
    Then you can set the production batch size, batch cost to produce, and item sale price for each one:
    ```python
    planner.product(
        "productname",
        batch_size = 4,
        batch_cost = 8.0,
        unit_sale_price = 3.0
    )
    ```
    This is for a product made in batches of 4, that costs $8 to make per batch, and sells for $3 each.

6. Train the model.
   ```
   planner.train()
   ```
7. Make predictions for the next week.
   ```
   planner.predict()
   ```
   The predictions are saved in the file `.../this_planner/forecast/forecast_production.csv`.
   
### Make predictions with an existing planner
1. Create the object for the existing planner.
    ```python
    from concha import Planner
  
    planner = Planner(planner_name="this_planner", model="ProfitMaximizer")
    ```
   This tells Concha which folder to look in for settings and training data.

2. Train and predict.
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
     
3. Set the API key and the best weather station:
    ```python
    planner.noaa_key("Yourkeyhere")
    planner.noaa_station("GHCND:USC00448084")
    ```
    Now every time you run `.train()` on your model, Concha will look up the 
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
4. Compare model metrics visually.
    - View the loss function outputs during training of the deep learning models.
    ```python
    sim_planner.plot_validation_loss()
    ```
    - Compare the average daily profits for each product and cross validation fold.
    ```python
    sim_planner.plot_profits()
    ```
    - Compare the average daily waste numbers.
    ```python
    sim_planner.plot_wastes()
    ```
5. Compare model metrics with a paired t-test.
   ```python
    sim_planner.compare_grid_results()
   ```  
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
    "notebooks" folder and open up [concha_examples.ipynb](/notebooks/concha_examples.ipynb) .
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
