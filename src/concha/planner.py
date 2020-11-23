import os
import pandas as pd
import numpy as np
from scipy import stats
from scipy.stats import norm, skewnorm
from datetime import datetime, date, timedelta, timezone
from dateutil import parser
import pytz

from sklearn.model_selection import ParameterGrid

import matplotlib.pyplot as plt
import seaborn as sns

from concha import Product
from concha import ProfitMaximizer, QuantileRegressor, Mean, MeanWeekPart
from concha.environment import FileHandler
from concha.weather import NOAA
from concha.importers import Square

rgen = np.random.default_rng()

###########- PLANNER CLASS -###########


class Planner:
    """Top level object for creating optimization models"""

    def __init__(
        self,
        planner_name="example",
        estimate_missed_demand=True,
        model="ProfitMaximizer",
        model_layers=4,
        model_width=20,
        dropout=0.0,
        l2_penalty=0.001,
        epochs=200,
        model_batch_size=15,
        round_to_batch=True,
        demand_quantile=0.9,
        verbose=0,
        categorical_feature_cols=None,
        demand_estimation_quantiles=10,
        **product_settings,
    ):
        """Creates a planner object to learn from past sales of products, then predict optimal production numbers.

        Args:
            planner_name (str): The name of the folder (.../concha_planners/[planner_name]) where all sales transactions
                csv files should be placed for import, and where the settings file is written.
            estimate_missed_demand (bool): If true, estimates of actual demand constructed from transactions are
                used in training the prediction models. Defaults to True.
            model (str): The model to construct production predictions. Options:
                "ProfitMaximizer" (default): Maximizes profit by product batch_size, batch_cost, and unit_sale_price.
                "QuantileRegressor": Predicts production value to center on the demand_quantile level of demand.
                    Setting demand_quantile=0.9 means meeting or exceeding demand 90% of the time given the
                    training conditions (day of week, weather).
                "MeanWeekPart": Finds a mean of sales for weekdays and weekends separately, and uses them for predictions of future production.
                "Mean": Finds a mean of all past measured sales by day and uses it as prediction for production.
            model_layers (int): The number of dense layers in the multi-layer-perceptron models use by deep learning models. Higher
                makes the model able to understand complex relationships, lower is faster. Defaults to 4.
            model_width (int): The number of units in each densely connected neural network layer. Higher makes model able to
                "understand" more, but slows down training. Defaults to 20.
            dropout (float): Value between 0.0 and 1.0 for use in dropout layers. Giving non-zero values will slow down training,
                but may improve quality of patterns learned by the models. Defaults to 0.0 (which, at 0.0 means this isn't used by default).
            l2_penalty (float): l2_regularization paramater applied to each dense layer. Higher slows down training, and increases loss function
                values, and may improve what model can achieve. Default is 0.001.
            epochs (int): The maximum number of epochs (training steps) used in training deep learning models. The models use early stopping,
                so this is just an upper limit. Defaults to 200.
            model_batch_size (int): Size of the batches used in training models. Not to be confused with product batch_size, which is
                the size of product batches (i.e. six muffins per tray). Defaults to 15.
            round_to_batch (bool): Only applies to ProfitMaximizer model. True just rounds the optimal production regression output to
                the nearest batch size (so if batch_size=5, 12.2321 would be rounded to 10 units). When False, another deep learning model
                is trained to decide whether or not to round the regression output up or down to the nearest batch size. Defaults to True.
            demand_quantile (float): Value between 0.0 and 1.0. Only used by QuantileRegressor model. This is how much of demand to predict
                for production. Defaults to 0.9.
            verbose (0, 1, or 2): Verbosity of training process. 1 and 2 print a line for each epoch. Defaults to 0.
            categorical_feature_cols (List([str,])): Specifies which columns of features dataframe should be consired categorical
                and one hot encoded. If set to None, all fetures with less than 12 or fewer unique values are treated as categoricals.
                Defaults to None.
            demand_estimation_quantiles (int): Used when estimating demand from sales transactions.
                The number of quantiles in which to divide up transaction timestamps in order to project total demand
                for days when it seems supply ran out early (stockout days). Less is possibly more biased, but more
                stable. Defaults to 24.
        """
        self.planner_name = planner_name
        self.products = {}
        self.estimate_missed_demand = estimate_missed_demand
        self.model = model
        self.model_layers = model_layers
        self.model_width = model_width
        self.dropout = dropout
        self.l2_penalty = l2_penalty
        self.epochs = epochs
        self.model_batch_size = model_batch_size
        self.round_to_batch = round_to_batch
        self.demand_quantile = demand_quantile
        self.verbose = verbose
        self.categorical_feature_cols = categorical_feature_cols
        self.demand_estimation_quantiles = demand_estimation_quantiles
        self.product_settings = product_settings

        # These attributes track the columns in the transaction csv(s).
        self.time_column, self.product_column, self.quantity_column = None, None, None

        self.filehandler = FileHandler()
        self.planner_dir, self.settings_path = self.filehandler.check_planner_path(
            self.planner_name
        )

        # Creates a planner_settings.json file, or updates, if it already exists.
        self.update_settings()

        # Set up the weather if the right info is available
        if "weather" in self.settings:
            if self.settings["weather"]["type"] == "noaa":
                self.weather = NOAA(name=self.settings["weather"]["name"])

        if "importer" in self.settings:
            if self.settings["importer"]["type"] == "square":
                self.importer = Square(name=self.settings["importer"]["name"])

    ###########- TOP LEVEL METHODS -###########

    def train(self):
        """Wrapper for setup and train steps. This assumes each product already has the correct settings
        specified in ...concha_planners/[planner_name]/planner_settings.json."""
        if not hasattr(self, "transactions"):
            self.update_settings()
            self.import_transactions()
            self.update_settings()
        self.generate_daily_history_metadata()
        self.setup_products()
        self.train_models()

    def predict(self):
        """Wrapper for prediction steps

        Returns:
            production (pd.DataFrame): The forecast metadata used for the prediction, and the predicted values
            in the 'production' column.
        """
        self.generate_daily_forecast_metadata()
        production = self.predict_production()
        return production

    def update_history(self, products=None):
        """Wrapper that pulls in new transactions from the importer, then imports them to the planner.

        Args:
            products (list[str], or str): If a list of strings, the transactions are filtered to only
                include the listed products. If a string, only that product's transactions
                are imported. If None (the default), no filter is applied and all products
                available are imported.

        Returns:
            new_transactions (DataFrame): The new batch of transactions written to
                [planner_name]/history
        """

        # Check if an importer is attached first
        if not hasattr(self, "importer"):
            print("There isn't an importer attached to this planner")
            return

        # Get new transactions
        location = self.settings["location"]
        new_transactions = self.importer.get_orders(
            location=location["name"], last_timestamp=location["last_timestamp"]
        )
        most_recent_order = parser.parse(new_transactions.iloc[-1]["timestamp"])
        most_recent_order = most_recent_order.isoformat()

        # Write the history to file.
        current_timestamp = datetime.now().strftime("%Y-%m-%dT%H_%M_%S")
        history_path = os.path.join(
            self.planner_dir, "history", current_timestamp + ".csv"
        )
        new_transactions.to_csv(history_path, index=False)
        self.settings["location"]["last_timestamp"] = most_recent_order
        tz = self.settings["location"]["timezone"]
        self.filehandler.dict_to_file(self.settings, self.settings_path)
        self.import_transactions(products=products, tz=tz)
        self.update_settings()
        return new_transactions

    def import_transactions(
        self,
        products=None,
        time_column=None,
        product_column=None,
        quantity_column=None,
        tz=None,
    ):
        """Imports any csv files of transactions for use.

        All csv files in the history directory of the planner are assumed to be transaction files.
        All files are assumed to be in the same format (data dump from point of sale provider). They can possible overlap
        (duplicate rows are removed).

        Args:
            products (str or list): Optional filter for products imported. If string, only imports that
                product, if list, imports only those products. If none, no filter is applied
                and all products are imported.
            time_column (str): Name of the column with the timestamp for sales transactions used in csv(s).
            product_column (str): Name of the product identifier column.
            quantity_column (str): Name of column listing number of each product sold per timestamp.
            tz (str): Format like "US/Eastern" or "US/Pacific". If set, transactions will be imported at tz.
                If not set, the "square_location" location timezone will be used. Otherwise,
                "US/Eastern" is used.

        Attributes Set:
            transactions (pd.DataFrame): Dataframe of all transactions.

        Returns:
            transactions (pd.DataFrame): Dataframe of all transactions.

        """
        # Set a tz if none provided or in settings.
        if tz is None:
            if "location" in self.settings:
                tz = self.settings["location"]["timezone"]
            else:
                tz = "US/Eastern"

        # get the csv files in the history directory of the planner
        history_path = os.path.join(self.planner_dir, "history")
        self.transaction_csv_paths = [
            os.sep.join([history_path, path])
            for path in os.listdir(history_path)
            if path.endswith(".csv")
        ]

        if len(self.transaction_csv_paths) == 0:
            # Just exit if there are not files found to import
            print("No transaction csv files to import.")
            return

        print("Importing from: " + history_path)
        csv_data = pd.concat([pd.read_csv(path) for path in self.transaction_csv_paths])

        print("Imported csv columns: " + ", ".join(list(csv_data.columns)))
        use_columns = []

        # Check if the column names were specified as arguments, if so use those names.
        # If not, use what was specified in the planner_settings.json file.
        # If both are None, try the first three columns of the csvs.
        strans = self.settings["transactions"]
        if time_column is None and strans["time_column"] is not None:
            time_column = strans["time_column"]
            use_columns.append(time_column)
        if product_column is None and strans["product_column"] is not None:
            product_column = strans["product_column"]
            use_columns.append(product_column)
        if quantity_column is None and strans["quantity_column"] is not None:
            quantity_column = strans["quantity_column"]
            use_columns.append(quantity_column)
        if len(use_columns) < 3:
            use_columns = csv_data.columns[:3]

        # Write the updated transaction columns names back to settings.
        self.settings["transactions"] = {
            "time_column": use_columns[0],
            "product_column": use_columns[1],
            "quantity_column": use_columns[2],
        }
        self.filehandler.dict_to_file(self.settings, self.settings_path)

        csv_data[["timestamp", "product", "quantity"]] = csv_data[use_columns]
        csv_data["timestamp"] = csv_data["timestamp"].astype(str)

        # Get the date and minute in local time for estimating demand
        localtz = pytz.timezone(tz)
        csv_data["local_ts"] = csv_data["timestamp"].apply(
            lambda x: parser.parse(x).astimezone(localtz)
        )
        csv_data["date"] = csv_data["local_ts"].apply(lambda x: x.date())
        csv_data["date"] = pd.to_datetime(csv_data["date"])
        csv_data["minute"] = csv_data.apply(
            lambda row: row["local_ts"].hour * 60 + row["local_ts"].minute, axis=1
        )
        csv_data["product"] = csv_data["product"].astype(str)
        csv_data["quantity"] = pd.to_numeric(csv_data["quantity"])

        # If only one product was passed in as a string - convert it to a list
        if isinstance(products, str):
            products = [products]

        # Filter the transactions to only include the products specified.
        if products is not None:
            csv_data = csv_data[csv_data["product"].isin(products)]

        # Drop duplicate rows. Transaction data dumps are probably specific time periods, so this prevents the user having
        # to figure out which files overlap.
        self.transactions = csv_data.drop_duplicates(ignore_index=True)
        return self.transactions

    def update_settings(self):
        """Syncronizes the values in the settings file with planner object values

        If a ...concha_planners/[planner_name]/planner_settings.json file is not present, this creates one.
        If it is present, this syncs values in the planner with the file.

        Attributes Set:
            settings (Dict): Dict (synced) verson of what's present in the json file.
        """
        # If the file exists, get the current values.
        if os.path.exists(self.settings_path):
            self.settings = self.filehandler.dict_from_file(self.settings_path)
        else:
            # If file isn't present, make a fresh set of values
            self.settings = {
                "transactions": {
                    "time_column": self.time_column,
                    "product_column": self.product_column,
                    "quantity_column": self.quantity_column,
                },
                "product": {},
            }

        # if transactions have been imported, add any new product ids to settings['product']
        if hasattr(self, "transactions"):
            history_products = list(self.transactions["product"].unique())
            for product in history_products:
                # If settings don't exist write default values for a product
                if product not in self.settings["product"]:
                    dummy_prod = Product(product)
                    self.settings["product"][product] = dummy_prod.get_settings()

        self.filehandler.dict_to_file(self.settings, self.settings_path)

    ###########- METADATA METHODS -###########

    def generate_daily_history_metadata(self, load_from_file=False, write_csv=True):
        """Creates a dataframe of features for each date in the transaction history.

        Columns are ['date', 'day_of_week'] if no weather API info is included, and
        ['date', 'day_of_week', 'tmax', 'tmin', 'prcp', 'snow'] if it is included.

        Args:
            load_from_file (bool): True pulls the daily history from ...[planner_name]/metadata/daily_history_metadata.csv.
                False generates the metadata from the transactions and pulls the weather data from NOAA.

        Attributes Set:
            daily_history_metadata (pd.DataFrame): Metadata for each date present in the transactions.

        Returns:
            daily_history_metadata (pd.DataFrame)
        """
        metadata_file_path = os.sep.join(
            [self.planner_dir, "metadata", "daily_history_metadata.csv"]
        )
        if load_from_file:
            # load from daily_history_metadata.csv
            if os.path.exists(metadata_file_path):
                print("Loading history metadata from: " + metadata_file_path)
                self.daily_history_metadata = pd.read_csv(
                    metadata_file_path, parse_dates=["date"]
                )
            else:
                print(
                    f"file path: daily_history_metadata.csv isn't in {os.path.join(self.planner_dir, 'metadata')}."
                )
                print(
                    "Setting load_from_file=False will get historical weather data for dates (if noaa api key provided.)"
                )

            # load from simulated_demand_history.csv
            simulated_demand_path = os.sep.join(
                [self.planner_dir, "metadata", "simulated_demand_history.csv"]
            )
            if os.path.exists(simulated_demand_path):
                print("Loading simulated demand from: " + simulated_demand_path)
                self.simulated_demand_history = pd.read_csv(
                    simulated_demand_path, parse_dates=["date"], dtype={"product": str}
                )
        # generate the daily summaries directly from the transactions and the weather history API
        else:
            # Get the dates listed transactions and create day of week metadata from them.
            dates = pd.to_datetime(self.transactions["date"])
            dates = pd.Series(dates.unique(), name="date")
            dates = dates.to_frame()
            dates = dates.sort_values(by="date")
            dates["day_of_week"] = dates["date"].dt.strftime("%a")

            # Add the weather metadata if added to the planner
            if hasattr(self, "weather"):
                start_date = dates.head(1)["date"].dt.strftime("%Y-%m-%d").values[0]
                end_date = dates.tail(1)["date"].dt.strftime("%Y-%m-%d").values[0]
                station_id = self.settings["weather"]["station"]["id"]
                weather = self.weather.get_weather_history(
                    start_date, end_date, station_id
                )
                dates = dates.merge(weather, on="date")
            self.daily_history_metadata = dates

        if write_csv:
            self.daily_history_metadata.to_csv(metadata_file_path, index=False)

        #         self.daily_history_metadata['date'] = self.daily_history_metadata.dt.date
        return self.daily_history_metadata

    def generate_daily_forecast_metadata(self):
        """Creates a dataframe of features for each date in the coming week.

        Attributes Set:
            daily_forecast_metadata (pd.DataFrame): Dataframe used for making predictions
                of future optimal production. Takes same form as the daily_history_metadata:
                ['date', 'day_of_week']  and optionally ['tmax', 'tmin', 'prcp', 'snow'].

        Returns:
            daily_forecast_metadata (pd.DataFrame)

        """

        # Get next ten days
        today = date.today()
        dates = pd.Series(
            [today + timedelta(days=x) for x in range(10)], name="date"
        ).to_frame()
        dates["date"] = pd.to_datetime(dates["date"])
        dates["day_of_week"] = dates["date"].dt.strftime("%a")

        # Get the forecast at the location of the NOAA station
        if hasattr(self, "weather"):
            forecast_url = self.settings["weather"]["forecast_url"]
            forecast = self.weather.get_weather_forecast(forecast_url)
            dates = dates.merge(forecast, on="date")

        # Only provide a forecast for columns present in the history, if present
        if hasattr(self, "daily_history_metadata"):
            history_columns = self.daily_history_metadata.columns
            dates = dates[history_columns]
        self.daily_forecast_metadata = dates
        return self.daily_forecast_metadata

    ###########- PREDICTION METHODS -###########

    def setup_products(self):
        """Creates Product objects for each product present in settings and attaches a Model object to each.

        Attributes Set:
            products (Dict): Each key is the name of the product (whatever appeared in the transactions
            data). The value is the Product object.
        """

        for product, attr in self.settings["product"].items():
            # if a set of settings was provided, use that for all the products (used for evaluating models)
            if self.product_settings:
                prod = Product(product, **self.product_settings)
            else:
                prod = Product(product, **attr)
            prod.transactions = self.transactions[
                self.transactions["product"] == product
            ]

            # If a product has settings, but no transactions, it's not added to the planner
            if prod.transactions.shape[0] < 1:
                continue

            # Estimate demand from intraday transactions per product.
            demand_estimate = prod.estimate_demand(
                num_quantiles=self.demand_estimation_quantiles
            )

            # If simulated_demand_history exists, get the history for this particular product.
            if hasattr(self, "simulated_demand_history"):
                sim_demand = (
                    self.simulated_demand_history[
                        self.simulated_demand_history["product"] == product
                    ]
                    .copy()
                    .drop(columns=["product"])
                )

                # Merge in the estimated demand, and set daily_demand_history
                demand_estimate = demand_estimate.merge(sim_demand, on="date")
                prod.daily_demand_history = demand_estimate

            # history is merge of all demand histories
            prod.history = self.daily_history_metadata.merge(demand_estimate, on="date")

            # Set up the models according to top level specification
            if self.model == "ProfitMaximizer":
                prod.model = ProfitMaximizer(
                    batch_size=prod.batch_size,
                    batch_cost=prod.batch_cost,
                    unit_sale_price=prod.unit_sale_price,
                    model_layers=self.model_layers,
                    model_width=self.model_width,
                    dropout=self.dropout,
                    l2_penalty=self.l2_penalty,
                    epochs=self.epochs,
                    model_batch_size=self.model_batch_size,
                    round_to_batch=self.round_to_batch,
                    categorical_feature_cols=self.categorical_feature_cols,
                    verbose=self.verbose,
                )

            if self.model == "QuantileRegressor":
                prod.model = QuantileRegressor(
                    batch_size=prod.batch_size,
                    demand_quantile=self.demand_quantile,
                    model_layers=self.model_layers,
                    model_width=self.model_width,
                    dropout=self.dropout,
                    l2_penalty=self.l2_penalty,
                    epochs=self.epochs,
                    model_batch_size=self.model_batch_size,
                    categorical_feature_cols=self.categorical_feature_cols,
                    verbose=self.verbose,
                )

            if self.model == "Mean":
                prod.model = Mean(batch_size=prod.batch_size)
            if self.model == "MeanWeekPart":
                prod.model = MeanWeekPart(batch_size=prod.batch_size)

            # Only add daily_forecast_metadata to product if it exists in the planner.
            if hasattr(self, "daily_forecast_metadata"):
                prod.daily_forecast_metadata = self.daily_forecast_metadata
            self.products[product] = prod

    def train_models(self):
        """Set off training for all products"""

        if self.estimate_missed_demand:
            target = "estimated_demand"
        else:
            target = "measured_demand"

        for prod_name, prod in self.products.items():
            print(f"Training model for product: {prod_name}")
            prod.train_model(prod.history, target=target)

    def predict_production(self, sort_by_date=False):
        """Create production forecasts for each product and combine into a record.

        Args:
            sort_by_date (bool): True sorts the combined dataframe of predictions for each
                date by date, instead of by product.

        Attributes Set:
            forecast_production (pd.DataFrame): Columns are ['date', 'product', ...metadata..., 'production']
                for all products, together.

        Returns:
            forecast_production (pd.DataFrame)
        """
        forecasts = []
        for prod_name, prod in self.products.items():
            print(f"Predicting production for product: {prod_name}")
            forecast = prod.predict_production(self.daily_forecast_metadata)
            forecast.insert(0, "product", prod_name)
            forecast.insert(0, "date", self.daily_forecast_metadata["date"])
            forecasts.append(forecast)

        # combine all product forecasts
        current_date = str(date.today())
        self.forecast_production = pd.concat(forecasts)
        if sort_by_date:
            self.forecast_production = self.forecast_production.sort_values(by="date")
        self.forecast_production.to_csv(
            os.sep.join([self.planner_dir, "forecast", f"{current_date}_forecast.csv"]),
            index=False,
        )
        return self.forecast_production

    ###########- ANALYSIS METHODS -###########

    def score_products(
        self,
        target="estimated_demand",
        true_demand="estimated_demand",
        cv=5,
        verbose=True,
    ):
        """Analyze performance of each product given the prediction model.

        For each product, do a [cv] fold cross validation where a the model is trained on the training
        set and then performance is evaluated on the validation set. This combines the analysis for each
        product individually into one dataframe. It also averages the validation loss function histories
        of all trainings to give a sense of the loss curve over epochs. Finally, more raw data from each
        fold is retained to do more detailed analysis for comparison of models. (That would mostly be
        used with simulated data - for actual data with heterogeneous products - it would be harder to
        interpret.

        Args:
            target (str): The demand value used to train the models. Possible values are:
                "estimated_demand (default)": Trains models on demand estimated from transaction data.
                "measured_demand": Trains models on recorded sales w/out estimate of missed sales
                    included.
            true_target (str): The values to use for evaluation of predictions. Possible values are:
                "estimated_demand (default)": Evaluate performance of predictions based on the estimated demand
                    for each day. This would be used with real world data where there is no way to know
                    how much product would have actually sold if a stockout had not occurred.
                "demand": When the transaction data is simulated, it is known how many possible sales
                    could have ocurred, because they were simulated, and comparison against this actual
                    "demand" value makes sense.
            cv (int): The number of cross validation folds to run for each product score. Default is 5.
            verbose (bool): True prints when each product is being scored. False does not. Default is True.

        Returns:
            product_summaries (pd.DataFrame): Summary of performance of each product using the specified
                model.
            val_loss (np.array): Mean of validation set loss function values for each epoch of training.
            cv_outputs (Dict): Raw outputs from each product score run. Used by grid_search
                for evaluating comparative performance between models.
        """

        # update settings with any transactions first
        self.update_settings()

        # Setup the product objects and store them in the self.products Dict
        self.setup_products()

        # values to record for each product in the product_summaries dataframe
        summary_fields = [
            "revenue_avg",
            "profit_margin",
            "profit_avg",
            "waste_pct",
            "waste_avg",
        ]
        val_loss = []
        prod_summaries = []

        # Values that are tracked per fold of cross validation
        cv_outputs = {"produced": [], "profit": [], "waste": [], "days": []}

        # Loop through products and record score values
        for prod_name, prod in self.products.items():
            if verbose:
                print(f"Scoring product: {prod_name}")
            score = prod.score_model(target=target, true_demand=true_demand, cv=cv)
            for key in cv_outputs.keys():
                # add cv fold results to each of the cv_output fields
                cv_outputs[key].extend(score[key])
            summary = {key: score[key] for key in summary_fields if key in score}
            summary["product"] = prod_name
            prod_summaries.append(summary)

            # Add all val_loss arrays into one list
            if "val_loss" in score:
                val_loss = val_loss + score["val_loss"]

        # create the combined df for all products
        self.product_summaries = pd.DataFrame(prod_summaries)

        # Each val_loss list may be a different length because the deep learning models use early stopping.
        # This pads each array with NaN values so they are all the same length, then finds the nanmean
        if len(val_loss) > 0:
            max_epochs = max([len(vl) for vl in val_loss])
            padded_losses = [
                np.hstack([np.array(vl), np.full(max_epochs - len(vl), np.nan)])
                for vl in val_loss
            ]
            val_loss = np.vstack(padded_losses)
            val_loss = np.nanmean(val_loss, axis=0)
        self.val_loss = val_loss
        return self.product_summaries, self.val_loss, cv_outputs

    def grid_search(
        self, param_grid=None, param_set=None, cv=5, true_demand="estimated_demand"
    ):
        """Evaluates performance over multiple paramater sets to tune models, and compare between them.

        This runs multiple comparisons, then records the results to a dataframe with the parameters
        that were changed as columns and the values used as the field values. Some metrics are recorded as
        scalars, but others are stored as np.arrays inside of a list of lenght one. (This is a hack to get
        pandas to store a np.array as a value.) For parameter set, there are num products x cv runs. So this
        can end up using a lot of processor!

        Args:
            param_grid (Dict): Dictionary of parameters and values. All possible combinations are run.
                Example:
                    {
                        'model': ['ProfitMaximizer', 'MeanWeekPart'],
                        'l2_penalty': [0.001, 0.005]
                    }
                This would run four score_products four times.
            param_set (List[Dict,]): List of parameter sets to run. This does not enumerate through all possible
                values, just runs each dictionary of parameters. Why? This is useful if you want to change batch_size
                and batch_cost together:
                [{'batch_size': 6, 'batch_cost': 18}, {'batch_size': 12, 'batch_cost': 36}]
                This compares efficiency with batch sizes of 6 and 12 with a marginal cost of 3 for each.
            cv (int): Number of folds in cross validation. Passed to score_products.
            true_demand (str): The demand used for evaluating performance. "estimated_demand" for any
                real world data, "demand" is better for any simulated transaction data.

        Attributes Set:
            grid_results (pd.DataFrame): Results of running the score_products, for all products:
                "label": string form of parameters changed for the run (used for plots).
                "profit_avg": The average daily profit for all products and all cv folds.
                "waste_pct": The average daily percentage of production that was not sold.
                "profit": Average daily profit for each individual cv run (a numpy array).
                "val_loss": Average nanmean loss values for each epoch of each run.
                "waste": Average daily wasted products for each run.
        Returns:
            grid_results (pd.DataFrame)
        """
        records = []
        if param_grid is None:
            param_grid = {}
        if param_set is None:
            param_set = [{}]
        for set_params in param_set:
            for grid_params in ParameterGrid(param_grid):
                # combine params set in param_set and param_grid
                params = {**grid_params, **set_params}
                print(f"Evaluating: {params}")
                self.update_settings()
                record = params.copy()
                record_as_string = "_".join(
                    [f"{key}={val}" for key, val in params.items()]
                )
                for key, val in params.items():
                    # set each param value in the planner object
                    setattr(self, key, val)
                self.setup_products()

                # Only use estimated demand for actual ML models, it's not reasonable to
                # assume simple models would have a demand estimation system.
                if getattr(self, "model") in ["ProfitMaximizer", "QuantileRegressor"]:
                    target = "estimated_demand"
                else:
                    target = "measured_demand"
                product_summary, val_loss, scores = self.score_products(
                    cv=cv, target=target, true_demand=true_demand, verbose=False
                )
                record["label"] = record_as_string
                record["profit_avg"] = np.sum(scores["profit"]) / np.sum(scores["days"])
                record["waste_pct"] = (
                    np.sum(scores["waste"]) * 100.0 / np.sum(scores["produced"])
                )
                record["profit"] = [
                    np.around(np.divide(scores["profit"], scores["days"]), decimals=2)
                ]
                record["val_loss"] = [np.around(val_loss, decimals=4)]
                record["waste"] = [
                    np.around(np.divide(scores["waste"], scores["days"]), decimals=2)
                ]
                records.append(record)
        self.grid_results = pd.DataFrame(records)
        return self.grid_results

    def plot_validation_loss(self):
        """Creates a line plot of validation loss(es) labeled with the run parameters."""
        if hasattr(self, "val_loss"):
            plt.plot(self.val_loss)
        if hasattr(self, "grid_results"):
            for idx, row in self.grid_results.iterrows():
                val_loss = row["val_loss"][0]
                if val_loss.shape[0] > 0:
                    plt.plot(val_loss, label=row["label"])
            plt.legend()
        plt.title("Model Training History")
        plt.xlabel("Training Steps (Epochs)")
        plt.ylabel("Validation Loss")

    def plot_cv_results(self, field="profit"):
        """Creates layered histogram plots of array results from grid_search."""
        if hasattr(self, "grid_results"):
            for idx, row in self.grid_results.iterrows():
                vals = row[field][0]
                label = row["label"]
                sns.distplot(vals, bins=10, label=label)
            plt.legend()
            plt.title(f"{field.capitalize()} (all products)")
            plt.xlabel("Cross Validation Results")
            plt.ylabel("Frequency")

    def plot_profits(self):
        """Creates a layered histogram view of profits."""
        self.plot_cv_results(field="profit")

    def plot_wastes(self):
        """Creates a layered histogram view of average daily wastes."""
        self.plot_cv_results(field="waste")

    @staticmethod
    def compare_paired_samples(s1, s2):
        """Evaluate paired t-test and 95 % bounds of the mean of the difference.

        During the grid search, multiple sets of model parameters are tried for
        each cross validation fold for each product. The folds and products are
        checked in the same order, so two different approaches can be compared for
        a significance of difference with a paired t-test. The p_value
        is the probability that the two sets of samples (s1, s2) are drawn from
        the same population. If the p_value is low, ( < .05 is a standard threshold),
        then it is unlikely the the two sets are the same, i.e. the difference
        between them is significant.

        The difference is calculated with pairwise subtraction: s1 - s2. So if the
        bounds of the mean of the difference are positive, then s1 is larger than s2.

        Args:
            s1 (np.Array): Set of scalar values for comparison against s2.
            s2 (np.Array): Set of scalar values of same length as s1.

        Returns:
            p_value (float): Probability the two samples are the drawn
                from the same population
            bounds (Tuple(float)): 95 % Confidence bounds on the
                mean difference (s1 - s2) between populations
        """
        # Find the difference between paired values
        diff = np.subtract(s1, s2)
        n = diff.shape[0]

        # Find the mean difference
        diff_mean = diff.mean()

        # Find the standard deviation of the sample
        diff_std = diff.std(ddof=1)

        # Find the x value at which P(T > x) = .025
        t_alpha2 = stats.t.ppf(0.975, n - 1)
        half_bound = t_alpha2 * diff_std / np.sqrt(n)

        # Get bounds
        bounds = (diff_mean - half_bound, diff_mean + half_bound)
        t_stat = abs(diff_mean / (diff_std / np.sqrt(n)))

        # get prob of standardized mean of differences if sample are from samp pop
        p_value = 2 * stats.t.cdf(-t_stat, n - 1)
        return p_value, bounds

    def compare_grid_results(self):
        """Get pairwise paired t-test and mean of difference bounds for all params
            run during grid_search.

        Returns:
            comparisons (pd.DataFrame): columns:
                params_1: String description of first params set
                params_2: String description of second params set
                profit_p_value: pairwise t-test between cross validaiton fold avg profits
                profit_difference_bounds: 95% confidence bound on mean of difference.
                waste_p_value: pairwise t-test between cross validation fold avg wastes
                waste_difference_bounds: 95% confidence bound on mean of difference of avg wastes.
        """
        if hasattr(self, "grid_results"):
            comparisons = []
            for idx_1, row_1 in self.grid_results.iterrows():
                for idx_2, row_2 in self.grid_results.iterrows():
                    if idx_2 > idx_1:
                        profit_p_value, profit_bounds = self.compare_paired_samples(
                            row_1["profit"][0], row_2["profit"][0]
                        )
                        waste_p_value, waste_bounds = self.compare_paired_samples(
                            row_1["waste"][0], row_2["waste"][0]
                        )
                        comparison = {
                            "params_1": row_1["label"],
                            "params_2": row_2["label"],
                            "profit_p_value": np.round(profit_p_value, 5),
                            "profit_difference_bounds": np.round(profit_bounds, 3),
                            "waste_p_value": np.round(waste_p_value, 5),
                            "waste_difference_bounds": np.round(waste_bounds, 3),
                        }
                        comparisons.append(comparison)
            comparisons = pd.DataFrame(comparisons)
            return comparisons

    ###########- SIMULATION METHODS -###########

    def simulate_history(
        self,
        num_days=90,
        demand_mean=100,
        demand_std=10,
        num_products=5,
        start_hour=6,
        end_hour=18,
        stockout_prob=0.5,
        tz=None,
        write_csv=False,
    ):
        """Top level wrapper for whole simulation process.

        Creates a set of metadata history by date, and of simulated transactions for a number of products.

        Args:
            num_days (int): The number of days over which to generate a simulated history.
            demand_mean (float): The mean demand for each day for the product. This isn't exact - the mean is
                affected by the simulated day of the week and weather.
            demand_std (float): The standard deviation of the (Skew) Gaussian random variable used to generate a number for
                 the actual demand per day.
            num_products (int): Number of products to model. Each product is assigned a "type" randomly (uniform) that determines
                how the mean demand is affected by the temperature.
            start_hour (float): Hour (in 24 hr time) when simulated days start. (6.0 would mean 6:00 AM)
            end_hour (float): Hour when simulated days end (17.0 would mean 5:00 PM)
            stockout_prob (float): Probability value between 0 and 1. For the simulated data, the probability
                that any given day is one where demand outstrips supply (stockout). 0.1 means demand is greater than supply
                90% of the time. 0.9 means stockouts occur on 90% of days.
            tz (str): The descriptor for the time zone for the timstamps, which are time zone aware.
            write_csv (bool): True writes daily_history_metadata, simulated_demand_history, and transactions to csv in folder
                ...concha_planners/[planner_name]/metadata
        """

        # Simulates weather over a series of of num_days
        self.simulate_daily_history_metadata(num_days=num_days, write_csv=write_csv)

        # Simulates the demand for each product type (hot weather, cold weather, and balmy weather products)
        self.simulate_daily_demand(
            demand_mean=demand_mean, demand_std=demand_std, num_products=num_products
        )

        # The tricky thing with timezones is that pandas won't convert timestamps in the transactions unless they
        # are all the same timezone. So these simulated transactions should be whatever is set as a kwarg,
        # otherwise it should match whatever the importer location is listed as
        # and then finally "US/Eastern" as a default
        if tz is None:
            if "location" in self.settings:
                tz = self.settings["location"]["timezone"]
            else:
                tz = "US/Eastern"

        # Turns the demand numbers into potential transactions and actual ones limited by supply.
        self.simulate_transactions(
            start_hour=start_hour,
            end_hour=end_hour,
            stockout_prob=stockout_prob,
            tz=tz,
            write_csv=write_csv,
        )

    def simulate_daily_history_metadata(self, num_days=14, write_csv=False):
        """Simulate the weather for num_days number of dates.

        Args:
            num_days (int): Number of days to simulate.
            write_csv: True writes the dates and simulated weather history to daily_history_metadata.csv

        Attributes Set:
            daily_history_metadata (pd.DataFrame): ['date', 'day_of_week', 'tmin', 'tmax', 'prcp', 'snow']
                as columns. Each weather field is drawn randomly.

        """
        dates = pd.date_range(end=pd.Timestamp.today(), periods=num_days, freq="D")
        daily_metadata = []
        for dte in dates:
            # tmin is a Gaussian rv centered on 60.
            tmin = int(rgen.normal(loc=60, scale=15))

            # tmax is a Gaussian rv distance above tmin.
            tmax = int(tmin + rgen.normal(loc=15, scale=10))

            # prcp has a 20% of being True
            prcp = rgen.random() > 0.8

            # snow has a 5% of being True
            snow = rgen.random() > 0.95
            daily_metadata.append(
                {
                    "date": dte.date(),
                    "day_of_week": dte.strftime("%a"),
                    "tmin": tmin,
                    "tmax": tmax,
                    "prcp": prcp,
                    "snow": snow,
                }
            )
        self.daily_history_metadata = pd.DataFrame(daily_metadata)
        self.daily_history_metadata["date"] = pd.to_datetime(
            self.daily_history_metadata["date"]
        )
        if write_csv:
            self.daily_history_metadata.to_csv(
                os.sep.join(
                    [self.planner_dir, "metadata", "daily_history_metadata.csv"]
                ),
                index=False,
            )
        return self.daily_history_metadata

    def simulate_daily_demand(
        self, demand_mean=100, demand_std=10, skew=3, num_products=1
    ):
        """Given a set of simulated metadata, determine the actual demand, drawn from a skew normal distribution.

            demand_mean (float): The (loose) center of random demand.
            demand_std (float): The standard deviation for the skew Gaussian random variable determining demand.
            skew (float): The skew passed to the random variable:
                (https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.skewnorm.html)
            num_products (int): The number of products for which to simulate demand.

        Attributes Set:
            simulated_demand_history (pd.DataFrame): The date, product and level of demand chosen randomly based on
                simulated day_of_week and weather conditions.

        """
        simulated_demand_history = []
        for product in range(1, num_products + 1):

            # Choose one of three product types
            product_type = rgen.integers(low=0, high=3)
            for idx, row in self.daily_history_metadata.iterrows():
                dow = row["date"].weekday()

                # Choose the mean demand to be lowest at start of week, and then to grow until the weekend
                date_mean = (0.65 + dow * 0.1) * demand_mean

                if product_type == 0:
                    # simulates a product people want more of when it's hot, like ice cream
                    temp_effect = (
                        1
                        + (max(row["tmax"], 80) - 80) / 40
                        - (60 - min(row["tmax"], 60)) / 40
                    )
                elif product_type == 1:
                    # simulates something like soup
                    temp_effect = (
                        1
                        + (50 - min(row["tmin"], 50)) / 40
                        - (max(row["tmin"], 60) - 60) / 40
                    )
                else:
                    # something people like most when weather is 78, and less otherwise
                    temp_effect = (
                        1
                        - (78 - min(row["tmax"], 78)) / 30
                        - (max(row["tmax"], 78) - 78) / 30
                    )

                # combine effects and reduced demand by 10% for rain, 20% for snow, and 30% if both occur.
                # "temp_effect" is short for "temperature_effect".
                temp_effect = max(
                    temp_effect - int(row["prcp"]) * 0.1 - int(row["snow"]) * 0.2, 0.1
                )
                adjusted_mean = date_mean * temp_effect
                demand = max(
                    int(skewnorm.rvs(skew, loc=adjusted_mean, scale=demand_std)), 0
                )
                simulated_demand_history.append(
                    {"date": row["date"], "product": str(product), "demand": demand}
                )
        self.simulated_demand_history = pd.DataFrame(simulated_demand_history)
        self.simulated_demand_history["date"] = pd.to_datetime(
            self.simulated_demand_history["date"]
        )
        return self.simulated_demand_history.sort_values(by="date")

    def simulate_transactions(
        self,
        start_hour=6,
        end_hour=18,
        stockout_prob=0.5,
        tz="US/Eastern",
        write_csv=False,
    ):
        """Given a set of demands per product and date, creates a set of timestamps for sales of each product.

        Transactions are modeled as a Poisson process with intervals between transactions having an
        exponential distribution. Also, because the Poisson process doesn't quite land on the same
        demand number created in simulate_daily_demand, the actual number of transactions that ended up
        landing between start_hour and end_hour is recorded as "demand" or the actual number of possible
        sales if supply hadn't run out (stockout).

        Args:
            start_hour (float): Start hour for simulated transactions in local time. (6.5 would mean 6:30 AM)
                Transactions will start after this time for every simulated day.
            end_hour (float): End hour for simulated transactions in 24 hr local time (17.25 would mean 5:15 PM)
                Transactions will end before this time for every simulated day.
            stockout_prob (float): Value between 0.0 and 1.0. For the simulated days, a (constant) supply
                value is chosen such that the probability of a stockout over all days is stockout_prob.
            tz (str): Time zone for timestamps, which are timezone aware.
            write_csv (bool): True writes transactions and simulated_demand_history to csvs of the same name.
                in the "metadata" folder.

        """

        minutes_per_day = (end_hour - start_hour) * 60

        # Keep track of how much the supply should be for each product. Supply is how many items this simulator
        # assumes are available for sale each day. If demand is 80, but only 70 things are made, only 70 transactions
        # are possible.
        supply = {}
        for prod, group in self.simulated_demand_history.groupby("product"):
            # Demand is modeled as a Gaussian, and supply is chosen as the 1 - stockout_prob quantile of demand.
            supply[prod] = int(
                norm(loc=group["demand"].mean(), scale=group["demand"].std()).ppf(
                    1 - stockout_prob
                )
            )

        # For each day, product, and demand, find the total possible transactions, and actual transactions
        # that would have ocurred before supply ran out.
        transactions = []
        updated_demand = []
        for idx, row in self.simulated_demand_history.iterrows():

            # start_naive is a tz naive time at the start hour of business
            start_naive = row["date"] + timedelta(hours=start_hour)
            localtz = pytz.timezone(tz)
            utc = pytz.utc

            # Start is set the start hour at local time, then as UTC.
            # This is essential to account for daylight savings.
            start = localtz.localize(start_naive).astimezone(utc)
            # find the expected transactions per min given the demand, and prevent divide by zeros with +0.01
            lmda = (row["demand"] + 0.01) / minutes_per_day

            # generate the minutes at which transactions occur given the supply that will give the stockout_prob
            # The demand is scaled by 1.5 to make sure there are enough possible transactions to go well past end_hour.
            spacings = rgen.exponential(
                scale=(1 / lmda), size=int(row["demand"] * 1.5)
            ).cumsum()
            
            # Ensure the sign of all the spacings is positive to prevent "leading negative sign" errors.
            spacings = np.absolute(spacings)

            # limit the transactions that occur within business hours
            filt_arr = spacings < minutes_per_day
            spacings = spacings[filt_arr]

            # record how many transactions would have happen without limited supply
            row["demand"] = len(spacings)

            # limit the size by the supply
            spacings = spacings[: supply[prod]]

            # Construct the actual timestamps from the spacings modeled as exponential rvs.
            times = [
                start + pd.Timedelta(f"{i} minutes").round(freq="S") for i in spacings
            ]

            times_naive = [
                start_naive + pd.Timedelta(f"{i} minutes").round(freq="S")
                for i in spacings
            ]

            # make the dataframe and add in the product and quantity columns
            prod_trans = pd.DataFrame({"timestamp": times, "local_ts": times_naive})
            prod_trans["date"] = prod_trans["local_ts"].apply(lambda x: x.date())
            prod_trans["date"] = pd.to_datetime(prod_trans["date"])
            prod_trans["minute"] = prod_trans.apply(
                lambda xrow: xrow["local_ts"].hour * 60 + xrow["local_ts"].minute,
                axis=1,
            )
            prod_trans["product"] = row["product"]
            prod_trans["quantity"] = 1

            transactions.append(prod_trans)
            updated_demand.append(row)

        # combine transactions for each product into one dataframe
        transactions = pd.concat(transactions)

        # store updated demand history and transactions.
        self.simulated_demand_history = pd.DataFrame(updated_demand)
        self.transactions = transactions

        if write_csv:
            self.transactions.to_csv(
                os.sep.join(
                    [self.planner_dir, "metadata", "simulated_transactions.csv"]
                ),
                columns=["timestamp", "product", "quantity"],
                index=False,
            )
            self.simulated_demand_history.to_csv(
                os.sep.join(
                    [self.planner_dir, "metadata", "simulated_demand_history.csv"]
                ),
                index=False,
            )
        return transactions

        ###########- Getter/Setters -###########

    def product(self, *args, batch_size=None, batch_cost=None, unit_sale_price=None):
        """Gets/sets a single product, or returns a list of current products.

        Args:
            product (str): Optional. The product to get/set for which to set
                the properties.
            batch_size (int): The product batch size. (Default: None)
            batch_cost (float): The marginal cost to produce a batch. (Default: None)
            unit_sale_price (float): The sale price of one item. (Default: None)

        Returns:
            Products (list): If no product is provided, a list of all current
                products in the planner are returned.
            Product Settings (Dict): If a product name is specified, but no
                settings, the current settings are returned.
            None: If a product is specified and settings are specified as
                kwargs, the new settings are set and saved to the settings json.
        """
        if len(args) == 0:
            return list(self.settings["product"].keys())
        else:
            product = args[0]
            if product not in self.settings["product"]:
                print("That's not one of the products in the planner")
            elif batch_size is None:
                return self.settings["product"][product]
            else:
                self.update_settings()
                self.settings["product"][product]["batch_size"] = int(batch_size)
                self.settings["product"][product]["batch_cost"] = float(batch_cost)
                self.settings["product"][product]["unit_sale_price"] = float(
                    unit_sale_price
                )
                self.filehandler.dict_to_file(self.settings, self.settings_path)

    ########### Importers Interfaces #############

    def set_importer(self, importer):
        """Set the sales history importer.

        Args:
            importer (importer object): At this point only the Square class is implemented.
            The importer name and type are stored in the planner settings. A new importer
            is instantiated for the planner object when the planner object is created using
            the name specified.
        """
        self.importer = importer
        self.settings["importer"] = {
            "name": importer.settings["name"],
            "type": importer.settings["type"],
        }
        # Set these becuase an importer will be used. Not some other csv with differnt names.
        self.settings["transactions"] = {
            "time_column": "timestamp",
            "product_column": "product",
            "quantity_column": "quantity",
        }
        self.filehandler.dict_to_file(self.settings, self.settings_path)

    def set_location(self, location_name):
        """Set the location of the planner from the existing planners on the Square account.

        Args:
            location_name (str): The "name" of the location from all the locations extant on the
                Square account.

        Returns:
            location (dict): Metadata for the cafe/restaurant location taken from the
                square locations API.
        """

        # Check that an importer is attached.
        if not hasattr(self, "importer"):
            print("This method requires the planner to have an importer attached")
        else:
            # Get the specific location settings from the locations dict
            location = self.importer.settings["locations"][location_name]

            # Because the location is being set. Assume we need to pull as much
            # data as possible, but within 6 months because pre-pandemic numbers
            # won't make sense now.
            half_year_ago = datetime.now(timezone.utc) - timedelta(days=180)
            location["last_timestamp"] = half_year_ago.isoformat()
            self.settings["location"] = location

            # Set the 'location' in the attached planner settings.
            self.filehandler.dict_to_file(self.settings, self.settings_path)
            print(f"location for {self.planner_name} set to {location_name}.")
            return location

    def set_weather_agent(self, agent):
        """Specifies a named weather agent and saves it in the planner's settings.

        Args:
            agent (concha.weather.NOAA class object): The the weather agent which may or may
                not have an api_key set yet.
        """

        # Save the information needed to instantiate the weather agent when the planner object is created.
        self.settings["weather"] = {
            "name": agent.settings["name"],
            "type": agent.settings["type"],
        }
        self.weather = agent

        # Save the settings to file.
        self.filehandler.dict_to_file(self.settings, self.settings_path)

    def set_weather_station(self, lat=None, lng=None):
        """Given a set of coordinates, find the nearest best weather station, then set for the planner.

        If no lat/lng are given, grab one from the attached planner square_location if set.

        Args:
            lat (float): Latitude of the store location.
            lng (float): Longitude of the store location
        """

        # If no coordinates are given, get on from the location assigned to the planner
        # from the square importer.
        if lat is None:
            if "location" in self.settings:
                lat, lng = (
                    self.settings["location"]["lat"],
                    self.settings["location"]["lng"],
                )

        # Get the nearest station
        station = self.weather.get_station(lat=lat, lng=lng)
        self.settings["weather"]["forecast_url"] = self.weather.get_forecast_url(
            station
        )
        self.settings["weather"]["station"] = station

        # Write the station stuff to file
        self.filehandler.dict_to_file(self.settings, self.settings_path)
