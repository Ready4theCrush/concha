import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import ShuffleSplit, KFold


class Product:
    """Product object for all operations on an individual product"""

    def __init__(self, id, batch_size=1, batch_cost=1, unit_sale_price=1.3):
        """Sets up the product parameters. The actual model is attached to self.model at the planner level.

        Args:
            id (str): The identifier for the product. Set to be the same as the key in the planner.products dict.
            batch_size (int): Smallest group of products made at once.
            batch_cost (float): The cost to produce one batch of size batch_size.
            unit_sale_price (float): The sale price of an individual product.

        Attributes Set:
            marginal_cost (float): The marginal cost to produce an individual unit.

        """
        self.id = id
        self.batch_size = batch_size
        self.batch_cost = batch_cost
        self.unit_sale_price = unit_sale_price
        self.marginal_cost = float(batch_cost) / batch_size

    def get_settings(self):
        """Returns price attributes as dict."""

        return {
            "batch_size": self.batch_size,
            "batch_cost": self.batch_cost,
            "unit_sale_price": self.unit_sale_price,
        }

    def score_model(
        self, target="estimated_demand", true_demand="estimated_demand", cv=5
    ):
        """Use cross validation to determine the performance of a production prediction model on the product demand history.

        Args:
            target (str): The demand field on which the model should be trained. Options:
                "estimated_demand": Demand estimated from existing demand on stockout days.
                "measured_demand": Only measured sales (if the product ever sells out, this metric is negatively biased.)
            true_demand (str): The demand field against which the production values will be evaluated. Options:
                "estimated_demand": The demand estimated from transactions of how many sales were possible.
                "demand": Only applies to when data was simulated and the number of possible sales that would have
                    taken place without supply limits is recorded.
            cv (int): Number of cross validation folds. Example: 4 means the history will be split into 4 parts - 3
                will be used to train the model and the predictions will be evaluated on the last fourth. This is repeated
                another 3 times so that each fourth gets a chance to be evaluated as the validation data.

        Returns:
            score (Dict): A record of many metrics from the cross validation evaluation process.

        """

        # Get the product history and drop the date and any possible demand columns
        records = self.history
        X = records.drop(
            columns=["date", "estimated_demand", "measured_demand", "demand"],
            errors="ignore",
        )

        # Get the specified column as the target data for training
        y = records[[target]]

        # Get the specified column for evaluation of production predictions
        demand = records[[true_demand]]

        # set up lists for recording values for each fold of the cross validation
        score = {
            "val_loss": [],
            "days": [],
            "demand": [],
            "produced": [],
            "revenue": [],
            "cost": [],
            "sales": [],
            "profit": [],
            "waste": [],
        }

        # split is done in groups, but could be switched to random samples
        #         splitter = ShuffleSplit(n_splits=cv, test_size=1/cv)
        splitter = KFold(n_splits=cv, shuffle=False)
        for train_indices, test_indices in splitter.split(X):
            X_train, X_test = X.iloc[train_indices], X.iloc[test_indices]
            y_train, y_test = y.iloc[train_indices], y.iloc[test_indices]
            y_true = demand.iloc[test_indices].to_numpy()
            score["demand"].append(y_true)

            # only look for fit_history if epochs is present, i.e. if model is a deep learning model.
            if hasattr(self.model, "epochs"):
                self.model.fit(
                    X_train, y_train, validation_data=(X_test, y_test), reset_model=True
                )
                score["val_loss"].append(self.model.fit_history["val_loss"])
            else:
                self.model.fit(X_train, y_train)

            # Get productions on test set.
            production = self.model.predict(X_test)

            # Get total produced over all days.
            produced = np.sum(production)
            score["produced"].append(produced)

            # Find actual sales
            sales = np.sum(np.min(np.hstack([y_true, production]), axis=1))
            score["sales"].append(sales)
            revenue = sales * self.unit_sale_price
            score["revenue"].append(revenue)
            cost = np.sum(production) / self.batch_size * self.batch_cost
            score["cost"].append(cost)
            score["days"].append(X_test.shape[0])
            profit = revenue - cost
            score["profit"].append(profit)
            waste = produced - sales
            score["waste"].append(waste)

        total_days = np.sum(score["days"])
        score["revenue_avg"] = np.sum(score["revenue"]) / total_days
        score["profit_avg"] = np.sum(score["profit"]) / total_days
        score["profit_margin"] = (
            np.sum(score["profit"]) * 100.0 / np.sum(score["revenue"])
        )
        score["waste_pct"] = np.sum(score["waste"]) * 100 / np.sum(score["produced"])
        score["waste_avg"] = np.sum(score["waste"]) / total_days
        return score

    def train_model(self, records, target="estimated_demand"):
        """Train the product.model on the given dataframe with features and target.

        Args:
            records (pd.DataFrame): The history on which to train the model features are
                a subset of the columns, and the target is also a column.
            target (str): The column to use for the target for training the model.
                "estimated_demand": The estimated of actual demand based on transaction timing each day.
                "measured_demand": A count of actual sales of the product per day.

        """
        X = records.drop(
            columns=["date", "estimated_demand", "measured_demand", "demand"],
            errors="ignore",
        )
        y = records[[target]]
        self.model.fit(X, y)

    def predict_production(self, records):
        """Create predictions of optimal production from a set of features.

        Args:
            records (pd.Dataframe): A dataframe that includes all the metadata used to train the model.

        Attributes Set:
            daily_forecast_production (pd.DataFrame): The features provided for production with a column "production"
                added with the predicted production values.

        Returns:
            daily_forecast_production (pd.DataFrame): same
        """
        X = records.copy()
        X = X.drop(
            columns=["date", "estimated_demand", "measured_demand", "demand"],
            errors="ignore",
        )
        predicted = self.model.predict(X)
        X["production"] = predicted
        self.daily_forecast_production = X
        return X

    def estimate_demand(self, num_quantiles=24):
        """Analyzes a transactions history to estimate fulfilled and missed sales (demand) for each date.

        Args:
            num_quantiles (int): The number of quantiles in which to divide the intraday transaction times
                in order to predict total demand. More could mean less biased estimates, but possibly less
                stable.

        Attributes Set:
            daily_demand_history (pd.DataFrame): see below
        Returns:
            daily_demand_history (pd.DataFrame): The date, the measured number of sales ("measured_demand") and the
                "estimated_demand" or the extrapolated number of sales given when supply ran out.
        """

        # get the minute of each transaction in order to divide the transactions into quantiles
        # based on the minute within the day in which they occurred.
        trans = self.transactions.copy()

        # train quantile bins on dates least likely to have a stockout (days with latest sales)

        # find the lastest minute of a sale for each date.
        by_max_minute = trans.groupby("date")["minute"].max().to_frame()

        # Divide all the dates into five groups.
        by_max_minute["qnt"] = pd.qcut(
            by_max_minute["minute"], 5, labels=False, duplicates="drop"
        )

        # Use the dates with the latest minute of sale to figure out the intraday demand curve for the product.
        train_dates = by_max_minute[by_max_minute["qnt"] == 4].index

        # find quantile bins based on 20% of dates with longest time before stockout
        ser, bins = pd.qcut(
            trans[trans["date"].isin(train_dates)]["minute"],
            num_quantiles,
            retbins=True,
            labels=False,
            duplicates="drop",
        )
        # make sure the first quantile includes all early times
        bins[0] = float(0)

        # Bin all transactions into the quantiles. Days with stockouts likely won't have sales in the latest quantiles.
        trans["quantile"] = pd.cut(
            trans["minute"], bins=bins, labels=False, include_lowest=True
        )

        # Get the number of sales in each quantile by date.
        sales_by_quantile = (
            trans.groupby(["product", "date", "quantile"])["quantity"]
            .sum()
            .reset_index()
        )
        sales_by_quantile["cumulative_sales"] = sales_by_quantile.groupby(
            ["product", "date"]
        )["quantity"].transform(pd.Series.cumsum)

        # Create a linear regression model for extrapolation from each quantile. Example, if the last known sale ocurred at
        # a time in quantile = 3, how many sales would we expect for the rest of the day?
        quantile_models = {}
        by_date = []
        for date, dt_sales in sales_by_quantile.groupby("date"):
            record = {
                "date": date,
                "max_quantile": dt_sales[
                    "quantile"
                ].max(),  # max quantile transaction time.
                "is_all_demand": dt_sales["quantile"].max()
                == num_quantiles - 1,  # if last transaction happened at end of day.
                "max_known_quantile": dt_sales[
                    dt_sales["quantile"] < dt_sales["quantile"].max()
                ]["quantile"].max(),
                # last quantile where we know demand was not cutoff by stockout.
                "known_demand": dt_sales[
                    dt_sales["quantile"] < dt_sales["quantile"].max()
                ][
                    "cumulative_sales"
                ].max(),  # the amount at that quantile.
                "total_demand": dt_sales["cumulative_sales"].max(),
            }

            by_date.append(record)
        by_date = pd.DataFrame(by_date)

        # filter out rows where no quantile is complete, so there's nothing known to extrapolate from
        by_date = by_date[by_date["max_known_quantile"] > 0]

        self.daily_demand_history = by_date
        by_date = by_date.dropna()
        by_date[by_date["is_all_demand"]]

        # get dates where sales kept happening until the end of the day.
        known_dates = by_date[by_date["is_all_demand"]]["date"].unique()
        training_set = sales_by_quantile[sales_by_quantile["date"].isin(known_dates)]

        # For each quantile, look at how many sales have ocurred up to that point, and then train on how many sales
        # actually ended up ocurring that day.
        for quantile in range(1, num_quantiles - 1):
            limited_by_quantile = training_set[
                training_set["quantile"] < quantile
            ]  # train_model_dates[train_model_dates['quantile'] <= quantile]
            X = limited_by_quantile.groupby("date")["cumulative_sales"].max().to_frame()
            y = by_date[by_date["date"].isin(X.index)]["total_demand"]
            lr = LinearRegression()
            lr.fit(X, y)
            quantile_models[quantile] = lr

        # For each date, us recorded sales, unless sales stopped before end of the day, in which case extrapolate
        # total sales based on when the product sold out.
        by_date["estimated_demand"] = by_date.apply(
            lambda row: row["total_demand"]
            if row["is_all_demand"]
            else quantile_models[row["max_known_quantile"]].predict(
                np.array([[row["known_demand"]]])
            )[0],
            axis=1,
        )
        by_date["estimated_demand"] = by_date["estimated_demand"].round()
        by_date["measured_demand"] = by_date["total_demand"]
        self.daily_demand_history = by_date[
            ["date", "estimated_demand", "measured_demand"]
        ]
        return self.daily_demand_history
