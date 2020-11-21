import numpy as np
import tensorflow as tf

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.initializers import he_uniform
from tensorflow.keras.callbacks import EarlyStopping
import tensorflow.keras.backend as K

from sklearn.dummy import DummyRegressor
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer

# Setting verbosity to "ERROR", otherwise, TF gives a thousand
# impossible to track down "WARNING" level notifications.
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)


class DeepLearningEstimator:
    """Base class for deep learning estimators.

    There are some static methods useful for both, and they are
    defined in this class.
    """

    def __init__(self):
        pass

    @staticmethod
    def create_mlp_regression_model(num_units, num_layers, dropout, l2_penalty):
        """Creates a multilayer perceptron model of num_units per layer, with num_layers.

        Args:
            num_units (int): The number of units in each Dense layer (or the "width").
            num_layers (int): The number of densely connected layers in the model (the "depth").
            dropout (float): Rate of dropout between 0.0 and 1.0 for
                a dropout layer placed after every dense layer.
                If set to 0.0, the layer isn't added.
            l2_penalty (float): The l2 regularization factor for each
                dense layer. Smaller adds less to the loss function and
                affects the model less. More slows learning but can
                prevent overfitting.

        Returns:
            model (tensorflow.keras.Sequential): The Sequential keras model.
        """
        model = Sequential()
        for i in range(num_layers):
            model.add(
                Dense(
                    num_units,
                    activation="relu",
                    kernel_initializer=he_uniform(),
                    kernel_regularizer=tf.keras.regularizers.l2(l2_penalty),
                )
            )
            if dropout > 0.0:
                model.add(Dropout(dropout))
        model.add(Dense(1))
        return model

    @staticmethod
    def get_output_at_layer(model, X, layer_index=0):
        """Get the values at a layer in the model.

        This is used for debugging/understanding of the models. It can be used for
        figuring out if layers concatenated correctly.

        Args:
            model (keras.Sequential): The model from which to grab
                unit values.
            X (keras input, np.array in this case): The input tensor/array to train the model.
            layer_index (int): The layer at which to get the output of the model.

        Returns:
            layer_output (array): The layer values at after model is trained with X.

        """
        get_n_layer_output = K.function(
            [model.layers[0].input], [model.layers[layer_index].output]
        )
        layer_output = get_n_layer_output([X])
        return layer_output[0]

    @staticmethod
    def prepare_features(features_df, categorical_feature_cols=None):
        """Possibly infer categorical columns, then one hot encode them, or standardize the others.

        Args:
            features_df (pd.DataFrame): The features on which to infer the data type (categorical, numeric),
                and to train the preprocessors.
            categorical_feature_cols (List): The columns that should be treated as categorical columns.

        Returns:
            converted_features (np.Array): The features converted by the preprocessors. This is ready
                for use as an input into the deep learning models.
            feature_converter (sklearn.compose.ColumnTransformer): A fitted transformer that one-hot encodes
                the categorical columns, and uses StandardScalar on the numeric columns.
        """

        if categorical_feature_cols is None:
            # Infer which columns should be categoricals based on number of unique values present.
            num_unique = features_df.nunique()
            categorical_feature_cols = list(num_unique[num_unique < 13].index)
            categorical_feature_cols = [
                col for col in categorical_feature_cols if col not in ["tmin", "tmax"]
            ]

        continuous_feature_cols = [
            col for col in features_df.columns if col not in categorical_feature_cols
        ]

        feature_converter = ColumnTransformer(
            [
                (
                    "cat_converted",
                    OneHotEncoder(sparse=False, handle_unknown="ignore"),
                    categorical_feature_cols,
                ),
                ("cont_converted", StandardScaler(), continuous_feature_cols),
            ]
        )
        converted_features = feature_converter.fit_transform(features_df)
        return converted_features, feature_converter

    @staticmethod
    def get_layer_weights(model):
        return [layer.get_weights() for layer in model.layers]


class ProfitMaximizer(BaseEstimator, RegressorMixin, DeepLearningEstimator):
    """Model to maximize profit for random demand."""

    def __init__(
        self,
        batch_size=1,
        batch_cost=1,
        unit_sale_price=2,
        model_layers=4,
        model_width=20,
        dropout=0.0,
        l2_penalty=0.001,
        epochs=20,
        model_batch_size=20,
        round_to_batch=False,
        categorical_feature_cols=None,
        verbose=0,
    ):
        """Set up the parameters, all of which have reasonable default values.

        Args:
            batch_size (float): The size of batches when the product is produced, e.g. muffins are
                made in a tin with six slots, conchas on a tray of 20, etc.
            batch_cost (float): The cost to produce one batch, e.g. takes 10 min + $5 of supplies to make
                 a tray of conchas, so cost is (1/6 hour)*($30/hour labor) + $5 = $10, so 10.0.
            unit_sale_price (float): The sale price of one unit. Conchas sell for $1 each, so 1.0.
            model_layers (int): The number of dense layers in the deep learning model.
            model_width (int): The numer of units per dense layer in the model.
            dropout (float): dropout rate between 0.0 and 1.0. Higher prevents overfitting more,
                but also slows down training, and may not imporove performance.
            l2_penalty (float): The l2 regularization factor for each dense layer. Smaller adds less to the
                loss function and affects the model less. More slows learning but can prevent overfitting.
            epochs (int): The upper limit on number of training iterations to take. The model uses early stopping
                so it will cease training when the validation loss stops going down.
            model_batch_size (int): The size of the the batches over which the training data is evaluated. Higher is faster,
                perhaps slower convergence of model.
            round_to_batch (bool): True rounds the optimal production value, which is a float, to the nearest batch of production.
                If batch_size = 10, this would just round to 0, 10, 20, etc. False creates a deep learning model to decied whether
                to round up or down.
            categorical_feature_cols (List[str,]): A list of columns in the features dataframe which should be treated as categorical
                variables.
            verbose (int): 0 is no output from training of models. 1, 2 show progress for each epoch.
        """
        self.batch_size = float(batch_size)
        self.batch_cost = float(batch_cost)
        self.unit_sale_price = float(unit_sale_price)
        self.marginal_cost = self.batch_cost / self.batch_size
        self.model_layers = model_layers
        self.model_width = model_width
        self.dropout = dropout
        self.l2_penalty = l2_penalty
        self.epochs = epochs
        self.model_batch_size = model_batch_size
        self.round_to_batch = round_to_batch
        self.categorical_feature_cols = categorical_feature_cols
        self.verbose = verbose

        # Create loss function to maximize profit given gross margin of the product.
        self.regr_loss_ = self.create_maximize_profit_regression_loss_function(
            self.marginal_cost, self.unit_sale_price
        )

        # Create the deep learning model
        self.regression_model_ = self.create_mlp_regression_model(
            self.model_width, self.model_layers, self.dropout, self.l2_penalty
        )

        # Compile the keras model.
        self.regression_model_.compile(loss=self.regr_loss_, optimizer="adam")

        if self.round_to_batch:
            return

        # If round_to_batch = False, we need to make a second deep learning model.
        # This one learns the optimal direction to round (up or down) given
        # the output of the regression model.

        # Create the optimal rounding direction loss function
        self.optimal_rounding_fn_ = self.make_rounding_direction_function(
            self.batch_size, self.batch_cost, self.unit_sale_price
        )

        # Create discretizer (rounding) model
        self.discretizer_model_ = self.create_mlp_regression_model(
            self.model_width, self.model_layers, self.dropout, self.l2_penalty
        )

        # Compile the discretizer model
        self.discretizer_model_.compile(loss="mae", optimizer="adam")

    def fit(self, X, y=None, validation_data=None, reset_model=False):
        """Transforms features and target, then fits.

        Args:
            X (pd.DataFrame): Features on which to train the model.
            y (pd.Series): The "target" column from the history dataframe used as a target.
            validation_data (Tuple(pd.DataFrame, pd.Series)): The same as X, and y, just the sets
                used for validation. If None, then early stopping uses the training loss. (not the
                'val_loss').

        Attributes Set:
            X_converter_ (sklearn.compose.ColumnTransformer): Fitted transformer for features.
            y_converter_ (sklearn.preprocessing.StandardScaler): Fitted target transformer.
            regression_model_ (keras.Sequential): Trained optimal production regression model.
            fit_history (np.Array): History 'val_loss' or 'loss' of training regressor model.
            discretizer_model_ (keras.Sequential): Trained optimal round-up profit delta model.
        """

        # Convert y to numpy array
        y_original = y.to_numpy().astype(np.float32)

        # Get transformed training features and the ColumnTransformer
        X_conv, self.X_converter_ = self.prepare_features(
            X, categorical_feature_cols=self.categorical_feature_cols
        )

        # Tensorflow complains if the input is float64, so we change it.
        X_conv = X_conv.astype(np.float32)

        # Scale the target.
        self.y_converter_ = StandardScaler()
        y_conv = self.y_converter_.fit_transform(y_original)

        if validation_data is not None:
            # Scale the validation data using the transformers fitted on the training set.
            X_conv_test = self.X_converter_.transform(validation_data[0])
            y_conv_test = self.y_converter_.transform(validation_data[1])
            validation_data = (X_conv_test, y_conv_test)
            earlystopping_monitor = "val_loss"
        else:
            earlystopping_monitor = "loss"

        # callback checks if the model should stop early
        callback = EarlyStopping(monitor=earlystopping_monitor, patience=7)

        # If each fit needs to be from scratch, reconstruct and recompile the model
        if reset_model:
            self.regression_model_ = self.create_mlp_regression_model(
                self.model_width, self.model_layers, self.dropout, self.l2_penalty
            )
            self.regression_model_.compile(loss=self.regr_loss_, optimizer="adam")

        history = self.regression_model_.fit(
            X_conv,
            y_conv,
            epochs=self.epochs,
            batch_size=self.model_batch_size,
            validation_data=validation_data,
            callbacks=[callback],
            verbose=self.verbose,
        )
        self.fit_history = history.history

        # Get the fractional optimal production value
        regr_prediction = self.y_converter_.inverse_transform(
            self.regression_model_.predict(X_conv)
        )

        # We're done if round_to_batch == True
        if self.round_to_batch:
            return self

        ################ This part only for round_to_batch == False #########################

        # If round_to_batch == False, a model is trained to predict the delta in profit if we rounded up.
        disc_target = self.optimal_rounding_fn_(y_original, regr_prediction)

        # find how how far the regression output is between batch sizes
        batch_fraction = (
            regr_prediction / self.batch_size
            - np.floor(regr_prediction / self.batch_size)
            - 0.5
        )

        # concat the original inputs, and the batch_fraction
        disc_input = np.hstack([X_conv, batch_fraction])

        # Create the earlystopping callback
        disc_callback = EarlyStopping(monitor="loss", patience=7)

        # Reset the discretizer model so it's a fresh train (for cross validation usually)
        if reset_model:
            self.discretizer_model_ = self.create_mlp_regression_model(
                self.model_width, self.model_layers, self.dropout, self.l2_penalty
            )
            self.discretizer_model_.compile(loss="mae", optimizer="adam")

        # Train the discretizer model
        self.discretizer_model_.fit(
            disc_input,
            disc_target,
            epochs=self.epochs,
            batch_size=self.model_batch_size,
            callbacks=[disc_callback],
            verbose=self.verbose,
        )

        return self

    def predict(self, X):
        """Predict optimal production, then also round up/down predictions to nearest batch.

        Args:
            X (pd.DataFrame): The metadata over which to make optimal production predictions.

        Returns:
            batch_rounded_production (np.Array): The production values predicted in the same
                order as the input features dataframe.
        """

        # tranform features using fit from training data
        X_conv = self.X_converter_.transform(X).astype(np.float32)

        # get the optimal production float.
        regr_prediction = self.y_converter_.inverse_transform(
            self.regression_model_.predict(X_conv)
        )
        self.regression_prediction_ = regr_prediction

        # Round it to nearest batch
        if self.round_to_batch:
            return np.round(regr_prediction / self.batch_size) * self.batch_size

        # If round_to_batch = False, combine the batch fraction and training data.
        batch_fraction = (
            regr_prediction / self.batch_size
            - np.floor(regr_prediction / self.batch_size)
            - 0.5
        )
        disc_input = np.hstack([X_conv, batch_fraction])
        disc_input = tf.convert_to_tensor(disc_input)
        disc_prediction = self.discretizer_model_.predict(disc_input)
        self.discretizer_prediction_ = disc_prediction

        # Get production of rounded only down.
        rounded_down_output = (
            np.floor(regr_prediction / self.batch_size) * self.batch_size
        )

        # if discretizer predicts more profit from rounding up, round up to next batch.
        batch_rounded_prediction = np.where(
            disc_prediction <= 0,
            rounded_down_output,
            rounded_down_output + self.batch_size,
        )
        return batch_rounded_prediction

    @staticmethod
    def create_maximize_profit_regression_loss_function(marginal_cost, sale_price):
        """Given a marginal_cost, sale_price, create a standard loss function that is the inverse of profit.

        Args:
            marginal_cost (float): Cost to produce one unit. (batch_cost/batch_size)
            sale_price (float): Price of one unit.

        Returns:
            loss (function): A function that finds the inverse of profit.
        """

        @tf.function
        def loss(y_true, y_pred):
            """Profit maximizing loss function.

            Args:
                y_true (tf.Tensor): The target values used for training.
                y_pred (tf.Tensor): The current predicted optimal production values.

            Returns:
                loss (float): The mean of the negative of profit for the given y_true, y_pred.
            """

            # find how many could have sold (the minimum of the number produced and number people wanted) * price
            gross_profit = K.minimum(y_true, y_pred) * np.array([sale_price])

            # Cost only depends on number produced, not how many sold.
            production_cost = y_pred * np.array([marginal_cost])
            return K.mean(production_cost - gross_profit)

        return loss

    @staticmethod
    def make_rounding_direction_function(batch_size, batch_cost, unit_sale_price):
        """Create function to find profit of rounding up vs. down.

        Args:
            batch_size (int): Size of the production batches.
            batch_cost (float): Marginal cost to produce a batch.
            unit_sale_price (float): What you get from selling a unit.

        Returns:
            The difference in profit between rounding up vs rounding down.
        """

        @tf.function
        def make_target(y_true, y_regr):
            """The delta in profit from rounding up.

            Args:
                y_true: The actual demand.
                y_regr: The predicted optimal production amount.

            Returns:
                relative_profit (np.Array): The profit delta from rounding up for each y_true, y_regr comparison.
            """

            # Amounts of production if rounded down and up
            lower_output = K.round(y_regr / batch_size - 0.5) * batch_size
            upper_output = lower_output + batch_size

            # profits from each scenario given target demand
            lower_profit = K.minimum(y_true, lower_output) * np.array(
                [unit_sale_price]
            ) - lower_output / batch_size * np.array([batch_cost])
            upper_profit = K.minimum(y_true, upper_output) * np.array(
                [unit_sale_price]
            ) - upper_output / batch_size * np.array([batch_cost])
            relative_profit = upper_profit - lower_profit
            return relative_profit

        return make_target


class QuantileRegressor(BaseEstimator, RegressorMixin, DeepLearningEstimator):
    """Quantile Regression to find the production that will be the nth quantile of demand"""

    def __init__(
        self,
        batch_size=1,
        demand_quantile=0.9,
        model_layers=4,
        model_width=20,
        dropout=None,
        l2_penalty=0.01,
        epochs=20,
        model_batch_size=20,
        categorical_feature_cols=None,
        verbose=0,
    ):
        """Sets up the parameters to create quantile regression model.

        Args:
            batch_size (int): Size of production batches.
            demand_quantile (float): Value between 0.0 an 1.0. The proportion of times demand exceeds production
                for the model given metadata for each date.
            model_layers (int): The number of dense layers in the deep learning model.
            model_width (int): The numer of units per dense layer in the model.
            dropout (float): dropout rate between 0.0 and 1.0. Higher prevents overfitting more,
                but also slows down training, and may not imporove performance.
            l2_penalty (float): The l2 regularization factor for each dense layer. Smaller adds less to the
                loss function and affects the model less. More slows learning but can prevent overfitting.
            epochs (int): The upper limit on number of training iterations to take. The model uses early stopping
                so it will cease training when the validation loss stops going down.
            model_batch_size (int): The size of the the batches over which the training data is evaluated. Higher is faster,
                perhaps slower convergence of model.
            round_to_batch (bool): True rounds the optimal production value, which is a float, to the nearest batch of production.
                If batch_size = 10, this would just round to 0, 10, 20, etc. False creates a deep learning model to decied whether
                to round up or down.
            categorical_feature_cols (List[str,]): A list of columns in the features dataframe which should be treated as categorical
                variables.
            verbose (int): 0 is no output from training of models. 1, 2 show progress for each epoch.
        """
        self.batch_size = batch_size
        self.demand_quantile = demand_quantile
        self.model_layers = model_layers
        self.model_width = model_width
        self.dropout = dropout
        self.l2_penalty = l2_penalty
        self.epochs = epochs
        self.model_batch_size = model_batch_size
        self.verbose = verbose
        self.categorical_feature_cols = categorical_feature_cols

        # Create mlp model.
        self.regression_model_ = self.create_mlp_regression_model(
            self.model_width, self.model_layers, self.dropout, self.l2_penalty
        )

        # Create the loss function that weights the losses to find the right quantile.
        self.regr_loss_ = self.create_quantile_regression_loss_function(
            self.demand_quantile
        )

        # Compile the model.
        self.regression_model_.compile(loss=self.regr_loss_, optimizer="adam")

    def fit(self, X, y=None, validation_data=None, reset_model=False):
        """Transforms features and target, then fits.

        Args:
            X (pd.DataFrame): Features on which to train the model.
            y (pd.Series): The "target" column from the history dataframe used as a target.
            validation_data (Tuple(pd.DataFrame, pd.Series)): The same as X, and y, just the sets
                used for validation. If None, then early stopping uses the training loss. (not the
                'val_loss').

        Attributes Set:
            X_converter_ (sklearn.compose.ColumnTransformer): Fitted transformer for features.
            y_converter_ (sklearn.preprocessing.StandardScaler): Fitted target transformer.
            regression_model_ (keras.Sequential): Trained quantile production regression model.
            fit_history (np.Array): History 'val_loss' or 'loss' of training regressor model.
        """

        # Train ColumnTransformer for features and StandardScaler for target.
        y_original = y.to_numpy().astype(np.float32)
        X_conv, self.X_converter_ = self.prepare_features(
            X, categorical_feature_cols=self.categorical_feature_cols
        )
        X_conv = X_conv.astype(np.float32)
        self.y_converter_ = StandardScaler()
        y_conv = self.y_converter_.fit_transform(y_original)

        if validation_data is not None:
            X_conv_test = self.X_converter_.transform(validation_data[0]).astype(
                np.float32
            )
            y_conv_test = self.y_converter_.transform(validation_data[1]).astype(
                np.float32
            )
            validation_data = (X_conv_test, y_conv_test)
            earlystopping_monitor = "val_loss"
        else:
            earlystopping_monitor = "loss"

        # If reset_model = True, recompile to reset weights in the model each time it's fit.
        if reset_model:
            self.regression_model_.compile(loss=self.regr_loss_, optimizer="adam")

        callback = EarlyStopping(monitor=earlystopping_monitor, patience=7)
        history = self.regression_model_.fit(
            X_conv,
            y_conv,
            epochs=self.epochs,
            batch_size=self.model_batch_size,
            validation_data=validation_data,
            callbacks=[callback],
            verbose=self.verbose,
        )
        self.fit_history = history.history

    def predict(self, X):
        """Predict optimal production, then also round up/down predictions to nearest batch.

        Args:
            X (pd.DataFrame): The metadata over which to make optimal production predictions.

        Returns:
            batch_rounded_production (np.Array): The production values predicted in the same
                order as the input features dataframe.
        """
        X_conv = self.X_converter_.transform(X).astype(np.float32)
        regr_prediction = self.y_converter_.inverse_transform(
            self.regression_model_.predict(X_conv)
        )
        rounded_to_batch = np.round(regr_prediction / self.batch_size) * self.batch_size
        return rounded_to_batch

    @staticmethod
    def create_quantile_regression_loss_function(quantile):
        """Creates the quantile loss function.

        Args:
            quantile (float): Proportion between 0.0 and 1.0 of demand probability
                density to predict. 0.5 predicts the median (half the time demand
                is lower, half the time it's higher).

        Returns:
            loss (function): loss function for quantile regression.
        """

        @tf.function
        def loss(y_true, y_pred):
            """Quantile loss function. The higher the quantile,
                the more the model penalizes predicting too low,
                and the less it penalizes being too high.

            Args:
                y_true (tf.Tensor): Target demand
                y_pred (tf.Tensor): Predicted demand

            Returns:
                loss (float): Mean of loss for the batch
            """
            error = y_true - y_pred
            return K.mean(K.maximum(quantile * error, (quantile - 1) * error))

        return loss


class Mean:
    """Finds the mean of all the days in the history and uses them as the prediction"""

    def __init__(self, batch_size=1):
        self.batch_size = batch_size
        self.regressor_ = DummyRegressor(strategy="mean")

    def fit(self, X, y=None):
        self.regressor_.fit(X, y)

    def predict(self, X):
        predictions = self.regressor_.predict(X)
        predictions = np.round(predictions / self.batch_size) * self.batch_size
        return np.expand_dims(predictions, -1)


class MeanWeekPart:
    """Finds the mean of target by weekends and weekdays, (two averages), then
    uses them to predict the target.
    """

    def __init__(self, batch_size=1):
        self.batch_size = batch_size
        self.weekends = ["Sat", "Sun"]

    def fit(self, X, y=None):
        X = X.copy()
        X["target"] = y
        self.weekend_mean = X[X["day_of_week"].isin(self.weekends)]["target"].mean()
        self.weekday_mean = X[~X["day_of_week"].isin(self.weekends)]["target"].mean()

    def predict(self, X):
        y = X["day_of_week"].apply(
            lambda dow: self.weekend_mean if dow in self.weekends else self.weekday_mean
        )
        y = np.round(y / self.batch_size) * self.batch_size
        y = np.expand_dims(y.to_numpy(), -1)

        return y
