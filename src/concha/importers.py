import os
from urllib.parse import urlparse, parse_qs
from datetime import datetime, timezone
import dateutil
import geocoder
import pandas as pd
import requests
from square.client import Client
from concha.environment import FileHandler


class Square:
    "Class to handle setting cafe locations and importing transaction history for products using Square POS." ""

    def __init__(self, name="square"):
        """Creates the importer.

        Args:
            name: (str) The name of the importer and to define the path to find it's settings in
                concha_planners/importers/[name].json.
        """

        # Create the file agent
        self.filehandler = FileHandler()
        importers_path = self.filehandler.check_importer_path()
        self.settings_path = os.path.join(importers_path, f"{name}.json")

        # Load the existing settings (including the access_token for the square client if set.)
        if os.path.exists(self.settings_path):
            self.settings = self.filehandler.dict_from_file(self.settings_path)
        else:
            self.settings = {"name": name, "type": "square"}
            self.filehandler.dict_to_file(self.settings, self.settings_path)

        if "refresh_token" in self.settings:
            # If the access token is expiring in less than 15 days, refresh it.
            self.refresh_access_token(check_expiration=True)
        if "access_token" in self.settings:
            self.connect()
        return

    def connect(self):
        """Create the client and necessary mini-apis if access_token is set."""

        self.client = Client(
            access_token=self.settings["access_token"], environment="production"
        )
        self.orders_api = self.client.orders
        self.locations_api = self.client.locations

    def get_orders(self, location=None, last_timestamp=None):
        """Get the transaction history for a location from the last_timestamp.

        Args:
            location (str): The "name" of the vendor as listed in the square locations.

            last_timestamp (str): Last timestamp recorded in the last import (or six months
                ago if no previous import exists. Format: "%Y-%m-%dT%H:%M:%S%z" or like
                "2020-10-19T19:24:03-07:00"

        Returns:
            trans_df (DataFrame): The transactions as a dataframe with columns:
                "timestamp", "product", "quantity".
        """

        # Look up the metadata for the location by name from the locations dict
        location_id = self.settings["locations"][location]["id"]

        # Create the query dict for the orders search of the square api
        body = {
            "location_ids": [location_id],
            "query": {"filter": {"state_filter": {"states": ["COMPLETED"]}}},
            "state": ["Completed"],
        }

        # the 'sort' part is required when a 'date_time_filter' is included. Otherwise the API
        # throws an error.
        if last_timestamp is not None:
            body["query"]["filter"]["date_time_filter"] = {
                "closed_at": {"start_at": last_timestamp}
            }
            body["query"]["sort"] = {"sort_field": "CLOSED_AT", "sort_order": "ASC"}

        # Results are paginated, so we need to loop through to get them all.
        transactions = []
        while True:
            response = self.orders_api.search_orders(body)
            result = self.check_response(response)
            orders = result["orders"]
            for order in orders:
                ts = order["closed_at"]
                for item in order["line_items"]:
                    product = " - ".join([item["name"], item["variation_name"]])
                    transactions.append(
                        {
                            "timestamp": ts,
                            "product": product,
                            "quantity": float(item["quantity"]),
                        }
                    )
            if "cursor" in result:
                body["cursor"] = result["cursor"]
            else:
                break
        trans_df = pd.DataFrame(transactions)

        return trans_df

    @staticmethod
    def check_response(response):
        """Checks if the square api call was a success and returns the response body if so.

        Args:
            response (Requests response): Response from the requests query call.

        Returns:
            response.body or response.errors depending on what happened.
        """
        if response.is_success():
            return response.body
        elif response.is_error():
            print(response.errors)
        else:
            print("looks like check_response needs more work")

    def get_locations(self):
        """Look up the locations listed on the Square account.

        If the lat,lng for a square location aren't recorded, this will geocode the address
        listed with OSM.

        Returns:
            locations (dict): A dict where the key is the "name" of the location and the
                value is the metadata for each location.
        """
        res = self.locations_api.list_locations()
        locs = self.check_response(res)["locations"]
        locations_dict = {}

        for lc in locs:
            name = lc["name"]
            dc = {"id": lc["id"], "name": name}
            if "address" in lc:
                dc["address"] = ", ".join(
                    [value for key, value in lc["address"].items()]
                )
            if "coordinates" in lc:
                dc["lat"] = lc["coordinates"]["latitude"]
                dc["lng"] = lc["coordinates"]["longitude"]
            else:
                # Look up coordinates for an address if coordinates not looked up by Square.
                osm_lookup = geocoder.osm(dc["address"])
                dc["lat"] = osm_lookup.osm["y"]
                dc["lng"] = osm_lookup.osm["x"]
            if "timezone" in lc:
                dc["timezone"] = lc["timezone"]
            locations_dict[name] = dc
        self.settings["locations"] = locations_dict
        self.filehandler.dict_to_file(self.settings, self.settings_path)
        return locations_dict

    def set_access_token(self, token):
        """
        Sets the access token for the importer. Then connects to the Square APIs.

        Args:
            token (str): The access token
        """

        self.settings["access_token"] = token
        self.filehandler.dict_to_file(self.settings, self.settings_path)
        self.connect()

    def create_access_request_url(self, client_id, client_secret):
        """Create the url the user will need to use to grant access to an app.

        The scope is limited to "ORDERS_READ", which is the orders history, and
        "MERCHANT_PROFILE_READ" which is information about the vendor locations.
        Once the user clicks to grant access the app, the user will be redirected
        to the redirect url they specified in the app.

        Args:
            client_id (str): In the Square Oath2 flow, this identifies an
                app publicly.

            client_secret (str): This is used by the app to confirm its
                identity to the Square servers when requesting an access code.
                It isn't used in this step, but added to the importer with
                this method.

        Returns:
            access_request_url (str): The url required to request access for the
                app identified by client_id to whichever Square account the user uses to login.
        """

        # Set the information in the settings
        self.settings["client_id"] = client_id
        self.settings["client_secret"] = client_secret
        self.filehandler.dict_to_file(self.settings, self.settings_path)

        # Create the url
        authorize_url = "https://connect.squareup.com/oauth2/authorize"
        scope = "ORDERS_READ MERCHANT_PROFILE_READ"
        authorize_body = {"client_id": client_id, "scope": scope}
        headers = {"Accept": "application/json"}
        req = requests.Request(
            method=None, url=authorize_url, headers=headers, params=authorize_body
        ).prepare()
        return req.url

    def get_access_token(self, code=None):
        """Gets the access token from the authorization code.

        Accepts either the full url after redirect, or just the authorization code provided.
        Then gets the access token from Square. This is the second step in the Oath2 flow.

        Args:
            code (str): Either the full url, or the just the code. So either
                "www.google.com?code=123abc" or "123abc".

        Returns:
            result (Square Obtain Token Response Object):
                (https://github.com/square/square-python-sdk/blob/master/doc/models/obtain-token-response.md)
        """
        # Get the access token (initial step using the authorization code)
        client = Client()
        o_auth_api = client.o_auth

        if "?" in code:
            # if '?' present, assume the whole url was included, including the query string
            # and parse to get the code.
            psd = urlparse(code)
            query_dict = parse_qs(psd.query)
            code = query_dict["code"][0]
            # else, assume the code is the only thing included.

        body = {
            "client_id": self.settings["client_id"],
            "client_secret": self.settings["client_secret"],
            "code": code,
            "grant_type": "authorization_code",
        }
        result = o_auth_api.obtain_token(body)
        if result.is_success():
            self.settings["access_token"] = result.body["access_token"]
            self.settings["access_token_expiration"] = result.body["expires_at"]
            self.settings["refresh_token"] = result.body["refresh_token"]
            self.filehandler.dict_to_file(self.settings, self.settings_path)
            self.connect()
        else:
            print(result.errors)
        return result

    def refresh_access_token(self, check_expiration=True):
        """Refreshes the app's access token to the square account.

        Args:
            check_experiation (bool): If True, only refreshes if the expiration
                is more than 15 days away. If False, refresh the access token.
        """
        # if check_expiration - only refresh if token will expire in less than 15 days.
        if check_expiration:
            expiration = dateutil.parser.parse(self.settings["access_token_expiration"])
            time_since = datetime.now(timezone.utc) - expiration
            # just exit if expiration is more than 15 days away.
            if time_since.days < 15:
                return

        client = Client()
        o_auth_api = client.o_auth
        body = {
            "client_id": self.settings["client_id"],
            "client_secret": self.settings["client_secret"],
            "grant_type": "refresh_token",
            "refresh_token": self.settings["refresh_token"],
        }
        result = o_auth_api.obtain_token(body)

        # Save the new token and update the expiration
        self.settings["access_token"] = result.body["access_token"]
        self.settings["access_token_expiration"] = result.body["expires_at"]
        self.filehandler.dict_to_file(self.settings, self.settings_path)
        return result.body
