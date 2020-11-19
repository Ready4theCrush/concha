import os
import time
import re
import requests
import pandas as pd
from datetime import datetime, timedelta
from dateutil import parser

from concha.environment import FileHandler


class NOAA:
    """Handles NOAA weather operations for finding stations, getting historical weather, and forecasts.

    The only setting for this class is the NOAA api key which the user needs to set, and which is saved
    in ".../concha_planners/importers/[name from __init__].json
    """

    def __init__(self, name="noaa"):
        """
        Initializes the NOAA service.

        Args:
            noaa (str): Name of the weather profile.
        """

        # Assign a filehandler
        self.filehandler = FileHandler()
        importers_path = self.filehandler.check_importer_path()
        self.settings_path = os.path.join(importers_path, f"{name}.json")

        # Get the settings (i.e. the api if it is set)
        if os.path.exists(self.settings_path):
            self.settings = self.filehandler.dict_from_file(self.settings_path)
        else:
            # If not, make a [name].json file
            self.settings = {"type": "noaa", "name": name, "api_key": None}
            self.filehandler.dict_to_file(self.settings, self.settings_path)
        return

    def set_api_key(self, api_key):
        """Setter for the NOAA api key."""
        self.settings["api_key"] = api_key
        self.filehandler.dict_to_file(self.settings, self.settings_path)

    def get_weather_history(self, start_date, end_date, station_id):
        """Looks up the weather within a date range at the station.

        Args:
            start_date (str): Format from: ['date'].dt.strftime('%Y-%m-%d'), so like '2020-07-15'
            end_date (str): End date of range in which to find weather history.
            station_id (str): The NOAA GHCND station name. (https://gis.ncdc.noaa.gov/maps/ncei/summaries/daily)

        Return:
            weather_history (pd.DataFrame): fields: ['date', 'tmin', 'tmax'] and possibly ['prcp', 'snow']

        """

        # Assemble actual request to NOAA. "GHCND" is the Global Historical Climate Network Database
        try:
            api_key = self.settings["api_key"]
        except KeyError:
            print("An api_key not set up.")
            print(self.settings)
        headers = {"Accept": "application/json", "token": api_key}
        url = "https://www.ncdc.noaa.gov/cdo-web/api/v2/data"
        params = {
            "stationid": station_id,
            "startdate": start_date,
            "enddate": end_date,
            "datasetid": "GHCND",
            "units": "standard",
            "datatypeid": ["TMIN", "TMAX", "PRCP", "SNOW"],
            "sortorder": "DESC",
            "offset": 0,
            "limit": 1000,
        }

        # Loop through requests to the API if more than 1000 results are required.
        records = []
        for i in range(10):
            req = requests.get(url, headers=headers, params=params)
            res = req.json()
            recs = pd.DataFrame(res["results"])
            records.append(recs)
            count = res["metadata"]["resultset"]["count"]
            if count < params["limit"]:
                break
            else:
                params["offset"] += params["limit"]
        records = pd.concat(records)

        # Format the results and turn precipitation and snow levels into just yes/no booleans
        records["datatype"] = records["datatype"].str.lower()
        records = records.pivot(
            index="date", columns="datatype", values="value"
        ).reset_index()
        records["date"] = pd.to_datetime(records["date"])
        use_fields = ["date", "tmin", "tmax"]
        for field in ["prcp", "snow"]:
            if field in records.columns:
                records[field] = records[field].apply(lambda x: x > 0)
                use_fields.append(field)
        return records[use_fields]

    def get_weather_forecast(self, forecast_url):
        """Finds the weather forecast at the grid covering the station location.

        The forecast API gives a day and an overnight forecast. This parses the min
        temperature from the *overnight/morning of* instead of that night. This was done
        because the morning before temperature has potentially more effect on demand than
        the min temperature after the store has closed.

        Args:
            forecast_url (str): The api.weather.gov forecast url for the given location.

        Returns:
            by_date (pd.DataFrame): The high/low temp by date with the precipitation
                and snow as booleans.
        """

        # The forecast api is weird and sometimes won't respond for the first minute or so.
        headers = {"User-Agent": "project concha python application"}

        # Try to get the forecast 10 times.
        for i in range(5):
            req = requests.get(forecast_url, headers=headers)
            try:
                req.raise_for_status()
            except requests.exceptions.HTTPError:
                # Just ignore and try again later
                time.sleep(5)

        # If the first tries don't work - just tell the user to try again later.
        # The API is weird and sometimes just doesn't work.
        try:
            req.raise_for_status()
        except requests.exceptions.HTTPError as err:
            print(
                f"""The NOAA forecast site is weird and sometimes doesn't respond. Try the url in your browser,
                Then you get a respose, you should be able to run this again and get the forecast. URL:
                {forecast_url}"""
            )
            print(err)

        res = req.json()
        forecast = pd.DataFrame(res["properties"]["periods"])

        # Many fields are returned, this limits them to the ones needed.
        forecast = forecast[
            ["startTime", "endTime", "isDaytime", "temperature", "shortForecast"]
        ]

        # Date is chosen such that the previous overnight temp, and day temp are assigned to the date
        forecast["date"] = forecast["endTime"].apply(
            lambda x: parser.parse(x).strftime("%Y-%m-%d")
        )

        # String search used to figure out of rain is in the forecast
        forecast["prcp"] = forecast["shortForecast"].str.contains(
            "showers|rain|thunderstorms", flags=re.IGNORECASE, regex=True
        )
        forecast["snow"] = forecast["shortForecast"].str.contains(
            "snow|blizzard|flurries", flags=re.IGNORECASE, regex=True
        )

        # Because two values exist for each date, the they are aggregated to find one value for each date.
        by_date = forecast.groupby("date").agg(
            tmin=pd.NamedAgg(column="temperature", aggfunc="min"),
            tmax=pd.NamedAgg(column="temperature", aggfunc="max"),
            count=pd.NamedAgg(column="temperature", aggfunc="count"),
            prcp=pd.NamedAgg(column="prcp", aggfunc="any"),
            snow=pd.NamedAgg(column="snow", aggfunc="any"),
        )

        # Only include dates with two values
        by_date = by_date[by_date["count"] > 1].drop(columns="count").reset_index()
        by_date[["tmin", "tmax"]] = by_date[["tmin", "tmax"]].astype(float)
        by_date["date"] = pd.to_datetime(by_date["date"])

        return by_date

    def check_key(self):
        """Quick check function to make sure NOAA api key is present."""

        if self.settings["api_key"] is None:
            message = """
    This won't work without a NOAA api key...that's the bad news.
    The good news is that they're free!
    Go here to get one emailed to you. Then add it like this:
        from concha.weather import NOAA
        weather = NOAA()
        weather.set_api_key("whatevertheysentyou")
            """
            print(message)

    def get_station(self, lat=47.624642, lng=-122.3261102):
        """Hits the NOAA API for the nearest/best weather station recording historical data

        The NOAA api does not search from a point, so this method creates a series of expanding
        bounding boxes to prioritize closer stations.

        Args:
            lat (float): Latitude of vendor location
            lng (float): Longitude of cafe/restaurant

        Returns:
            station (dict): The metadata for a weather station recording
                historical daily weather summaries.
        """
        # make sure the api key is present
        self.check_key()
        headers = {"Accept": "application/json", "token": self.settings["api_key"]}
        url = "https://www.ncdc.noaa.gov/cdo-web/api/v2/stations"
        delta = 0.1

        # the stations never have current to-the-day weather, but they should have it at
        # least from three weeks ago if they are "current"
        recent_date = datetime.today() - timedelta(weeks=3)

        # extent is the bounding box for the search.
        params = {
            "extent": [lat - delta, lng - delta, lat + delta, lng + delta],
            "datasetid": "GHCND",
            "datacategoryid": ["TEMP", "PRCP"],
            "startdate": recent_date.strftime("%Y-%m-%d"),
            "sortfield": "datacoverage",
            "sortorder": "desc",
        }

        # keep expanding the bounding box if a current station with
        # datacoverage > 0.85 isn't found.
        for i in range(10):
            req = requests.get(url, headers=headers, params=params)
            res = req.json()
            if "results" in res:
                stations = pd.DataFrame(res["results"])
                if stations.iloc[0]["datacoverage"] > 0.85:
                    station = stations.iloc[0].to_dict()
                    break
            delta = delta * 2
            params["extent"] = [lat - delta, lng - delta, lat + delta, lng + delta]
            print("Expanding bounds to:")
            print(params["extent"])
        return station

    def get_forecast_url(self, station):
        """Asks the forecasting API what url to use for the station's location

        Args:
            station (dict): The station metadata returned from self.get_station
                and the https://www.ncdc.noaa.gov/cdo-web/api/v2/stations api.

        Returns:
            forecast_url (str): The url that provides forecasts from api.weather.gov
        """

        url = f"https://api.weather.gov/points/{station['latitude']},{station['longitude']}"
        headers = {"User-Agent": "project concha python application"}
        req = requests.get(url, headers=headers)
        res = req.json()
        forecast_url = res["properties"]["forecast"]
        return forecast_url
