import pandas as pd
from workalendar.core import Calendar
from workalendar.registry import registry

from hcl_model.labels import LabelsCommon, LabelsHolidaysOut


class CalendarReader:
    lbl = LabelsCommon()
    lbl_holidays_out = LabelsHolidaysOut()

    def get_holidays(self, holiday_name: str, country_code: str, from_year: int, to_year: int) -> pd.DataFrame:
        """Import holidays.

        The reader uses 'workalendar' library available here: https://pypi.org/project/workalendar/.

        :param holiday_name: holiday name
        :param country_code: two letter country code, e.g. `DE` or `CN`
        :param from_year: the year from which to start the calendar
        :param to_year: the year to end the calendar
        :return: DataFrame with holiday dates for specific countries.

        Typical output:

        .. code-block:: python

                                 holiday country
            date
            2010-04-05     Easter Monday      DE
            2002-04-01     Easter Monday      DE
            2030-04-22     Easter Monday      DE
            2009-04-13     Easter Monday      DE

        The DataFrame is indexed by date of the holiday

        Two columns:

        - holiday: name of the holiday
        - country: iso code of the country

        """
        cal = self._get_calendar(country_code)

        # Go over all holidays in each year find those that match the request
        holiday_list = [x[0] for year in range(from_year, to_year) for x in cal.holidays(year) if holiday_name == x[1]]
        if not holiday_list:
            raise RuntimeError('Holiday "{}" was not found in "{}" calendar!'.format(holiday_name, country_code))

        # Create a DataFrame with date as index, country and holiday name as columns
        return pd.DataFrame(
            {
                self.lbl_holidays_out.lbl_holiday: holiday_name,
                self.lbl_holidays_out.lbl_country: country_code,
            },
            index=pd.Index(pd.to_datetime(holiday_list), name=self.lbl.date),
        )

    @staticmethod
    def _get_calendar(country_code: str = "CN") -> Calendar:
        """Get calendar object given two-letter code of a country.

        :param country_code: two letter country code, e.g. `DE` or `CN`
        :return: Calendar class for specific country
        """
        if country_code in registry.region_registry.keys():
            cal_class = registry.get(country_code)
            return cal_class()
        else:
            raise RuntimeError("{} is an unknown country code!".format(country_code))
