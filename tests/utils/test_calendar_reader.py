from workalendar.core import Calendar

from hcl_model.utils.calendar_reader import CalendarReader


class TestCalendarReader:

    from_year = 2005
    to_year = 2030
    lbl_new_year = "New year"

    lbl_country_code_de = "DE"
    lbl_country_code_cn = "CN"
    lbl_country_code_tw = "TW"

    def test_get_holidays(self):
        cal_reader = CalendarReader()

        df = cal_reader.get_holidays(self.lbl_new_year, self.lbl_country_code_de, self.from_year, self.to_year)

        assert df.shape[0] == self.to_year - self.from_year
        assert set(df.values.ravel()) == {self.lbl_new_year, self.lbl_country_code_de}
        assert df.index.year[0] == self.from_year
        assert df.index.year[-1] == self.to_year - 1

    def test_get_calendar(self):
        cal_reader = CalendarReader()

        assert isinstance(cal_reader._get_calendar(self.lbl_country_code_de), Calendar)
