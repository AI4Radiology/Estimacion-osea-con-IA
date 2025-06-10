import unittest
from datetime import datetime
from dateutil.relativedelta import relativedelta
import re

def convert_to_days_2(age_str):
    years = months = weeks = days = 0
    match = re.match(r'(?:(\d+)A)?(?:(\d+)M)?(?:(\d+)S)?(?:(\d+)D)?$', age_str)

    if not match:
        raise ValueError(f"Formato de edad inválido: '{age_str}'")

    if match.group(1):
        years = int(match.group(1))
    if match.group(2):
        months = int(match.group(2))
    if match.group(3):
        weeks = int(match.group(3))
    if match.group(4):
        days = int(match.group(4))

    start_date = datetime(1, 1, 1)
    end_date = start_date + relativedelta(years=years, months=months, weeks=weeks, days=days)
    total_days = (end_date - start_date).days
    return total_days

class TestConvertToDays2(unittest.TestCase):
    # Casos válidos
    def test_years_only(self):
        self.assertEqual(convert_to_days_2("10A"), 3652)

    def test_months_only(self):
        self.assertEqual(convert_to_days_2("6M"), 181)

    def test_days_only(self):
        self.assertEqual(convert_to_days_2("15D"), 15)

    def test_combination(self):
        self.assertEqual(convert_to_days_2("2A3M10D"), (datetime(3, 4, 11) - datetime(1, 1, 1)).days)

    def test_empty_string(self):
        self.assertEqual(convert_to_days_2(""), 0)

    # 🚨 Casos de error
    def test_invalid_letter(self):
        with self.assertRaises(ValueError):
            convert_to_days_2("5X")

    def test_missing_number(self):
        with self.assertRaises(ValueError):
            convert_to_days_2("AD")

    def test_invalid_format(self):
        with self.assertRaises(ValueError):
            convert_to_days_2("2A-3M")

    def test_non_digit_values(self):
        with self.assertRaises(ValueError):
            convert_to_days_2("aA")

    def test_partial_match(self):
        with self.assertRaises(ValueError):
            convert_to_days_2("2A3")

if __name__ == "__main__":
    unittest.main()
