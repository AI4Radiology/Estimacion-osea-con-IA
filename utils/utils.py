from datetime import datetime
from dateutil.relativedelta import relativedelta
import re

'''
    Converts a text string representing an age (in years, months, weeks, and days)
    into the total number of days.

    This function interprets the age string using a specific format where:
    - 'A' represents years
    - 'M' represents months
    - 'S' represents weeks
    - 'D' represents days
    Nuumbers are expected to precede their corresponding letters 
    (e.g., "1A2M" for 1 year and 2 months).
'''

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