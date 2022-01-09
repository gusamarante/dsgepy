from pydsge import FRED

fred = FRED()

# ===== Grab and organize Data ===== #
series_dict = {'CPIAUCSL': 'CPI',
               'GDP': 'GDP',
               'DFF': 'Fed Funds Rate'}

df = fred.fetch(series_id=series_dict)

print(df)
