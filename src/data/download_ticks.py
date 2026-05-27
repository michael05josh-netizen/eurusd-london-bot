from histdata import download_hist_data as dl
from histdata.api import Platform as P, TimeFrame as TF
import os

os.makedirs("data/raw/ticks", exist_ok=True)

year   = "2023"
months = range(1, 13)

for month in months:
    try:
        print(f"Downloading {year}-{month:02d}...")
        dl(
            year             = year,
            month            = str(month),
            pair             = "eurusd",
            platform         = P.GENERIC_ASCII,
            time_frame       = TF.TICK_DATA,
            output_directory = "data/raw/ticks"
        )
        print(f"  Done.")
    except Exception as e:
        print(f"  [SKIP] {year}-{month:02d}: {e}")

print("\nAll done.")