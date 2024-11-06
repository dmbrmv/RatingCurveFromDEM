"""Parse water level data from RusHydro website.

This script parses the water level data from RusHydro website
using Selenium. It opens the website, completes the CAPTCHA
manually, selects the date range, and extracts the data for
each point in the date range.

The script stores the data in a Pandas DataFrame and sets the
date as the index.

The data is stored in a CSV file named 'dynamic_data_points.csv'
in the same directory as the script.

The script uses tqdm to display a progress bar while parsing
the data.

The script uses WebDriverWait to wait for the elements to be
present before interacting with them.

The script uses ActionChains to hover over the elements to
trigger the update of the displayed data.

The script uses time.sleep to wait for a short time to allow
the data to update.
"""

import time
from datetime import datetime

import pandas as pd
from bs4 import BeautifulSoup
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.support.ui import WebDriverWait
from tqdm.notebook import tqdm

# Initialize Selenium WebDriver
driver = webdriver.Chrome()  # Or use webdriver for other browsers

# Open the target URL
driver.get("https://rushydro.ru/informer/")

# Wait for the CAPTCHA to be completed manually
input("Complete the CAPTCHA and press Enter to continue...")

# Parse page content with BeautifulSoup
soup = BeautifulSoup(driver.page_source, "html.parser")

# Wait until the trigger elements are present
trigger_wrap = WebDriverWait(driver, 1).until(
    EC.presence_of_element_located((By.CLASS_NAME, "water-day__trigger-wrap"))
)

# Find all elements with the class 'water-day__trigger' within the wrapper
triggers = trigger_wrap.find_elements(By.CLASS_NAME, "water-day__trigger")
# TODO add triger to select specific Hydro Power station
# Loop through each trigger element to find the one with the text '30'
for trigger in triggers:
    if trigger.text.strip() == "30":  # Check if the text matches '30'
        trigger.click()  # Click the element
        break  # Exit the loop once found and clicked
# TODO Add loop by dates
# Target date details
target_day = "1"
target_month = "9"  # Note: Months are 0-indexed in the data attributes, so August = 7
target_year = "2013"

# Step 1: Open the date picker by clicking on the input field
date_input = WebDriverWait(driver, 0).until(EC.element_to_be_clickable((By.ID, "water-date")))
date_input.click()

# Step 2: Click on the year title to open the decade view
year_title_element = WebDriverWait(driver, 1).until(
    EC.element_to_be_clickable((By.CLASS_NAME, "datepicker--nav-title"))
)
year_title_element.click()

# Step 3: Navigate to the correct decade, if needed
# while True:
# Get the currently displayed decade range from the title
current_decade_range = (
    WebDriverWait(driver, 1)
    .until(EC.visibility_of_element_located((By.CLASS_NAME, "datepicker--nav-title")))
    .text
)

# Wait until the element with class 'water-day__trigger' and text '30' is present
# Split the range text to get the start and end years of the decade range
start_year, end_year = map(int, current_decade_range.split(" - "))

# Check if the target year is within this range
if start_year <= int(target_year) <= end_year:
    # Step 4: Select the specific year
    try:
        year_element = WebDriverWait(driver, 1).until(
            EC.element_to_be_clickable(
                (
                    By.XPATH,
                    f"//div[@class='datepicker--cell datepicker--cell-year -current-' and @data-year='{target_year}']",
                )
            )
        )
    except Exception:
        year_element = WebDriverWait(driver, 1).until(
            EC.element_to_be_clickable(
                (
                    By.XPATH,
                    f"//div[@class='datepicker--cell datepicker--cell-year' and @data-year='{target_year}']",
                )
            )
        )

    year_element.click()
elif int(target_year) < start_year:
    # If the target year is earlier than the displayed range, click the "prev" button
    prev_button = driver.find_element(By.CSS_SELECTOR, "div.datepicker--nav-action[data-action='prev']")
    prev_button.click()
    try:
        year_element = WebDriverWait(driver, 1).until(
            EC.element_to_be_clickable(
                (
                    By.XPATH,
                    f"//div[@class='datepicker--cell datepicker--cell-year -current-' and @data-year='{target_year}']",
                )
            )
        )
    except Exception:
        year_element = WebDriverWait(driver, 1).until(
            EC.element_to_be_clickable(
                (
                    By.XPATH,
                    f"//div[@class='datepicker--cell datepicker--cell-year' and @data-year='{target_year}']",
                )
            )
        )
    year_element.click()
else:
    # If the target year is later than the displayed range, click the "next" button
    next_button = driver.find_element(By.CSS_SELECTOR, "div.datepicker--nav-action[data-action='next']")
    next_button.click()
# Step 4: Select the month
# After selecting the year, locate the month in the date picker
month_element = WebDriverWait(driver, 10).until(
    EC.element_to_be_clickable(
        (
            By.XPATH,
            f"//div[@class='datepicker--cell datepicker--cell-month' and @data-month='{target_month}']",
        )
    )
)
month_element.click()

# Step 5: Select the day
# After selecting the month, locate the day in the date picker
day_element = WebDriverWait(driver, 10).until(
    EC.element_to_be_clickable(
        (
            By.XPATH,
            f"//div[@class='datepicker--cell datepicker--cell-day' and @data-date='{target_day}' and @data-month='{target_month}' and @data-year='{target_year}']",
        )
    )
)
day_element.click()

# Wait until the ranges_group is present
ranges_group = WebDriverWait(driver, 1).until(EC.presence_of_element_located((By.CLASS_NAME, "ranges")))

# Find all <rect> elements within ranges_group
rectangles = ranges_group.find_elements(By.TAG_NAME, "rect")

# List to store data for each point
dynamic_data_points = []

# Iterate through each rectangle
for rect in tqdm(rectangles):
    # Get the data-num attribute as the unique identifier for each rectangle
    data_num = rect.get_attribute("data-num")

    # Click or hover over the rectangle to update the displayed level data
    driver.execute_script("arguments[0].scrollIntoView();", rect)  # Scroll to the element if needed
    rect.click()  # This may trigger the update; alternatively, use ActionChains to hover if hover is needed

    # Wait a short time to allow the data to update (adjust if necessary)
    time.sleep(
        1
    )  # Simple wait; adjust based on page response time or use WebDriverWait for dynamic checks

    # Extract the updated data from ges-levels__more
    polemk = driver.find_element(By.CLASS_NAME, "polemk").text
    pritok = driver.find_element(By.CLASS_NAME, "pritok").text
    rashod = driver.find_element(By.CLASS_NAME, "rashod").text
    sbros = driver.find_element(By.CLASS_NAME, "sbros").text

    # Find the <text> element with the same data-num to get day and month
    text_element = WebDriverWait(driver, 0).until(
        EC.presence_of_element_located((By.XPATH, f"//*[name()='text'][@data-num='{data_num}']"))
    )

    # Extract day and month from the text element
    day = text_element.text.split(".")[0]  # Extract day from the main text before "."
    month = text_element.find_element(
        By.CLASS_NAME, "level-month"
    ).text  # Extract month within the <tspan>

    # Build the datetime object using year_of_interest, month, and day
    date_obj = pd.to_datetime(datetime(int(target_year), int(month), int(day)))

    # Store the data for the current rectangle
    dynamic_data_points.append(
        {
            "date": date_obj,
            "Capacity, qm mln": polemk,
            "Inflow, qms": pritok,
            "Outflow, qms": rashod,
            "Outflow spillways, qms": sbros,
        }
    )
# TODO merge dataframes from every possible period
dynamic_data_points = pd.DataFrame(dynamic_data_points)
dynamic_data_points = dynamic_data_points.set_index("date").sort_index()
dynamic_data_points
