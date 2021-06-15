# Zillow Regression Project 

## Deliverables
- Slide deck (link here)
- Repo with work 
    - [Final Notebook](https://github.com/HeatherOrtegaMcMillan/zillow_regression_project/blob/main/final_notebook.ipynb)

## Questions and Goals
- To predict the values of **single unit properties** that the tax disctrict assesses using the property data from those with a transaction during the "hot months" of May-August, 2017.
- What are the drivers of these single unit property values

- What states and counties these properties are located in
- What is the distribution rate of the tax rates for each county

- The first iteration of my model, I will use only square feet of the home, number of bedrooms, and number of bathrooms

### Inital Ideas and Hypotheses 

- Some Inital Ideas I had that homes with more square footage would be more expensive
- I also thought that having a pool would make the value of the home go up
- When I found out these houses were in California, I also knew that we would be dealing with very expensive homes, probably some of the most expensive in the country. It was something I kept in mind.

H0: There is no linear coorrelation between total square footage and tax value (home value)

Ha: There is a significant linear correlation between total square footage and tax value (home value)

Test: Pearson R

H0: There is no linear correlation between number of bathrooms and tax value (home value) 

Ha: There is a linear correlation between number of bathrooms and tax value (home value)

(in this case I am considering number of bathroom a continous variable)

Test: Pearson R

## About the Data
- The data in this project comes from the Zillow data prize competition in 2017. The two tables used here are `properties_2017` and `predictions_2017`

### Data Dictionary
| Column Name       | Description                                                                              | Use                   |
|-------------------|------------------------------------------------------------------------------------------|-----------------------|
| `parcel_id`       | Unique Identifier for each property                                                      | Identifier            |
| `tax_value`       | Tax appraised value of home. Original column name taxvaluedollaramount.                  | Target                |
| `bathroom_cnt`    | Number of bathrooms                                                                      | Variable (continuous) |
| `bedroom_cnt`     | Number of bedrooms                                                                       | Variable (continuous) |
| `sqft_calculated` | Total Square footage of home. Original column name calculatedfinishedsquarefeet.         | Variable (continuous) |
| `has_pool`        | Indicator on whether the property has a pool or not. Engineered from poolcnt             | Variable (boolean)    |
| `has_garage`      | Indicator on whether the property has a garage or not. Engineered from garagecarcnt      | Variable (boolean)    |
| `tax_amount`      | Amount of tax collected for property                                                     | Information           |
| `fips`            | Federal Information Processing Standards Code. Indicator of county.                      | Information           |
| `county`          | County where the property is located. Generated from `fips`                              | Information           |
| `tax_rate`        | Percentage of value that was paid in taxes. Calculated from `tax_value` and `tax_amount` | Information           |
 

### Plan and Process
- [Trello Board](https://trello.com/b/ElVvHjKs/zillow-regression-project)

### How to recreate this project?

You will need your own env file with database credentials along with all the necessary files listed below to run my final project notebook.

1. Read the README.md
2. Download the wrangle.py, evaluate.py, explore.py and final_notebook.ipynb files into your working directory, or clone this repository 
3. Add your own env file to your directory. (user, password, host)
4. Run the final_notebook.ipynb notebook

## Skills Required
Technical Skills
- Python
    - Pandas
    - Seaborn
    - Matplotlib
    - Numpy 
    - Sklearn

- SQL

- Statistical Analysis
    - Descriptive Stats
    - Hypothesis Testing
    - T-test
    - Chi^2 Test

- Regression Modeling
    - Linear Regression Evaluation Methods
        - RMSE, R2, etc
    - Tweedie Regressor
    - Lasso Lars