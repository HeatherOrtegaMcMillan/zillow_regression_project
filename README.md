# Zillow Regression Project 

## Deliverables
- Slide deck (link here)
- Repo with work 
    - Final Notebook (link here)

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

${H_0}$ There is no linear coorrelation between total square footage and tax value (home value)
${H_a}$ There is a significant linear correlation between total square footage and tax value (home value)

Test: Pearson R

${H_0}$ There is no linear correlation between number of bathrooms and tax value (home value) 
${H_a}$ There is a linear correlation between number of bathrooms and tax value (home value)

(in this case I am considering number of bathroom a continous variable)

Test: Pearson R

## About the Data
- The data in this project comes from the Zillow data prize competition in 2017. The two tables used here are `properties_2017` and `predictions_2017`

### Data Dictionary
*insert data dictionary here* 

### Plan and Process
- [Trello Board](https://trello.com/b/ElVvHjKs/zillow-regression-project)

### How to recreate this project?