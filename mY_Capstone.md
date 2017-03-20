### Breast Cancer Detection
http://archive.ics.uci.edu/ml/datasets/Breast+Cancer+Wisconsin+%28Diagnostic%29


#### Blurb:

This project is to train an alogorithm to predict "Diagnosis" result of given case from readings generated in Breat Cancer screening. Data includes 32 features of real-world dignostics of "Breast Cancer" tests. 

Expected Accuracy levels of algorithm is to be decided.

###### Features / Attributes information:
Number of attributes: 32 (ID, diagnosis, 30 real-valued input features)
1) ID number
2) Diagnosis (M = malignant, B = benign)
3-32)

Ten real-valued features are computed for each cell nucleus:

	a) radius (mean of distances from center to points on the perimeter)
	b) texture (standard deviation of gray-scale values)
	c) perimeter
	d) area
	e) smoothness (local variation in radius lengths)
	f) compactness (perimeter^2 / area - 1.0)
	g) concavity (severity of concave portions of the contour)
	h) concave points (number of concave portions of the contour)
	i) symmetry 
	j) fractal dimension ("coastline approximation" - 1)

The mean, standard error, and "worst" or largest (mean of the three
largest values) of these features were computed for each image,
resulting in 30 features.  For instance, field 3 is Mean Radius, field
13 is Radius SE, field 23 is Worst Radius.



### Predicting House Prices
https://www.kaggle.com/c/house-prices-advanced-regression-techniques


#### Blurb:

This is project is to deduce a price-prediction equation from list of 80+ attributes those describe every aspect of residential home. 

Expected Accuracy levels of Prediction Equation is to be decided.

###### Attributes information:

1. Here's a brief version of what you'll find in the data description file.
2. SalePrice - the property's sale price in dollars. This is the target variable that you're trying to predict.
3. MSSubClass: The building class
4. MSZoning: The general zoning classification
5. LotFrontage: Linear feet of street connected to property
6. LotArea: Lot size in square feet
7. Street: Type of road access
8. Alley: Type of alley access
9. LotShape: General shape of property
10. LandContour: Flatness of the property
11. Utilities: Type of utilities available
12. LotConfig: Lot configuration
13. LandSlope: Slope of property
14. Neighborhood: Physical locations within Ames city limits
15. Condition1: Proximity to main road or railroad
16. Condition2: Proximity to main road or railroad (if a second is present)
17. BldgType: Type of dwelling
18. HouseStyle: Style of dwelling
19. OverallQual: Overall material and finish quality
20. OverallCond: Overall condition rating
21. YearBuilt: Original construction date
22. YearRemodAdd: Remodel date
23. RoofStyle: Type of roof
24. RoofMatl: Roof material
25. Exterior1st: Exterior covering on house
26. Exterior2nd: Exterior covering on house (if more than one material)
27. MasVnrType: Masonry veneer type
28. MasVnrArea: Masonry veneer area in square feet
29. ExterQual: Exterior material quality
30. ExterCond: Present condition of the material on the exterior
31. Foundation: Type of foundation
32. BsmtQual: Height of the basement
33. BsmtCond: General condition of the basement
34. BsmtExposure: Walkout or garden level basement walls
35. BsmtFinType1: Quality of basement finished area
36. BsmtFinSF1: Type 1 finished square feet
37. BsmtFinType2: Quality of second finished area (if present)
38. BsmtFinSF2: Type 2 finished square feet
39. BsmtUnfSF: Unfinished square feet of basement area
40. TotalBsmtSF: Total square feet of basement area
41. Heating: Type of heating
42. HeatingQC: Heating quality and condition
43. CentralAir: Central air conditioning
44. Electrical: Electrical system
45. 1stFlrSF: First Floor square feet
46. 2ndFlrSF: Second floor square feet
47. LowQualFinSF: Low quality finished square feet (all floors)
48. GrLivArea: Above grade (ground) living area square feet
49. BsmtFullBath: Basement full bathrooms
50. BsmtHalfBath: Basement half bathrooms
51. FullBath: Full bathrooms above grade
52. HalfBath: Half baths above grade
53. Bedroom: Number of bedrooms above basement level
54. Kitchen: Number of kitchens
55. KitchenQual: Kitchen quality
56. TotRmsAbvGrd: Total rooms above grade (does not include bathrooms)
57. Functional: Home functionality rating
58. Fireplaces: Number of fireplaces
59. FireplaceQu: Fireplace quality
60. GarageType: Garage location
61. GarageYrBlt: Year garage was built
62. GarageFinish: Interior finish of the garage
63. GarageCars: Size of garage in car capacity
64. GarageArea: Size of garage in square feet
65. GarageQual: Garage quality
66. GarageCond: Garage condition
67. PavedDrive: Paved driveway
68. WoodDeckSF: Wood deck area in square feet
69. OpenPorchSF: Open porch area in square feet
70. EnclosedPorch: Enclosed porch area in square feet
71. 3SsnPorch: Three season porch area in square feet
72. ScreenPorch: Screen porch area in square feet
73. PoolArea: Pool area in square feet
74. PoolQC: Pool quality
75. Fence: Fence quality
76. MiscFeature: Miscellaneous feature not covered in other categories
77. MiscVal: Value of miscellaneous feature
78. MoSold: Month Sold
79. YrSold: Year Sold
80. SaleType: Type of sale
81. SaleCondition: Condition of sale


### Predicting the outcomes of Basket ball tornament
https://www.kaggle.com/c/march-machine-learning-mania-2017


#### Blurb:

This is project is to train an algorithm to predict the outcomes (winners and losers) of US men's college basketball tournament using rich historical data.

Expected Accuracy levels of Prediction Equation is to be decided.

###### Data information:

* Teams
This file identifies the different college teams present in the dataset. Each team has a 4 digit id number.

* Seasons:
This file identifies the different seasons included in the historical data, along with certain season-level properties.

* RegularSeasonCompactResults
This file identifies the game-by-game results for 32 seasons of historical data, from 1985 to 2015. Each year, it includes all games played from daynum 0 through 132 (which by definition is "Selection Sunday," the day that tournament pairings are announced). Each row in the file represents a single game played.

* RegularSeasonDetailedResults
This file is a more detailed set of game results, covering seasons 2003-2016. This includes team-level total statistics for each game (total field goals attempted, offensive rebounds, etc.) The column names should be self-explanatory to basketball fans (as above, "w" or "l" refers to the winning or losing team):

* TourneyCompactResults
This file identifies the game-by-game NCAA tournament results for all seasons of historical data. The data is formatted exactly like the regular_season_compact_results.csv data. Note that these games also include the play-in games (which always occurred on day 134/135) for those years that had play-in games.

* TourneyDetailedResults
This file contains the more detailed results for tournament games from 2003 onward.

* TourneySeeds
This file identifies the seeds for all teams in each NCAA tournament, for all seasons of historical data. Thus, there are between 64-68 rows for each year, depending on the bracket structure.

* TourneySlots
This file identifies the mechanism by which teams are paired against each other, depending upon their seeds. Because of the existence of play-in games for particular seed numbers, the pairings have small differences from year to year. If there were N teams in the tournament during a particular year, there were N-1 teams eliminated (leaving one champion) and therefore N-1 games played, as well as N-1 slots in the tournament bracket, and thus there will be N-1 records in this file for that season.

