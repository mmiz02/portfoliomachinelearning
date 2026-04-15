# Team Project: Airbnb Business Analysis Data Science

### Project Overview
This team project focused on analysing Airbnb listings in New York City using a classical machine learning approach (Track 1: Regression and Clustering). The objective was to apply data science techniques to a real-world business problem and generate actionable pricing insights for Airbnb hosts.

The researcg question our team came up with was: <br>
What is the optimal price range per neighbourhood_group × room_type segment that maximises booking possibility while preserving competitive host earnings? 

We used the AB_NYC_2019 dataset (Dgomonov, 2019), which includes listing information such as price, location, availability, room type, etc. Since actual booking data was not available we used reviews per month as a proxy for demand.

### Data Preparation
We began by cleaning and preparing the dataset to ensure reliable analysis including:<br>
- Removing invalid or missing price values<br>
- Winsorising extreme price outliers to reduce distortion<br>
- Handling missing review data by imputing zero values<br>
- Applying log transformations to price to address skewness<br>

Exploratory analydid was done to understand how pricing varies across boroughs and room types. This revealed strong structural differences, particularly between Manhattan and outer boroughs, as well as between entire homes, entire rooms, and shared rooms. This is shown in Figure 1 below.

### Machine Learning Approach
We applied multiple classical machine learning techniques:

#### Regression Models
We trained Ridge Regression and Random Forest models to predict price based on features such as location, room type, etc.

![My Image](type.png)
*Figure 1: Room Type and neighbourhood group median nightly price (winsorised) *

The Random Forest model performed better than the Ridge model with lower error (RMSE ~ 0.42 in log scale). Using this model, the top drivers of the model were depicted in Figure 2. Again, room or home type was shown as being the main price driver.

![My Image](type.png)
*Figure 1: The main top 20 price drivers – Importance of Random Forest feature *


#### Clustering (K-Means)
We used K-Means clustering to segment listings into distinct market groups. Optimal *k = 8* was deducted based on silhouette scores. Clear segments were identified ranging from budget listings to premium listings.






Clustering (K-Means)

We used K-Means clustering to segment listings into distinct market groups.

Optimal k = 8 based on silhouette scores

Identified clear segments ranging from budget listings to premium listings

Demand Modelling

We used Logistic Regression to estimate the probability of high demand, defined using reviews per month within each segment.

This allowed us to combine:

Expected Value = Price × Probability of High Demand

4. Pricing Strategy and Business Insights

We developed a pricing framework that balanced:

Booking probability

Expected revenue

Market segment characteristics

This produced three pricing strategies:

Booking-maximising price

Revenue-maximising price

Compromise price (recommended strategy)

Overall, results showed that mid-range pricing often achieved the best balance between demand and earnings, and that pricing must be tailored by both neighbourhood and room type.

5. Key Findings

Location and room type are the strongest pricing determinants

Demand follows a non-linear relationship with price

The market can be meaningfully segmented into distinct clusters

Data-driven pricing can improve both occupancy and revenue outcomes

6. Team Working and Process Reflection (INSERT HERE – IMPORTANT)

👉 This is where you describe how your group actually worked.

Include:

How tasks were divided (e.g., modelling, cleaning, visuals, report writing)

Tools used (Python, Jupyter, sklearn, etc.)

Collaboration style (weekly meetings, GitHub, shared notebooks, etc.)

Any communication issues or coordination strategies

Insert here:

[Your description of team workflow, collaboration, and division of labour]

7. Challenges and Limitations (INSERT HERE)

👉 This section is important for reflection marks.

You can include things like:

Missing real booking/occupancy data

Using reviews as a proxy for demand

Outlier handling decisions (and why they were needed)

Model limitations (e.g., no temporal data, limited features)

Clustering interpretability issues

Insert here:

[Your specific challenges and how you addressed them]

8. Lecturer Feedback and Improvements (INSERT HERE)

👉 This is where you show learning and critical improvement.

Include:

What feedback you received

What you would improve (e.g., better validation, more features, cross-validation, feature engineering)

How you would extend the project

Insert here:

[Lecturer feedback + your response to it]

9. Conclusion

This project demonstrated how classical machine learning can be applied to a real-world platform economy problem. By combining regression, clustering, and classification techniques, we were able to generate actionable insights for Airbnb pricing strategy.

The work highlights the importance of data preprocessing, feature selection, and segment-specific modelling when dealing with heterogeneous marketplace data.














[← Back to Home](https://mmiz02.github.io/portfoliomachinelearning/)

