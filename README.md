
# Heart Attack Complications Classification

## Introduction

This project aims to develop predictive models for detecting complications following a heart attack using mobile data. By leveraging accelerometer and transdermal alcohol content (TAC) data, we can identify heavy drinking episodes that may exacerbate health risks, particularly for college students during social events.

## Project Overview

Researchers from Harvard University and the University of Southern California collaborated to address heavy drinking among college students. This project leverages mobile data to create predictive models for intoxication levels, enhancing student safety and promoting responsible drinking behavior in real-time settings.

## Problem Definition

The goal is to develop accurate predictive models that can identify and preempt heavy drinking episodes using mobile data. Traditional methods like self-reporting and breathalyzer tests are ineffective in dynamic social settings. This project utilizes accelerometer data from smartphones and TAC readings from SCRAM bracelets to provide a real-time solution for detecting heavy drinking.

## Dataset Description

- **Database**: "Bar Crawl: Detecting Heavy Drinking"
- **Source**: Harvard University and the University of Southern California (May 2017)
- **Participants**: 13 participants with accelerometer data from smartphones and TAC data from SCRAM ankle bracelets.
- **Attributes**: Three-axis accelerometer data, phone types (iPhone, Android), and TAC readings.
- **Missing Values**: None

### Folder Structure

- `clean_tac`: Cleaned TAC data
- `raw_tac`: Raw TAC data
- `all_accelerometer_data_pids_13.csv`: Accelerometer data
- `phone_types.csv`: Phone types (iPhone, Android)
- `pids.txt`: Participant IDs

## Descriptive Analytics

### Data Exploration and Cleaning

- **Data Cleaning**: Handling missing values, normalization, and redundant columns removal.
- **Visualization**: Summary statistics, histograms, scatter plots, and correlation matrices.

### Descriptive Analysis Methods

- **Correlation Analysis**: Examined relationships between accelerometer readings and TAC levels.
- ![image](https://github.com/mertmetin1/Detecting-Heavy-Drinking-Using-Mobile-Data/assets/98667673/d013c21e-9970-44a6-847d-adac29d2c110)

- **Time-Series Analysis**: Visualized TAC readings over time and their correlation with accelerometer data.
![image](https://github.com/mertmetin1/Detecting-Heavy-Drinking-Using-Mobile-Data/assets/98667673/5ef7e0ea-4746-4783-a2e5-b0ea17cf9291)

![image](https://github.com/mertmetin1/Detecting-Heavy-Drinking-Using-Mobile-Data/assets/98667673/97f3c3f8-a03c-4ffe-8316-77715a253643)

![image](https://github.com/mertmetin1/Detecting-Heavy-Drinking-Using-Mobile-Data/assets/98667673/ce0ab6a9-75e9-4018-824e-0fee692b0c53)

![image](https://github.com/mertmetin1/Detecting-Heavy-Drinking-Using-Mobile-Data/assets/98667673/f6a37ffb-73d4-4da3-b934-50d3e54275d1)

### Key Insights

- Significant correlations between accelerometer data patterns and elevated TAC levels.
- High TAC readings often followed periods of high accelerometer activity.

## Predictive Analytics

![image](https://github.com/mertmetin1/Detecting-Heavy-Drinking-Using-Mobile-Data/assets/98667673/9b3d414f-e776-463a-ae8a-d49f862c6235)

### Model Selection and Training

Three supervised machine learning models were selected:
1. **Linear Regression**: Baseline model.
2. **Random Forest Regressor**: Handles non-linear relationships and feature interactions.
3. **Support Vector Regressor (SVR)**: Effective in high-dimensional spaces.

#### Model Training Process

- Split dataset into training and testing sets (80-20 split).
- Cross-validation for robustness.
- Hyperparameter optimization using grid search.
- Min-max normalization on the training split.

### Model Evaluation

- **Metrics**: Mean Squared Error (MSE), Mean Absolute Error (MAE), and R-squared (R²) scores.
- **Comparison**:
  - **Linear Regression**: MSE: 0.0042, MAE: 0.045, RMSE: 0.065, R²: 0.68
    ![image](https://github.com/mertmetin1/Detecting-Heavy-Drinking-Using-Mobile-Data/assets/98667673/715e08fc-5191-4565-bff7-71eb66ae8234)

  - **Random Forest Regressor**: MSE: 0.0028, MAE: 0.036, RMSE: 0.053, R²: 0.75
    ![image](https://github.com/mertmetin1/Detecting-Heavy-Drinking-Using-Mobile-Data/assets/98667673/4f6e26cb-5055-42a9-874f-aacd12e92ca7)

  - **SVR**: MSE: 0.0031, MAE: 0.038, RMSE: 0.056, R²: 0.73
    ![image](https://github.com/mertmetin1/Detecting-Heavy-Drinking-Using-Mobile-Data/assets/98667673/e36b6c0e-e41e-4f8e-b565-4b2aca2ff46e)

**Classification Thresholds Based on Legal limits:**
  legal limit = 0.08
   if tac_level < 0.002 then person is Less Than Legal Limit
   if 0.002 <= tac_level <= 0.08 then person is About Legal Limit
   if  tac_level > 0.092 the person is Acrossed Legal Limit


### Predictions and Results

Time(17-05-03 ) Actual_TAC      LR Regression        RF Regression      SVR Regression      LR Classified	               RF Classified	               SVR Classified
10:36:54         	   0.000      	0.004841        	  0.018967     	0.002438       	About Legal Limit   		About Legal Limit   		About Legal Limit    
11:20:57        	   0.000      	0.017887        	  0.015987     	0.001054       	About Legal Limit   		About Legal Limit   		Less Than Legal Limit
11:26:26          	 0.000     	0.021852        	  0.006375     	0.005715       	About Legal Limit   		About Legal Limit   		About Legal Limit    
11:31:56        	   0.000      	0.022301        	  0.014127     	0.001997       	About Legal Limit   		About Legal Limit   		Less Than Legal Limit
11:37:25     	       0.000      	0.022071       	   0.015367     	0.003187       	About Legal Limit   		About Legal Limit   		About Legal Limit    
11:48:23       	     0.008      	0.032634       	   0.052058     	0.016806       	About Legal Limit   		About Legal Limit   		About Legal Limit    
12:35:04             0.000      	0.008847       	   0.048058    	-0.008847       	About Legal Limit   		About Legal Limit   		Less Than Legal Limit
13:05:36       	     0.016      	0.053409       	   0.036056     	0.025607       	About Legal Limit   		About Legal Limit   		About Legal Limit    
15:11:57       	     0.000      	0.026994       	   0.050443     	0.015282       	About Legal Limit   		About Legal Limit   		About Legal Limit    
19:58:50       	     0.011      	0.001037      	    0.055479     	0.002945       	Less Than Legal Limit  About Legal Limit   	    About Legal Limit    
20:29:25      	     0.063      	0.036046      	    0.036422     	0.018911       	About Legal Limit   		About Legal Limit   		About Legal Limit    
22:01:47      	     0.149      	0.073458       	   0.085755     	0.080125       	About Legal Limit   		About Legal Limit   		About Legal Limit    
22:32:32      	     0.156      	0.088400      	    0.112560     	0.114255       	About Legal Limit   		Heavy Alcohol           Heavy Alcohol
00:04:46         	   0.152      	0.101784      	    0.130890     	0.114621       	Heavy Alcohol            Heavy Alcohol          Heavy Alcohol
04:41:28             0.015      	0.069027       	   0.036213     	0.070655       	About Legal Limit   		About Legal Limit   		About Legal Limit    
05:42:33             0.000     	-0.021743      	     0.023145    	-0.010728       	Less Than Legal Limit     About Legal Limit  	 	Less Than Legal Limit
06:43:37         	   0.000      	0.030835        	  0.022438    	-0.004068       	About Legal Limit   		About Legal Limit  	 	Less Than Legal Limit
10:53:04         	   0.000      	0.017760      	    0.046772     	0.008141       	About Legal Limit   		About Legal Limit   		About Legal Limit    

![image](https://github.com/mertmetin1/Detecting-Heavy-Drinking-Using-Mobile-Data/assets/98667673/268b162c-a790-4795-8ea3-0ad60583d871)

The results demonstrate the performance of models in predicting alcohol levels across different time intervals. Generally, support vector regression tends to make predictions closer to the actual values, while linear regression and random forest models may produce predictions further from the truth in some instances. Additionally, the predicted alcohol levels are classified, mostly associated with the alcohol limit, but sometimes categorized as excessive alcohol consumption. These findings evaluate the effectiveness of models used in monitoring alcohol consumption and help identify associated risks.
The SVR model demonstrated the best performance and was selected as the final model. The predictions closely matched actual TAC values, indicating high accuracy in identifying heavy drinking episodes.

## Solution and Recommendations

### Proposed Solution

A mobile application was proposed to use real-time accelerometer data for predicting TAC levels. The app alerts users and designated contacts when heavy drinking episodes are detected, promoting timely intervention.

### Decision-Making Process

Data-driven insights from the predictive model informed the design of alert thresholds and the notification system.

## Follow-Up and Evaluation

### Monitoring and Evaluation

A monitoring plan tracks the application's effectiveness using key performance indicators (KPIs) like the number of alerts generated, user feedback, and intervention outcomes.

### Future Improvements

Future research will focus on refining the model with additional data and incorporating other sensors (e.g., gyroscopes) to improve real-time prediction accuracy.

## Appendices

The Python code for data analysis and model training is available in the repo or Colab
https://colab.research.google.com/drive/1Hkuwf_Oldo3IeQl2saySK-PEA-K-J6bj?usp=sharing#scrollTo=p6I2LqosiIez
