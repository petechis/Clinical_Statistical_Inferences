import streamlit as st

# constant Variables
star_and_space = '&#x2605;&nbsp;'
square_bullet_point ='&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&#x2751'


theory_explained=[
    "Inferential statistics allow us to make inferences about a population based on a sample. In clinical trials, this helps determine whether a new drug is effective.",
    "Regression Analysis is statistical processes for estimating the relationships between a dependent variable, often called the outcome or response variable, and one or more independent variables.",
    "Analysis of Variance (ANOVA) is a statistical method used to test whether there are any statistically significant differences between the means of three or more independent groups.", 
    "The Chi-Square test is a statistical method used to determine if there is a significant association between two categorical variables. It compares the observed frequencies in each category to the frequencies expected if there were no association.",
    "Survival analysis, also known as time-to-event analysis, is a branch of statistics that deals with analyzing the expected duration until one or more events happen, such as death in biological organisms or failure in mechanical systems",
    "Meta-analysis is a statistical technique that combines the results of multiple scientific studies to identify patterns, discrepancies, or other interesting insights",
    "Bayesian statistics is a mathematical approach that uses probability to represent uncertainty in statistical models. It incorporates prior knowledge (prior distribution) and updates this knowledge with new evidence (likelihood) to form a posterior distribution.",
    "Cluster analysis is a statistical method used to group similar objects into clusters. In the context of clinical trials, it helps identify subgroups of patients who respond similarly to treatments. This can lead to more personalized and effective therapies.",
    "Time series analysis involves analyzing data points collected or recorded at specific time intervals. In clinical trials, it helps monitor changes in patient responses to treatments over time, identifying trends, patterns, and potential anomalies.",
    "Intention-to-treat (ITT) analysis is a method used in randomized controlled trials (RCTs) where all participants who are randomized are included in the statistical analysis and analyzed according to the group they were originally assigned, regardless of what treatment (if any) they received1. This approach helps to maintain the benefits of randomization and provides an unbiased estimate of treatment effect."
]    

purpose_expertise = [
    "The Project objective was to analyze the effectiveness of a new drug developed by Chisamba Pharmaceuticals using inferential statistics. The analysis will include calculating confidence intervals and p-values to determine the drug’s efficacy.",
    "The purpose of this project is to analyze the dose-response relationship in a clinical trial conducted by Chisamba Pharmaceuticals. Dose-response analysis helps in understanding the effect of different doses of a drug on patients. This is crucial for determining the optimal dose that maximizes efficacy while minimizing side effects.",
    "ANOVA helps in determining if the observed variations among group means are due to actual differences or just random chance. Its purpose is to compare the means of multiple groups, to understand if at least one group mean is different from the others and to control for Type I errors that can occur when multiple t-tests are conducted.",
    "Its purpose in clinical trials is, understanding the relationship between treatment types and patient outcomes is crucial for evaluating the effectiveness of treatments.",
    "The primary purpose is to estimate the survival function, compare survival rates between different groups, and assess the relationship between covariates and survival time2.",
    "In the context of pharmaceutical and clinical trials, Meta Analysis helps in: (1) Aggregating Evidence: Combining data from various studies to provide a more robust conclusion, (2) Identifying Trends: Detecting trends that might not be apparent in individual studies, (3) Improving Precision: Increasing the statistical power by pooling data and (4) Guiding Decision-Making: Informing clinical guidelines and policy decisions based on comprehensive evidence.",
    "The purpose of this project is to showcase how Bayesian methods can be used in clinical trials to continuously update the probability of a drug’s efficacy as new data is collected. This approach allows for more flexible and informed decision-making compared to traditional frequentist methods.",
    "The context of this project is to demonstrate how cluster analysis can be applied to clinical trial data to identify patient subgroups with similar responses to a therapy. This can help in tailoring treatments to specific patient groups, improving outcomes and reducing side effects.",
    "The purpose of this project is to demonstrate how time series analysis can be applied to clinical trial data to monitor treatment efficacy over time. This can help in understanding the long-term effects of treatments and making informed decisions about patient care.",
    "This project demonstrates how ITT analysis can be applied to clinical trial data to ensure the validity of the results and prevent bias due to dropouts or non-compliance. This helps in making reliable conclusions about the efficacy of treatments."
]




