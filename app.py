import streamlit as st
import matplotlib.pyplot as plt
import plots as ps
import utilities as us

st.set_page_config(page_title="Statistical Inferences!", layout="wide")

def generate_chart(theory):
    fig, ax = plt.subplots()
    # Example: Generate a simple bar chart
    ax.bar(["Feature 1", "Feature 2", "Feature 3"], [10, 20, 15])
    ax.set_title(f"Chart for {theory}")
    return fig

# Define the list of marketing statistical theories
theories = [
    "Theory 1: Inferential Statistical Analysis.",
    "Theory 2: Regression Analysis.",
    "Theory 3: Analysis Of Variance (ANOVA).",
    "Theory 4: Chi-Square Test.",
    "Theory 5: Survival Analysis.",
    "Theory 6: Meta-Analysis.",
    "Theory 7: Bayesian statistics.",
    "Theory 8: Cluster Analysis.",
    "Theory 9: Time-Series-Analysis.",
    "Theory 10: Intention-To-Treat (ITT) analysis."
   ]

def custom_title(title, color, fsize, weight):
    st.markdown(f"<p style='text-align: center; color : {color}; font-size:{fsize}px; font-weight:{weight}'>{title}</p>", unsafe_allow_html=True)

def custom_sidebar_title(title, color, fsize, weight):
    st.sidebar.markdown(f"<p style='text-align: left; color : {color}; font-size:{fsize}px; font-weight:{weight}'>{title}</p>", unsafe_allow_html=True)

def custom_text(text, color, fsize, weight):
    st.sidebar.markdown(f"<p style='text-align: left; color : {color}; font-size:{fsize}px; font-weight:{weight}'>{text}</p>", unsafe_allow_html=True)

def custom_text_main(title, color, fsize, weight, align):
    st.markdown(f"<p style='text-align: {align}; color : {color}; font-size:{fsize}px; font-weight:{weight}'>{title}</p>", unsafe_allow_html=True)


st.sidebar.image('img/chisamba_clinical_trials.png')

# Sidebar for selecting a theory
selected_theory = st.sidebar.selectbox("Select a Statistical Analysis Theory.", theories)
#st.sidebar.divider()
custom_sidebar_title(f'{us.star_and_space}Theory Explained:','lightgray',24,'normal')

with st.sidebar:
    if "1:" in selected_theory:
        custom_text(us.theory_explained[0],'white',12,'normal')  
    elif "2:" in selected_theory:
        custom_text(us.theory_explained[1],'white',12,'normal')  
    elif "3:" in selected_theory:
        custom_text(us.theory_explained[2],'white',12,'normal') 
    elif "4:" in selected_theory:
        custom_text(us.theory_explained[3],'white',12,'normal') 
    elif "5:" in selected_theory:
        custom_text(us.theory_explained[4],'white',12,'normal')
    elif "6:" in selected_theory:
        custom_text(us.theory_explained[5],'white',12,'normal')
    elif "7:" in selected_theory:
        custom_text(us.theory_explained[6],'white',12,'normal')
    elif "8:" in selected_theory:
        custom_text(us.theory_explained[7],'white',12,'normal')
    elif "9:" in selected_theory:
        custom_text(us.theory_explained[8],'white',12,'normal')
    elif "10:" in selected_theory:
        custom_text(us.theory_explained[9],'white',12,'normal')
    else:
        st.write("No more data to analyse ...")    
    
custom_sidebar_title(f'{us.star_and_space} Purpose & Expertise:','lightgray',24,'normal')
with st.sidebar:
    if "1:" in selected_theory:
        custom_text(us.purpose_expertise[0],'white',12,'normal')
    elif "2:" in selected_theory:
        custom_text(us.purpose_expertise[1],'white',12,'normal')
    elif "3:" in selected_theory:
        custom_text(us.purpose_expertise[2],'white',12,'normal') 
    elif "4:" in selected_theory:
        custom_text(us.purpose_expertise[3],'white',12,'normal')  
    elif "5:" in selected_theory:
        custom_text(us.purpose_expertise[4],'white',12,'normal')
    elif "6:" in selected_theory:
        custom_text(us.purpose_expertise[5],'white',12,'normal')  
    elif "7:" in selected_theory:
        custom_text(us.purpose_expertise[6],'white',12,'normal') 
    elif "8:" in selected_theory:
        custom_text(us.purpose_expertise[7],'white',12,'normal') 
    elif "9:" in selected_theory:
        custom_text(us.purpose_expertise[8],'white',12,'normal')
    elif "10:" in selected_theory:
        custom_text(us.purpose_expertise[9],'white',12,'normal')
    else:
        st.write("Work in progress!")


# Function to generate a chart based on the selected theory
def generate_chart(theory):
    fig, ax = plt.subplots()
    # Example: Generate a simple bar chart
    ax.bar(["Feature 1", "Feature 2", "Feature 3"], [10, 20, 15])
    ax.set_title(f"Chart for {theory}")
    return fig

# Function to explain the theory
def explain_theory(theory):
    explanations = {  
        "Theory 1: Inferential Statistical Analysis.": f"The graph shows the measurements before and after treatment for each patient. The confidence interval lines indicate the range within which we expect the true mean difference to lie. The p-value helps us determine the statistical significance of the observed difference. By visualizing the data, we can effectively communicate the impact of the new drug. The confidence intervals and p-values provide a statistical basis for our conclusions, making the analysis robust and credible. <br><b>Key Terminologies about Inferential Statistics:</b><br>{us.square_bullet_point} Inferential Statistics: Inferential statistics allow us to make inferences about a population based on a sample. In clinical trials, this helps determine whether a new drug is effective. <br>{us.square_bullet_point} Confidence Interval: A confidence interval gives a range of values within which we expect the true population parameter to lie, with a certain level of confidence (e.g., 95%). <br>{us.square_bullet_point} P-Value: The p-value helps us determine the significance of our results. A p-value less than 0.05 typically indicates that the results.",
        "Theory 2: Regression Analysis." : f"The graph above illustrates the relationship between the dose of a drug and the response observed in patients. The scatter plot represents the actual responses recorded during the clinical trial, while the line plot shows the predicted responses based on our regression model. The graph above illustrates the relationship between the dose of a drug and the response observed in patients. The scatter plot represents the actual responses recorded during the clinical trial, while the line plot shows the predicted responses based on our regression model.<br><b>Key Takeaways:</b><br>{us.square_bullet_point}Trend Identification: The positive slope of the regression line indicates a positive correlation between dose and response, suggesting that higher doses lead to increased responses.<br>{us.square_bullet_point}Optimal Dose: By analyzing the graph, we can identify the dose at which the response begins to plateau, indicating the optimal dose for maximum efficacy.<br>{us.square_bullet_point} Model Accuracy: The closeness of the actual data points to the regression line demonstrates the model’s accuracy in predicting responses.<br> This project not only showcases the application of regression analysis in a pharmaceutical context but also emphasizes the importance of data visualization and storytelling in conveying complex information effectively.",
        "Theory 3: Analysis Of Variance (ANOVA)." : f"The data science team at Chisamba pharmaceutical company analysed the response of three different drugs (Drug A, Drug B, and Drug C) on patients. The ANOVA test was conducted to determine if there are statistically significant differences between the mean responses of these drugs. <br><b>Outcome:</b><br>{us.square_bullet_point} The F-statistic and p-value from the ANOVA test indicate whether the differences between the group means are statistically significant. <br>{us.square_bullet_point} The box plot visualizes the distribution of responses for each drug, helping to identify any apparent differences.<br> By interpreting the ANOVA results and visualizing the data, they can effectively communicate the findings and make informed decisions about the efficacy of the drugs tested.",
        "Theory 4: Chi-Square Test.":f"From the graph, we can observe the comparison between the observed and expected frequencies of treatment outcomes. If the observed frequencies significantly differ from the expected frequencies, it suggests a potential relationship between the treatment type and the outcome. <br><br><b>Takeaways</b><br>{us.square_bullet_point} Statistical Insight: The Chi-Square test helps determine if there is a significant association between treatment and outcome.",
        "Theory 5: Survival Analysis." : f"The narrative: The graph shows the survival probability over time for patients treated with Drug A and Drug B. By comparing the survival curves, we can infer which treatment is more effective in prolonging patient survival. The outcome is, if the survival curve for Drug A is consistently above that of Drug B, it suggests that Drug A is more effective. Conversely, if the curves cross or Drug B’s curve is higher, Drug B might be the better option. <br><br><b>In clinical trials held by Chisamba Clinical Co., survival analysis helped in:</b><br> {us.square_bullet_point} Evaluating Treatment Efficacy: Comparing the time-to-event between treatment groups. <br> {us.square_bullet_point} Understanding Risk Factors: Identifying factors that influence the time-to-event. <br> {us.square_bullet_point} Handling Censored Data: Managing incomplete data due to patients dropping out or the study ending before the event occurs.",
        "Theory 6: Meta-Analysis." : f"In project, Chisamba Clinical Trials Co. aggregated data from its data warehouse to estimate the overall effect size of a treatment. The combined effect size was represented by the red marker, and the confidence interval was shown as a red line. <br>This visualization helps in understanding the overall effectiveness of the treatment across different studies. By synthesizing the data, Chisamba Clinical Trials Co. could: <br>{us.square_bullet_point} make more informed decisions about the treatment’s efficacy, <br>{us.square_bullet_point} identify trends across studies, <br>{us.square_bullet_point} and improve the precision of our estimates. <br> This approach is crucial for guiding clinical and policy decisions in the pharmaceutical industry.",
        "Theory 7: Bayesian statistics." : f"The graph shows how our understanding of the treatment’s success rate evolves as we incorporate new data. The posterior distribution is narrower and more peaked, indicating increased confidence in the estimated success rate. This project demonstrates the power of Bayesian statistics in clinical trials, allowing for continuous learning and more informed decision-making. <br><br><b>Explanation and Takeaways for the project at Chisamba Clinical Co.</b><br>{us.square_bullet_point} <b>Prior Distribution:</b> Represents our initial belief about the success rate of the treatment before seeing the data. In this case, we assumed a prior success rate of 0.5 based on 10 prior trials. <br> {us.square_bullet_point} <b>Posterior Distribution:</b> Updates our belief about the success rate after incorporating the new data from the clinical trial. The posterior distribution is more informative and reflects the combined knowledge from prior information and new evidence. <br> {us.square_bullet_point} <b>Data Storytelling:</b> The graph shows how the team the understanding of the treatment’s success rate evolves as they incorporate new data. The posterior distribution is narrower and more peaked, indicating increased confidence in the estimated success rate.<br><br>This project demonstrates the power of Bayesian statistics in clinical trials, allowing for continuous learning and more informed decision-making.",
        "Theory 8: Cluster Analysis.":f"The visualization above shows the clustering of patients based on their: <br>{us.square_bullet_point} age, <br>{us.square_bullet_point} BMI, <br>{us.square_bullet_point} and response to therapy. <br><br>Each cluster represents a subgroup of patients with similar characteristics and responses. By identifying these subgroups, Chisamba Pharmaceutical Company can tailor their treatments to improve efficacy and reduce adverse effects for each specific group.",
        "Theory 9: Time-Series-Analysis." : f"The visualization above shows the average treatment efficacy over time for patients in the clinical trial. <br> {us.square_bullet_point} Each point represents the average efficacy on a given day, and the line connects these points to show the trend over time. <br> {us.square_bullet_point} By analyzing these trends and patterns, Chisamba Pharmaceutical Company can gain insights into the long-term efficacy of their treatments and make data-driven decisions to improve patient outcomes.",
        "Theory 10: Intention-To-Treat (ITT) analysis." : f"The bar chart above shows the mean outcome for the treatment and control groups based on ITT analysis. <br> {us.square_bullet_point} Each bar represents the average outcome for patients in the respective groups, including those who did not complete the treatment. <br> {us.square_bullet_point} By applying ITT analysis, Chisamba Pharmaceutical Company can ensure the reliability and validity of their clinical trial results, leading to more accurate conclusions about the efficacy of their treatments."        
    }
    return explanations.get(theory, "Explanation not available.")

def main():
    # Main window to display the chart and explanation
    custom_title('Medical, Clinical & Pharma Statistical Methods.','orange',32,'bold')
    custom_title(f'{selected_theory}','red',18,'bold')   
    if "1:" in selected_theory:          
        (ps.plot_clinical_statistical_inference())         
        st.write("Data Story-Telling:")   
        custom_text_main(explain_theory(selected_theory),'orange',14,'italic','justified')        
    elif "2:" in selected_theory:          
        (ps.plot_regression_analyis()) 
        st.write("Data Story-Telling:")                
        custom_text_main(explain_theory(selected_theory),'orange',14,'italic','justified') 
    elif "3:" in selected_theory:          
        (ps.plot_anova_analysis()) 
        st.write("Data Story-Telling:")                
        custom_text_main(explain_theory(selected_theory),'orange',14,'italic','justified')
    elif "4:" in selected_theory:          
        (ps.plot_chi_square_test()) 
        st.write("Data Story-Telling:")                
        custom_text_main(explain_theory(selected_theory),'orange',14,'italic','justified') 
    elif "5:" in selected_theory:          
        (ps.plot_survival_analysis()) 
        st.write("Data Story-Telling:")                
        custom_text_main(explain_theory(selected_theory),'orange',14,'italic','justified')    
    elif "6:" in selected_theory:          
        (ps.plot_meta_analysis()) 
        st.write("Data Story-Telling:")                
        custom_text_main(explain_theory(selected_theory),'orange',14,'italic','justified') 
    elif "7:" in selected_theory:          
        (ps.plot_bayesian_inference_statistics()) 
        st.write("Data Story-Telling:")                
        custom_text_main(explain_theory(selected_theory),'orange',14,'italic','justified') 
    elif "8:" in selected_theory:          
        (ps.plot_cluster_analysis()) 
        st.write("Data Story-Telling:")                
        custom_text_main(explain_theory(selected_theory),'orange',14,'italic','justified') 
    elif "9:" in selected_theory:          
        (ps.plot_times_series_analysis()) 
        st.write("Data Story-Telling:")                
        custom_text_main(explain_theory(selected_theory),'orange',14,'italic','justified') 
    elif "10:" in selected_theory:          
        (ps.plot_intention_to_treat_analysis()) 
        st.write("Data Story-Telling:")                
        custom_text_main(explain_theory(selected_theory),'orange',14,'italic','justified')         
    else:
        st.pyplot(generate_chart(selected_theory))
        st.write(explain_theory(selected_theory))

if __name__ == "__main__":
        main()


