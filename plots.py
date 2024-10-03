import numpy as np
import pandas as pd
import plotly.graph_objects as go

from scipy import stats
from scipy.stats import beta
from scipy.stats import norm
from sklearn.cluster import KMeans
from lifelines import KaplanMeierFitter
from scipy.stats import chi2_contingency 
from scipy.optimize import curve_fit
from sklearn.preprocessing import StandardScaler

import streamlit as st

def plot_clinical_statistical_inference():
    
    # Fictitious dataset
    np.random.seed(42)
    data = {
        'patient_id': range(1, 101),
        'before_treatment': np.random.normal(loc=50, scale=10, size=100),
        'after_treatment': np.random.normal(loc=55, scale=10, size=100)
    }
    df = pd.DataFrame(data)

    # Calculate the difference
    df['difference'] = df['after_treatment'] - df['before_treatment']

    # Calculate mean and standard deviation
    mean_diff = df['difference'].mean()
    std_diff = df['difference'].std()

    # Calculate confidence interval
    confidence_level = 0.95
    degrees_freedom = len(df) - 1
    confidence_interval = stats.t.interval(confidence_level, degrees_freedom, mean_diff, std_diff/np.sqrt(len(df)))

    # Perform t-test
    t_stat, p_value = stats.ttest_rel(df['before_treatment'], df['after_treatment'])

    # Data storytelling with Plotly
    fig = go.Figure()

    # Add traces
    fig.add_trace(go.Scatter(x=df['patient_id'], y=df['before_treatment'], mode='markers', name='Before Treatment'))
    fig.add_trace(go.Scatter(x=df['patient_id'], y=df['after_treatment'], mode='markers', name='After Treatment'))

    # Add confidence interval
    fig.add_trace(go.Scatter(
        x=[0, 100], y=[confidence_interval[0], confidence_interval[0]],
        mode='lines', name='Lower Confidence Interval', line=dict(dash='dash')
    ))
    fig.add_trace(go.Scatter(
        x=[0, 100], y=[confidence_interval[1], confidence_interval[1]],
        mode='lines', name='Upper Confidence Interval', line=dict(dash='dash')
    ))

    # Add layout
    fig.update_layout(
        title='Clinical Trial Results for Chisamba Pharmaceuticals.',
        title_x=0.22,  # Title aligned with grid
        title_y=0.90,  # Title positioned near the top vertically
        xaxis_title='Patient ID.',
        yaxis_title='Measurement.',
        legend_title='Legend.'
    )
    # Show plot
    st.plotly_chart(fig)

    # Print results
    #print(f"Mean Difference: {mean_diff}")
    #print(f"Confidence Interval: {confidence_interval}")
    #print(f"P-Value: {p_value}")

def plot_regression_analyis():  

    # Fictitious dataset
    data = {
        'Dose': [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
        'Response': [0, 1.2, 2.3, 3.1, 4.5, 5.1, 5.8, 6.2, 6.5, 6.8, 7.0]
    }

    df = pd.DataFrame(data)

    # Define the dose-response function
    def dose_response(dose, a, b, c):
        return a * np.exp(-b * dose) + c

    # Fit the model
    popt, pcov = curve_fit(dose_response, df['Dose'], df['Response'])

    # Generate predictions
    df['Predicted_Response'] = dose_response(df['Dose'], *popt)

    # Plotting the data
    fig = go.Figure()

    # Scatter plot of actual data
    fig.add_trace(go.Scatter(x=df['Dose'], y=df['Response'], mode='markers', name='Actual Response'))

    # Line plot of predicted data
    fig.add_trace(go.Scatter(x=df['Dose'], y=df['Predicted_Response'], mode='lines', name='Predicted Response'))

    # Adding titles and labels
    fig.update_layout(
        title='Dose-Response Analysis for Chisamba Pharmaceutical Co.',
        title_x=0.22,  # Title aligned with grid
        title_y=0.90,  # Title positioned near the top vertically
        xaxis_title='Dose.',
        yaxis_title='Response.',
        legend_title='Legend.'
    )

    st.plotly_chart(fig)

def plot_anova_analysis():

    # Fictitious dataset
    np.random.seed(42)
    data = {
        'Group': np.repeat(['Drug A', 'Drug B', 'Drug C'], 30),
        'Response': np.concatenate([
            np.random.normal(50, 10, 30),
            np.random.normal(55, 10, 30),
            np.random.normal(60, 10, 30)
        ])
    }

    df = pd.DataFrame(data)

    # Perform ANOVA
    anova_result = stats.f_oneway(
        df[df['Group'] == 'Drug A']['Response'],
        df[df['Group'] == 'Drug B']['Response'],
        df[df['Group'] == 'Drug C']['Response']
    )

    # Print ANOVA result
    print(f'F-statistic: {anova_result.statistic}, p-value: {anova_result.pvalue}')

    # Data visualization using Plotly
    fig = go.Figure()

    for group in df['Group'].unique():
        fig.add_trace(go.Box(
            y=df[df['Group'] == group]['Response'],
            name=group
        ))

    fig.update_layout(
        title='ANOVA Analysis of Drug Responses.',
        title_x=0.28,  # Title aligned with grid
        title_y=0.90,  # Title positioned near the top vertically
        yaxis_title='Response.',
        xaxis_title='Drug Type.'
    )

    st.plotly_chart(fig)
    
def plot_chi_square_test():

    # Create a fictitious dataset
    data = {
        'PatientID': range(1, 101),
        'Treatment': np.random.choice(['Drug A', 'Drug B'], 100),
        'Outcome': np.random.choice(['Improved', 'Not Improved'], 100)
    }

    df = pd.DataFrame(data)

    # Create a contingency table
    contingency_table = pd.crosstab(df['Treatment'], df['Outcome'])

    # Perform Chi-Square test
    chi2, p, dof, expected = chi2_contingency(contingency_table)

    # Print results
    #print(f"Chi-Square Statistic: {chi2}")
    #print(f"P-Value: {p}")
    #print(f"Degrees of Freedom: {dof}")
    #print("Expected Frequencies:")
    #print(expected)

    # Visualize the results
    fig = go.Figure(data=[
        go.Bar(name='Observed', x=contingency_table.columns, y=contingency_table.loc['Drug A'], marker_color='indianred'),
        go.Bar(name='Expected', x=contingency_table.columns, y=expected[0], marker_color='lightsalmon')
    ])

    fig.update_layout(
        title='Chi-Square Test Results for Treatment Outcomes.',
        title_x=0.25,  # Title aligned with grid
        title_y=0.90,  # Title positioned near the top vertically
        xaxis_title='Outcome.',
        yaxis_title='Frequency.',
        barmode='group'
    )
    st.plotly_chart(fig)


def plot_survival_analysis():   

    # Sample fictitious dataset
    np.random.seed(42)
    data = {
        'PatientID': range(1, 201),
        'Treatment': np.random.choice(['Drug A', 'Drug B'], size=200),
        'Time': np.random.exponential(scale=365, size=200),  # Time in days
        'Event': np.random.binomial(1, 0.7, size=200)  # Event occurred (1) or censored (0)
    }

    df = pd.DataFrame(data)

    # Separate data by treatment
    drug_a = df[df['Treatment'] == 'Drug A']
    drug_b = df[df['Treatment'] == 'Drug B']

    # Kaplan-Meier Fitter
    kmf_a = KaplanMeierFitter()
    kmf_b = KaplanMeierFitter()

    # Fit the data
    kmf_a.fit(durations=drug_a['Time'], event_observed=drug_a['Event'], label='Drug A')
    kmf_b.fit(durations=drug_b['Time'], event_observed=drug_b['Event'], label='Drug B')

    # Visualization using plotly.graph_objects
    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=kmf_a.survival_function_.index,
        y=kmf_a.survival_function_['Drug A'],
        mode='lines',
        name='Drug A'
    ))

    fig.add_trace(go.Scatter(
        x=kmf_b.survival_function_.index,
        y=kmf_b.survival_function_['Drug B'],
        mode='lines',
        name='Drug B'
    ))

    fig.update_layout(
        title='Survival Probability Over Time.',
        title_x=0.23,  # Title aligned with grid
        title_y=0.90,  # Title positioned near the top vertically
        xaxis_title='Time (days).',
        yaxis_title='Survival Probability.',
        template='plotly_dark'
    )
    st.plotly_chart(fig)


def plot_meta_analysis():   

    # Fictitious dataset
    data = {
        'Study': ['Study 1', 'Study 2', 'Study 3', 'Study 4', 'Study 5'],
        'Effect_Size': [0.2, 0.5, 0.3, 0.4, 0.6],
        'Standard_Error': [0.1, 0.15, 0.1, 0.2, 0.15]
    }

    df = pd.DataFrame(data)

    # Calculate weights
    df['Weight'] = 1 / df['Standard_Error']**2

    # Calculate combined effect size
    combined_effect_size = (df['Effect_Size'] * df['Weight']).sum() / df['Weight'].sum()

    # Calculate combined standard error
    combined_se = (1 / df['Weight'].sum())**0.5

    # Confidence interval
    ci_lower = combined_effect_size - 1.96 * combined_se
    ci_upper = combined_effect_size + 1.96 * combined_se

    # Create plot
    fig = go.Figure()

    # Add individual studies
    for i, row in df.iterrows():
        fig.add_trace(go.Scatter(
            x=[row['Effect_Size']],
            y=[row['Study']],
            mode='markers',
            marker=dict(size=10),
            name=row['Study']
        ))

    # Add combined effect size
    fig.add_trace(go.Scatter(
        x=[combined_effect_size],
        y=[-1],
        mode='markers',
        marker=dict(size=15, color='red'),
        name='Combined Effect Size'
    ))

    # Add confidence interval
    fig.add_trace(go.Scatter(
        x=[ci_lower, ci_upper],
        y=[-1, -1],
        mode='lines',
        line=dict(color='red', width=2),
        name='95% CI'
    ))

    fig.update_layout(
        title='Meta-Analysis of Clinical Trials',
        xaxis_title='Effect Size',
        yaxis_title='Study',
        yaxis=dict(tickvals=list(range(-1, len(df))), ticktext=['Combined'] + df['Study'].tolist())
    )
    st.plotly_chart(fig)

    
def plot_bayesian_inference_statistics():

    # Generate a fictitious dataset
    np.random.seed(42)
    data = pd.DataFrame({
        'patient_id': range(1, 101),
        'treatment': np.random.binomial(1, 0.5, 100),
        'outcome': np.random.binomial(1, 0.6, 100)
    })

    # Prior knowledge: Assume prior success rate is 0.5 with 10 prior trials
    prior_successes = 5
    prior_failures = 5

    # Calculate posterior distribution parameters
    successes = data[data['treatment'] == 1]['outcome'].sum()
    failures = data[data['treatment'] == 1]['outcome'].count() - successes

    posterior_successes = prior_successes + successes
    posterior_failures = prior_failures + failures

    # Plot the prior and posterior distributions
    x = np.linspace(0, 1, 100)
    prior_dist = beta(prior_successes, prior_failures).pdf(x)
    posterior_dist = beta(posterior_successes, posterior_failures).pdf(x)

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=x, y=prior_dist, mode='lines', name='Prior Distribution'))
    fig.add_trace(go.Scatter(x=x, y=posterior_dist, mode='lines', name='Posterior Distribution'))

    fig.update_layout(
        title='Prior and Posterior Distributions of Treatment Success Rate.',
        title_x=0.22,  # Title aligned with grid
        title_y=0.90,  # Title positioned near the top vertically
        xaxis_title='Success Rate.',
        yaxis_title='Density.'
    )

    st.plotly_chart(fig)


def plot_cluster_analysis():
    # Sample fictitious dataset
    data = {
        'PatientID': range(1, 101),
        'Age': np.random.randint(20, 70, 100),
        'BMI': np.random.uniform(18.5, 35, 100),
        'BloodPressure': np.random.uniform(120, 180, 100),
        'Cholesterol': np.random.uniform(150, 250, 100),
        'ResponseToTherapy': np.random.uniform(0, 1, 100)
    }

    df = pd.DataFrame(data)

    # Data preprocessing
    features = ['Age', 'BMI', 'BloodPressure', 'Cholesterol', 'ResponseToTherapy']
    X = df[features]
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # KMeans clustering
    kmeans = KMeans(n_clusters=3, random_state=42)
    df['Cluster'] = kmeans.fit_predict(X_scaled)

    # Visualization using plotly.graph_objects
    fig = go.Figure()

    for cluster in df['Cluster'].unique():
        cluster_data = df[df['Cluster'] == cluster]
        fig.add_trace(go.Scatter3d(
            x=cluster_data['Age'],
            y=cluster_data['BMI'],
            z=cluster_data['ResponseToTherapy'],
            mode='markers',
            marker=dict(size=5),
            name=f'Cluster {cluster}'
        ))

    fig.update_layout(
        title='Patient Clusters Based on Therapy Response.',
        title_x=0.28,  # Title aligned with grid
        title_y=0.90,  # Title positioned near the top vertically
        scene=dict(
            xaxis_title='Age.',
            yaxis_title='BMI.',
            zaxis_title='Response to Therapy.'
        )
    )

    st.plotly_chart(fig)

def plot_times_series_analysis():
    # Sample fictitious dataset
    np.random.seed(42)
    dates = pd.date_range(start='2024-10-01', periods=100, freq='D')
    data = {
        'Date': dates,
        'PatientID': np.random.choice(range(1, 21), size=100),
        'TreatmentEfficacy': np.random.uniform(0.5, 1.5, size=100) + np.sin(np.linspace(0, 10, 100))
    }

    df = pd.DataFrame(data)

    # Aggregate data by date
    df_agg = df.groupby('Date').mean().reset_index()

    # Visualization using plotly.graph_objects
    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=df_agg['Date'],
        y=df_agg['TreatmentEfficacy'],
        mode='lines+markers',
        name='Treatment Efficacy'
    ))

    fig.update_layout(
        title='Time Series Analysis of Treatment Efficacy.',
        title_x=0.25,  # Title aligned with grid
        title_y=0.90,  # Title positioned near the top vertically
        xaxis_title='Date.',
        yaxis_title='Average Treatment Efficacy.',
        template='plotly_dark'
    )

    st.plotly_chart(fig)
    

def plot_intention_to_treat_analysis():
   
    # Sample fictitious dataset
    np.random.seed(42)
    data = {
        'PatientID': range(1, 101),
        'AssignedGroup': np.random.choice(['Treatment', 'Control'], size=100),
        'CompletedTreatment': np.random.choice([True, False], size=100, p=[0.8, 0.2]),
        'Outcome': np.random.normal(loc=0, scale=1, size=100)
    }

    df = pd.DataFrame(data)

    # Introduce some missing data to simulate dropouts
    df.loc[df['CompletedTreatment'] == False, 'Outcome'] = np.nan

    # Impute missing data for ITT analysis (e.g., using mean imputation)
    df['Outcome'].fillna(df['Outcome'].mean(), inplace=True)

    # Group by AssignedGroup and calculate mean outcome
    grouped = df.groupby('AssignedGroup')['Outcome'].mean().reset_index()

    # Visualization using plotly.graph_objects
    fig = go.Figure()

    fig.add_trace(go.Bar(
        x=grouped['AssignedGroup'],
        y=grouped['Outcome'],
        name='Mean Outcome',
        marker_color=['blue', 'red']
    ))

    fig.update_layout(
        title='Intention-to-Treat Analysis of Treatment Efficacy.',
        title_x=0.25,  # Title aligned with grid
        title_y=0.90,  # Title positioned near the top vertically
        xaxis_title='Assigned Group.',
        yaxis_title='Mean Outcome.',
        template='plotly_dark'
        
    )

    st.plotly_chart(fig)