import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from wordcloud import WordCloud, STOPWORDS
import matplotlib.pyplot as plt
import numpy as np
import re 
import kaleido
from sklearn.feature_extraction.text import CountVectorizer
import streamlit as st
from PIL import Image
import requests
from io import BytesIO
from textwrap import wrap

from utils import *

st.set_page_config(
   layout='wide',
   initial_sidebar_state='collapsed'
)

#reading the data file as dataframe
url = 'https://raw.githubusercontent.com/COBA-NIH/2023_ImageAnalysisSurvey/main/Bridging%20Imaging%20Users%20to%20Imaging%20Analysis%20-%202022%20(Responses)%20-%20Form%20Responses%201_copy.csv'
data = pd.read_csv(url)
print(data.columns)

#Creating a dictionary to rename
dict_rename = {
    'Which of the following roles best describes you?':'Role', 
    'Which of the following do you have significant formal training in or experience with? Select all that apply.':'Training', 
    'Where do you currently primarily work?':'Location', 
    'How would you describe your work?':'Work description', 
    'How would you rate your computational skills?':'Level of computational skills', 
    'How would you rate your comfort in developing new computational skills?':'Comfort in developing computational skills', 
    'How do you generally go about solving an image analysis problem? Check the approach(es) you use the most.':'Approach to solutions', 
    'How frequently do you use scripting to solve image analysis problems?':'Frequency of script usage', 
    'In regards to learning more about image analysis, how preferable do you find each of these instructional methods? [Scholarly "best practices" article]':'Best practices articles',
    'In regards to learning more about image analysis, how preferable do you find each of these instructional methods? [Written step-by-step tutorial]':'Written tutorials',
    'In regards to learning more about image analysis, how preferable do you find each of these instructional methods? [Video tutorial]':'Video tutorial',
    'In regards to learning more about image analysis, how preferable do you find each of these instructional methods? [Interactive webinar]':'Interactive webinar',
    'In regards to learning more about image analysis, how preferable do you find each of these instructional methods? [One-on-one "office hours" with an expert]':'Office hours',
    'In regards to learning more about image analysis, how preferable do you find each of these instructional methods? [In person seminar/tutorial lasting <1 day]':'One day seminar',
    'In regards to learning more about image analysis, how preferable do you find each of these instructional methods? [Multiday workshop]':'Multiday workshop',
    'How interested are you in learning more about the following topics? [Image analysis theory]':'Image analysis theory',
    'How interested are you in learning more about the following topics? [General image analysis practices]':'Image analysis practices',
    'How interested are you in learning more about the following topics? [Image analysis practices related to my (sub) discipline]':'Topics related to sub discipline',
    'How interested are you in learning more about the following topics? [Learning to use a particular software tool]':'Specific software tool',
    'How interested are you in learning more about the following topics? [Deep learning as applied to image analysis]':'Deep learning for image analysis',
    'How interested are you in learning more about the following topics? [Analyzing large images/large numbers of images]':'Analyzing large images',
    'How interested are you in learning more about the following topics? [Visualizing image analysis results]':'Visualization of results',
    'The next question will ask you about particular image analysis tools and techniques. Do you want to answer questions about microscopy in the field/area of life sciences or physical sciences?':'Microscopy for life sciences physical sciences',
    'What image analysis tools have you used before? (check all that apply)':'usage of image analysis tools',
    'What image analysis tools do you use the most?':'Most used image analysis tools',
    'What kinds of images do you commonly want to analyze (select all that apply?* [Brightfield/DIC/phase-contrast of cells or organisms from manually selected fields]':'Brightfield/DIC/phase-contrast manually acquired',
    'What kinds of images do you commonly want to analyze (select all that apply?* [Brightfield/DIC/phase-contrast of cells or organisms from an automated microscope (such as a high content imager)]':'Brightfield/DIC/phase-contrast automated',
    'What kinds of images do you commonly want to analyze (select all that apply?* [Fluorescent images of cells/organisms from manually selected fields]':'Fluorescent images manually acquired',
    'What kinds of images do you commonly want to analyze (select all that apply?* [Fluorescent images of cells/organisms from an automated microscope (such as a high content imager)]':'Fluorescent images automated',
    'What kinds of images do you commonly want to analyze (select all that apply?* [Histologically stained tissue sections]':'Histological tissue sections',
    'What kinds of images do you commonly want to analyze (select all that apply?* [Electron microscopy]':'Electron microscopy',
    'What kinds of images do you commonly want to analyze (select all that apply?* [Imaging mass spectrometry]':'Imaging mass spectrometry',
    'What kinds of images do you commonly want to analyze (select all that apply?* [Imaging flow cytometry]':'Imaging flow cytometry',
    'What kinds of images do you commonly want to analyze (select all that apply?* [Superresolution (PALM/STORM)]':'Superresolution (PALM/STORM)',
    'What kinds of images do you commonly want to analyze (select all that apply?* [Autofluorescence imaging (ie FLIM)]':'Autofluorescence imaging (ie FLIM)',
    'What kinds of images do you commonly want to analyze (select all that apply?* [Other]':'Other',
    'What image analysis problems (i.e. finding nuclei, tissue analysis, analysis of super-resolution data, etc) do you think are generally well-solved?':'Well solved image analysis problems',
    'What image analysis problems (i.e. finding nuclei, tissue analysis, analysis of super-resolution data, etc) do you wish had easier/better solutions?':'Image analysis problems which needs better solutions',
    'What image analysis tools have you used before? (check all that apply)':'Commonly used image analysis tools',
    'What kinds of images do you commonly want to analyze (select all that apply)? [Optical microscopy/DIC/fractography]':'Optical microscopy/DIC/fractography',
    'What kinds of images do you commonly want to analyze (select all that apply)? [Scanning electron microscopy (secondary electron or back scattered imaging)]':'Scanning electron microscopy',
    'What kinds of images do you commonly want to analyze (select all that apply)? [Transmission electron microscopy (including electron diffraction and STEM imaging, e.g. HAADF-STEM)]':'Transmission electron microscopy',
    'What kinds of images do you commonly want to analyze (select all that apply)? [Spectroscopy/diffractive imaging in the SEM/TEM (eg. EDS, EBSD, EELS, CL)]':'Spectroscopy/diffractive imaging in the SEM/TEM',
    'What kinds of images do you commonly want to analyze (select all that apply)? [Imaging with mass spectrometry (eg SIMS/APT)]':'Imaging with mass spectrometry',
    'What kinds of images do you commonly want to analyze (select all that apply)? [X-ray microscopy (including tomography)]':'X-ray microscopy',
    'What kinds of images do you commonly want to analyze (select all that apply)? [FM imaging, force spectroscopy, single molecule force spectroscopy]':'Atomic force microscopy',
    'What kinds of images do you commonly want to analyze (select all that apply)? [Fluorescence microscopy]':'luorescence microscopy',
    'What kinds of images do you commonly want to analyze (select all that apply)? [Other]':'Other',
    'What image analysis problems (i.e. segmenting particles, 3D reconstruction, spectroscopic analysis, extracting force/mechanical property measurements, etc) do you think are generally well-solved?':'Well solved image analysis problems-physical sciences',
    'What image analysis problems (i.e. segmenting particles, 3D reconstruction, spectroscopic analysis, extracting force/mechanical property measurements, etc) do you wish had easier/better solutions?':'Image analysis problems which needs better solutions physical sciences',
    'Where did you hear about this survey? Please select all that apply.':'hear about this survey',
    'Please select any of the following you have attended in the past':'Previous experience',
    'Are there any image analysis workshops, tutorials, or conferences you are aware of and attended or considered attending? If so, how many?':'Number of conferences/workshops attended',
    'How would you most prefer to be notified about image analysis workshops, sessions, or conferences being planned?':'Notification of image analysis workshop',
    'Are there any image analysis workshops, tutorials, or conferences that you have participated in and found particularly helpful?  If yes, what made them beneficial?':'Experience with previous workshops/conferences',
    'Are there any conferences you\'ve attended in the past that you think would particularly benefit from the addition/expansion of image analysis offerings?':'Recommended conferences/workshops',
    'What specific topics (i.e. overviews of a particular tool, comparisons between pieces of software, or how to use a certain tool for a certain kind of experiment) would you like to see prioritized for future image analysis workshop and tutorial offerings?':'Topics of interest',
    'What do you think analysis tool CREATORS (such as software developers) could/should do to make image analysis better and more successful? How best could we encourage them to do it?':'Creators role',
    'What do you think analysis tool USERS (such as microscopists) could/should do to make image analysis better and more successful?  How best could we encourage them to do it?':'Users role',
    'Any other thoughts?':'Other thoughts',
    'Would you be interested in subscribing to a mailing list (announcing workshops, new tools, collaboration opportunities, etc) for EITHER the Center for Open Bioimage Analysis OR Bioimaging North America OR the Royal Microscopical society? If yes, you will be taken to a page to subscribe, if not this form will submit.':'Subscription'
    }
 
#renaming the columns of the dictionary
data.rename(columns=dict_rename, inplace=True)

role_val_counts = pd.DataFrame(data.Role.value_counts())
role_val_counts=role_val_counts.reset_index()
print(role_val_counts.columns)

role_val_counts = role_val_counts.replace('Undergraduate/Graduate student', 'Student - Graduate/Undergraduate')
role_val_counts = role_val_counts.rename(columns={'index':'roles', 'Role':'counts'})

# pie chart for the roles of the survey participants 
role_pie_chart = px.pie(data, values=role_val_counts['counts'], names=role_val_counts['roles'], title="Role", width=800, height=500)
role_pie_chart.update_traces(insidetextorientation = 'radial', textinfo='value+percent')
role_pie_chart.update_layout(title_x=0.8,title_y = 0.85, font=dict(family='Helvetica', color="Black", size=16), legend=dict(title_font_family = 'Helvetica', font=dict(size=16, color="Black")))

# The domains in which the participants work with respect to the roles
data["Training"] = data["Training"].astype(str) # this was done to avoid the NaN rows for further analysis 
# Based on the values in a particular query column('Training'), a new column was created with boolean values based on the presence of the keyterms in the query column;,done using lambda function
data['Cell/Molecular Biology'] = data['Training'].apply(lambda x: True if 'Cell/Molecular Biology' in x else False)
data['Physics/Biophysics'] = data["Training"].apply(lambda x: True if 'Physics/Biophysics'in x else False)
data['Chemistry/Biochemistry'] = data["Training"].apply(lambda x: True if 'Chemistry/Biochemistry'in x else False)
data['Statistics/Biostatistics'] =data['Training'].apply(lambda x: True if 'Statistics/Biostatistics' in x else False)
data['Medicine'] = data['Training'].apply(lambda x: True if 'Medicine' in x else False)
data['Computer science'] = data['Training'].apply(lambda x: True if 'Computer science' in x else False)
data['Computer vision'] = data['Training'].apply(lambda x: True if 'Computer vision' in x else False)
data['Deep learning'] = data['Training'].apply(lambda x: True if 'Deep learning' in x else False)
data['Developmental Biology'] = data['Training'].apply(lambda x: True if 'Developmental Biology' in x else False)

#slicing the columns of interest and performing counts to get the respective counts
df_Cell_Molecular_Biology = data.loc[:, ["Role","Cell/Molecular Biology"]]
Cell_Molecular_Biology = df_Cell_Molecular_Biology.groupby("Role").sum().reset_index()

df_Physics_Biophysics = data.loc[:, ["Role", "Physics/Biophysics"]]
Physics_Biophysics = df_Physics_Biophysics.groupby("Role").sum().reset_index()

df_Chemistry_Biochemistry =data.loc[:, ["Role", "Chemistry/Biochemistry"]]
Chemistry_Biochemistry = df_Chemistry_Biochemistry.groupby("Role").sum().reset_index()

df_Statistics_Biostatistics = data.loc[:, ["Role","Statistics/Biostatistics"]]
Statistics_Biostatistics = df_Statistics_Biostatistics.groupby("Role").sum().reset_index()

df_Medicine = data.loc[:, ["Role", "Medicine"]]
Medicine = df_Medicine.groupby("Role").sum().reset_index()

df_Computer_science = data.loc[:,["Role", "Computer science"]]
Computer_science = df_Computer_science.groupby("Role").sum().reset_index()

df_Computer_vision = data.loc[:, ["Role", "Computer vision"]]
Computer_vision = df_Computer_vision.groupby("Role").sum().reset_index()

df_Deep_learning = data.loc[:, ["Role", "Deep learning"]]
Deep_learning = df_Deep_learning.groupby("Role").sum().reset_index()

df_Developmental_Biology = data.loc[:, ["Role", "Developmental Biology"]]
Developmental_Biology = df_Developmental_Biology.groupby("Role").sum().reset_index()

#concatenating the dataframes with respect to roles and domains
df_role_domain = pd.concat([Cell_Molecular_Biology, Physics_Biophysics, Chemistry_Biochemistry, Statistics_Biostatistics, Medicine, Computer_science, Computer_vision, Deep_learning, Developmental_Biology], axis=1)

#since the roles column was repetitive, the duplicate 'role' columns were removed
df_role_domain = df_role_domain.T.drop_duplicates().T

role=df_role_domain["Role"]


# Stacked bar chart - role with domains on what the participants are working
x=role
layout = go.Layout(margin=go.layout.Margin(l=250))
fig = go.Figure(go.Bar(name='Developmental Biology', y=role, x=df_role_domain["Developmental Biology"], orientation='h', insidetextanchor='end', text=df_role_domain["Developmental Biology"]), layout=layout)
fig.add_trace(go.Bar(name='Deep learning', y=role, x=df_role_domain["Deep learning"], orientation='h', text=df_role_domain["Deep learning"]))
fig.add_trace(go.Bar(name='Computer vision', y=role, x=df_role_domain["Computer vision"],  orientation='h', text = df_role_domain["Computer vision"]))
fig.add_trace(go.Bar(name="Computer science", y=role, x=df_role_domain["Computer science"],  orientation='h', text=df_role_domain["Computer science"]))
fig.add_trace(go.Bar(name="Medicine", y=role, x=df_role_domain["Medicine"],  orientation='h', text =df_role_domain["Medicine"]))
fig.add_trace(go.Bar(name="Statistics/Biostatistics", y=role, x=df_role_domain["Statistics/Biostatistics"],  orientation='h', text =df_role_domain["Statistics/Biostatistics"]))
fig.add_trace(go.Bar(name="Chemistry/Biochemistry", y=role, x=df_role_domain["Chemistry/Biochemistry"],  orientation='h', text=df_role_domain["Chemistry/Biochemistry"]))
fig.add_trace(go.Bar(name="Physics/Biophysics", y=role, x=df_role_domain["Physics/Biophysics"],  orientation='h', text=df_role_domain["Physics/Biophysics"]))
fig.add_trace(go.Bar(name="Cell/Molecular Biology", y=role, x=df_role_domain["Cell/Molecular Biology"],  orientation='h', text=df_role_domain["Cell/Molecular Biology"]))

fig.update_layout(barmode='stack', title='Role Based Training & Experience', title_x=0.5, title_y=0.9, font=dict(family='Helvetica', color="Black", size=16), legend=dict(title_font_family = 'Helvetica', font=dict(size=16, color="Black")))
fig.update_layout(width=1500,legend=dict(yanchor="bottom",y=0.02,xanchor="right",x=0.99))
fig.update_yaxes(categoryorder = 'total ascending')


st.title('Bridging Imaging Users to Imaging Analysis - A community survey')

## Figure 1
with st.container():
    st.subheader('Figure 1')
    st.subheader('Survey respondents roles and training histories vary across the sampled responses')
    st.write('A)')
    st.plotly_chart(role_pie_chart,theme=None)
    st.write('Figure 1A) Answers to the multiple-choice question “Which of the following roles best describes you?”')
    st.write('B)')
    st.plotly_chart(fig, theme=None, use_container_width=False)
    st.write('Figure 1B) Answers to the check-box question “Which of the following do you have significant formal training in or experience with? Select all that apply.” Responses were categorized based on the answers provided for part 1A.')

# categorizing trainees vs nontrainees
data['Trainee status'] = data['Role'].dropna().apply(lambda x: 'Trainee' if x in ['Undergraduate/Graduate student', 'Postdoctoral fellow'] else 'Nontrainee')

#Geo chart for the location of the participants; locations were given based on a country that is centrally located in a continent
demographics = data["Location"]
demographics_chart = px.scatter_geo(demographics, locations=['UKR', 'USA', 'KGZ', 'BRA', 'AUS','TCD'],size_max=20,opacity=0.2, projection="natural earth", color = data.Location.value_counts().values,text=data.Location.value_counts(), title="Location", labels={'color':'Continent'}, width=600, height=400)
demographics_chart.update_layout(title_x=0.50,title_y=0.80, font=dict(family='Helvetica', color="Black", size=16), legend=dict(title_font_family = 'Helvetica', font=dict(size=16, color="Black")))
demographics_chart.update_layout(showlegend=False)
demographics_chart.update_coloraxes(showscale=False)
demographics_chart.update_traces(marker=dict(color='blue'))
demographics_chart.update_traces(marker={'size':25})

## Supplementary figure S1
st.subheader('Figure S1')
st.subheader('Survey respondents were drawn primarily from North America and Europe')
st.write('A)')
st.plotly_chart(demographics_chart)
st.write('Figure S1A)- Answers to the multiple-choice question “Where do you currently primarily work?”')
url = 'https://github.com/COBA-NIH/2023_ImageAnalysisSurvey/raw/main/Graphs/cellprofiler_website_access_2022.tif'
response = requests.get(url)
img = Image.open(BytesIO(response.content))
st.write('B)')
st.image(img)
st.write('Figure S1B) Image from Google Analytics showing the number of visitors to the Cellprofiler website in the year 2022. The scale bar indicates the number of visitors')

#Work description - creating new columns based on the values provided by the participants in the work description
data["Imaging"] = data.eval("`Work description` < 3")
data["Balanced"] = data.eval("`Work description` in [3,4,5]")
data["Analyst"] = data.eval("`Work description` > 5")
Work_type = [sum(data["Imaging"]), sum(data["Balanced"]), sum(data["Analyst"])]

# creating a column for work type
data.loc[data["Work description"] < 3, "Work type" ] = "Imaging"
data.loc[(data["Work description"] >= 3) & (data["Work description"] <= 5), "Work type"] = "Balanced"
data.loc[data["Work description"] > 5, "Work type"] = "Analyst"
work_type_counts = data["Work type"].value_counts()

# creating a column for knowledge in computational skills 
data.loc[data["Level of computational skills"] < 3, "Knowledge of computational skills"] = "Low skill"
data.loc[(data["Level of computational skills"] >= 3) & (data["Level of computational skills"] <= 5), "Knowledge of computational skills"] = "Medium skill"
data.loc[data["Level of computational skills"] > 5, "Knowledge of computational skills"] = "High skill"
computational_knowledge = data["Knowledge of computational skills"].value_counts()

#Dataframe for work description - distribution 
work_cat = data['Work description'].value_counts().to_frame().reset_index()
work_cat['Percentage'] = (work_cat['Work description']/work_cat['Work description'].sum())*100
work_cat['Percentage'] = work_cat['Percentage'].round(decimals=1)
work_cat.loc[work_cat['index'] <3, 'Category'] ='Imaging'
work_cat.loc[(work_cat['index'] >= 3) & (work_cat['index'] <= 5), 'Category'] ='Balanced'
work_cat.loc[work_cat['index'] >5 , 'Category'] ='Analyst'
work_cat = work_cat.sort_values(by='index')

#graph for work description -distribution 
work_cat_bar = px.bar(work_cat, x=work_cat['index'], y=work_cat['Work description'], labels={'index':'', 'Work description':'Counts'},text=work_cat['Percentage'],text_auto=True, color='Category', color_discrete_map = {'Imaging':'lightskyblue', 'Balanced':'darkseagreen', 'Analyst':'orchid'})
work_cat_bar.update_layout(width=700, height=500, legend_traceorder = 'normal' )
work_cat_bar.update_layout(title='Work description', title_x=0.4, title_y=0.95, font=dict(family='Helvetica', color="Black", size=16), legend=dict(title_font_family = 'Helvetica', font=dict(size=16, color="Black")))

#dataframe for categorization of computational skills 
comp_skill_distbn = data["Level of computational skills"].value_counts().to_frame().reset_index()
comp_skill_distbn['Percentage'] = (comp_skill_distbn['Level of computational skills']/comp_skill_distbn['Level of computational skills'].sum())*100
comp_skill_distbn['Percentage'] = comp_skill_distbn['Percentage'].round(decimals=1)
comp_skill_distbn.loc[comp_skill_distbn['index'] <3, 'Category'] = 'Low skill'
comp_skill_distbn.loc[(comp_skill_distbn['index'] >= 3) & (comp_skill_distbn['index'] <= 5), 'Category'] = 'Medium skill'
comp_skill_distbn.loc[comp_skill_distbn['index'] >5, 'Category'] = 'High skill'
comp_skill_distbn = comp_skill_distbn.sort_values(by='index')

#graph for computational skill- distribution 
comp_skill_distbn_bar = px.bar(comp_skill_distbn, x=comp_skill_distbn['index'], y=comp_skill_distbn['Level of computational skills'], labels={'index':'', 'Level of computational skills':'Counts'},text=comp_skill_distbn['Percentage'],text_auto=True, color='Category',
color_discrete_map = {'Low skill':'lightskyblue', 'Medium skill':'darkseagreen', 'High skill':'orchid'})
comp_skill_distbn_bar.update_layout(width=750, height=500)
comp_skill_distbn_bar.update_layout(title='Level of computational skills', title_x=0.3, title_y=0.95, font=dict(family='Helvetica', color="Black", size=16), legend=dict(title_font_family = 'Helvetica', font=dict(size=16, color="Black")))


# Comfort in developing new computational skills
data.loc[data["Comfort in developing computational skills"] < 3, 'Comfort'] = 'Low comfort'
data.loc[(data['Comfort in developing computational skills'] >= 3) & (data['Comfort in developing computational skills'] <= 5), 'Comfort'] = "Medium comfort"
data.loc[data['Comfort in developing computational skills'] > 5 , 'Comfort'] = "High comfort"

#dataframe for categorization of comfort in developing new computational skills
comf_distbn = data['Comfort in developing computational skills'].value_counts().to_frame().reset_index()
comf_distbn['Percentage'] = (comf_distbn['Comfort in developing computational skills']/comf_distbn['Comfort in developing computational skills'].sum())*100
comf_distbn['Percentage'] = comf_distbn['Percentage'].round(decimals=1)
comf_distbn.loc[comf_distbn['index'] <3, 'Category'] ='Low comfort'
comf_distbn.loc[(comf_distbn['index'] >= 3) & (comf_distbn['index'] <= 5), 'Category'] ='Medium comfort'
comf_distbn.loc[comf_distbn['index'] >5 , 'Category'] ='High  comfort'
comf_distbn = comf_distbn.sort_values(by='index')

#graph for the comfort in developing new computational skills - distribution 
comf_distbn_bar = px.bar(comf_distbn, x=comf_distbn['index'], y=comf_distbn['Comfort in developing computational skills'], labels={'index':'', 'Comfort in developing computational skills':'Counts'},text=comf_distbn['Percentage'],text_auto=True, color='Category', 
color_discrete_map = {'Low comfort':'lightskyblue', 'Medium comfort':'darkseagreen', 'High  comfort':'orchid'})
comf_distbn_bar.update_layout(width=750, height=500)
comf_distbn_bar.update_layout(title='Comfort in developing new computational skills', title_x=0.2, title_y=0.95, font=dict(family='Helvetica', color="Black", size=16), legend=dict(title_font_family = 'Helvetica', font=dict(size=16, color="Black")))


#combining the work type, comp skills and comfort 
work_comp_com_df = data.loc[:, ['Work type', 'Knowledge of computational skills', 'Comfort', 'Microscopy for life sciences physical sciences']]
work_comp_com_lif_df = work_comp_com_df.groupby('Microscopy for life sciences physical sciences').get_group('Life Sciences')
work_comp_com_phy_df = work_comp_com_df.groupby('Microscopy for life sciences physical sciences').get_group('Physical Sciences')

#grouping life science participants
work_comp_com_lif_df_grp = work_comp_com_lif_df.groupby('Comfort').value_counts().reset_index()
work_comp_com_lif_df_grp = work_comp_com_lif_df_grp.rename(columns={0:'counts'})

#grouping physical science participants 
work_comp_com_phy_df_grp = work_comp_com_phy_df.groupby('Comfort').value_counts().reset_index()
work_comp_com_phy_df_grp = work_comp_com_phy_df_grp.rename(columns={0:'counts'})

#Combined - both life and physical 
work_comp_com_df_grp = work_comp_com_df.groupby('Comfort').value_counts().reset_index()
work_comp_com_df_grp = work_comp_com_df_grp.rename(columns={0:'counts'})

#sunburst for life science participants 
sun_lif = px.sunburst(work_comp_com_lif_df_grp, path=["Work type", 'Knowledge of computational skills', 'Comfort'],color = 'Work type', color_discrete_map = {'Imaging':'lightskyblue', 'Balanced':'darkseagreen', 'Analyst':'orchid'}, values='counts', width=500, height=500)
sun_lif.update_traces(textinfo="label+percent parent", insidetextorientation = 'radial')
sun_lif.update_layout(title="Skills of the participants (Life Sciences)", title_x=0.3, font=dict(family='Helvetica', color="Black", size=16))


#sunburst for physical science participants 
sun_phy = px.sunburst(work_comp_com_phy_df_grp, path=["Work type", 'Knowledge of computational skills', 'Comfort'],color='Work type', color_discrete_map = {'Imaging':'lightskyblue', 'Balanced':'darkseagreen', 'Analyst':'orchid'}, values='counts', width=500, height=500)
sun_phy.update_traces(textinfo="label+percent parent", insidetextorientation = 'radial')
sun_phy.update_layout(title="Skills of the participants (Physical Sciences)", title_x=0.3, font=dict(family='Helvetica', color="Black", size=16))

work_comp_com_df_grp = work_comp_com_df_grp.sort_values(['Work type', 'Comfort', 'Knowledge of computational skills']).reset_index()

#sunburst for combined group 
lif_phy = px.sunburst(work_comp_com_df_grp, path=["Work type", 'Knowledge of computational skills', 'Comfort'], color = 'Work type', color_discrete_map = {'Imaging':'lightskyblue', 'Balanced':'darkseagreen', 'Analyst':'orchid'}, values='counts', width=500, height=500)
lif_phy.update_traces(textinfo="label+percent parent", insidetextorientation = 'radial', sort=False, selector=dict(type='sunburst'))
lif_phy.update_layout(title="Skills of the participants", title_x=0.35, font=dict(family='Helvetica', color="Black", size=16))
lif_phy.update_traces(leaf=dict(opacity = 1))
lif_phy.update_traces(sort=False, selector=dict(type='sunburst')) 


trainee_df = data.loc[:, ("Work type", 'Trainee status')]
trainee_worktype_fig  = sunburst(trainee_df, order_list=['Trainee status', "Work type"], color_column='Trainee status', custom_colors={'Trainee':'lightskyblue', 'Nontrainee':'darkseagreen'}, title='Worktype categorized based on trainee status')
trainee_worktype_fig.update_layout(title_x=0.3)
trainee_comp_comf_df = data.loc[:, ('Knowledge of computational skills','Comfort', 'Trainee status')]
trainee_comp_comf_fig = sunburst(trainee_comp_comf_df, order_list=['Trainee status', 'Knowledge of computational skills', 'Comfort'], color_column='Trainee status', custom_colors={'Trainee':'lightskyblue', 'Nontrainee':'darkseagreen'}, title='Computational skills of the trainees and nontrainees') 
trainee_comp_comf_fig.update_layout(title_x=0.25)


## Figure 2
with st.container():
    st.subheader('Figure 2')
    st.subheader('Skills of the participants')
    st.plotly_chart(lif_phy)
    st.write('Figure 2- Breakdown of answers to the multiple-choice questions “How would you describe your work?”, “How would you rate your computational skills?” and “How would you rate your comfort in developing new computational skills?” Percentages were rounded to the nearest percent; in outer wedges, percentages are of the adjacent inner wedge population. See methods and Supplemental Figure 2 for fuller descriptions of each category.')

## Supplementary figure 2

with st.container():
 st.subheader('Supplementary figure 2')
 st.subheader('Work description and skills of the participant')
 st.write('A)')
 st.plotly_chart(work_cat_bar, use_container_width=False)
 st.write('Supplementary figure 2A) Answers to the question “How would you describe your work?” on a linear scale of 1 to 7 with 1 defined as ‘Nearly entirely imaging (sample prep, optimizing/deciding on imaging modalities, acquiring images and data, etc)’ and 7 defined as ‘Nearly entirely image analysis (finding the right tools to analyze a particular experiment, optimizing the analysis, data mining)’. Scale values 1 and 2 were categorized as “Imaging” work type, values 3 to 5 were categorized as “Balanced” work type, and values 6 and 7 were categorized as “Analyst”. ')
 st.write('B)')
 st.plotly_chart(comp_skill_distbn_bar, use_container_width=False)
 st.write('Supplementary figure 2B) Answers to the question “How would you rate your computational skills?” on a linear scale of 1 to 7 with 1 defined as ‘Very poor’ and 7 defined as ‘Excellent’. Scale values 1 and 2 were categorized as “Low skill”, values 3 to 5 were categorized as “Medium skill” and values 6 and 7 were categorized as “High skill”. ')
 st.write('C)')
 st.plotly_chart(comf_distbn_bar, use_container_width=False)
 st.write('Supplementary figure 2C) Answers to the question “How would you rate your comfort in developing new computational skills?” on a linear scale of 1 to 7 with 1 defined as “Very uncomfortable” and 7 defined as “Very comfortable”. Scale values 1 and 2 were categorized as “Low comfort”, values 3 to 5 were categorized as “Medium comfort” and values 6 and 7 were categorized as “High comfort”.')
 st.write('D)')
 st.plotly_chart(trainee_worktype_fig, use_container_width=False)
 st.write('Supplementary figure 2D) Answers to the question “How would you describe your work?” were categorized as described in part A and the answers - ‘‘Undergraduate/Graduate student’ and ‘Postdoctoral fellow’ to the question “Which of the following roles best describes you?” were considered as ‘Trainees’ and other roles were considered as ‘Nontrainees’.')
 st.write('E)')
 st.plotly_chart(trainee_comp_comf_fig, use_container_width=False)
 st.write('Supplementary figure 2E) Answers to the question “How would you rate your computational skills?” and “How would you rate your comfort in developing new computational skills?” were categorized based on the trainee status.')
 st.write('F)')
 st.plotly_chart(sun_lif, use_container_width=False)
 st.write('Supplementary figure 2F) Answers to part A, B, and C were classified based on the responses to the question “The next question will ask you about particular image analysis tools and techniques. Do you want to answer questions about microscopy in the field/area of life sciences or physical sciences?” with only ‘Life Sciences’ participants replies represented in the figure. ')
 st.write('G)')
 st.plotly_chart(sun_phy, use_container_width=False)
 st.write('Supplementary figure 2G) Answers to part A, B, and C were classified based on the responses to the question “The next question will ask you about particular image analysis tools and techniques. Do you want to answer questions about microscopy in the field/area of life sciences or physical sciences?” with only Physical Sciences’ participants replies represented in the figure')
 

## Figure 3
kinds_of_images = data.iloc[:, 27:38]
kinds_life = stacked_barchart(kinds_of_images,title="What kinds of images do you commonly want to analyze (life sciences)", order_of_stacks=['2D', '2D+time','3D(<3000x3000x100)', '3D+time','3D(SPIM/largevolume)','3Dlargevolume+time'])
kinds_of_images_phy = data.iloc[:, 42:51]
kinds_phy = stacked_barchart(kinds_of_images_phy,title="What kinds of images do you commonly want to analyze (physical sciences)", order_of_stacks=['2D', '2D+time','3D(<3000x3000x100)', '3D+time','3D(largevolume)','3Dlargevolume+time'])

with st.container():
   st.subheader('Figure 3')
   st.subheader('Kinds of images analyzed')
   st.write('A)')
   st.plotly_chart(kinds_life, theme=None, use_container_width=True)
   st.write('Figure 3 A) Answers to the checkbox grid question “What kinds of images do you commonly want to analyze (select all that apply)?” under the “Life Sciences Image Analysis” section.')
   st.write('B)')
   st.plotly_chart(kinds_phy, theme=None, use_container_width=True)
   st.write('Figure 3 B) Answers to the checkbox grid question “What kinds of images do you commonly want to analyze (select all that apply)?” under the “Physical Sciences Image Analysis” section')


## Figure 4
# commonly and most used tools -life sciences
com_lif = tools_count(data.iloc[:,25], title='Commonly used image analysis tools (life sciences)')
com_lif.update_layout(title_x=0.3)
most_lif = tools_count(data['Most used image analysis tools'], title='Most used image analysis tools (life sciences)')
most_lif.update_layout(title_x=0.3)
# commonly and most used tools -physical sciences
def tools_count_phy(series, title=''):
    """
    Creates bar chart for commonly used and most used image analysis tools
    Parameters:
    series:pd.Series. Takes a dataframe column as input
    title: title to be given for the graph
    """

    #creating the dcitionary of the unique values with the counts
    list = series.dropna(how='all').tolist()
    list_split = [re.split(r',\s*(?![^()]*\))', i) for i in list]   #splitting the list based on comma but not the ones inside the bracket
    list_flat = [i for innerlist in list_split for i in innerlist]  #converting a nested list to a flat list
    list_wo_spaces = [i.lstrip() for i in list_flat] #removing the leading spaces in the string
    dict_count = dict((x,list_wo_spaces.count(x)) for x in set(list_wo_spaces)) # creating a dictionary with counts of each unique values
    print(dict_count.values)

    #shortening the index to avoid having lengthy axis labels
    final_key=[]
    for i in dict_count.keys():
        new_key = i.split('(')
        new_key = [new_key[0]]
        final_key = final_key + new_key

    #changing the key values in a dictionary
    list = dict_count.values()
    final_dict = dict(zip(final_key, list))

    #creating a dataframe from the modified dictionary
    tools_df = pd.DataFrame.from_dict(final_dict, orient='index')
    tools_df = tools_df.rename(columns={0:'counts'})
    
    #plotting a graph
    fig = px.bar(x=tools_df['counts'], y=tools_df.index, labels ={'x':'counts', 'y':''}, text_auto=True, orientation='h')
    fig.update_layout(width=700, height=400, title=title, title_x=0.3, title_y=0.95, font=dict(family='Helvetica', color="Black", size=16), legend=dict(title_font_family = 'Helvetica', font=dict(size=16, color="Black")))
    fig.update_yaxes(categoryorder = 'total ascending')

    #saving the figure as svg file
    fig.write_image(title+'.svg')

    return fig


com_phy = tools_count_phy(data.iloc[:,40], title='Commonly used image analysis tools (physical sciences)')
most_phy = tools_count_phy(data.iloc[:, 41], title='Most used image analysis tools (physical sciences)')

#frequency of script usage - life sciences 
freq_df  = data.loc[:, ('Frequency of script usage','Microscopy for life sciences physical sciences')]
freq_lif_df = freq_df.groupby('Microscopy for life sciences physical sciences').get_group('Life Sciences')
freq_lif_df = freq_lif_df.drop(['Microscopy for life sciences physical sciences'], axis=1)
freq_lif = barchart_horizontal(freq_lif_df["Frequency of script usage"],title='Frequency of script usage (life sciences)', order_of_axes=['Most of the time','Often','Sometimes','Never'])
freq_lif.update_layout(title_x=0.3)
#frequency of script usage - physical sciences 
freq_phy_df = freq_df.groupby('Microscopy for life sciences physical sciences').get_group('Physical Sciences')
freq_phy_df = freq_phy_df.drop(['Microscopy for life sciences physical sciences'], axis=1)
freq_phy = barchart_horizontal(freq_phy_df["Frequency of script usage"],title='Frequency of script usage (physical sciences)', order_of_axes=['Most of the time','Often','Sometimes','Never'])
freq_phy.update_layout(title_x=0.3)
with st.container():
   st.subheader('Figure 4')
   st.subheader('The community prefers open-source point-and-click software')
   st.write('A)')
   st.plotly_chart(com_lif)
   st.write('Figure 4A) Answers to the multiple-choice question “What image analysis tools have you used before? (check all that apply)” under the “Life Sciences Image Analysis” section.')
   st.write('B)')
   st.plotly_chart(most_lif)
   st.write('Figure 4B) Answers to the checkbox question “What image analysis tools do you use the most?” under the “Life Sciences Image Analysis” section.')
   st.write('C)')
   st.plotly_chart(freq_lif)
   st.write('Figure 4C) Answers to the question “How frequently do you use scripting to solve image analysis problems?” by ‘Life Science’ participants.')
   st.write('D)')
   st.plotly_chart(com_phy)
   st.write('Figure 4D) Answers to the multiple-choice question “What image analysis tools have you used before? (check all that apply)” under the “Physical Sciences Image Analysis” section.')
   st.write('E)')
   st.plotly_chart(most_phy)
   st.write('Figure 4E) Answers to the checkbox question “What image analysis tools do you use the most?” under the “Physical Sciences Image Analysis” section.')
   st.write('F)')
   st.plotly_chart(freq_phy)
   st.write('Figure 4F) Answers to the question “How frequently do you use scripting to solve image analysis problems?” by ‘Physical Science’ participants.')

## Figure S3 
scp_work_lif = pd.concat((data['Most used image analysis tools'], data['Work type']), axis=1)
scp_df = scp_work_lif.groupby('Most used image analysis tools').get_group('Computational libraries and scripts (python (scikit-image), MATLAB, etc)')
scp_df_1 = scp_df.groupby('Work type').value_counts().to_frame().reset_index().rename(columns={0:'counts'})
scp_lif = px.bar(scp_df_1, x=scp_df_1['Work type'], y=scp_df_1['counts'], title='Script usage (life sciences)', width=500, height=500, text_auto=True, labels={'Work type':''})
scp_lif.update_layout(title_x =0.5, font=dict(family='Helvetica', color='Black', size=16), legend=dict(title_font_family = 'Helvetica', font=dict(size=16, color="Black")))
scp_lif.update_xaxes(categoryorder='array', categoryarray =['Imaging', 'Balanced', 'Analyst'])


scp_work_phy = pd.concat((data.iloc[:, 41], data['Work type']), axis=1)
scp_df = scp_work_phy.groupby('What image analysis tools do you use the most?.1').get_group('Computational libraries and scripts (python (scikit-image), MATLAB, etc)')
scp_df_1 = scp_df.groupby('Work type').value_counts().to_frame().reset_index().rename(columns={0:'counts'})
scp_phy = px.bar(scp_df_1, x=scp_df_1['Work type'], y=scp_df_1['counts'], title='Script usage (physical sciences)', width=500, height=500, text_auto=True, labels={'Work type':''})
scp_phy.update_layout(title_x =0.5, font=dict(family='Helvetica', color='Black', size=16), legend=dict(title_font_family = 'Helvetica', font=dict(size=16, color="Black")))
scp_phy.update_xaxes(categoryorder='array', categoryarray =['Balanced', 'Analyst'])

## Figure S3 
st.subheader('Figure S3')
st.write('A)')
st.plotly_chart(scp_lif)
st.write('Figure S3 A) Usage of computational scripts and libraries among life science participants was categorized based on the ‘work type’ as described in supplementary figure 2.')
st.write('B)')
st.plotly_chart(scp_phy)
st.write('Figure S3 B) Usage of computational scripts and libraries among physical science participants was categorized based on the ‘work type’ as described in supplementary figure 2.')


## Figure 5 
approach = barchart_horizontal(data["Approach to solutions"], title='Approach to solutions for image analysis problems')
approach.update_layout(title_x=0.3)
data_1 = data.rename(columns={'Well solved image analysis problems':'Well solved image analysis problems (life sciences)'})
#generating word clouds


with st.container():
   st.subheader('Figure 5')
   st.subheader('Solving image analysis problems')
   st.write('A)')
   st.plotly_chart(approach)
   st.write('Figure 5A) Answers to the checkbox question “How do you generally go about solving an image analysis problem? Check the approach(es) you use the most.” ')
   st.write('B)')
   st.set_option('deprecation.showPyplotGlobalUse', False)
   well_solved_lif = wordcloud(data_1["Well solved image analysis problems (life sciences)"], extra_stopwords=['finding nuclei','finding', 'nuclei', 'tissue','analysis', 'analysis of super-resolution data', 'cell', 'cells', 'etc', 'image', 'images', 'well', 
             'solved', 'imaging', 'better', 'simple', 'different', 'non', 'within', 'depend', 'none', 'things', 'easy', 'many', 'almost', 'common', 
             'though', 'n', 'depends', 'number', 'made', 'seem', 'show','little','clearly','need','often', 'particularly','em','way', 'co', 'size', 'types','using', 'set', 'every', 'possible', 'based', 'generally', 'semi', 'work', 'don', 't', 'basic', 's', 'e', 'g', 'esp', 'right'])
   st.pyplot(well_solved_lif, use_container_width=False)
   st.write('Figure 5B) Wordcloud representation of the unigrams of the answers by ‘life science’ participants to an open-ended question “What image analysis problems (i.e. finding nuclei, tissue analysis, analysis of super-resolution data, etc) do you think are generally well-solved?”')
   st.set_option('deprecation.showPyplotGlobalUse', False)
   st.write('C)')
   data_2 = data.rename(columns={'Image analysis problems which needs better solutions':'Image analysis problems which needs better solutions (life sciences)'})
   better_lif = wordcloud(data_2["Image analysis problems which needs better solutions (life sciences)"], extra_stopwords=['finding nuclei','finding', 'nuclei', 'tissue','analysis', 'analysis of super-resolution data', 'cell', 'cells', 'etc', 'image', 'images', 'well', 
             'solved', 'imaging', 'better', 'simple', 'different', 'non', 'within', 'depend', 'none', 'things', 'easy', 'many', 'almost', 'common', 
             'though', 'n', 'depends', 'number', 'made', 'seem', 'show','little','clearly','need','often', 'particularly','em','way', 'co', 'size', 'types','using', 'set', 'every', 'possible', 'based', 'generally', 'semi', 'work', 'don', 't', 'basic', 's', 'e', 'g', 'esp', 'right'])
   st.pyplot(better_lif, use_container_width=False)
   st.write('Figure 5C) Wordcloud representation of the unigrams of the answers by ‘life science’ participants to an open-ended question “What image analysis problems (i.e. finding nuclei, tissue analysis, analysis of super-resolution data, etc) do you wish had easier/better solutions?”')
   st.write('D)')
   st.set_option('deprecation.showPyplotGlobalUse', False)
   data_3 = data.rename(columns={'Well solved image analysis problems-physical sciences':'Well solved image analysis problems (physical sciences)'})
   well_solved_phy = wordcloud(data_3['Well solved image analysis problems (physical sciences)'], extra_stopwords=['recons', 'causes', 'gratitude', 'haadf', 'round', 'follows', 'varying','due', 'mostly', 'answer', 'round', 'program', 'super', 'separate', 'nature','difficult', 'low', 'admiration', 'highly', 'rule', 'thunderstorm', 'stem'])
   st.pyplot(well_solved_phy, use_container_width=False)
   st.write('Figure 5D) Wordcloud representation of the unigrams of the answers by ‘physical science’ participants to an open-ended question “What image analysis problems (i.e. finding nuclei, tissue analysis, analysis of super-resolution data, etc) do you think are generally well-solved?”')
   st.write('E)')
   st.set_option('deprecation.showPyplotGlobalUse', False)
   data_4 = data.rename(columns={'Image analysis problems which needs better solutions physical sciences':'Image analysis problems which needs better solutions (physical sciences)'})
   better_phy = wordcloud(data_4['Image analysis problems which needs better solutions (physical sciences)'], extra_stopwords=['don', 'sims', 'wish', 'bad', 'terrible', 'across','deep', 'end', 'trust','vs', 'etc', 'look', 'sometimes','individual', 'producing', 'specifically','done', 'round', 'from', 'non', 'still', 'front', 'think'])
   st.pyplot(better_phy, use_container_width=False)
   st.write('Figure 5E) Wordcloud representation of the unigrams of the answers by ‘’physical science’ participants to an open-ended question “What image analysis problems (i.e. finding nuclei, tissue analysis, analysis of super-resolution data, etc) do you wish had easier/better solutions?”')

   ## Figure S4

   #Usage of image sc forum 
data['Usage of image sc forum'] = data['Approach to solutions'].dropna().apply(lambda x:True if 'Ask on forum.image.sc' in x else False)
forum_work_df = data.loc[:, ('Work type', 'Usage of image sc forum')]
forum_usage_df = forum_work_df.groupby('Usage of image sc forum').get_group(True).value_counts().reset_index()
forum_usage_df= forum_usage_df.rename(columns={0:'counts', 'Work type':'work type_1'})
# work type counts 
work_type_counts = data['Work type'].value_counts().reset_index()
#dataframe to calculate percentage 
per_sc_usage_df = pd.concat([forum_usage_df, work_type_counts], axis=1)
per_sc_usage_df['percentage'] = (per_sc_usage_df['counts']/per_sc_usage_df['Work type'])*100
per_sc_usage_df['percentage']= per_sc_usage_df['percentage'].round(decimals=2)
per_sc_usage_df= per_sc_usage_df.sort_values(by='percentage', ascending=True)
sc_usage_fig = px.bar(per_sc_usage_df, per_sc_usage_df['index'], y=per_sc_usage_df['percentage'], text_auto=True, width=500, height=500,labels={'index':'', 'percentage':'Percent'}, title='Usage of image sc forum')
sc_usage_fig.update_layout(width=500, height=500, title_x=0.4, title_y=0.95, font=dict(family='Helvetica', color="Black", size=16), legend=dict(title_font_family = 'Helvetica', font=dict(size=16, color="Black")))


with st.container():
   st.subheader('Figure S4')
   st.subheader('Image analysis problems as stated by 2020 survey participants')
   st.write('A)')
   st.plotly_chart(sc_usage_fig, use_container_width=False)
   st.write('Figure S4A) Percentage usage of image sc forum was calculated based on answers provided for the question ‘How do you generally go about solving an image analysis problem? Check the approach(es) you use the most’ normalized with the work type that was categorized based on the answers provided to “How would you describe your work?”. Categorization is given in detail in Figure S2A.')
   st.write('B)')
   st.set_option('deprecation.showPyplotGlobalUse', False)
   url_1 = 'https://raw.githubusercontent.com/ciminilab/2021_Jamali_BiologicalImaging/main/AnonymizedSurveyQuestions/What_image_analysis_problems_ie_finding_nuclei_tissue_analysis_analysis_of_super-resolution_data_etc_do_you_think_are_generally_well-solved.csv'
   well_solved_2021 = pd.read_csv(url_1)
   well_solved_2021 = well_solved_2021.rename(columns= {'0':'Well-solved image analysis problems-2020'})
   well_solved_2021 =wordcloud(well_solved_2021["Well-solved image analysis problems-2020"], extra_stopwords=['finding nuclei','finding', 'nuclei', 'tissue','analysis', 'analysis of super-resolution data', 'cell', 'cells', 'etc', 'image', 'images', 'well', 
             'solved', 'imaging', 'better', 'simple', 'different', 'non', 'within', 'depend', 'none', 'things', 'easy', 'many', 'almost', 'common', 
             'though', 'n', 'depends', 'number', 'made', 'seem', 'show', 'particularly', 'co', 'size', 'types', 'every', 'possible', 'based', 'generally', 'semi', 'work', 'don', 't', 'basic'])
   st.pyplot(well_solved_2021, use_container_width=False)
   st.write('Figure S4B) Wordcloud representation of the unigrams of the answers provided by the 2020 bioimage analysis survey participants to an open-ended question “What image analysis problems (i.e. finding nuclei, tissue analysis, analysis of super-resolution data, etc) do you think are generally well-solved?”.')
   st.write('C)')
   st.set_option('deprecation.showPyplotGlobalUse', False)
   url_2 = 'https://raw.githubusercontent.com/ciminilab/2021_Jamali_BiologicalImaging/main/AnonymizedSurveyQuestions/What_image_analysis_problems_ie_finding_nuclei_tissue_analysis_analysis_of_super-resolution_data_etc_do_you_wish_had_easier_better_solutions.csv'
   better_sol_2021 = pd.read_csv(url_2)
   better_sol_2021 = better_sol_2021.rename(columns= {'0':'Image analysis problems that need better solutions-2020'})
   better_2021 = wordcloud(better_sol_2021["Image analysis problems that need better solutions-2020"], extra_stopwords=['finding nuclei','finding', 'nuclei', 'tissue', 'analysis', 'analysis of super-resolution data', 'image', 'images', 'imaging', 'e', 'g', 
             'cell', 'em', 'non', 'etc', 'cells', 'better', 'find', 'easy', 'sample', 'well', 'don', 't', 'different', 'without','many', 'change', 'high', 'especially', 
             'low', 'good', 'tool', 'based', 'things', 'changes','within', 'problem', 'small', 'working', 'data', 
             'analyzing','set', 'often', 'easier', 'clearly', 'types', 'general', 'anything', 'co', 'work', 'highly', 'need', 'way', 're'])
   st.pyplot(better_2021, use_container_width=False)
   st.write('Figure S4C) Wordcloud representation of the unigrams of the answers provided by the 2020 bioimage analysis survey participants to an open-ended question “What image analysis problems (i.e. finding nuclei, tissue analysis, analysis of super-resolution data, etc) do you wish had easier/better solutions?”')


   ## Figure 6

   pre_exp= barchart_horizontal(data['Previous experience'], title='Previous experience')
   conf_att = barchart_horizontal(data["Number of conferences/workshops attended"], title='Conferences attended by the survey participants', order_of_axes=['Many','Some','Few','None'])
   conf_att.update_layout(title_x=0.3)
   #Are there any image analysis workshops, tutorials, or conferences that you have participated in and found particularly helpful? If yes, what made them beneficial?
workshops = data["Experience with previous workshops/conferences"].str.lower().str.strip()
workshops = workshops.dropna(how='all').to_frame()

#creating new columns based on the values in the query column
workshops["NEUBIAS"] = workshops['Experience with previous workshops/conferences'].apply(lambda x: True if 'neubias' in x else False)
workshops["Fiji/ImageJ"] = workshops['Experience with previous workshops/conferences'].apply(lambda x: True if ('imagej' in x) or ('fiji' in x) else False)
workshops["Cellprofiler"] = workshops['Experience with previous workshops/conferences'].apply(lambda x: True if 'cellprofiler' in x else False)
workshops["Imaris"] = workshops['Experience with previous workshops/conferences'].apply(lambda x: True if 'imaris' in x else False)
workshops["Robert Haase"] = workshops['Experience with previous workshops/conferences'].apply(lambda x: True if 'robert' in x else False)
workshops["AQLM"] = workshops['Experience with previous workshops/conferences'].apply(lambda x: True if 'aqlm' in x else False)
workshops["CSHL"] = workshops['Experience with previous workshops/conferences'].apply(lambda x: True if 'cshl' in x else False)
workshops["I2K"] = workshops['Experience with previous workshops/conferences'].apply(lambda x: True if 'i2k' in x else False)
workshops["EMBL"] = workshops['Experience with previous workshops/conferences'].apply(lambda x: True if 'embl' in x else False)

#Chart
workshops_for_chart = workshops.drop(columns=['Experience with previous workshops/conferences', 'CSHL'])
workshops_for_chart = workshops_for_chart.sum().to_frame()
workshops_chart = px.bar(workshops_for_chart, labels={'value':'counts', 'index':''},orientation='h', text_auto=True)
workshops_chart.update_layout(title='Workshops attended by the participants', width=500, height=500, showlegend=False, font=dict(family='Helvetica', color="Black", size=16))
workshops_chart.update_yaxes(categoryorder = 'total ascending')
workshops_chart.update_layout(title_x=0.3)


with st.container():
   st.subheader('Figure 6')
   st.subheader('Experience in image analysis')
   st.write('A)')
   st.plotly_chart(pre_exp)
   st.write('Figure 6A) Answers to a multiple-choice question “Please select any of the following you have attended in the past”.')
   st.write('B)')
   st.plotly_chart(conf_att)
   st.write('Figure 6B) Answers to the checkbox question “Are there any image analysis workshops, tutorials, or conferences you are aware of and attended or considered attending? If so, how many?”')
   st.write('C)')
   st.plotly_chart(workshops_chart)
   st.write('Figure 6C) Answers to an open-ended question “Are there any image analysis workshops, tutorials, or conferences that you have participated in and found particularly helpful? If yes, what made them beneficial?”  ')


## Figure 7 
pre_top = data.iloc[:,17:24] 
top_int = percentage_stackedcharts(pre_top, title='How interested are you in learning more about the following topics',order_of_stacks=['Veryinterested', 'Moderatelyinterested', 'Somewhatinterested', 'Notatallinterested'], 
order_of_axes=['Topics related to sub discipline', 'Visualization of results', 'Analyzing large images', 'Image analysis practices', 'Specific software tool', 'Deep learning for image analysis', 'Image analysis theory'], colors={'Veryinterested':'royalblue','Moderatelyinterested':'dodgerblue','Somewhatinterested':'cornflowerblue','Notatallinterested':'skyblue'})
top_int.update_layout(title='How interested are you in learning more about the following topics?', title_x=0.2)
cols_mtds = data.iloc[:,10:17] 
pre_mtd = percentage_stackedcharts(cols_mtds, title='How preferable do you find each of these instructional methods',order_of_stacks=['Verypreferable', 'Moderatelypreferable', 'Somewhatpreferable', 'Notatallpreferable'], 
order_of_axes=['Written tutorials', 'Video tutorial', 'Office hours', 'One day seminar', 'Interactive webinar','Best practices articles', 'Multiday workshop'], colors={'Verypreferable':'royalblue','Moderatelypreferable':'dodgerblue','Somewhatpreferable':'cornflowerblue','Notatallpreferable':'skyblue'})
pre_mtd.update_layout(title='How preferable do you find each of these instructional methods?', title_x=0.2)
## Fold change figure 
# Topics of interest for the tutorials - 2020 and 2022
theory = 'https://raw.githubusercontent.com/ciminilab/2021_Jamali_BiologicalImaging/main/AnonymizedSurveyQuestions/How_interested_are_you_in_learning_more_about_the_following_topics_Image_analysis_theory_.csv'
int_2020_theory = pd.read_csv(theory)
int_2020_theory = int_2020_theory.rename(columns={'0':'theory'})
int_2020_theory = int_2020_theory['theory'].value_counts().to_frame()

practices = 'https://raw.githubusercontent.com/ciminilab/2021_Jamali_BiologicalImaging/main/AnonymizedSurveyQuestions/How_interested_are_you_in_learning_more_about_the_following_topics_General_image_analysis_practices.csv'
int_2020_practices = pd.read_csv(practices)
int_2020_practices = int_2020_practices.rename(columns={'0':'practices'})
int_2020_practices = int_2020_practices['practices'].value_counts().to_frame()

par_field = 'https://raw.githubusercontent.com/ciminilab/2021_Jamali_BiologicalImaging/main/AnonymizedSurveyQuestions/How_interested_are_you_in_learning_more_about_the_following_topics_Image_analysis_practices_particular_to_my_field.csv'
int_2020_par_field = pd.read_csv(par_field)
int_2020_par_field = int_2020_par_field.rename(columns={'0':'Practices related to particular field'})
int_2020_par_field = int_2020_par_field['Practices related to particular field'].value_counts().to_frame()

spc_tool = 'https://raw.githubusercontent.com/ciminilab/2021_Jamali_BiologicalImaging/main/AnonymizedSurveyQuestions/How_interested_are_you_in_learning_more_about_the_following_topics_Learning_to_use_a_particular_software_tool.csv'
int_2020_spc_tool = pd.read_csv(spc_tool)
int_2020_spc_tool = int_2020_spc_tool.rename(columns={'0':'Specific software tool'})
int_2020_spc_tool = int_2020_spc_tool['Specific software tool'].value_counts().to_frame()

DL = 'https://raw.githubusercontent.com/ciminilab/2021_Jamali_BiologicalImaging/main/AnonymizedSurveyQuestions/How_interested_are_you_in_learning_more_about_the_following_topics_Deep_learning_as_applied_to_image_analysis.csv'
int_2020_DL = pd.read_csv(DL)
int_2020_DL = int_2020_DL.rename(columns={'0':'Deep learning for image analysis'})
int_2020_DL = int_2020_DL['Deep learning for image analysis'].value_counts().to_frame()
int_2020_com = pd.concat([int_2020_theory, int_2020_practices, int_2020_par_field, int_2020_spc_tool, int_2020_DL], axis=1)
int_2020_com = int_2020_com.rename(columns={'theory':'Image analysis theory', 'practices':'Image analysis practices'})
int_2020_com = int_2020_com.reset_index()
int_2020_com = int_2020_com.rename(columns={'index':'interest'})

#changing the value of the row since the it is 'A little interested' in 2020 and 'somewhat interested' in 2022
int_2020_com.loc[2,'interest'] = 'Somewhat interested'

#getting the percentage 
int_2020_com['percent_theory'] = (int_2020_com['Image analysis theory']/int_2020_com['Image analysis theory'].sum()) *100
int_2020_com['percent_practices'] = (int_2020_com['Image analysis practices']/int_2020_com['Image analysis practices'].sum()) *100
int_2020_com['percent_particular'] = (int_2020_com['Practices related to particular field']/int_2020_com['Practices related to particular field'].sum()) *100
int_2020_com['percent_sptool'] = (int_2020_com['Specific software tool']/int_2020_com['Specific software tool'].sum()) *100
int_2020_com['percent_DL'] = (int_2020_com['Deep learning for image analysis']/int_2020_com['Deep learning for image analysis'].sum()) *100

#rounding up the decimals
int_2020_com['percent_theory']= int_2020_com['percent_theory'].round(decimals=1)
int_2020_com['percent_practices'] = int_2020_com['percent_practices'].round(decimals=1)
int_2020_com['percent_particular'] = int_2020_com['percent_particular'].round(decimals=1)
int_2020_com['percent_sptool'] = int_2020_com['percent_sptool'].round(decimals=1)
int_2020_com['percent_DL'] = int_2020_com['percent_DL'].round(decimals=1)

#2022 - topics of interest
trial_radar = data["Image analysis theory"].value_counts().to_frame()
trial_radar_1=data["Image analysis practices"].value_counts().to_frame()
trial_radar_2=data["Topics related to sub discipline"].value_counts().to_frame()
trial_radar_3=data["Specific software tool"].value_counts().to_frame()
trial_radar_4=data["Deep learning for image analysis"].value_counts().to_frame()
trial_radar_5=data["Analyzing large images"].value_counts().to_frame()
trial_radar_6=data["Visualization of results"].value_counts().to_frame()

#concatenating all the related columns for topics of interest
trial = pd.concat([trial_radar, trial_radar_1, trial_radar_2, trial_radar_3, trial_radar_4, trial_radar_5, trial_radar_6], axis=1)

#Reseting the index so that the column can be renamed
trial =trial.reset_index()
trial = trial.rename(columns={'index':'interest'})

#editing the 2022 column 
trial_subset = trial.iloc[:, 0:6]
trial_subset = trial_subset.rename(columns={'Interes in topics Image analysis theory':'Image analysis theory', 'Interest in topics Image analysis practices':'Image analysis practices','Interest in topics related to sub discipline':'Practices related to particular field','Interest in topics Specific software tool':'Specific software tool', 'Interest in topics Deep learning for image analysis':'Deep learning for image analysis'})

#getting the percentage 
trial_subset['percent_theory'] = (trial_subset['Image analysis theory']/trial_subset['Image analysis theory'].sum()) *100
trial_subset['percent_practices'] = (trial_subset['Image analysis practices']/trial_subset['Image analysis practices'].sum()) *100
trial_subset['percent_particular'] = (trial_subset['Topics related to sub discipline']/trial_subset['Topics related to sub discipline'].sum()) *100
trial_subset['percent_sptool'] = (trial_subset['Specific software tool']/trial_subset['Specific software tool'].sum()) *100
trial_subset['percent_DL'] = (trial_subset['Deep learning for image analysis']/trial_subset['Deep learning for image analysis'].sum()) *100

#rounding up the decimals
trial_subset['percent_theory']= trial_subset['percent_theory'].round(decimals=1)
trial_subset['percent_practices'] = trial_subset['percent_practices'].round(decimals=1)
trial_subset['percent_particular'] = trial_subset['percent_particular'].round(decimals=1)
trial_subset['percent_sptool'] = trial_subset['percent_sptool'].round(decimals=1)
trial_subset['percent_DL'] = trial_subset['percent_DL'].round(decimals=1)

## fold change in the interest 
# Creating dataframes to understand the fold change in the topics of interest for future workshops
#2020 columns 
int_2020_com_sub = int_2020_com.drop(int_2020_com.columns[[1,2,3,4,5]], axis=1)
int_2020_com_sub.columns = [i+'20' for i in int_2020_com_sub.columns]

#2022 columns 
trial_subset_sub = trial_subset.drop(trial_subset.columns[[1,2,3,4,5]], axis=1)
trial_subset_sub.columns = [i+'22' for i in trial_subset_sub.columns]

#combining dataframes 
int_22_20 = pd.concat([trial_subset_sub, int_2020_com_sub], axis=1)

# calculating the fold change
for i in range(1,6):
    col_name=int_22_20.columns[i]
    int_22_20[col_name+'fold']=(int_22_20.iloc[:, i]-int_22_20.iloc[:, i+6])/int_22_20.iloc[:, i+6]
    int_22_20[col_name+'fold']  = int_22_20[col_name+'fold'].round(decimals=2)
#selecting the desired columns from the dataframe 
int_22_20_1 = int_22_20.iloc[:,0]
int_22_20_2 = int_22_20.iloc[:,12:18]
combined = pd.concat([int_22_20_1, int_22_20_2], axis=1)
combined = combined.set_index('interest22').transpose()
combined = combined.reset_index()

#replacing the values
combined = combined.replace('percent_theory22fold', 'Image analysis theory')
combined = combined.replace('percent_practices22fold', 'Image analysis practices')
combined = combined.replace('percent_particular22fold', 'Image analysis practices related to particular subdiscipline')
combined = combined.replace('percent_sptool22fold', 'Particular software tool')
combined = combined.replace('percent_DL22fold', 'Deep learning for image analysis')

#Preferable method comparison 

#Preferable instructional methods - 2020
art = 'https://raw.githubusercontent.com/ciminilab/2021_Jamali_BiologicalImaging/main/AnonymizedSurveyQuestions/For_any_topics_you_re_interested_in_how_interested_would_you_be_in_learning_about_them_in_the_following_ways_Scholarly_best_practices_article.csv'
art_20 = pd.read_csv(art)
art_20 = art_20.rename(columns={'0':'article'})
art_20 = art_20['article'].value_counts().to_frame()

wrt = 'https://raw.githubusercontent.com/ciminilab/2021_Jamali_BiologicalImaging/main/AnonymizedSurveyQuestions/For_any_topics_you_re_interested_in_how_interested_would_you_be_in_learning_about_them_in_the_following_ways_Written_step-by-step_tutorial.csv'
wrt_tut_20 = pd.read_csv(wrt)
wrt_tut_20 = wrt_tut_20.rename(columns={'0':'written tutorial'})
wrt_tut_20 = wrt_tut_20['written tutorial'].value_counts().to_frame()

vid = 'https://raw.githubusercontent.com/ciminilab/2021_Jamali_BiologicalImaging/main/AnonymizedSurveyQuestions/For_any_topics_you_re_interested_in_how_interested_would_you_be_in_learning_about_them_in_the_following_ways_Video_tutorial.csv'
vid_tut_20 = pd.read_csv(vid)
vid_tut_20 = vid_tut_20.rename(columns={'0':'video tutorial'})
vid_tut_20 = vid_tut_20['video tutorial'].value_counts().to_frame()

int_web = 'https://raw.githubusercontent.com/ciminilab/2021_Jamali_BiologicalImaging/main/AnonymizedSurveyQuestions/For_any_topics_you_re_interested_in_how_interested_would_you_be_in_learning_about_them_in_the_following_ways_Interactive_webinar.csv'
int_web_20 = pd.read_csv(int_web)
int_web_20 = int_web_20.rename(columns={'0':'interactive webinar'})
int_web_20 = int_web_20['interactive webinar'].value_counts().to_frame()

off_hr = 'https://raw.githubusercontent.com/ciminilab/2021_Jamali_BiologicalImaging/main/AnonymizedSurveyQuestions/For_any_topics_you_re_interested_in_how_interested_would_you_be_in_learning_about_them_in_the_following_ways_One-on-one_office_hours_with_an_expert.csv'
off_hr_20 = pd.read_csv(off_hr)
off_hr_20 = off_hr_20.rename(columns={'0':'office hour'})
off_hr_20 = off_hr_20['office hour'].value_counts().to_frame()

one_day = 'https://raw.githubusercontent.com/ciminilab/2021_Jamali_BiologicalImaging/main/AnonymizedSurveyQuestions/For_any_topics_you_re_interested_in_how_interested_would_you_be_in_learning_about_them_in_the_following_ways_In_person_seminar_tutorial_lasting_lessthan1_day.csv'
one_day_20 = pd.read_csv(one_day)
one_day_20 = one_day_20.rename(columns={'0':'one day'})
one_day_20 = one_day_20['one day'].value_counts().to_frame()

mul_day = 'https://raw.githubusercontent.com/ciminilab/2021_Jamali_BiologicalImaging/main/AnonymizedSurveyQuestions/For_any_topics_you_re_interested_in_how_interested_would_you_be_in_learning_about_them_in_the_following_ways_Multiday_workshop.csv'
mul_day_20 = pd.read_csv(mul_day)
mul_day_20 = mul_day_20.rename(columns={'0':'multi day'})
mul_day_20 = mul_day_20['multi day'].value_counts().to_frame()

#Combining all the dataframes 
pre_tut_20 = pd.concat([art_20,wrt_tut_20,vid_tut_20,int_web_20,off_hr_20,one_day_20,mul_day_20], axis=1)
pre_tut_20 = pre_tut_20.reset_index()
pre_tut_20.loc[2, 'index'] = 'Somewhat interested'

#creating percentage columns 
pre_tut_20['percent_article'] = (pre_tut_20['article']/pre_tut_20['article'].sum())* 100
pre_tut_20['percent_article']= pre_tut_20['percent_article'].round(decimals=1)

pre_tut_20['percent_written'] = (pre_tut_20['written tutorial']/pre_tut_20['written tutorial'].sum())*100
pre_tut_20['percent_written']=pre_tut_20['percent_written'].round(decimals=1)

pre_tut_20['percent_video'] = (pre_tut_20['video tutorial']/pre_tut_20['video tutorial'].sum())*100
pre_tut_20['percent_video']= pre_tut_20['percent_video'].round(decimals=1)

pre_tut_20['percent_web'] = (pre_tut_20['interactive webinar']/pre_tut_20['interactive webinar'].sum())*100
pre_tut_20['percent_web']=pre_tut_20['percent_web'].round(decimals=1)

pre_tut_20['percent_off'] = (pre_tut_20['office hour']/pre_tut_20['office hour'].sum())*100
pre_tut_20['percent_off']=pre_tut_20['percent_off'].round(decimals=1)

pre_tut_20['percent_one'] = (pre_tut_20['one day']/pre_tut_20['one day'].sum())*100
pre_tut_20['percent_one']=pre_tut_20['percent_one'].round(decimals=1)

pre_tut_20['percent_mul'] = (pre_tut_20['multi day']/pre_tut_20['multi day'].sum())*100
pre_tut_20['percent_mul']=pre_tut_20['percent_mul'].round(decimals=1)


#preferable instructional methods - 2022
art = data["Best practices articles"].value_counts().to_frame()
wrt = data["Written tutorials"].value_counts().to_frame()
vid = data["Video tutorial"].value_counts().to_frame()
web = data["Interactive webinar"].value_counts().to_frame()
off = data["Office hours"].value_counts().to_frame()
one = data["One day seminar"].value_counts().to_frame()
mul = data["Multiday workshop"].value_counts().to_frame()

#combining all the dataframes
mtds = pd.concat([art, wrt, vid, web, off, one, mul], axis=1)
mtds = mtds.reset_index()
mtds['index'] = ['Somewhat interested', 'Moderately interested', 'Very interested', 'Not at all interested']

#creating percentage columns 
mtds['per_art'] = (mtds['Best practices articles']/mtds['Best practices articles'].sum())*100
mtds['per_art'] = mtds['per_art'].round(decimals=1)

mtds['per_wrt'] = (mtds['Written tutorials']/mtds['Written tutorials'].sum())*100
mtds['per_wrt'] =mtds['per_wrt'].round(decimals=1)

mtds['per_vid'] = (mtds['Video tutorial']/mtds['Video tutorial'].sum())*100
mtds['per_vid'] = mtds['per_vid'].round(decimals=1)

mtds['per_web'] = (mtds['Interactive webinar']/mtds['Interactive webinar'].sum())*100
mtds['per_web'] =mtds['per_web'].round(decimals=1)

mtds['per_off'] = (mtds['Office hours']/mtds['Office hours'].sum())*100
mtds['per_off'] = mtds['per_off'].round(decimals=1)

mtds['per_one'] = (mtds['One day seminar']/mtds['One day seminar'].sum())*100
mtds['per_one'] = mtds['per_one'].round(decimals=1)

mtds['per_mul'] = (mtds['Multiday workshop']/mtds['Multiday workshop'].sum())*100
mtds['per_mul'] = mtds['per_mul'].round(decimals=1)

#percent change in the preferable mtds 
# Dipalying the fold change - 2020 and 2022
#creating dataframes for the fold change 
fold_df_20 = pre_tut_20.drop(pre_tut_20.columns[[1, 2,3,4,5,6,7]], axis=1)
fold_df_22 = mtds.drop(mtds.columns[[1,2,3,4,5,6,7]], axis=1)


fold_df_20 = fold_df_20.sort_values(by='index').reset_index(drop=True)
fold_df_22.columns = [(i+'22') for i in fold_df_22.columns]
fold_df_22 = fold_df_22.sort_values(by='index22').reset_index(drop=True)

fold_df_22_20 = pd.concat([fold_df_22, fold_df_20], axis=1)

#Calculating the fold change 
for i in range(1,8):
    col_name=fold_df_22_20.columns[i]
    fold_df_22_20[col_name+'fold']=(fold_df_22_20.iloc[:, i]-fold_df_22_20.iloc[:, i+8])/fold_df_22_20.iloc[:, i+8]
    fold_df_22_20[col_name+'fold']  = fold_df_22_20[col_name+'fold'].round(decimals=2)

fold_df_22_20_sub_1 = fold_df_22_20.iloc[:, 0]
fold_df_22_20_sub_2 = fold_df_22_20.iloc[:, 16:]
fold_df_22_20_sub = pd.concat([fold_df_22_20_sub_1, fold_df_22_20_sub_2], axis=1)
fold_df_22_20_sub = fold_df_22_20_sub.set_index('index22').transpose()
fold_df_22_20_sub = fold_df_22_20_sub.reset_index()

#Changing the column names 
fold_df_22_20_sub = fold_df_22_20_sub.replace('per_art22fold', 'Best practices articles')
fold_df_22_20_sub = fold_df_22_20_sub.replace('per_wrt22fold', 'Written tutorials')
fold_df_22_20_sub = fold_df_22_20_sub.replace('per_vid22fold', 'Video tutorials')
fold_df_22_20_sub = fold_df_22_20_sub.replace('per_web22fold', 'Interactive webinar')
fold_df_22_20_sub = fold_df_22_20_sub.replace('per_off22fold', 'Office hours')
fold_df_22_20_sub = fold_df_22_20_sub.replace('per_one22fold', 'In-person seminar/ One day workshop')
fold_df_22_20_sub = fold_df_22_20_sub.replace('per_mul22fold', 'Multiday workshop')

## combining the percent change 
#Combinbing the fold change charts of the preferable methods and the topic of interest for the tutorial 
comb_df_1 = combined.loc[:,['index','Very interested']]
comb_df_1['category'] = 'Interest for future workshops'
comb_df_2 = fold_df_22_20_sub.loc[:, ['index', 'Very interested']]
comb_df_2['category'] = 'Preferable instructional methods'
comb_fold_df = pd.concat([comb_df_1, comb_df_2])
comb_df_1 = combined.loc[:,['index','Very interested']]

#creating bar chart for the combined fold
comb_fold_bar= px.bar(x=comb_fold_df['index'], y=comb_fold_df['Very interested'], color = comb_fold_df['category'],labels={'y':'Percent change', 'x':' '}, orientation='v')
comb_fold_bar.update_layout(barmode='group',width=700, height=700, font=dict(family='Helvetica', color="Black", size=14), legend=dict(title_font_family = 'Helvetica', font=dict(size=16, color="Black")))
comb_fold_bar.update_layout(legend=dict(yanchor="bottom",y=0.02,xanchor="left",x=0.49), legend_title = 'Category')
comb_fold_bar.update_xaxes(categoryorder ='total ascending')

#separate charts for the percent change in the interests
int_top_subplots = make_subplots(rows=1, cols=2, shared_yaxes='all', subplot_titles=('Interest for future workshops', 'Preferable instructional methods'))

int_top_subplots.add_trace(go.Bar(x=comb_df_1['index'], y=comb_df_1['Very interested'], orientation='v',showlegend=False, marker_color='royalblue', yaxis='y1'), 1,1)
int_top_subplots.add_trace(go.Bar(x=comb_df_2['index'], y=comb_df_2['Very interested'],orientation='v',showlegend=False, marker_color='royalblue'), 1,2)

int_top_subplots.update_layout(width=1000, height=500)
int_top_subplots.update_xaxes(categoryorder ='total ascending', tickangle = 90)
int_top_subplots.update_layout(yaxis1 =(dict(title='Percent change')))
int_top_subplots['data'][0]['x'][2] = 'Image analysis practices <br> related to particular<br> subdiscipline'
int_top_subplots['data'][0]['x'][4] = 'Deep learning <br> for image analysis'
int_top_subplots['data'][1]['x'][0] = 'Best practices <br> articles'
int_top_subplots['data'][1]['x'][5] = 'In-person seminar/<br> One day workshop'
int_top_subplots.update_layout(font=dict(family='Helvetica', color="Black", size=14), legend=dict(title_font_family = 'Helvetica', font=dict(size=16, color="Black")))
int_top_subplots.write_image('fold_sharedaxes.svg')

with st.container():
   st.subheader('Figure 7')
   st.subheader('The imaging community prefers to learn about customized image analysis tools at their own pace')
   st.write('A)')
   st.plotly_chart(top_int)
   st.write('Figure 7A) Answers to a multiple-choice grid question “How interested are you in learning more about the following topics?”')
   st.write('B)')
   st.plotly_chart(pre_mtd)
   st.write('Figure 7B) Answers to a multiple-choice grid question “In regards to learning more about image analysis, how preferable do you find each of these instructional methods?”')
   st.write('C)')
   st.plotly_chart(int_top_subplots)
   st.write('Figure 7C) Percent change in the “Very interested/Very preferable” category of part A and B compared to the results from the 2020 bioimage analysis survey.')
   

## Figure S5
#Comparing the skills with interest for the topics of image analysis tutorials 
int_skills_df_1 = data.iloc[:,17:24]
int_skills_df_2 = data.loc[:,'Knowledge of computational skills']
int_skills_df = pd.concat([int_skills_df_1, int_skills_df_2], axis=1)


##categories -theory 
df_theory = int_skills_df.loc[:, ["Image analysis theory", "Knowledge of computational skills"]]

df_the_grp = df_theory.groupby('Knowledge of computational skills')
#High skilled- theory
df_the_high = df_the_grp.get_group('High skill')
df_the_high = df_the_high['Image analysis theory'].value_counts().to_frame().reset_index()
df_the_high['per_high'] =(df_the_high['Image analysis theory']/df_the_high['Image analysis theory'].sum())*100
df_the_high['per_high'] = df_the_high['per_high'].round(decimals=1)

#Medium skill - theory 
df_the_med = df_the_grp.get_group('Medium skill')
df_the_med = df_the_med['Image analysis theory'].value_counts().to_frame().reset_index()
df_the_med['per_med'] =(df_the_med['Image analysis theory']/df_the_med['Image analysis theory'].sum())*100
df_the_med['per_med'] = df_the_med['per_med'].round(decimals=1)

#low skill -theory 
df_the_low = df_the_grp.get_group('Low skill')
df_the_low = df_the_low['Image analysis theory'].value_counts().to_frame().reset_index()
df_the_low['per_low'] =(df_the_low['Image analysis theory']/df_the_low['Image analysis theory'].sum())*100
df_the_low['per_low'] = df_the_low['per_low'].round(decimals=1)

##categories - practices
df_practices = int_skills_df.loc[:, ["Image analysis practices", "Knowledge of computational skills"]]

df_the_grp_1 = df_practices.groupby('Knowledge of computational skills')
#High skilled- practices
df_the_high_p = df_the_grp_1.get_group('High skill')
df_the_high_p = df_the_high_p['Image analysis practices'].value_counts().to_frame().reset_index()
df_the_high_p['per_high'] =(df_the_high_p['Image analysis practices']/df_the_high_p['Image analysis practices'].sum())*100
df_the_high_p['per_high'] = df_the_high_p['per_high'].round(decimals=1)

#Medium skill - practices 
df_the_med_p = df_the_grp_1.get_group('Medium skill')
df_the_med_p = df_the_med_p['Image analysis practices'].value_counts().to_frame().reset_index()
df_the_med_p['per_med'] =(df_the_med_p['Image analysis practices']/df_the_med_p['Image analysis practices'].sum())*100
df_the_med_p['per_med'] = df_the_med_p['per_med'].round(decimals=1)

#low skill -practices 
df_the_low_p = df_the_grp_1.get_group('Low skill')
df_the_low_p = df_the_low_p['Image analysis practices'].value_counts().to_frame().reset_index()
df_the_low_p['per_low'] =(df_the_low_p['Image analysis practices']/df_the_low_p['Image analysis practices'].sum())*100
df_the_low_p['per_low'] = df_the_low_p['per_low'].round(decimals=1)

#categories - particular field
df_sub = int_skills_df.loc[:, ["Topics related to sub discipline", "Knowledge of computational skills"]]

df_the_grp_sub = df_sub.groupby('Knowledge of computational skills')

#High skilled- particular field
df_the_high_sub = df_the_grp_sub.get_group('High skill')
df_the_high_sub = df_the_high_sub['Topics related to sub discipline'].value_counts().to_frame().reset_index()
df_the_high_sub['per_high'] =(df_the_high_sub['Topics related to sub discipline']/df_the_high_sub['Topics related to sub discipline'].sum())*100
df_the_high_sub['per_high'] = df_the_high_sub['per_high'].round(decimals=1)

#Medium skill - particular field 
df_the_med_sub = df_the_grp_sub.get_group('Medium skill')
df_the_med_sub = df_the_med_sub['Topics related to sub discipline'].value_counts().to_frame().reset_index()
df_the_med_sub['per_med'] =(df_the_med_sub['Topics related to sub discipline']/df_the_med_sub['Topics related to sub discipline'].sum())*100
df_the_med_sub['per_med'] = df_the_med_sub['per_med'].round(decimals=1)

#low skill -particular field 
df_the_low_sub = df_the_grp_sub.get_group('Low skill')
df_the_low_sub = df_the_low_sub['Topics related to sub discipline'].value_counts().to_frame().reset_index()
df_the_low_sub['per_low'] =(df_the_low_sub['Topics related to sub discipline']/df_the_low_sub['Topics related to sub discipline'].sum())*100
df_the_low_sub['per_low'] = df_the_low_sub['per_low'].round(decimals=1)

#categories - spc tool
df_tool = int_skills_df.loc[:, ["Specific software tool", "Knowledge of computational skills"]]

df_the_grp_tool = df_tool.groupby('Knowledge of computational skills')

#High skilled- spc tool
df_the_high_tool = df_the_grp_tool.get_group('High skill')
df_the_high_tool = df_the_high_tool['Specific software tool'].value_counts().to_frame().reset_index()
df_the_high_tool['per_high'] =(df_the_high_tool['Specific software tool']/df_the_high_tool['Specific software tool'].sum())*100
df_the_high_tool['per_high'] = df_the_high_tool['per_high'].round(decimals=1)

#Medium skill - spc tool 
df_the_med_tool = df_the_grp_tool.get_group('Medium skill')
df_the_med_tool = df_the_med_tool['Specific software tool'].value_counts().to_frame().reset_index()
df_the_med_tool['per_med'] =(df_the_med_tool['Specific software tool']/df_the_med_tool['Specific software tool'].sum())*100
df_the_med_tool['per_med'] = df_the_med_tool['per_med'].round(decimals=1)

#low skill -spc tool 
df_the_low_tool = df_the_grp_tool.get_group('Low skill')
df_the_low_tool = df_the_low_tool['Specific software tool'].value_counts().to_frame().reset_index()
df_the_low_tool['per_low'] =(df_the_low_tool['Specific software tool']/df_the_low_tool['Specific software tool'].sum())*100
df_the_low_tool['per_low'] = df_the_low_tool['per_low'].round(decimals=1)

#categories - DL
df_DL = int_skills_df.loc[:, ["Deep learning for image analysis", "Knowledge of computational skills"]]

df_the_grp_DL = df_DL.groupby('Knowledge of computational skills')

#High skilled- DL
df_the_high_DL = df_the_grp_DL.get_group('High skill')
df_the_high_DL = df_the_high_DL['Deep learning for image analysis'].value_counts().to_frame().reset_index()
df_the_high_DL['per_high'] =(df_the_high_DL['Deep learning for image analysis']/df_the_high_DL['Deep learning for image analysis'].sum())*100
df_the_high_DL['per_high'] = df_the_high_DL['per_high'].round(decimals=1)

#Medium skill - DL 
df_the_med_DL = df_the_grp_DL.get_group('Medium skill')
df_the_med_DL = df_the_med_DL['Deep learning for image analysis'].value_counts().to_frame().reset_index()
df_the_med_DL['per_med'] =(df_the_med_DL['Deep learning for image analysis']/df_the_med_DL['Deep learning for image analysis'].sum())*100
df_the_med_DL['per_med'] = df_the_med_DL['per_med'].round(decimals=1)

#low skill -DL 
df_the_low_DL = df_the_grp_DL.get_group('Low skill')
df_the_low_DL = df_the_low_DL['Deep learning for image analysis'].value_counts().to_frame().reset_index()
df_the_low_DL['per_low'] =(df_the_low_DL['Deep learning for image analysis']/df_the_low_DL['Deep learning for image analysis'].sum())*100
df_the_low_DL['per_low'] = df_the_low_DL['per_low'].round(decimals=1)

#categories - large images
df_large = int_skills_df.loc[:, ["Analyzing large images", "Knowledge of computational skills"]]

df_the_grp_large = df_large.groupby('Knowledge of computational skills')

#High skilled- large images
df_the_high_large = df_the_grp_large.get_group('High skill')
df_the_high_large = df_the_high_large['Analyzing large images'].value_counts().to_frame().reset_index()
df_the_high_large['per_high'] =(df_the_high_large['Analyzing large images']/df_the_high_large['Analyzing large images'].sum())*100
df_the_high_large['per_high'] = df_the_high_large['per_high'].round(decimals=1)

#Medium skill - large images 
df_the_med_large = df_the_grp_large.get_group('Medium skill')
df_the_med_large = df_the_med_large['Analyzing large images'].value_counts().to_frame().reset_index()
df_the_med_large['per_med'] =(df_the_med_large['Analyzing large images']/df_the_med_large['Analyzing large images'].sum())*100
df_the_med_large['per_med'] = df_the_med_large['per_med'].round(decimals=1)

#low skill -large images 
df_the_low_large = df_the_grp_large.get_group('Low skill')
df_the_low_large = df_the_low_large['Analyzing large images'].value_counts().to_frame().reset_index()
df_the_low_large['per_low'] =(df_the_low_large['Analyzing large images']/df_the_low_large['Analyzing large images'].sum())*100
df_the_low_large['per_low'] = df_the_low_large['per_low'].round(decimals=1)

#categories - Visualization of results
df_viz = int_skills_df.loc[:, ["Visualization of results", "Knowledge of computational skills"]]

df_the_grp_viz = df_viz.groupby('Knowledge of computational skills')

#High skilled- Visualization of results
df_the_high_viz = df_the_grp_viz.get_group('High skill')
df_the_high_viz = df_the_high_viz['Visualization of results'].value_counts().to_frame().reset_index()
df_the_high_viz['per_high'] =(df_the_high_viz['Visualization of results']/df_the_high_viz['Visualization of results'].sum())*100
df_the_high_viz['per_high'] = df_the_high_viz['per_high'].round(decimals=1)

#Medium skill - Visualization of results 
df_the_med_viz = df_the_grp_viz.get_group('Medium skill')
df_the_med_viz = df_the_med_viz['Visualization of results'].value_counts().to_frame().reset_index()
df_the_med_viz['per_med'] =(df_the_med_viz['Visualization of results']/df_the_med_viz['Visualization of results'].sum())*100
df_the_med_viz['per_med'] = df_the_med_viz['per_med'].round(decimals=1)

#low skill -Visualization of results 
df_the_low_viz = df_the_grp_viz.get_group('Low skill')
df_the_low_viz = df_the_low_viz['Visualization of results'].value_counts().to_frame().reset_index()
df_the_low_viz['per_low'] =(df_the_low_viz['Visualization of results']/df_the_low_viz['Visualization of results'].sum())*100
df_the_low_viz['per_low'] = df_the_low_viz['per_low'].round(decimals=1)

#Topics for interest for the tutorials based on skills 
int_ski = go.Figure()

int_ski = make_subplots(rows=4, cols=2,shared_yaxes='all',subplot_titles=('Image analysis theory', 'Image analysis practices', 'Image analysis practices <br> related to particular field', 'Specific software tool', 'Deep learning for image analysis', 'Analyzing large images', 'Visualization of results'))


int_ski.add_trace(go.Bar(name="High skill", x=df_the_high['index'], y=df_the_high['per_high'], text=df_the_high['per_high'],marker_color= 'cornflowerblue', legendgroup='group1', yaxis='y1'), row=1, col=1)
int_ski.add_trace(go.Bar(name="Medium skill", x=df_the_med['index'], y=df_the_med['per_med'], text=df_the_med['per_med'],marker_color= 'deepskyblue', legendgroup='group1'), row=1, col=1)
int_ski.add_trace(go.Bar(name="Low skill", x=df_the_low['index'], y=df_the_low['per_low'], text=df_the_low['per_low'],marker_color= 'lightskyblue', legendgroup='group1'), row=1, col=1)

int_ski.add_trace(go.Bar(x=df_the_high_p['index'], y=df_the_high_p['per_high'], text=df_the_high_p['per_high'],marker_color= 'cornflowerblue',showlegend=False, legendgroup='group1'), row=1, col=2)
int_ski.add_trace(go.Bar(x=df_the_med_p['index'], y=df_the_med_p['per_med'], text=df_the_med_p['per_med'],marker_color= 'deepskyblue',showlegend=False, legendgroup='group1'), row=1, col=2)
int_ski.add_trace(go.Bar(x=df_the_low_p['index'], y=df_the_low_p['per_low'], text=df_the_low_p['per_low'],marker_color= 'lightskyblue',showlegend=False, legendgroup='group1'), row=1, col=2)

int_ski.add_trace(go.Bar(x=df_the_high_sub['index'], y=df_the_high_sub['per_high'], text=df_the_high_sub['per_high'],marker_color= 'cornflowerblue',showlegend=False, legendgroup='group1', yaxis='y3'), row=2, col=1)
int_ski.add_trace(go.Bar(x=df_the_med_sub['index'], y=df_the_med_sub['per_med'], text=df_the_med_sub['per_med'],marker_color= 'deepskyblue',showlegend=False, legendgroup='group1'), row=2, col=1)
int_ski.add_trace(go.Bar(x=df_the_low_sub['index'], y=df_the_low_sub['per_low'], text=df_the_low_sub['per_low'],marker_color= 'lightskyblue',showlegend=False, legendgroup='group1'), row=2, col=1)

int_ski.add_trace(go.Bar(x=df_the_high_tool['index'], y=df_the_high_tool['per_high'], text=df_the_high_tool['per_high'],marker_color= 'cornflowerblue',showlegend=False, legendgroup='group1'), row=2, col=2)
int_ski.add_trace(go.Bar(x=df_the_med_tool['index'], y=df_the_med_tool['per_med'], text=df_the_med_tool['per_med'],marker_color= 'deepskyblue',showlegend=False, legendgroup='group1'), 2,2)
int_ski.add_trace(go.Bar(x=df_the_low_tool['index'], y=df_the_low_tool['per_low'], text=df_the_low_tool['per_low'],marker_color= 'lightskyblue',showlegend=False, legendgroup='group1'), 2,2)

int_ski.add_trace(go.Bar(x=df_the_high_DL['index'], y=df_the_high_DL['per_high'], text=df_the_high_DL['per_high'],marker_color= 'cornflowerblue',showlegend=False, legendgroup='group1', yaxis='y5'), row=3, col=1)
int_ski.add_trace(go.Bar(x=df_the_med_DL['index'], y=df_the_med_DL['per_med'], text=df_the_med_DL['per_med'],marker_color= 'deepskyblue',showlegend=False, legendgroup='group1'),3,1)
int_ski.add_trace(go.Bar(x=df_the_low_DL['index'], y=df_the_low_DL['per_low'], text=df_the_low_DL['per_low'],marker_color= 'lightskyblue',showlegend=False, legendgroup='group1'),3,1)

int_ski.add_trace(go.Bar(x=df_the_high_large['index'], y=df_the_high_large['per_high'], text=df_the_high_large['per_high'],marker_color= 'cornflowerblue',showlegend=False, legendgroup='group1'), row=3, col=2)
int_ski.add_trace(go.Bar(x=df_the_med_large['index'], y=df_the_med_large['per_med'], text=df_the_med_large['per_med'],marker_color= 'deepskyblue',showlegend=False, legendgroup='group1'),3,2)
int_ski.add_trace(go.Bar(x=df_the_low_large['index'], y=df_the_low_large['per_low'], text=df_the_low_large['per_low'],marker_color= 'lightskyblue',showlegend=False, legendgroup='group1'),3,2)

int_ski.add_trace(go.Bar(x=df_the_high_viz['index'], y=df_the_high_viz['per_high'], text=df_the_high_viz['per_high'],marker_color= 'cornflowerblue',showlegend=False, legendgroup='group1',yaxis='y7'), row=4, col=1)
int_ski.add_trace(go.Bar(x=df_the_med_viz['index'], y=df_the_med_viz['per_med'], text=df_the_med_viz['per_med'],marker_color= 'deepskyblue',showlegend=False, legendgroup='group1'),4,1)
int_ski.add_trace(go.Bar(x=df_the_low_viz['index'], y=df_the_low_viz['per_low'], text=df_the_low_viz['per_low'],marker_color= 'lightskyblue',showlegend=False, legendgroup='group1'),4,1)

int_ski.update_layout(width=650, height=1250, font=dict(family='Helvetica', color="Black", size=14), legend=dict(title_font_family = 'Helvetica', font=dict(size=16, color="Black")))
int_ski.update_layout(yaxis=dict(title='Percent'),yaxis3=dict(title='Percent'),yaxis5=dict(title='Percent'), yaxis7=dict(title="Percent"))
int_ski.update_layout(title='Topics of interest for the future workshops', title_x =0.2, title_y =0.99)


with st.container():
   st.subheader("Figure S5")
   st.subheader('‘Image analysis practices related to sub-discipline’ is the most preferred regardless of the computational skills')
   st.plotly_chart(int_ski)
   st.write('Answers to a multiple-choice grid question “How interested are you in learning more about the following topics?” are categorized based on ‘Level of computational skills’ as described in supplementary S2B.')
## Figure S6
#Creating dataframes for preferable instructional methods vs comfort in developing new comp skills 
dct_com ={}
list_of_col = ['Best practices articles', 'Written tutorials', 'Video tutorial', 'Interactive webinar', 'Office hours', 'One day seminar', 'Multiday workshop']
for i in list_of_col:
    df_i= data.loc[:,[i, 'Comfort']]
    df_group_i = df_i.groupby('Comfort')
    df_i_high = df_group_i.get_group('High comfort')
    df_i_high = df_i_high.value_counts().to_frame().reset_index()
    df_i_high['per_high'] = (df_i_high[0]/df_i_high[0].sum())*100
    df_i_high['per_high'] = df_i_high['per_high'].round(decimals=1)
       
    df_i_med = df_group_i.get_group('Medium comfort')
    df_i_med = df_i_med.value_counts().to_frame().reset_index()
    df_i_med['per_med'] = (df_i_med[0]/df_i_med[0].sum())*100
    df_i_med['per_med'] = df_i_med['per_med'].round(decimals=1)
    
    df_i_low = df_group_i.get_group('Low comfort')
    df_i_low = df_i_low.value_counts().to_frame().reset_index()
    df_i_low['per_low'] = (df_i_low[0]/df_i_low[0].sum())*100
    df_i_low['per_low'] = df_i_low['per_low'].round(decimals=1)
    
    columnnames = list(df_i.columns)
    df_name = columnnames[0]
    namelist = df_name.split()
    dfnamehigh = 'High' + str(namelist[-1])
    dfnamemed = 'Med' + str(namelist[-1])
    dfnamelow = 'Low' + str(namelist[-1])
    dct_com[dfnamehigh] = df_i_high
    dct_com[dfnamemed] = df_i_med
    dct_com[dfnamelow] = df_i_low
#chart - preferable instructional methods vs comfort in developing new comp skills

ins_com = go.Figure()
ins_com = make_subplots(rows=4, cols=2,shared_yaxes="all", subplot_titles=('Best practices articles', 'Written tutorials', 'Video tutorial', 'Interactive webinar','Office hours','In-person workshop <br> /One day workshop', 'Multi-day workshop'))

ins_com.add_trace(go.Bar(name='High comfort', x=dct_com['Higharticles']['Best practices articles'], y=dct_com['Higharticles']['per_high'],text=dct_com['Higharticles']['per_high'], marker_color ='darkviolet', yaxis='y1'),1,1)
ins_com.add_trace(go.Bar(name='Medium comfort', x=dct_com['Medarticles']['Best practices articles'], y=dct_com['Medarticles']['per_med'],text=dct_com['Medarticles']['per_med'], marker_color = 'blueviolet'),1,1)
ins_com.add_trace(go.Bar(name='Low comfort', x=dct_com['Lowarticles']['Best practices articles'], y=dct_com['Lowarticles']['per_low'],text=dct_com['Lowarticles']['per_low'], marker_color = 'violet'),1,1)

ins_com.add_trace(go.Bar(x=dct_com['Hightutorials']['Written tutorials'], y=dct_com['Hightutorials']['per_high'],text=dct_com['Hightutorials']['per_high'], marker_color ='darkviolet',showlegend =False),1,2)
ins_com.add_trace(go.Bar(x=dct_com['Medtutorials']['Written tutorials'], y=dct_com['Medtutorials']['per_med'],text=dct_com['Medtutorials']['per_med'], marker_color = 'blueviolet', showlegend =False),1,2)
ins_com.add_trace(go.Bar(x=dct_com['Lowtutorials']['Written tutorials'], y=dct_com['Lowtutorials']['per_low'],text=dct_com['Lowtutorials']['per_low'], marker_color = 'violet', showlegend =False),1,2)

ins_com.add_trace(go.Bar(x=dct_com['Hightutorial']['Video tutorial'], y=dct_com['Hightutorial']['per_high'],text=dct_com['Hightutorial']['per_high'], marker_color ='darkviolet',showlegend =False, yaxis='y3'),2,1)
ins_com.add_trace(go.Bar(x=dct_com['Medtutorial']['Video tutorial'], y=dct_com['Medtutorial']['per_med'],text=dct_com['Medtutorial']['per_med'], marker_color = 'blueviolet', showlegend =False),2,1)
ins_com.add_trace(go.Bar(x=dct_com['Lowtutorial']['Video tutorial'], y=dct_com['Lowtutorial']['per_low'],text=dct_com['Lowtutorial']['per_low'], marker_color = 'violet', showlegend =False),2,1)

ins_com.add_trace(go.Bar(x=dct_com['Highwebinar']['Interactive webinar'], y=dct_com['Highwebinar']['per_high'],text=dct_com['Highwebinar']['per_high'], marker_color ='darkviolet',showlegend =False),2,2)
ins_com.add_trace(go.Bar(x=dct_com['Medwebinar']['Interactive webinar'], y=dct_com['Medwebinar']['per_med'],text=dct_com['Medwebinar']['per_med'], marker_color = 'blueviolet', showlegend =False),2,2)
ins_com.add_trace(go.Bar(x=dct_com['Lowwebinar']['Interactive webinar'], y=dct_com['Lowwebinar']['per_low'],text=dct_com['Lowwebinar']['per_low'], marker_color = 'violet', showlegend =False),2,2)

ins_com.add_trace(go.Bar(x=dct_com['Highhours']['Office hours'], y=dct_com['Highhours']['per_high'],text=dct_com['Highhours']['per_high'], marker_color ='darkviolet',showlegend =False, yaxis='y5'),3,1)
ins_com.add_trace(go.Bar(x=dct_com['Medhours']['Office hours'], y=dct_com['Medhours']['per_med'],text=dct_com['Medhours']['per_med'], marker_color = 'blueviolet', showlegend =False),3,1)
ins_com.add_trace(go.Bar(x=dct_com['Lowhours']['Office hours'], y=dct_com['Lowhours']['per_low'],text=dct_com['Lowhours']['per_low'], marker_color = 'violet', showlegend =False),3,1)

ins_com.add_trace(go.Bar(x=dct_com['Highseminar']['One day seminar'], y=dct_com['Highseminar']['per_high'],text=dct_com['Highseminar']['per_high'], marker_color ='darkviolet',showlegend =False),3,2)
ins_com.add_trace(go.Bar(x=dct_com['Medseminar']['One day seminar'], y=dct_com['Medseminar']['per_med'],text=dct_com['Medseminar']['per_med'], marker_color = 'blueviolet', showlegend =False),3,2)
ins_com.add_trace(go.Bar(x=dct_com['Lowseminar']['One day seminar'], y=dct_com['Lowseminar']['per_low'],text=dct_com['Lowseminar']['per_low'], marker_color = 'violet', showlegend =False),3,2)

ins_com.add_trace(go.Bar(x=dct_com['Highworkshop']['Multiday workshop'], y=dct_com['Highworkshop']['per_high'],text=dct_com['Highworkshop']['per_high'], marker_color ='darkviolet',showlegend =False, yaxis='y7'),4,1)
ins_com.add_trace(go.Bar(x=dct_com['Medworkshop']['Multiday workshop'], y=dct_com['Medworkshop']['per_med'],text=dct_com['Medworkshop']['per_med'], marker_color = 'blueviolet', showlegend =False),4,1)
ins_com.add_trace(go.Bar(x=dct_com['Lowworkshop']['Multiday workshop'], y=dct_com['Lowworkshop']['per_low'],text=dct_com['Lowworkshop']['per_low'], marker_color = 'violet', showlegend =False),4,1)


ins_com.update_layout(width=650, height=1250, font=dict(family='Helvetica', color="Black", size=14), legend=dict(title_font_family = 'Helvetica', font=dict(size=14, color="Black")))
ins_com.update_layout(yaxis=dict(title='Percent'),yaxis3=dict(title='Percent'),yaxis5=dict(title='Percent'), yaxis7=dict(title='Percent'))
ins_com.update_layout(title='Preferable instructional methods', title_x =0.2)
ins_com.update_xaxes(categoryorder='array', categoryarray = ['Very preferable', 'Moderately preferable', 'Somewhat preferable', 'Not at all preferable'] )

#Creating dataframes for preferable instructional methods vs comp skills 
dct ={}
list_of_col = ['Best practices articles', 'Written tutorials', 'Video tutorial', 'Interactive webinar', 'Office hours', 'One day seminar', 'Multiday workshop']
for i in list_of_col:
    df_i= data.loc[:,[i, 'Knowledge of computational skills']]
    df_group_i = df_i.groupby('Knowledge of computational skills')
    df_i_high = df_group_i.get_group('High skill')
    df_i_high = df_i_high.value_counts().to_frame().reset_index()
    df_i_high['per_high'] = (df_i_high[0]/df_i_high[0].sum())*100
    df_i_high['per_high'] = df_i_high['per_high'].round(decimals=1)
       
    df_i_med = df_group_i.get_group('Medium skill')
    df_i_med = df_i_med.value_counts().to_frame().reset_index()
    df_i_med['per_med'] = (df_i_med[0]/df_i_med[0].sum())*100
    df_i_med['per_med'] = df_i_med['per_med'].round(decimals=1)
    
    df_i_low = df_group_i.get_group('Low skill')
    df_i_low = df_i_low.value_counts().to_frame().reset_index()
    df_i_low['per_low'] = (df_i_low[0]/df_i_low[0].sum())*100
    df_i_low['per_low'] = df_i_low['per_low'].round(decimals=1)
    
    columnnames = list(df_i.columns)
    df_name = columnnames[0]
    namelist = df_name.split()
    dfnamehigh = 'High' + str(namelist[-1])
    dfnamemed = 'Med' + str(namelist[-1])
    dfnamelow = 'Low' + str(namelist[-1])
    dct[dfnamehigh] = df_i_high
    dct[dfnamemed] = df_i_med
    dct[dfnamelow] = df_i_low
   

#chart - preferable instructional methods vs comp skill 

ins_skill = go.Figure()
ins_skill = make_subplots(rows=4, cols=2,shared_yaxes='all',subplot_titles=('Best practices articles', 'Written tutorials', 'Video tutorial', 'Interactive webinar','Office hours','In-person workshop <br> /One day workshop', 'Multi-day workshop'))

ins_skill.add_trace(go.Bar(name='High skill', x=dct['Higharticles']['Best practices articles'], y=dct['Higharticles']['per_high'],text=dct['Higharticles']['per_high'], marker_color ='darkgreen', yaxis='y1'),1,1)
ins_skill.add_trace(go.Bar(name='Medium skill', x=dct['Medarticles']['Best practices articles'], y=dct['Medarticles']['per_med'],text=dct['Medarticles']['per_med'], marker_color = 'green'),1,1)
ins_skill.add_trace(go.Bar(name='Low skill', x=dct['Lowarticles']['Best practices articles'], y=dct['Lowarticles']['per_low'],text=dct['Lowarticles']['per_low'], marker_color = 'lightgreen'),1,1)

ins_skill.add_trace(go.Bar(x=dct['Hightutorials']['Written tutorials'], y=dct['Hightutorials']['per_high'],text=dct['Hightutorials']['per_high'], marker_color ='darkgreen',showlegend =False),1,2)
ins_skill.add_trace(go.Bar(x=dct['Medtutorials']['Written tutorials'], y=dct['Medtutorials']['per_med'],text=dct['Medtutorials']['per_med'], marker_color = 'green', showlegend =False),1,2)
ins_skill.add_trace(go.Bar(x=dct['Lowtutorials']['Written tutorials'], y=dct['Lowtutorials']['per_low'],text=dct['Lowtutorials']['per_low'], marker_color = 'lightgreen', showlegend =False),1,2)

ins_skill.add_trace(go.Bar(x=dct['Hightutorial']['Video tutorial'], y=dct['Hightutorial']['per_high'],text=dct['Hightutorial']['per_high'], marker_color ='darkgreen',showlegend =False, yaxis='y3'),2,1)
ins_skill.add_trace(go.Bar(x=dct['Medtutorial']['Video tutorial'], y=dct['Medtutorial']['per_med'],text=dct['Medtutorial']['per_med'], marker_color = 'green', showlegend =False),2,1)
ins_skill.add_trace(go.Bar(x=dct['Lowtutorial']['Video tutorial'], y=dct['Lowtutorial']['per_low'],text=dct['Lowtutorial']['per_low'], marker_color = 'lightgreen', showlegend =False),2,1)

ins_skill.add_trace(go.Bar(x=dct['Highwebinar']['Interactive webinar'], y=dct['Highwebinar']['per_high'],text=dct['Highwebinar']['per_high'], marker_color ='darkgreen',showlegend =False),2,2)
ins_skill.add_trace(go.Bar(x=dct['Medwebinar']['Interactive webinar'], y=dct['Medwebinar']['per_med'],text=dct['Medwebinar']['per_med'], marker_color = 'green', showlegend =False),2,2)
ins_skill.add_trace(go.Bar(x=dct['Lowwebinar']['Interactive webinar'], y=dct['Lowwebinar']['per_low'],text=dct['Lowwebinar']['per_low'], marker_color = 'lightgreen', showlegend =False),2,2)

ins_skill.add_trace(go.Bar(x=dct['Highhours']['Office hours'], y=dct['Highhours']['per_high'],text=dct['Highhours']['per_high'], marker_color ='darkgreen',showlegend =False, yaxis='y5'),3,1)
ins_skill.add_trace(go.Bar(x=dct['Medhours']['Office hours'], y=dct['Medhours']['per_med'],text=dct['Medhours']['per_med'], marker_color = 'green', showlegend =False),3,1)
ins_skill.add_trace(go.Bar(x=dct['Lowhours']['Office hours'], y=dct['Lowhours']['per_low'],text=dct['Lowhours']['per_low'], marker_color = 'lightgreen', showlegend =False),3,1)

ins_skill.add_trace(go.Bar(x=dct['Highseminar']['One day seminar'], y=dct['Highseminar']['per_high'],text=dct['Highseminar']['per_high'], marker_color ='darkgreen',showlegend =False),3,2)
ins_skill.add_trace(go.Bar(x=dct['Medseminar']['One day seminar'], y=dct['Medseminar']['per_med'],text=dct['Medseminar']['per_med'], marker_color = 'green', showlegend =False),3,2)
ins_skill.add_trace(go.Bar(x=dct['Lowseminar']['One day seminar'], y=dct['Lowseminar']['per_low'],text=dct['Lowseminar']['per_low'], marker_color = 'lightgreen', showlegend =False),3,2)

ins_skill.add_trace(go.Bar(x=dct['Highworkshop']['Multiday workshop'], y=dct['Highworkshop']['per_high'],text=dct['Highworkshop']['per_high'], marker_color ='darkgreen',showlegend =False, yaxis='y7'),4,1)
ins_skill.add_trace(go.Bar(x=dct['Medworkshop']['Multiday workshop'], y=dct['Medworkshop']['per_med'],text=dct['Medworkshop']['per_med'], marker_color = 'green', showlegend =False),4,1)
ins_skill.add_trace(go.Bar(x=dct['Lowworkshop']['Multiday workshop'], y=dct['Lowworkshop']['per_low'],text=dct['Lowworkshop']['per_low'], marker_color = 'lightgreen', showlegend =False),4,1)


ins_skill.update_layout(width=650, height=1250, font=dict(family='Helvetica', color="Black", size=14), legend=dict(title_font_family = 'Helvetica', font=dict(size=16, color="Black")))
ins_skill.update_layout(yaxis=dict(title='Percent'),yaxis3=dict(title='Percent'),yaxis5=dict(title='Percent'),yaxis7=dict(title='Percent'))
ins_skill.update_layout(title='Preferable instructional methods', title_x =0.2)
ins_skill.update_xaxes(categoryorder='array', categoryarray = ['Very preferable', 'Moderately preferable' , 'Somewhat preferable', 'Not at all preferable'] )

#Creating dataframes for preferable instructional methods vs worktype
mtd_work ={}
list_of_col = ['Best practices articles', 'Written tutorials', 'Video tutorial', 'Interactive webinar', 'Office hours', 'One day seminar', 'Multiday workshop']
for i in list_of_col:
    df_i= data.loc[:,[i, 'Work type']]
    df_group_i = df_i.groupby('Work type')
    df_i_high = df_group_i.get_group('Imaging')
    df_i_high = df_i_high.value_counts().to_frame().reset_index()
    df_i_high['per_ima'] = (df_i_high[0]/df_i_high[0].sum())*100
    df_i_high['per_ima'] = df_i_high['per_ima'].round(decimals=1)
       
    df_i_med = df_group_i.get_group('Balanced')
    df_i_med = df_i_med.value_counts().to_frame().reset_index()
    df_i_med['per_bal'] = (df_i_med[0]/df_i_med[0].sum())*100
    df_i_med['per_bal'] = df_i_med['per_bal'].round(decimals=1)
    
    df_i_low = df_group_i.get_group('Analyst')
    df_i_low = df_i_low.value_counts().to_frame().reset_index()
    df_i_low['per_ana'] = (df_i_low[0]/df_i_low[0].sum())*100
    df_i_low['per_ana'] = df_i_low['per_ana'].round(decimals=1)
    
    columnnames = list(df_i.columns)
    df_name = columnnames[0]
    namelist = df_name.split()
    dfnamehigh = 'Imaging' + str(namelist[-1])
    dfnamemed = 'Balanced' + str(namelist[-1])
    dfnamelow = 'Analyst' + str(namelist[-1])
    mtd_work[dfnamehigh] = df_i_high
    mtd_work[dfnamemed] = df_i_med
    mtd_work[dfnamelow] = df_i_low
#chart - preferable instructional methods vs work type

wrk_chart = go.Figure()
wrk_chart = make_subplots(rows=4, cols=2, subplot_titles=('Best practices articles', 'Written tutorials', 'Video tutorial', 'Interactive webinar','Office hours','In-person workshop <br> /One day workshop', 'Multi-day workshop'), shared_yaxes='all')

wrk_chart.add_trace(go.Bar(name='Analyst',x=mtd_work['Analystarticles']['Best practices articles'], y=mtd_work['Analystarticles']['per_ana'],text=mtd_work['Analystarticles']['per_ana'], marker_color = 'darkturquoise', yaxis='y1'),1,1)
wrk_chart.add_trace(go.Bar(name='Balanced',x=mtd_work['Balancedarticles']['Best practices articles'], y=mtd_work['Balancedarticles']['per_bal'],text=mtd_work['Balancedarticles']['per_bal'], marker_color = 'plum'),1,1)
wrk_chart.add_trace(go.Bar(name='Imaging',x=mtd_work['Imagingarticles']['Best practices articles'], y=mtd_work['Imagingarticles']['per_ima'],text=mtd_work['Imagingarticles']['per_ima'], marker_color ='royalblue'),1,1)

wrk_chart.add_trace(go.Bar(x=mtd_work['Analysttutorials']['Written tutorials'], y=mtd_work['Analysttutorials']['per_ana'],text=mtd_work['Analysttutorials']['per_ana'], marker_color = 'darkturquoise', showlegend =False),1,2)
wrk_chart.add_trace(go.Bar(x=mtd_work['Balancedtutorials']['Written tutorials'], y=mtd_work['Balancedtutorials']['per_bal'],text=mtd_work['Balancedtutorials']['per_bal'], marker_color = 'plum', showlegend =False),1,2)
wrk_chart.add_trace(go.Bar(x=mtd_work['Imagingtutorials']['Written tutorials'], y=mtd_work['Imagingtutorials']['per_ima'],text=mtd_work['Imagingtutorials']['per_ima'], marker_color ='royalblue',showlegend =False),1,2)

wrk_chart.add_trace(go.Bar(x=mtd_work['Analysttutorial']['Video tutorial'], y=mtd_work['Analysttutorial']['per_ana'],text=mtd_work['Analysttutorial']['per_ana'], marker_color = 'darkturquoise', showlegend =False, yaxis='y3'),2,1)
wrk_chart.add_trace(go.Bar(x=mtd_work['Balancedtutorial']['Video tutorial'], y=mtd_work['Balancedtutorial']['per_bal'],text=mtd_work['Balancedtutorial']['per_bal'], marker_color = 'plum', showlegend =False),2,1)
wrk_chart.add_trace(go.Bar(x=mtd_work['Imagingtutorial']['Video tutorial'], y=mtd_work['Imagingtutorial']['per_ima'],text=mtd_work['Imagingtutorial']['per_ima'], marker_color ='royalblue',showlegend =False),2,1)

wrk_chart.add_trace(go.Bar(x=mtd_work['Analystwebinar']['Interactive webinar'], y=mtd_work['Analystwebinar']['per_ana'],text=mtd_work['Analystwebinar']['per_ana'], marker_color = 'darkturquoise', showlegend =False),2,2)
wrk_chart.add_trace(go.Bar(x=mtd_work['Balancedwebinar']['Interactive webinar'], y=mtd_work['Balancedwebinar']['per_bal'],text=mtd_work['Balancedwebinar']['per_bal'], marker_color = 'plum', showlegend =False),2,2)
wrk_chart.add_trace(go.Bar(x=mtd_work['Imagingwebinar']['Interactive webinar'], y=mtd_work['Imagingwebinar']['per_ima'],text=mtd_work['Imagingwebinar']['per_ima'], marker_color ='royalblue',showlegend =False),2,2)

wrk_chart.add_trace(go.Bar(x=mtd_work['Analysthours']['Office hours'], y=mtd_work['Analysthours']['per_ana'],text=mtd_work['Analysthours']['per_ana'], marker_color = 'darkturquoise', showlegend =False, yaxis ='y5'),3,1)
wrk_chart.add_trace(go.Bar(x=mtd_work['Balancedhours']['Office hours'], y=mtd_work['Balancedhours']['per_bal'],text=mtd_work['Balancedhours']['per_bal'], marker_color = 'plum', showlegend =False),3,1)
wrk_chart.add_trace(go.Bar(x=mtd_work['Imaginghours']['Office hours'], y=mtd_work['Imaginghours']['per_ima'],text=mtd_work['Imaginghours']['per_ima'], marker_color ='royalblue',showlegend =False),3,1)

wrk_chart.add_trace(go.Bar(x=mtd_work['Analystseminar']['One day seminar'], y=mtd_work['Analystseminar']['per_ana'],text=mtd_work['Analystseminar']['per_ana'], marker_color = 'darkturquoise', showlegend =False),3,2)
wrk_chart.add_trace(go.Bar(x=mtd_work['Balancedseminar']['One day seminar'], y=mtd_work['Balancedseminar']['per_bal'],text=mtd_work['Balancedseminar']['per_bal'], marker_color = 'plum', showlegend =False),3,2)
wrk_chart.add_trace(go.Bar(x=mtd_work['Imagingseminar']['One day seminar'], y=mtd_work['Imagingseminar']['per_ima'],text=mtd_work['Imagingseminar']['per_ima'], marker_color ='royalblue',showlegend =False),3,2)

wrk_chart.add_trace(go.Bar(x=mtd_work['Analystworkshop']['Multiday workshop'], y=mtd_work['Analystworkshop']['per_ana'],text=mtd_work['Analystworkshop']['per_ana'], marker_color = 'darkturquoise', showlegend =False, yaxis='y7'),4,1)
wrk_chart.add_trace(go.Bar(x=mtd_work['Balancedworkshop']['Multiday workshop'], y=mtd_work['Balancedworkshop']['per_bal'],text=mtd_work['Balancedworkshop']['per_bal'], marker_color = 'plum', showlegend =False),4,1)
wrk_chart.add_trace(go.Bar(x=mtd_work['Imagingworkshop']['Multiday workshop'], y=mtd_work['Imagingworkshop']['per_ima'],text=mtd_work['Imagingworkshop']['per_ima'], marker_color ='royalblue',showlegend =False),4,1)


wrk_chart.update_layout(width=650, height=1250, font=dict(family='Helvetica', color="Black", size=14), legend=dict(title_font_family = 'Helvetica', font=dict(size=14, color="Black")))
wrk_chart.update_layout(yaxis=dict(title='Percent'), yaxis3=dict(title='Percent'),yaxis5=dict(title='Percent'), yaxis7=dict(title='Percent'))
wrk_chart.update_layout(title='Preferable instructional methods', title_x =0.2)
wrk_chart.update_xaxes(categoryorder='array', categoryarray = ['Very preferable', 'Moderately preferable', 'Somewhat preferable', 'Not at all preferable'] )

with st.container():
   st.subheader('Figure S6')
   st.subheader('‘Written tutorials’ are highly preferred among all groups')
   st.write('A)')
   st.plotly_chart(ins_com)
   st.write('Figure S6 A) Answers to a multiple-choice grid question “In regards to learning more about image analysis, how preferable do you find each of these instructional methods?” are categorized based on ‘Comfort level in developing new computational skills’ as described in supplementary S2C')
   st.write('B)')
   st.plotly_chart(wrk_chart)
   st.write('Figure S6 B) Answers to a multiple-choice grid question “In regards to learning more about image analysis, how preferable do you find each of these instructional methods?” are categorized based on ‘work type’ as described in supplementary S2A.')
   st.write('C)')
   st.plotly_chart(ins_skill)
   st.write('Figure S6 C) Answers to a multiple-choice grid question “In regards to learning more about image analysis, how preferable do you find each of these instructional methods?” are categorized based on ‘Level of computational skills’ as described in supplementary S2B. ')


## Figure 8 
fig_8a = word_counts(data['Recommended conferences/workshops'],synonym_dict={'ASCB':['ascb'], 
'MMC':['mmc'], 
'ELMI':['elmi'], 
'ABRF':['abrf'], 
'NEUBIAS':['neubias'], 
'FOM':['fom'], 
'Biophysical society':['biophysical society'],
'Developmental Biology':['developmental biology']}, title='Conferences that would benefit image analysis offerings')
fig_8a.update_layout(title_x=0.2)

fig_8b = word_counts(data['Topics of interest'], synonym_dict ={'Python/coding':['python', 'script','scripts', 'scripting', 'coding', 'scriptimg'], 
'Deep/machine learning':['deep learning','machine learning','deep', 'deeplearning'], 
'FIJI':['fiji', 'imagej', 'macro', 'macros', 'imagej2'], 
'Open source':['open source'], 
'Certain tools':['certain tools', 'certain tool'], 
'Cellprofiler':['cellprofiler'], 
'Best practices':['best practices'], 
'Comparisons':['comparisons', 'comparison'], 
'Segmentation':['segmentation', 'cell segmentation'], 
'Napari':['napari'], 
}, title='Topics of interest for image analysis workshop')
fig_8b.update_layout(title_x=0.3)

with st.container():
   st.subheader('Figure 8')
   st.subheader('Topics of interest for the image analysis sessions in a conference')
   st.write('A)')
   st.plotly_chart(fig_8a)
   st.write('Figure 8A) Answers to an open-ended question “Are there any image analysis workshops, tutorials, or conferences that you have participated in and found particularly helpful? If yes, what made them beneficial?” Unigrams and bigrams were counted from the answers and the meaningful words were plotted. ')
   st.write('B)')
   st.plotly_chart(fig_8b)
   st.write('Figure 8B) Answers to an open-ended question “What specific topics (i.e. overviews of a particular tool, comparisons between pieces of software, or how to use a certain tool for a certain kind of experiment) would you like to see prioritized for future image analysis workshop and tutorial offerings?” Unigrams and bigrams were counted from the answers and the meaningful words were plotted.')


## Figure 9 
with st.container():
   st.subheader('Figure 9')
   st.subheader('More documentation and feedback are necessary')
   st.write('A)')
   #Creating a dataframe for creators role from imaging participants
   creator_role_based = data.loc[:, ['Creators role', 'Work type']]
   creator_role_based = creator_role_based.dropna()
   creator_role_based= creator_role_based.groupby('Work type')
   creator_imaging = creator_role_based.get_group('Imaging')
   creator_imaging = creator_imaging.rename(columns={'Creators role':'Creators role-Imaging'})

   cre_img = wordcloud_50(creator_imaging["Creators role-Imaging"], extra_stopwords=['tool','make','analysis', 'better', 'more', 'software', 'tools', 'user', 'step', 'users', 'use', 'image', 'don', 't', 's', 'etc', 
                 'used', 'clearly', 'doesn', 'best', 'touch', 'give', 'non', 'less', 'images', 'asap', 'high','using', 'well', 'work', 'even', 'end',
                 'others','everyone','everything','know','really','without','something','always','one','want', 'exactly', 've','m','certain', 're','instead',
                 'us','maybe','sc', 'kind','going', 'different','see','especially', 'lack','much', 'past', 'selves', 'running', 'either','require', 'happens', 'lot', 'put', 'perhaps', 
                 'properly', 'many', 'behave', 'good', 'data', 'away', 'tell', 'will'])
   st.set_option('deprecation.showPyplotGlobalUse', False)
   st.pyplot(cre_img, use_container_width=False)
   st.write('Figure 9A) Answers to an open-ended question “What do you think analysis tool CREATORS (such as software developers)could/should do to make image analysis better and more successful? How best could we encourage them to do it?” was categorized based on the “work type” as described in supplementary S2A and the unigrams of the answers from the "Imaging participants" are represented as wordclouds.')
   st.write('B)')
   #Creating a dataframe for creators role from balanced participants
   creator_balanced = creator_role_based.get_group('Balanced')
   creator_balanced = creator_balanced.rename(columns={'Creators role':'Creators role- Balanced'})

   cre_bal = wordcloud_50(creator_balanced['Creators role- Balanced'], extra_stopwords=['tool','make','analysis', 'better', 'more', 'software', 'tools', 'user', 'step', 'users', 'use', 'image', 'don', 't', 's', 'etc', 
                 'used', 'clearly', 'doesn', 'best', 'touch', 'give', 'non', 'less', 'images', 'asap', 'high','using', 'well', 'work', 'even', 'end',
                 'others','everyone','everything','know','really','without','something','always','one','want', 'exactly', 've','m','certain', 're','instead',
                 'us','maybe','sc', 'kind','going', 'different','see','especially', 'lack','much', 'past', 'selves', 'running', 'either','require', 'happens', 'lot', 'put', 'perhaps', 
                 'properly', 'many', 'behave', 'good', 'data', 'away', 'tell', 'will'])
   st.set_option('deprecation.showPyplotGlobalUse', False)
   st.pyplot(cre_bal, use_container_width=False)
   st.write('Figure 9B) Answers to an open-ended question “What do you think analysis tool CREATORS (such as software developers)could/should do to make image analysis better and more successful? How best could we encourage them to do it?” was categorized based on the “work type” as described in supplementary S2A and the unigrams of the answers from the "Balanced" work type are represented as wordclouds.')
   #Creating a dataframe for creators role from analyst 
   creator_analyst = creator_role_based.get_group('Analyst')
   creator_analyst = creator_analyst.rename(columns={'Creators role':'Creators role-Analyst'})
   cre_ana = wordcloud_50(creator_analyst['Creators role-Analyst'], extra_stopwords=['tool','make','analysis', 'better', 'more', 'software', 'tools', 'user', 'step', 'users', 'use', 'image', 'don', 't', 's', 'etc', 
                 'used', 'clearly', 'doesn', 'best', 'touch', 'give', 'non', 'less', 'images', 'asap', 'high','using', 'well', 'work', 'even', 'end',
                 'others','everyone','everything','know','really','without','something','always','one','want', 'exactly', 've','m','certain', 're','instead',
                 'us','maybe','sc', 'kind','going', 'different','see','especially', 'lack','much', 'past', 'selves', 'running', 'either','require', 'happens', 'lot', 'put', 'perhaps', 
                 'properly', 'many', 'behave', 'good', 'data', 'away', 'tell', 'will', 'd'])
   st.set_option('deprecation.showPyplotGlobalUse', False)
   st.pyplot(cre_ana, use_container_width=False)
   st.write('Figure 9C) Answers to an open-ended question “What do you think analysis tool CREATORS (such as software developers)could/should do to make image analysis better and more successful? How best could we encourage them to do it?” was categorized based on the “work type” as described in supplementary S2A and the unigrams of the answers from the "Analyst" work type are represented as wordclouds.')
   st.write('D)')
   # Grouping based on work type and users role 
   user_role_based = data.loc[:, ['Users role', 'Work type']]
   user_role_based = user_role_based.dropna()
   user_role_based= user_role_based.groupby('Work type')

   user_role_imaging = user_role_based.get_group('Imaging')
   user_role_imaging =user_role_imaging.rename(columns={'Users role':'Users role - Imaging'})

   # wordcloud for users role - imaging participants 
   user_img = wordcloud_50(user_role_imaging["Users role - Imaging"], extra_stopwords=['tool','make','analysis', 'better', 'more', 'software', 'tools', 'user', 'step', 'users', 'use', 'don', 'image', 'imaging', 'images', 
                  'data', 'think', 'best', 'need', 'encourage', 'etc', 't', 'needs', 'basic', 'able', 'keep', 'used', 'help', 'high', 'try', 'first', 'clearly',
                  'bit', 'will', 'different', 'field', 'non', 'ask']) 
   st.set_option('deprecation.showPyplotGlobalUse', False)
   st.pyplot(user_img, use_container_width=False)
   st.write('Figure 9D) Answers to an open-ended question “What do you think analysis tool USERS (such as microscopists) could/should do to make image analysis better and more successful? How best could we encourage them to do it?” was categorized based on the “work type” as described in supplementary S2A and the unigrams of the answers from the "Imaging" work type are represented as wordclouds')
   st.write("E)")
   #creating dataframe for the 'Users role-Balanced'
   user_role_balanced = user_role_based.get_group('Balanced')
   user_role_balanced = user_role_balanced.rename(columns ={'Users role':'Users role-Balanced'})
   # wordcloud for users role - balanced participants 
   user_bal = wordcloud_50(user_role_balanced["Users role-Balanced"], extra_stopwords=['tool','make','analysis', 'better', 'more', 'software', 'tools', 'user', 'step', 'users', 'use', 'don', 'image', 'imaging', 'images', 
                  'data', 'think', 'best', 'need', 'encourage', 'etc', 't', 'needs', 'basic', 'able', 'keep', 'used', 'help', 'high', 'try', 'first', 'clearly',
                  'bit', 'will', 'different', 'field', 'non', 'ask'])
   st.set_option('deprecation.showPyplotGlobalUse', False)
   st.pyplot(user_bal, use_container_width=False)
   st.write('Figure 9E) Answers to an open-ended question “What do you think analysis tool USERS (such as microscopists) could/should do to make image analysis better and more successful? How best could we encourage them to do it?” was categorized based on the “work type” as described in supplementary S2A and the unigrams of the answers from the "Balanced" work type are represented as wordclouds')
   st.write('F)')
   #Users role - Analyst
   user_role_analyst = user_role_based.get_group('Analyst')
   user_role_analyst = user_role_analyst.rename(columns={'Users role':'Users role - Analyst'})

   # wordcloud for users role - analyst  
   user_ana = wordcloud_50(user_role_analyst["Users role - Analyst"], extra_stopwords=['tool','make','analysis', 'better', 'more', 'software', 'tools', 'user', 'step', 'users', 'use', 'don', 'image', 'imaging', 'images', 
                  'data', 'think', 'best', 'need', 'encourage', 'etc', 't', 'needs', 'basic', 'able', 'keep', 'used', 'help', 'high', 'try', 'first', 'clearly',
                  'bit', 'will', 'different', 'field', 'non', 'ask'])
   st.set_option('deprecation.showPyplotGlobalUse', False)
   st.pyplot(user_ana, use_container_width=False)
   st.write('Figure 9F) Answers to an open-ended question “What do you think analysis tool USERS (such as microscopists) could/should do to make image analysis better and more successful? How best could we encourage them to do it?” was categorized based on the “work type” as described in supplementary S2A and the unigrams of the answers from the "Analyst" work type are represented as wordclouds')

## Figure S7
with st.container():
   st.subheader('Figure S7')
   st.subheader('Creators and users role as suggested by the participants ')
   st.write('A)')
   cre_rol = wordcloud_50(data["Creators role"], extra_stopwords=['tool','make','analysis', 'better', 'more', 'software', 'tools', 'user', 'step', 'users', 'use', 'image', 'don', 't', 's', 'etc', 
                 'used', 'clearly', 'doesn', 'best', 'touch', 'give', 'non', 'less', 'images', 'asap', 'high','using', 'well', 'work', 'even', 'end',
                 'others','everyone','everything','know','really','without','something','always','one','want', 'exactly', 've','m','certain', 're','instead',
                 'us','maybe','sc', 'kind','going', 'different','see','especially', 'lack','much', 'past', 'selves', 'running', 'either','require', 'happens', 'lot', 'put', 'perhaps', 
                 'properly', 'many', 'behave', 'good', 'data', 'away', 'tell', 'will'])
   st.set_option('deprecation.showPyplotGlobalUse', False)
   st.pyplot(cre_rol, use_container_width=False)
   st.write('Figure S7 A) Wordcloud representation of the unigrams of the answers to an open-ended question “What do you think analysis tool CREATORS (such as software developers)could/should do to make image analysis better and more successful? How best could we encourage them to do it?”')
   st.write('B)')
   user_rol = wordcloud_50(data["Users role"], extra_stopwords=['tool','make','analysis', 'better', 'more', 'software', 'tools', 'user', 'step', 'users', 'use', 'don', 'image', 'imaging', 'images', 
                  'data', 'think', 'best', 'need', 'encourage', 'etc', 't', 'needs', 'basic', 'able', 'keep', 'used', 'help', 'high', 'try', 'first', 'clearly',
                  'bit', 'will', 'different', 'field', 'non', 'ask'])
   st.set_option('deprecation.showPyplotGlobalUse', False)
   st.pyplot(user_rol, use_container_width=False)
   st.write('Figure S7 B) Wordcloud representation of the unigrams of the answers to an open-ended question “What do you think analysis tool USERS (such as microscopists) could/should do to make image analysis better and more successful? How best could we encourage them to do it?”')

## Figure S8 
#comparison with the literature 
source = {'Term':['Artificial intelligence', 'Deep learning', 'Machine learning', 'Survey-Deep/Machine learning'], 'Fold':[1.69, 2.11, 1.7, 0.88]}
comparison_df = pd.DataFrame(source)
S7_fig = go.Figure()
S7_fig = px.bar(x=comparison_df['Term'], y=comparison_df['Fold'], text_auto=True,labels={'y':'Fold', 'x':''})
S7_fig.update_layout(width=500, height=500, font=dict(family='Helvetica', color="Black", size=16), legend=dict(title_font_family = 'Helvetica', font=dict(size=16, color="Black")))
S7_fig.write_image('discussion.svg')

st.subheader('Figure S8')
st.subheader('Interest in machine learning/deep learning')
st.plotly_chart(S7_fig)
st.write('Figure S8 Fold change in the number of articles that were published in PubMed with the terms - ‘Artificial Intelligence’, ‘Machine learning’, ‘Deep learning’ in 2020 and 2022 were plotted along with the interest level in ‘Deep learning as applied to image analysis’ as shown in the Figure 7C and described in the methods section. A fold change of less than 1 represents decreased interest. ')


   


   
