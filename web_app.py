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
import textwrap 

from utils import *

st.set_page_config(
   layout='wide',
   initial_sidebar_state='collapsed'
)

url = 'https://raw.githubusercontent.com/COBA-NIH/2023_ImageAnalysisSurvey/main/csv%20files/data.csv'
data = pd.read_csv(url)

#Figure 1A -graphs
role_url = 'https://raw.githubusercontent.com/COBA-NIH/2023_ImageAnalysisSurvey/main/csv%20files/role.csv'
role_val_counts =pd.read_csv(role_url)

# pie chart for the roles of the survey participants 
role_pie_chart = px.pie(role_val_counts, values=role_val_counts['counts'], names=role_val_counts['role'], title="Role", width=800, height=500)
role_pie_chart.update_traces(insidetextorientation = 'radial', textinfo='value+percent')
role_pie_chart.update_layout(title_x=0.8,title_y = 0.85, font=dict(family='Helvetica', color='Black', size=16), legend=dict(title_font_family = 'Helvetica', font=dict(size=16, color="Black")))
# Including the trainee information in the legend 
role_pie_chart['data'][0]['labels'][0] = 'Student - Graduate/Undergraduate <br> (trainee)'
role_pie_chart['data'][0]['labels'][2] = 'Postdoctoral fellow (trainee)'

#Figure 1B 
#Geo chart for the location of the participants; locations were given based on a country that is centrally located in a continent
demographics = data["Location"]
demographics_chart = px.scatter_geo(demographics, locations=['UKR', 'USA', 'KGZ', 'BRA', 'AUS','TCD'],size_max=20,opacity=0.2, projection="natural earth", color = data.Location.value_counts().values,text=data.Location.value_counts(), title="Location", labels={'color':'Continent'}, width=600, height=400)
demographics_chart.update_layout(title_x=0.50,title_y=0.80, font=dict(family='Helvetica', color="Black", size=16), legend=dict(title_font_family = 'Helvetica', font=dict(size=16, color="Black")))
demographics_chart.update_layout(showlegend=False)
demographics_chart.update_coloraxes(showscale=False)
demographics_chart.update_traces(marker=dict(color='blue'))
demographics_chart.update_traces(marker={'size':25})

#Figure 1C -graphs
df_domains_url = 'https://raw.githubusercontent.com/COBA-NIH/2023_ImageAnalysisSurvey/main/csv%20files/role_and_domains.csv'
df_domains = pd.read_csv(df_domains_url, index_col=0)

#Role and domains
role_domain_fig = stacked_barchart_fig(df_domains, title='Role Based Training & Experience', order_of_stacks=['Developmental Biology','Deep learning','Computer vision','Computer science','Medicine','Statistics/Biostatistics','Chemistry/Biochemistry','Physics/Biophysics','Cell/Molecular Biology' ])

st.title('Bridging Imaging Users to Imaging Analysis - A community survey')
## Figure 1
with st.container():
    st.subheader('Figure 1')
    st.subheader('Survey respondents roles and training histories vary across the sampled responses')
    st.write('A)')
    st.plotly_chart(role_pie_chart,theme=None)
    st.write('Figure 1A) Answers to the multiple-choice question “Which of the following roles best describes you?”')
    st.write('B)')
    st.plotly_chart(demographics_chart)
    st.write('Figure 1B)- Answers to the multiple-choice question “Where do you currently primarily work?”')
    st.write('C)')
    st.plotly_chart(role_domain_fig, theme=None, use_container_width=True)
    st.write('Figure 1C) Answers to the check-box question “Which of the following do you have significant formal training in or experience with? Select all that apply.” Responses were categorized based on the answers provided for part 1A.')

#Figure S1A -graphs 
training_csv_url = 'https://raw.githubusercontent.com/COBA-NIH/2023_ImageAnalysisSurvey/main/csv%20files/training_experience.csv'
training_df = pd.read_csv(training_csv_url)

#Figure S1B -graphs
url = 'https://github.com/COBA-NIH/2023_ImageAnalysisSurvey/raw/main/Graphs/cellprofiler_website_access_2022.tif'
response = requests.get(url)
img = Image.open(BytesIO(response.content))

#Creating the figures
training_pie =px.pie(training_df, values=training_df['counts'],names=training_df.index, title="Training and experience", width=700, height=500)
training_pie.update_traces(insidetextorientation = 'radial', textinfo='value+percent')
training_pie.update_layout(title_x=0.95,title_y = 0.85, font=dict(family='Helvetica', color='Black', size=16), legend=dict(title_font_family = 'Helvetica', font=dict(size=16, color="Black")))

## Supplementary figure S1
st.subheader('Figure S1')
st.subheader('Survey respondents were drawn primarily from North America and Europe')
st.write('A)')
st.image(training_pie)
st.write('Figure S1A) Training and experience of the survey respondents')
st.write('B)')
st.image(img)
st.write('Figure S1B) Image from Google Analytics showing the number of visitors to the Cellprofiler website in the year 2022. The scale bar indicates the number of visitors')

##Figure 2
work_comp_com_df_url = 'https://raw.githubusercontent.com/COBA-NIH/2023_ImageAnalysisSurvey/main/csv%20files/work_comp_com_df.csv'
work_comp_com_df = pd.read_csv(work_comp_com_df_url, index_col=0)

#Overall chart
lif_phy = sunburst_chart(work_comp_com_df,order_list=["Work type", 'Knowledge of computational skills', 'Comfort'], color_column = 'Work type', custom_colors= {'Imaging':'lightskyblue', 'Balanced':'darkseagreen', 'Analyst':'orchid'}, title='Skills of the participants' )
lif_phy.update_layout(title_x=0.35, width=1000, height=1000)

## Figure 2
with st.container():
    st.subheader('Figure 2')
    st.subheader('Skills of the participants')
    st.plotly_chart(lif_phy)
    st.write('Figure 2- Breakdown of answers to the multiple-choice questions “How would you describe your work?”, “How would you rate your computational skills?” and “How would you rate your comfort in developing new computational skills?” Percentages were rounded to the nearest percent; in outer wedges, percentages are of the adjacent inner wedge population. See methods and Supplemental Figure 2 for fuller descriptions of each category.')

###Figure S2
# Reading the CSVs
work_cat_url = 'https://raw.githubusercontent.com/COBA-NIH/2023_ImageAnalysisSurvey/main/csv%20files/work_des.csv'
work_cat = pd.read_csv(work_cat_url, index_col=0)

comp_skill_distbn_url = 'https://raw.githubusercontent.com/COBA-NIH/2023_ImageAnalysisSurvey/main/csv%20files/comp_skill_distbn.csv'
comp_skill_distbn = pd.read_csv(comp_skill_distbn_url, index_col=0)

comf_distbn_url = 'https://raw.githubusercontent.com/COBA-NIH/2023_ImageAnalysisSurvey/main/csv%20files/comf_distbn.csv'
comf_distbn = pd.read_csv(comf_distbn_url, index_col=0)

trainee_df_url = 'https://raw.githubusercontent.com/COBA-NIH/2023_ImageAnalysisSurvey/main/csv%20files/trainee_df.csv'
trainee_df = pd.read_csv(trainee_df_url, index_col=0)

trainee_comp_comf_df_url = 'https://raw.githubusercontent.com/COBA-NIH/2023_ImageAnalysisSurvey/main/csv%20files/trainee_comp_comf_df.csv'
trainee_comp_comf_df = pd.read_csv(trainee_comp_comf_df_url)

work_comp_com_lif_df_url ='https://raw.githubusercontent.com/COBA-NIH/2023_ImageAnalysisSurvey/main/csv%20files/work_comp_com_lif_df.csv'
work_comp_com_lif_df =pd.read_csv(work_comp_com_lif_df_url)

work_comp_com_phy_df_url ='https://raw.githubusercontent.com/COBA-NIH/2023_ImageAnalysisSurvey/main/csv%20files/work_comp_com_phy_df.csv'
work_comp_com_phy_df =pd.read_csv(work_comp_com_phy_df_url)

# Creating the figures
work_cat_bar = barchart_vertical_distbnfig(work_cat, title = 'Work description',color_by='Category', category_color={'Imaging':'lightskyblue', 'Balanced':'darkseagreen', 'Analyst':'orchid'})
work_cat_bar.update_layout(title_x=0.4)

comp_skill_distbn_bar = barchart_vertical_distbnfig(comp_skill_distbn, title='Level of computational skills', color_by='Category', category_color={'Low skill':'lightskyblue', 'Medium skill':'darkseagreen', 'High skill':'orchid'})
comp_skill_distbn_bar.update_layout(title_x=0.3)

comf_distbn_bar = barchart_vertical_distbnfig(comf_distbn, title='Comfort in developing new computational skills', color_by='Category', category_color={'Low comfort':'lightskyblue', 'Medium comfort':'darkseagreen', 'High  comfort':'orchid'})
comf_distbn_bar.update_layout(title_x=0.2)

# Charts for the tarinees vs non-trainees
trainee_worktype_fig  = sunburst_chart(trainee_df, order_list=['Trainee status', "Work type"], color_column='Trainee status', custom_colors={'Trainee':'lightskyblue', 'Nontrainee':'darkseagreen'}, title='Worktype categorized based on trainee status')
trainee_worktype_fig.update_layout(title_x=0.3)

trainee_comp_comf_fig = sunburst_chart(trainee_comp_comf_df, order_list=['Trainee status', 'Knowledge of computational skills', 'Comfort'], color_column='Trainee status', custom_colors={'Trainee':'lightskyblue', 'Nontrainee':'darkseagreen'}, title='Computational skills of the trainees and nontrainees') 
trainee_comp_comf_fig.update_layout(title_x=0.25)

#Charts for the life sciences vs physical sciences 
sun_lif = sunburst_chart(work_comp_com_lif_df, order_list=["Work type", 'Knowledge of computational skills', 'Comfort'], color_column='Work type', custom_colors={'Imaging':'lightskyblue', 'Balanced':'darkseagreen', 'Analyst':'orchid'}, title='Skills of the participants (Life Sciences)')
sun_lif.update_layout(title_x=0.3)

sun_phy = sunburst_chart(work_comp_com_phy_df, order_list=["Work type", 'Knowledge of computational skills', 'Comfort'], color_column='Work type', custom_colors={'Imaging':'lightskyblue', 'Balanced':'darkseagreen', 'Analyst':'orchid'}, title='Skills of the participants (Physical Sciences)')
sun_phy.update_layout(title_x=0.3)

###Figure S2 
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
 

 ### Figure 3
 images_lif_df_url = 'https://raw.githubusercontent.com/COBA-NIH/2023_ImageAnalysisSurvey/main/csv%20files/images_lif_df.csv'
 images_lif_df = pd.read_csv(images_lif_df_url, index_col=0)
 

 images_phy_df_url = 'https://raw.githubusercontent.com/COBA-NIH/2023_ImageAnalysisSurvey/main/csv%20files/images_phy_df.csv'
 images_phy_df = pd.read_csv(images_phy_df_url, index_col=0)
 

 kinds_life = stacked_barchart_fig(images_lif_df,title="What kinds of images do you commonly want to analyze (life sciences)", order_of_stacks=['2D', '2D+time','3D(<3000x3000x100)', '3D+time','3D(SPIM/largevolume)', '3Dlargevolume+time'])
 kinds_phy = stacked_barchart_fig(images_phy_df,title="What kinds of images do you commonly want to analyze (physical sciences)", order_of_stacks=['2D', '2D+time','3D(<3000x3000x100)', '3D+time','3D(largevolume)','3Dlargevolume+time'])

with st.container():
   st.subheader('Figure 3')
   st.subheader('Kinds of images analyzed')
   st.write('A)')
   st.plotly_chart(kinds_life, theme=None, use_container_width=True)
   st.write('Figure 3 A) Answers to the checkbox grid question “What kinds of images do you commonly want to analyze (select all that apply)?” under the “Life Sciences Image Analysis” section.')
   st.write('B)')
   st.plotly_chart(kinds_phy, theme=None, use_container_width=True)
   st.write('Figure 3 B) Answers to the checkbox grid question “What kinds of images do you commonly want to analyze (select all that apply)?” under the “Physical Sciences Image Analysis” section')

###Figure 4
### Reading the CSVs
com_lif_df_url = 'https://raw.githubusercontent.com/COBA-NIH/2023_ImageAnalysisSurvey/main/csv%20files/com_lif_df.csv'
com_lif_df = pd.read_csv(com_lif_df_url, index_col=0)

most_lif_df_url = 'https://raw.githubusercontent.com/COBA-NIH/2023_ImageAnalysisSurvey/main/csv%20files/most_lif_df.csv'
most_lif_df = pd.read_csv(most_lif_df_url, index_col=0)

fre_lif_df_counts_url = 'https://raw.githubusercontent.com/COBA-NIH/2023_ImageAnalysisSurvey/main/csv%20files/freq_lif_df_counts.csv'
fre_lif_df_counts = pd.read_csv(fre_lif_df_counts_url,index_col=0)

com_phy_df_url = 'https://raw.githubusercontent.com/COBA-NIH/2023_ImageAnalysisSurvey/main/csv%20files/com_phy_df.csv'
com_phy_df = pd.read_csv(com_phy_df_url,index_col=0)

most_phy_df_url = 'https://raw.githubusercontent.com/COBA-NIH/2023_ImageAnalysisSurvey/main/csv%20files/most_phy_df.csv'
most_phy_df = pd.read_csv(most_phy_df_url, index_col=0)

fre_phy_df_counts_url = 'https://raw.githubusercontent.com/COBA-NIH/2023_ImageAnalysisSurvey/main/csv%20files/freq_phy_df_counts.csv'
fre_phy_df_counts = pd.read_csv(fre_phy_df_counts_url, index_col=0)

#Life sciences
com_lif = barchart_horizontal_fig(com_lif_df, title='Commonly used image analysis tools (life sciences)')
com_lif.update_layout(title_x=0.3)

most_lif = barchart_horizontal_fig(most_lif_df, title='Most used image analysis tools (life sciences)')
most_lif.update_layout(title_x=0.3)

freq_lif = barchart_horizontal_fig(fre_lif_df_counts, title='Frequency of script usage (life sciences)', order_of_axes=['Most of the time','Often','Sometimes','Never'])
freq_lif.update_layout(title_x=0.3)

#Physical sciences
com_phy = barchart_horizontal_fig(com_phy_df, title='Commonly used image analysis tools (physical sciences)', order_of_axes=['Computational libraries<br>and scripts','Other commercial software','Commercial software that<br>comes with my microscope', 'Open source point-and-<br>click software'])
com_phy.update_layout(title_x=0.3)

most_phy = barchart_horizontal_fig(most_phy_df, title='Most used image analysis tools (physical sciences)', order_of_axes=['Other commercial software','Commercial software that<br>comes with my microscope','Computational libraries<br>and scripts','Open source point-and-<br>click software'])
most_phy.update_layout(title_x=0.3)

freq_phy = barchart_horizontal_fig(fre_phy_df_counts, title='Frequency of script usage (physical sciences)', order_of_axes=['Most of the time','Often','Sometimes','Never'])
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

#Figure S3 
#Reading dataframes 
scp_df_1_url = 'https://raw.githubusercontent.com/COBA-NIH/2023_ImageAnalysisSurvey/main/csv%20files/script_usage_lif.csv'
scp_df_1 = pd.read_csv(scp_df_1_url, index_col=0)

scp_df_phy_1_url  = 'https://raw.githubusercontent.com/COBA-NIH/2023_ImageAnalysisSurvey/main/csv%20files/script_usage_phy.csv'
scp_df_phy_1 = pd.read_csv(scp_df_phy_1_url, index_col=0)

#Creating the Fig S3 figures 
#Life sciences
scp_lif = px.bar(scp_df_1, x=scp_df_1['Work type'], y=scp_df_1['percent'], title='Script usage (life sciences)', width=500, height=500,text=[f'{val}%' for val in scp_df_1['percent']], labels={'Work type':'','percent':'Percent'})
scp_lif.update_layout(title_x =0.5, font=dict(family='Helvetica', color='Black', size=16), legend=dict(title_font_family = 'Helvetica', font=dict(size=16, color="Black")))
scp_lif.update_xaxes(categoryorder='array', categoryarray =['Imaging', 'Balanced', 'Analyst'])

#Physical sciences
scp_phy = px.bar(scp_df_phy_1, x=scp_df_phy_1['Work type'], y=scp_df_phy_1['percent'], title='Script usage (physical sciences)', width=500, height=500,text=[f'{val}%' for val in scp_df_phy_1['percent']],labels={'Work type':'', 'percent':'Percent'})
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
way_analyze_df = df_for_barcharts(data["Approach to solutions"])

approach = barchart_horizontal_fig(way_analyze_df, title='Approach to solutions')
approach.update_layout(height=550)

#Reading the images from github

with st.container():
   st.subheader('Figure 5')
   st.subheader('Solving image analysis problems')
   st.write('A)')
   st.plotly_chart(approach)
   st.write('Figure 5A) Answers to the checkbox question “How do you generally go about solving an image analysis problem? Check the approach(es) you use the most.” ')
   
   st.write('B)')
   st.image('https://raw.githubusercontent.com/COBA-NIH/2023_ImageAnalysisSurvey/main/Graphs/wordclouds/Image-Well%20solved%20image%20analysis%20problems%20(life%20sciences).svg')
   st.write('Figure 5B) Wordcloud representation of the unigrams of the answers by ‘life science’ participants to an open-ended question “What image analysis problems (i.e. finding nuclei, tissue analysis, analysis of super-resolution data, etc) do you think are generally well-solved?”')
   
   st.write('C)')
   st.image('https://raw.githubusercontent.com/COBA-NIH/2023_ImageAnalysisSurvey/main/Graphs/wordclouds/Image-Image%20analysis%20problems%20which%20needs%20better%20solutions%20(life%20sciences).svg')
   st.write('Figure 5C) Wordcloud representation of the unigrams of the answers by ‘life science’ participants to an open-ended question “What image analysis problems (i.e. finding nuclei, tissue analysis, analysis of super-resolution data, etc) do you wish had easier/better solutions?”')
   
   st.write('D)')
   st.image('https://raw.githubusercontent.com/COBA-NIH/2023_ImageAnalysisSurvey/main/Graphs/wordclouds/Image-Well%20solved%20image%20analysis%20problems%20(physical%20sciences).svg')
   st.write('Figure 5D) Wordcloud representation of the unigrams of the answers by ‘physical science’ participants to an open-ended question “What image analysis problems (i.e. finding nuclei, tissue analysis, analysis of super-resolution data, etc) do you think are generally well-solved?”')
   
   st.write('E)')
   st.image('https://raw.githubusercontent.com/COBA-NIH/2023_ImageAnalysisSurvey/main/Graphs/wordclouds/Image-Image%20analysis%20problems%20which%20needs%20better%20solutions%20(physical%20sciences).svg')
   st.write('Figure 5E) Wordcloud representation of the unigrams of the answers by ‘’physical science’ participants to an open-ended question “What image analysis problems (i.e. finding nuclei, tissue analysis, analysis of super-resolution data, etc) do you wish had easier/better solutions?”')

#Figure S4
per_sc_usage_df_url = 'https://raw.githubusercontent.com/COBA-NIH/2023_ImageAnalysisSurvey/main/csv%20files/usage_sc_forum.csv'
per_sc_usage_df = pd.read_csv(per_sc_usage_df_url, index_col=0)

sc_usage_fig = px.bar(per_sc_usage_df, per_sc_usage_df['index'], y=per_sc_usage_df['percentage'], text_auto=True, width=500, height=500,labels={'index':'', 'percentage':'Percent'}, title='Usage of image sc forum')
sc_usage_fig.update_layout(width=500, height=500, title_x=0.5, title_y=0.95, font=dict(family='Helvetica', color="Black", size=16), legend=dict(title_font_family = 'Helvetica', font=dict(size=16, color="Black")))

with st.container():
   st.subheader('Figure S4')
   st.subheader('Image analysis problems as stated by 2020 survey participants')
   st.write('A)')
   st.plotly_chart(sc_usage_fig, use_container_width=False)
   st.write('Figure S4A) Percentage usage of image sc forum was calculated based on answers provided for the question ‘How do you generally go about solving an image analysis problem? Check the approach(es) you use the most’ normalized with the work type that was categorized based on the answers provided to “How would you describe your work?”. Categorization is given in detail in Figure S2A.')
   
   st.write('B)')
   st.image('https://raw.githubusercontent.com/COBA-NIH/2023_ImageAnalysisSurvey/main/Graphs/wordclouds/Image-Well%20solved%20image%20analysis%20problems-2020.svg')
   st.write('Figure S4B) Wordcloud representation of the unigrams of the answers provided by the 2020 bioimage analysis survey participants to an open-ended question “What image analysis problems (i.e. finding nuclei, tissue analysis, analysis of super-resolution data, etc) do you think are generally well-solved?”.')
   
   st.write('C)')
   st.image('https://raw.githubusercontent.com/COBA-NIH/2023_ImageAnalysisSurvey/main/Graphs/wordclouds/Image-Image%20analysis%20problems%20that%20need%20better%20solutions-2020.svg')
   st.write('Figure S4C) Wordcloud representation of the unigrams of the answers provided by the 2020 bioimage analysis survey participants to an open-ended question “What image analysis problems (i.e. finding nuclei, tissue analysis, analysis of super-resolution data, etc) do you wish had easier/better solutions?”')


# Figure 6
#Reading the CSVs
pre_exp_url = 'https://raw.githubusercontent.com/COBA-NIH/2023_ImageAnalysisSurvey/main/csv%20files/previous_experience.csv'
pre_exp_df = pd.read_csv(pre_exp_url, index_col=0)

conf_att_df_url = 'https://raw.githubusercontent.com/COBA-NIH/2023_ImageAnalysisSurvey/main/csv%20files/conf_att_df.csv'
conf_att_df = pd.read_csv(conf_att_df_url, index_col=0)

workshops_for_chart_url = 'https://raw.githubusercontent.com/COBA-NIH/2023_ImageAnalysisSurvey/main/csv%20files/workshops_for_chart.csv'
workshops_for_chart = pd.read_csv(workshops_for_chart_url, index_col=0)

#Creating the figures 
pre_exp = barchart_horizontal_fig(pre_exp_df, title='Previous experience')

conf_att = barchart_horizontal_fig(conf_att_df, title='Conferences attended by the survey participants', order_of_axes=['Many','Some','Few','None'])

workshops_chart = barchart_horizontal_fig(workshops_for_chart, title='Workshops attended by the participants')

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


#Figure 7
#Reading the CSVs
top_int_df_url = 'https://raw.githubusercontent.com/COBA-NIH/2023_ImageAnalysisSurvey/main/csv%20files/topics_interest_df.csv'
top_int_df = pd.read_csv(top_int_df_url, index_col=0)


pre_mtd_url ='https://raw.githubusercontent.com/COBA-NIH/2023_ImageAnalysisSurvey/main/csv%20files/preferred_mtds_df.csv'
pre_mtd_df = pd.read_csv(pre_mtd_url, index_col=0)

fold_change_top_url =('https://raw.githubusercontent.com/COBA-NIH/2023_ImageAnalysisSurvey/main/csv%20files/fold_change_topics.csv')
fold_change_top = pd.read_csv(fold_change_top_url, index_col=0)

fold_change_int_url = ('https://raw.githubusercontent.com/COBA-NIH/2023_ImageAnalysisSurvey/main/csv%20files/fold_change_instructional.csv')
fold_change_int = pd.read_csv(fold_change_int_url, index_col=0)


#Creating the figures 
top_int = percentage_stackedcharts_fig_horizontal(top_int_df, title='How interested are you in learning more about the following topics',order_of_stacks=['Very interested', 'Moderately interested', 'Somewhat interested', 'Not at all interested'],order_of_axes=['Topics related to sub<br>discipline', 'Visualization of results', 'Analyzing large images', 'Image analysis practices', 'Specific software tool', 'Deep learning for image<br>analysis', 'Image analysis theory'], colors={'Very interested':'royalblue','Moderately interested':'dodgerblue','Somewhat interested':'cornflowerblue','Not at all interested':'skyblue'})
top_int.update_layout(title_x =0.2)

pre_mtd  = percentage_stackedcharts_fig_horizontal(pre_mtd_df, title='How preferable do you find each of these instructional methods',order_of_stacks=['Very preferable', 'Moderately preferable', 'Somewhat preferable', 'Not at all preferable'],order_of_axes=['Written tutorials', 'Video tutorial', 'Office hours', 'One day seminar', 'Interactive webinar','Best practices articles', 'Multiday workshop'], colors={'Very preferable':'royalblue','Moderately preferable':'dodgerblue','Somewhat preferable':'cornflowerblue','Not at all preferable':'skyblue'})
pre_mtd.update_layout(title_x =0.2)


int_top_subplots = make_subplots(rows=1, cols=2, shared_yaxes='all', subplot_titles=('Interest for future workshops', 'Preferable instructional methods'))
int_top_subplots.add_trace(go.Bar(x=fold_change_top.index.map(customwrap), y=fold_change_top['fold'], orientation='v',showlegend=False, marker_color='royalblue', yaxis='y1'), 1,1)
int_top_subplots.add_trace(go.Bar(x=fold_change_int.index.map(customwrap), y=fold_change_int['fold'],orientation='v',showlegend=False, marker_color='royalblue'), 1,2)
int_top_subplots.update_layout(width=1000, height=500)
int_top_subplots.update_xaxes(categoryorder ='total ascending', tickangle = 90)
int_top_subplots.update_layout(yaxis1 =(dict(title='Fold change')))
int_top_subplots.update_layout(font=dict(family='Helvetica', color="Black", size=14), legend=dict(title_font_family = 'Helvetica', font=dict(size=16, color="Black")))

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
   
#Figure S5
int_ski_df_url = ('https://raw.githubusercontent.com/COBA-NIH/2023_ImageAnalysisSurvey/main/csv%20files/Skill_topicscombined.csv')
int_ski_df = pd.read_csv(int_ski_df_url, index_col=0)

int_ski = fig_subgroups(int_ski_df,list_of_col=['Image analysis theory', 'Image analysis practices', 'Topics related to sub discipline', 'Specific software tool', 'Deep learning for image analysis', 'Analyzing large images', 'Visualization of results'],list_of_groups = ['High skill', 'Medium skill', 'Low skill'],colorkey=['cornflowerblue', 'deepskyblue', 'lightskyblue'], title='Topics of interest for the future workshops')

with st.container():
   st.subheader("Figure S5")
   st.subheader('‘Image analysis practices related to sub-discipline’ is the most preferred regardless of the computational skills')
   st.plotly_chart(int_ski)
   st.write('Answers to a multiple-choice grid question “How interested are you in learning more about the following topics?” are categorized based on ‘Level of computational skills’ as described in supplementary S2B.')


#Figure S6
ins_com_url =('https://raw.githubusercontent.com/COBA-NIH/2023_ImageAnalysisSurvey/main/csv%20files/Comfort_methodscombined.csv')
ins_com_df = pd.read_csv(ins_com_url, index_col=0)
ins_com = fig_subgroups(ins_com_df,list_of_col = ['Best practices articles', 'Written tutorials', 'Video tutorial', 'Interactive webinar', 'Office hours', 'One day seminar', 'Multiday workshop'], list_of_groups = ['High comfort', 'Medium comfort', 'Low comfort'],colorkey = ['darkviolet', 'blueviolet', 'violet'],title = 'Preferable instructional methods')

wrk_chart_url = ('https://raw.githubusercontent.com/COBA-NIH/2023_ImageAnalysisSurvey/main/csv%20files/Worktype_methodscombined.csv')
wrk_chart_df = pd.read_csv(wrk_chart_url, index_col=0)
wrk_chart = fig_subgroups(wrk_chart_df,list_of_col = ['Best practices articles', 'Written tutorials', 'Video tutorial', 'Interactive webinar', 'Office hours', 'One day seminar', 'Multiday workshop'], list_of_groups = ['Imaging', 'Balanced', 'Analyst'],colorkey = ['royalblue', 'plum', 'darkturquoise'],title = 'Preferable instructional methods')

ins_skill_url =('https://raw.githubusercontent.com/COBA-NIH/2023_ImageAnalysisSurvey/main/csv%20files/Skill_methodscombined.csv')
ins_skill_df = pd.read_csv(ins_skill_url, index_col=0)
ins_skill = fig_subgroups(ins_skill_df,list_of_col = ['Best practices articles', 'Written tutorials', 'Video tutorial', 'Interactive webinar', 'Office hours', 'One day seminar', 'Multiday workshop'], list_of_groups = ['High skill', 'Medium skill', 'Low skill'],colorkey = ['darkgreen', 'green', 'lightgreen'],title = 'Preferable instructional methods')

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
   st.image('https://raw.githubusercontent.com/COBA-NIH/2023_ImageAnalysisSurvey/main/Graphs/wordclouds/Image-50Creators%20role-Imaging.svg')
   st.write('Figure 9A) Answers to an open-ended question “What do you think analysis tool CREATORS (such as software developers)could/should do to make image analysis better and more successful? How best could we encourage them to do it?” was categorized based on the “work type” as described in supplementary S2A and the unigrams of the answers from the "Imaging participants" are represented as wordclouds.')
   
   st.write('B)')
   st.image('https://raw.githubusercontent.com/COBA-NIH/2023_ImageAnalysisSurvey/main/Graphs/wordclouds/Image-50Creators%20role-%20Balanced.svg')
   st.write('Figure 9B) Answers to an open-ended question “What do you think analysis tool CREATORS (such as software developers)could/should do to make image analysis better and more successful? How best could we encourage them to do it?” was categorized based on the “work type” as described in supplementary S2A and the unigrams of the answers from the "Balanced" work type are represented as wordclouds.')
   
   st.write('C)')
   st.image('https://raw.githubusercontent.com/COBA-NIH/2023_ImageAnalysisSurvey/main/Graphs/wordclouds/Image-50Creators%20role-Analyst.svg')
   st.write('Figure 9C) Answers to an open-ended question “What do you think analysis tool CREATORS (such as software developers)could/should do to make image analysis better and more successful? How best could we encourage them to do it?” was categorized based on the “work type” as described in supplementary S2A and the unigrams of the answers from the "Analyst" work type are represented as wordclouds.')
   
   st.write('D)')
   st.image('https://raw.githubusercontent.com/COBA-NIH/2023_ImageAnalysisSurvey/main/Graphs/wordclouds/Image-50Users%20role%20-%20Imaging.svg')
   st.write('Figure 9D) Answers to an open-ended question “What do you think analysis tool USERS (such as microscopists) could/should do to make image analysis better and more successful? How best could we encourage them to do it?” was categorized based on the “work type” as described in supplementary S2A and the unigrams of the answers from the "Imaging" work type are represented as wordclouds')
   
   st.write("E)")
   st.image('https://raw.githubusercontent.com/COBA-NIH/2023_ImageAnalysisSurvey/main/Graphs/wordclouds/Image-50Users%20role-Balanced.svg')
   st.write('Figure 9E) Answers to an open-ended question “What do you think analysis tool USERS (such as microscopists) could/should do to make image analysis better and more successful? How best could we encourage them to do it?” was categorized based on the “work type” as described in supplementary S2A and the unigrams of the answers from the "Balanced" work type are represented as wordclouds')
   
   st.write('F)')
   st.image('https://raw.githubusercontent.com/COBA-NIH/2023_ImageAnalysisSurvey/main/Graphs/wordclouds/Image-50Users%20role%20-%20Analyst.svg')
   st.write('Figure 9F) Answers to an open-ended question “What do you think analysis tool USERS (such as microscopists) could/should do to make image analysis better and more successful? How best could we encourage them to do it?” was categorized based on the “work type” as described in supplementary S2A and the unigrams of the answers from the "Analyst" work type are represented as wordclouds')


#Reading the dataframe for figure S7 
#Creator
#Creator-imaging 
creator_imaging_csv_url =  'https://raw.githubusercontent.com/COBA-NIH/2023_ImageAnalysisSurvey/main/csv%20files/creator_imaging.csv'
creator_imaging = pd.read_csv(creator_imaging_csv_url, index_col=0)

creator_imaging_ngrams_url = 'https://raw.githubusercontent.com/COBA-NIH/2023_ImageAnalysisSurvey/main/csv%20files/creator_imaging_ngrams.csv'
creator_imaging_ngrams = pd.read_csv(creator_imaging_ngrams_url, index_col=0)

#Creator- Balanced 
creator_balanced_csv_url = 'https://raw.githubusercontent.com/COBA-NIH/2023_ImageAnalysisSurvey/main/csv%20files/creator_balanced.csv'
creator_balanced = pd.read_csv(creator_balanced_csv_url, index_col =0)

creator_balanced_ngrams_url = 'https://raw.githubusercontent.com/COBA-NIH/2023_ImageAnalysisSurvey/main/csv%20files/creator_balanced_ngrams.csv'
creator_balanced_ngrams = pd.read_csv(creator_balanced_ngrams_url, index_col=0)

#Creator -Analyst 
creator_analyst_csv_url = 'https://raw.githubusercontent.com/COBA-NIH/2023_ImageAnalysisSurvey/main/csv%20files/creator_analyst.csv'
creator_analyst = pd.read_csv(creator_analyst_csv_url, index_col=0)

creator_analyst_ngrams_url = 'https://raw.githubusercontent.com/COBA-NIH/2023_ImageAnalysisSurvey/main/csv%20files/creator_analyst_ngrams.csv'
creator_analyst_ngrams = pd.read_csv(creator_analyst_ngrams_url, index_col=0)

#User 
#User -Imaging 
user_imaging_csv_url = 'https://raw.githubusercontent.com/COBA-NIH/2023_ImageAnalysisSurvey/main/csv%20files/user_role_imaging.csv'
user_imaging_csv = pd.read_csv(user_imaging_csv_url, index_col=0)

user_imaging_ngrams_url = 'https://raw.githubusercontent.com/COBA-NIH/2023_ImageAnalysisSurvey/main/csv%20files/user_imaging_ngrams.csv'
user_imaging_ngrams = pd.read_csv(user_imaging_ngrams_url, index_col=0)

#User - Balanced 
user_balanced_csv_url = 'https://raw.githubusercontent.com/COBA-NIH/2023_ImageAnalysisSurvey/main/csv%20files/user_role_balanced.csv'
user_balanced_csv = pd.read_csv(user_balanced_csv_url, index_col=0)

user_balanced_ngrams_url = 'https://raw.githubusercontent.com/COBA-NIH/2023_ImageAnalysisSurvey/main/csv%20files/user_balanced_ngrams.csv'
user_balanced_ngrams =pd.read_csv(user_balanced_ngrams_url, index_col=0)

#User-analyst
user_analyst_csv_url = 'https://raw.githubusercontent.com/COBA-NIH/2023_ImageAnalysisSurvey/main/csv%20files/user_role_analyst.csv'
user_analyst_csv = pd.read_csv(user_analyst_csv_url, index_col=0)

user_analyst_ngrams_url = 'https://github.com/COBA-NIH/2023_ImageAnalysisSurvey/blob/main/csv%20files/user_analyst_ngrams.csv'
user_analyst_ngrams = pd.read_csv(user_analyst_ngrams_url, index_col=0)

#Creating the barcharts for figure S7 
creator_imaging = wordcount_barchart(creator_imaging_ngrams, title='Creators role-Imaging', total=creator_imaging["Creators role-Imaging"])
creator_balanced = wordcount_barchart(creator_balanced_ngrams, title='Creators role-Balanced', total=creator_balanced['Creators role- Balanced'])
creator_analyst = wordcount_barchart(creator_analyst_ngrams, title='Creators role-Analyst', total = creator_analyst['Creators role-Analyst'])

user_imaging = wordcount_barchart(user_imaging_bigram, title='Users role-Imaging', total=user_role_imaging["Users role - Imaging"])
user_balanced  = wordcount_barchart(user_balanced_bigram, title='Users role-Balanced', total=user_role_balanced["Users role-Balanced"])
user_analyst = wordcount_barchart(user_analyst_bigram, title='Users role-Analyst', total=user_role_analyst["Users role - Analyst"])


#Figure S7 
with st.container():
   st.subheader('Figure S7')
   st.subheader('Creators and users role as suggested by the participants ')
   st.write('A)')
   st.image('https://raw.githubusercontent.com/COBA-NIH/2023_ImageAnalysisSurvey/main/Graphs/wordclouds/Image-50Creators%20role.svg')
   st.write('Figure S7 A) Wordcloud representation of the unigrams of the answers to an open-ended question “What do you think analysis tool CREATORS (such as software developers)could/should do to make image analysis better and more successful? How best could we encourage them to do it?”')
   
   st.write('B)')
   st.image('https://raw.githubusercontent.com/COBA-NIH/2023_ImageAnalysisSurvey/main/Graphs/wordclouds/Image-50Users%20role.svg')
   st.write('Figure S7 B) Wordcloud representation of the unigrams of the answers to an open-ended question “What do you think analysis tool USERS (such as microscopists) could/should do to make image analysis better and more successful? How best could we encourage them to do it?”')

   st.write('C)')
   st.image(creator_imaging)
   st.image(creator_balanced)
   st.image(creator_analyst)
   st.write('Figure S7 C) Answers to an open-ended question “What do you think analysis tool CREATORS (such as software developers)could/should do to make image analysis better and more successful? How best could we encourage them to do it?” was categorized based on the “work type” as described in supplementary S2A and the top ten bigrams of the responses are represented as bar charts.')

   st.write('D)')
   st.image(user_imaging)
   st.image(user_balanced)
   st.image(user_analyst)
   st.write('Figure S7 D) Answers to an open-ended question “What do you think analysis tool USERS (such as microscopists) could/should do to make image analysis better and more successful? How best could we encourage them to do it?” was categorized based on the “work type” as described in supplementary S2A and the top ten bigrams of the responses are represented as bar charts.')

#Figure S8 
comparison_df_url = 'https://raw.githubusercontent.com/COBA-NIH/2023_ImageAnalysisSurvey/main/csv%20files/Comparison_google_data_with_survey.csv'
comparison_df = pd.read_csv(comparison_df_url, index_col=0)

S8_fig = go.Figure()
S8_fig = px.bar(x=comparison_df['Index'], y=comparison_df['Fold'], text_auto=True,labels={'y':'Fold', 'x':''})
S8_fig.update_layout(width=500, height=500, font=dict(family='Helvetica', color="Black", size=16), legend=dict(title_font_family = 'Helvetica', font=dict(size=16, color="Black")))

st.subheader('Figure S8')
st.subheader('Interest in machine learning/deep learning')
st.plotly_chart(S8_fig)
st.write('Figure S8 Fold change in the number of articles that were published in PubMed with the terms - ‘Artificial Intelligence’, ‘Machine learning’, ‘Deep learning’ in 2020 and 2022 were plotted along with the interest level in ‘Deep learning as applied to image analysis’ as shown in the Figure 7C and described in the methods section. A fold change of less than 1 represents decreased interest. ')
