### Requirements
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from wordcloud import WordCloud, STOPWORDS
import matplotlib.pyplot as plt
import numpy as np
import re
import kaleido
import textwrap 
from pandas.api.types import CategoricalDtype
from plotly.subplots import make_subplots
from sklearn.feature_extraction.text import CountVectorizer

#### Templates for the graphs
def barchart_vertical(series, title='', order_of_axes=[]):
    """
    Creates a dataframe for the vertical bar chart and the vertical barchart.
    Parameters:
    series:pd.Series. Takes a dataframe column as input
    title: title to be given for the graph
    order_of_axes: list of index names as strings. It defines the order of the x axes (Optional)
    """

    #creating the dcitionary of the unique values with the counts
    list = series.dropna(how='all').tolist()
    list_split = [re.split(r',\s*(?![^()]*\))', i) for i in list]   #splitting the list based on comma but not the ones inside the bracket
    list_flat = [i for innerlist in list_split for i in innerlist]  #converting a nested list to a flat list
    list_wo_spaces = [i.lstrip() for i in list_flat] #removing the leading spaces in the string
    dict_count = dict((x,list_wo_spaces.count(x)) for x in set(list_wo_spaces)) # creating a dictionary with counts of each unique values



    #creating a dataframe from the modified dictionary
    df = pd.DataFrame.from_dict(dict_count, orient='index')
    df = df.rename(columns={0:'count'})


    #plotting a graph
    fig = px.bar(x=df.index, y=df['count'], labels ={'x':'', 'y':'count'}, text_auto=True, orientation='v')
    fig.update_layout(width=500, height=500, title=title, title_x=0.5, title_y=0.95, font=dict(family='Helvetica', color="Black", size=16), legend=dict(title_font_family = 'Helvetica', font=dict(size=16, color="Black")))
    if order_of_axes == []:
        fig.update_xaxes(categoryorder = 'total descending')
    else:
        fig.update_xaxes(categoryorder='array', categoryarray = order_of_axes)

    #saving the figure as svg file
    fig.write_image(title+'.svg')

    return fig
### Dataframe for the barcharts
def df_for_barcharts(series):
    """
    Dataframe is created for the creation of stacked barchart. 
    series: The columns for which the dataframe needs to be created. 
    """
#creating the dcitionary of the unique values with the counts
    list = series.dropna(how='all').tolist()
    list_split = [re.split(r',\s*(?![^()]*\))', i) for i in list]   #splitting the list based on comma but not the ones inside the bracket
    list_flat = [i for innerlist in list_split for i in innerlist]  #converting a nested list to a flat list
    list_wo_spaces = [i.lstrip() for i in list_flat] #removing the leading spaces in the string
    dict_count = dict((x,list_wo_spaces.count(x)) for x in set(list_wo_spaces)) # creating a dictionary with counts of each unique values



    #creating a dataframe from the modified dictionary
    df = pd.DataFrame.from_dict(dict_count, orient='index')
    df = df.rename(columns={0:'count'})
    return df 

### Vertical barchart - produces the figure and not the dataframe
def barchart_vertical_distbnfig(df, title='', order_of_axes=[], color_by='', category_color={}):
    """
    Creates a vertical bar chart (only the figure).
    Parameters:
    df:a dataframe. Takes a dataframe as input
    title: title to be given for the graph
    order_of_axes: list of index names as strings. It defines the order of the x axes (Optional)
    color_by: string is given as input. Takes a dataframe column name as input and the bars are colored based on the column values.
    category_color: a dictionary is given as input. The dictionary keys are the values in the dataframe columns and the dictionary values are the colors of the bars that a user needs. 
    """
    fig = px.bar(x=df.iloc[:,0], y=df.iloc[:,1], labels ={'x':'', 'y':'count'}, text_auto=True, orientation='v', text=df.iloc[:,2], color=df[color_by], color_discrete_map = category_color)
    fig.update_layout(width=700, height=500, legend_traceorder = 'normal', legend_title = color_by)
    fig.update_layout(title=title, title_x=0.5, title_y=0.95, font=dict(family='Helvetica', color="Black", size=16), legend=dict(title_font_family = 'Helvetica', font=dict(size=16, color="Black")))
    if order_of_axes == []:
        fig.update_xaxes(categoryorder = 'total descending')
    else:
        fig.update_xaxes(categoryorder='array', categoryarray = order_of_axes)
    return fig

def barchart_horizontal(series, title='', order_of_axes=[]):
    """
    Creates a horizontal bar chart.
    Parameters:
    series:pd.Series. Takes a dataframe column as input
    title: title to be given for the graph
    order_of_axes: list of index names as strings. It defines the order of the y axes (Optional)
    """

    #creating the dcitionary of the unique values with the counts
    list = series.dropna(how='all').tolist()
    list_split = [re.split(r',\s*(?![^()]*\))', i) for i in list]   #splitting the list based on comma but not the ones inside the bracket
    list_flat = [i for innerlist in list_split for i in innerlist]  #converting a nested list to a flat list
    list_wo_spaces = [i.lstrip() for i in list_flat] #removing the leading spaces in the string
    dict_count = dict((x,list_wo_spaces.count(x)) for x in set(list_wo_spaces)) # creating a dictionary with counts of each unique values



    #creating a dataframe from the modified dictionary
    df = pd.DataFrame.from_dict(dict_count, orient='index')
    df = df.rename(columns={0:'count'})
    
    
    #plotting a graph 
    fig = px.bar(y=df.index, x=df['count'], labels ={'x':'count', 'y':''}, text_auto=True, orientation='h')                                              
    fig.update_layout(width=700, height=400, title=title, title_x=0.5, title_y=0.95, font=dict(family='Helvetica', color="Black", size=16), legend=dict(title_font_family = 'Helvetica', font=dict(size=16, color="Black")))
    if order_of_axes == []:
        fig.update_yaxes(categoryorder = 'total ascending')
    else:
        fig.update_yaxes(categoryorder='array', categoryarray = order_of_axes)

    #saving the figure as svg file
    fig.write_image(title+'.svg')

    return fig

def customwrap(s,width=25):
    return "<br>".join(textwrap.wrap(str(s),width=width))

def barchart_horizontal_fig(df, title='', order_of_axes=[]):
    """
    Creates a vertical bar chart (only the figure).
    Parameters:
    df:a dataframe. Takes a dataframe as input
    title: title to be given for the graph
    order_of_axes: list of index names as strings. It defines the order of the x axes (Optional)
    """
    #plotting a graph 
    fig = px.bar(y=df.index.map(customwrap), x=df['count'], labels ={'x':'count', 'y':''}, text_auto=True, orientation='h')                                              
    fig.update_layout(width=700, height=400, title=title, title_x=0.5, title_y=0.95, font=dict(family='Helvetica', color="Black", size=16), legend=dict(title_font_family = 'Helvetica', font=dict(size=16, color="Black")))
    if order_of_axes == []:
        fig.update_yaxes(categoryorder = 'total ascending')
    else:
        fig.update_yaxes(categoryorder='array', categoryarray = order_of_axes)
    return fig 

def tools_count(series, title=''):
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
    tools_df = tools_df.rename(columns={0:'count'})
    tools_df = tools_df.drop('None')


    #plotting a graph
    fig = px.bar(x=tools_df['count'], y=tools_df.index, labels ={'x':'count', 'y':''}, text_auto=True, orientation='h')
    fig.update_layout(width=700, height=400, title=title, title_x=0.5, title_y=0.95, font=dict(family='Helvetica', color="Black", size=16), legend=dict(title_font_family = 'Helvetica', font=dict(size=16, color="Black")))
    fig.update_yaxes(categoryorder = 'total ascending')

    #saving the figure as svg file
    fig.write_image(title+'.svg')

    return fig

def tools_count_df(series):
    """
    Creates bar chart for commonly used and most used image analysis tools
    Parameters:
    series:pd.Series. Takes a dataframe column as input
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
    tools_df = tools_df.rename(columns={0:'count'})
    if 'None' in tools_df.index.values.tolist():
        tools_df = tools_df.drop('None')
    return tools_df 

##### Stacked barchart - creates the dataframe and the figure
def stacked_barchart(data, title='', order_of_axes=[], order_of_stacks=[]):
    """
    stacked bar chart for the types of images analyzed
    Parameters:
    data:pd.DataFrame
    title: title to be given to the plot.Images get saved in this name
    order_of_axes: list of index names as strings. It defines the order of the x axes.
    order_of_stacks: list of the column names as strings. It defines the order of the stacks
    """
    df = pd.DataFrame()
    for colnames, items in data.iteritems():
        items = items.dropna()
        df_to_list = items.tolist()
        list_split = [i.split(',') for i in df_to_list]
        list_flat = [i for innerlist in list_split for i in innerlist]
        list_wo_space = [i.replace(' ', '') for i in list_flat]
        dict_values = dict((x,list_wo_space.count(x)) for x in set(list_wo_space))
        dataframe = pd.DataFrame.from_dict(dict_values, orient='index')         #the dictionary with the unique values and the number of occurences are made into a dataframe
        dataframe = dataframe.transpose()
        dataframe['kinds'] = items.name
        df = pd.concat([df, dataframe], axis=0)


    #creating the stacked bar chart
    layout = go.Layout(margin=go.layout.Margin(l=500))
    fig = go.Figure(layout=layout)
    df = df.set_index('kinds')
    df = df[order_of_stacks]


    for i in df.columns:
        fig.add_trace(go.Bar(name=i, y=df.index,x=df[i],orientation='h', text=df[i]))

    fig.update_layout(barmode='stack')
    fig.update_layout(width=1200,legend=dict(yanchor="bottom",y=0.02,xanchor="right",x=0.99))
    fig.update_layout(title =title, title_x=0.5, height =500, title_y=0.9, font=dict(family='Helvetica', color="Black", size=16), legend=dict(title_font_family = 'Helvetica', font=dict(size=16, color="Black")))
    if order_of_axes == []:
        fig.update_layout(yaxis={'categoryorder':'total ascending'})
    else:
        fig.update_yaxes(categoryorder='array', categoryarray = order_of_axes)
    fig.write_image(title+'.svg')
    return fig
#Creation of dataframes for the stacked barchart
    """
    Dataframe is created for the creation of stacked barchart. 
    data: The columns for which the dataframe needs to be created. 
    """
def df_for_stackedchart(data):
    df = pd.DataFrame()
    for colnames, items in data.iteritems():
        items = items.dropna()
        df_to_list = items.tolist()
        list_split = [i.split(',') for i in df_to_list]
        list_flat = [i for innerlist in list_split for i in innerlist]
        list_wo_space = [i.replace(' ', '') for i in list_flat]
        dict_values = dict((x,list_wo_space.count(x)) for x in set(list_wo_space))
        dataframe = pd.DataFrame.from_dict(dict_values, orient='index')         #the dictionary with the unique values and the number of occurences are made into a dataframe
        dataframe = dataframe.transpose()
        dataframe['kinds'] = items.name
        df = pd.concat([df, dataframe], axis=0)
    df = df.set_index('kinds')
    return df 
#creating the stacked bar chart without creating the dataframe 
def stacked_barchart_fig(df, title='', order_of_axes=[], order_of_stacks=[]):
    """
    stacked bar chart figure alone is created. The final dataframe will be provided as the input
    Parameters:
    data:pd.DataFrame
    title: title to be given to the plot.Images get saved in this name
    order_of_axes: list of index names as strings. It defines the order of the x axes.
    order_of_stacks: list of the column names as strings. It defines the order of the stacks
    """
    layout = go.Layout(margin=go.layout.Margin(l=400))
    fig = go.Figure(layout=layout)
    df = df[order_of_stacks]


    for i in df.columns:
        fig.add_trace(go.Bar(name=i, y=df.index,x=df[i],orientation='h', text=df[i]))

    fig.update_layout(barmode='stack')
    fig.update_layout(width=1200,legend=dict(yanchor="bottom",y=0.02,xanchor="right",x=0.99))
    fig.update_layout(title =title, title_x=0.5, title_y=0.9, font=dict(family='Helvetica', color="Black", size=16), legend=dict(title_font_family = 'Helvetica', font=dict(size=16, color="Black")))
    if order_of_axes == []:
        fig.update_layout(yaxis={'categoryorder':'total ascending'})
    else:
        fig.update_yaxes(categoryorder='array', categoryarray = order_of_axes)
    return fig 


##### Percentage stacked bar charts
def percentage_stackedcharts(data, title='', order_of_axes=[], order_of_stacks=[], colors={}):
    """
    Percentage_stacked charts
    Parameters:
    data: pd.DataFrame
    title: title to be given to the plot.Images get saved in this name
    order_of_axes: list of index names as strings. It defines the order of the x axes.
    order_of_stacks: list of the column names as strings. It defines the order of the stacks
    """

    df = pd.DataFrame()
    for colnames, items in data.iteritems():
        dataframe = items.value_counts().to_frame() # counting the number of occurences of each unique values
        df = pd.concat([df,dataframe], axis=1)  # Appending to the values to the dataframe

    df = df.reset_index()           #resetting the index of the dataframe such that it is easier for making graphs
    df = df.rename(columns={'index':'interest'})
    df = df.set_index('interest').transpose()
    df.columns = df.columns.str.replace(' ', '')



    #creating a dataframe for the percentage values
    per_df = pd.DataFrame()         #creating a separate dataframe for percentage values
    for col in df.columns:
        per_df[col] = [((i/df.iloc[0, 0:4].sum())*100) for i in df[col]]  #calculating the percentage values
        per_df[col] = per_df[col].round(decimals=1)
    per_df.index =df.index  #defining the index of the dataframe to be similar as the previous dataframe


    #creating the chart
    fig = go.Figure()
    per_df = per_df[order_of_stacks]
    
    for i in per_df.columns:
        fig.add_trace(go.Bar(name=i,x=per_df.index, y=per_df[i], text=[f'{val}%' for val in per_df[i]], marker={'color':colors[i]}))

    fig.update_layout(width=800, height=700, barmode='stack')
    fig.update_yaxes(title='Percent')
    fig.update_layout(title=title, title_x = 0.5, font=dict(family='Helvetica', color="Black", size=16), legend=dict(title_font_family = 'Helvetica', font=dict(size=16, color="Black")))
    if order_of_axes == []:
        fig.update_layout(yaxis={'categoryorder':'total ascending'})
    else:
        fig.update_xaxes(categoryorder='array', categoryarray = order_of_axes, tickangle=90)
    fig.write_image(title+'.svg')
    return fig

def df_for_percentage_stackedcharts(data):
    """
    Creation of dataframe for percentage_stacked charts
    Parameters:
    data: pd.DataFrame
    """
    df = pd.DataFrame()
    for colnames, items in data.iteritems():
        dataframe = items.value_counts().to_frame() # counting the number of occurences of each unique values
        df = pd.concat([df,dataframe], axis=1)  # Appending to the values to the dataframe

    df = df.reset_index()           #resetting the index of the dataframe such that it is easier for making graphs
    df = df.rename(columns={'index':'interest'})
    df = df.set_index('interest').transpose()
    #df.columns = df.columns.str.replace(' ', '')

    #creating a dataframe for the percentage values
    per_df = pd.DataFrame()         #creating a separate dataframe for percentage values
    for col in df.columns:
        per_df[col] = [((i/df.iloc[0, 0:4].sum())*100) for i in df[col]]  #calculating the percentage values
        per_df[col] = per_df[col].round(decimals=1)
    per_df.index =df.index  #defining the index of the dataframe to be similar as the previous dataframe
    return per_df

def percentage_stackedcharts_fig(data, title='', order_of_axes=[], order_of_stacks=[], colors={}):
    """
    Percentage_stacked charts
    Parameters:
    data: pd.DataFrame
    title: title to be given to the plot.Images get saved in this name
    order_of_axes: list of index names as strings. It defines the order of the x axes.
    order_of_stacks: list of the column names as strings. It defines the order of the stacks
    """
    #creating the chart
    fig = go.Figure()
    data = data[order_of_stacks]
    
    for i in data.columns:
        fig.add_trace(go.Bar(name="<br>".join(textwrap.wrap(i,width=12)),x=data.index.map(customwrap), y=data[i], text=[f'{val}%' for val in data[i]], marker={'color':colors[i]}))

    fig.update_layout(width=800, height=700, barmode='stack')
    fig.update_yaxes(title='Percent')
    fig.update_layout(title=title, title_x = 0.5, font=dict(family='Helvetica', color="Black", size=16), legend=dict(title_font_family = 'Helvetica', font=dict(size=16, color="Black")))
    if order_of_axes == []:
        fig.update_layout(yaxis={'categoryorder':'total ascending'})
    else:
        fig.update_xaxes(categoryorder='array', categoryarray = order_of_axes, tickangle=90)
    fig.write_image(title+'.svg')
    return fig

def percentage_stackedcharts_fig_horizontal(data, title='', order_of_axes=[], order_of_stacks=[], colors={}):
    """
    Percentage_stacked charts
    Parameters:
    data: pd.DataFrame
    title: title to be given to the plot.Images get saved in this name
    order_of_axes: list of index names as strings. It defines the order of the x axes.
    order_of_stacks: list of the column names as strings. It defines the order of the stacks
    """
    #creating the chart
    fig = go.Figure()
    data = data[order_of_stacks]
    
    for i in data.columns:
        fig.add_trace(go.Bar(name="<br>".join(textwrap.wrap(i,width=12)),x=data[i], y=data.index.map(customwrap),orientation='h', text=[f'{val}%' for val in data[i]], marker={'color':colors[i]}))

    fig.update_layout(width=800, height=700, barmode='stack')
    fig.update_xaxes(title='Percent')
    fig.update_layout(title=title, title_x = 0.5, font=dict(family='Helvetica', color="Black", size=16), legend=dict(title_font_family = 'Helvetica', font=dict(size=16, color="Black")), xaxis_range= [0, 110])
    if order_of_axes == []:
        fig.update_layout(xaxis={'categoryorder':'total ascending'})
    else:
        fig.update_yaxes(categoryorder='array', categoryarray = order_of_axes)
    fig.write_image(title+'horizontal.svg')
    return fig

##### Wordcloud
#### Worcloud with wrapped titles 
def wordcloud(series,extra_stopwords=[]):
    """
    Makes a wordcloud based on the words in a given column
    Parameters:
    series: pd.Series. A dataframe column
    extra_stopwords: list of words that needs to be removed from the wordcloud
    """
    input = ''.join(series.str.lower().str.split().dropna(how='all').astype(str).str.replace(r'[-./?!,":;()\']',' ')) #splitting the strings and replacing the spaces or special characters
    words_to_remove = series.name.split()
    stopwords_new = words_to_remove + list(STOPWORDS) + extra_stopwords
    wc_image = WordCloud(stopwords=stopwords_new, background_color='white', width=600, height=600, random_state=4).generate(input)
    plt.imshow(wc_image)
    plt.tight_layout()
    plt.title("\n".join(textwrap.wrap(series.name, 40)))
    plt.axis('off')
    plt.savefig('Image-' +series.name+ '.svg', bbox_inches='tight', dpi=300)
    return 
#worcloud with 50 words 
def wordcloud_50(series,extra_stopwords=[]):
    """
    Makes a wordcloud based on the words in a given column
    Parameters:
    series: pd.Series. A dataframe column
    extra_stopwords: list of words that needs to be removed from the wordcloud
    """
    input = ''.join(series.str.lower().str.split().dropna(how='all').astype(str).str.replace(r'[-./?!,":;()\']',' ')) #splitting the strings and replacing the spaces or special characters
    words_to_remove = series.name.split()
    stopwords_new = words_to_remove + list(STOPWORDS) + extra_stopwords
    wc_image = WordCloud(stopwords=stopwords_new,max_words=50, background_color='white', width=600, height=600, random_state=4).generate(input)
    plt.imshow(wc_image)
    plt.tight_layout()
    plt.title(series.name)
    plt.axis('off')
    plt.savefig('Image-50' +series.name+ '.svg')
    return

#### Sunburst chart
def sunburst_chart(df, order_list, color_column='', custom_colors={}, title=''):
    """
    Makes a sunburst chart for comparing the work type, computational skill and level of comfort in developing new computational skill
    df: pd.DataFrame. A dataframe with all the columns that needs to be compared
    order_list: a list of the dataframe column names. This defines the order of the sunburst charts. The given dataframe will be grouped based on the last column name provided in this list.
    color_column: Column name in a string format. This defines the color of the sunburst
    custom_colors: A dictionary having the categories as keys and the colors used as the values.

    """
    #Creating the dataframe
    df = df.groupby(order_list[-1]).value_counts().reset_index()  #creating the counts based on grouping the column that was last in the 'order_list'
    df = df.rename(columns={0:'count'})
    #creating the sunburst chart
    fig = px.sunburst(df, path=order_list,color = color_column, color_discrete_map = custom_colors, values='count', width=500, height=500)
    fig.update_traces(textinfo="label+percent parent", insidetextorientation = 'radial')
    fig.update_layout(title=title, title_x=0.5, font=dict(family='Helvetica', color="Black", size=16))
    fig.write_image(title+'.svg')
    fig.write_image(title+'.png')
    return fig 
##### Creating CSVs for ngrams
def ngrams(series, range=(), name=''):
    """
    Creates a csv file of unigrams/bigrams/trigrams and their frequency of occurence.
    Parameters
    series:pd.Series. A dataframe column to be given as input
    range:tuple. If unigrams are expected then (1,1) is given as the parameter and for bigrams (2,2) is given as the parameter. If a mixture of unigrams and bigrams are expected then (1,2) can be given as the parameter.
    name: filename for the csv.
    """
    series = series.dropna()
    c_vec = CountVectorizer(stop_words='english', ngram_range=range)
    ngrams = c_vec.fit_transform(series)        # matrix of ngrams
    count_values = ngrams.toarray().sum(axis=0)  # count frequency of ngrams
    vocab = c_vec.vocabulary_       #creates a dictionary of the 'words' as the key and the position as 'values'
    df = pd.DataFrame(sorted([(count_values[i],k) for k,i in vocab.items()], reverse=True)).rename(columns={0: 'frequency', 1:'ngram'})       # list of ngrams
    df.to_csv(name + '.csv')
    return df 

#### Word counts
def word_counts(series, synonym_dict={}, title=''):
    """
    Counts the occurence of the words in the given series.
    Parameters
    series: pd.Series. A column of a dataframe which has to be analyzed.
    synoymn_dict: A dictionary where the keys are axis labels that we woud like to have in the graph and the values are the synonyms of the same word in the key/things want to combine.
    title:title to be given for the graph
    """

    series = series.dropna().tolist()

    vectorizer2 = CountVectorizer(analyzer='word', ngram_range=(1, 3))
    X2 = vectorizer2.fit_transform(series)
    question_keys = list(vectorizer2.get_feature_names_out())
    question_positions = X2.toarray()

    some_final_count_dict = {}
    for eachkey in synonym_dict.keys(): #for everything you want to graph
        keylist = []
        for eachsyn in synonym_dict[eachkey]:
            keylist.append(question_keys.index(eachsyn)) # make a list of the position of all of the keys we want to include for this entry
        question_totals = question_positions[:,keylist].sum(axis=1) # sum all the rows with the query terms we want
        some_final_count_dict[eachkey]= np.where(question_totals >0, 1,0).sum() # make a final count of nonzero answers, and sum it

    df= pd.DataFrame.from_dict(some_final_count_dict, orient='index')
    df = df.rename(columns={0:'count'})

    fig = go.Figure()

    fig= px.bar(y=df.index, x=df['count'], text=df['count'], labels={' ':'count', 'y':''}, orientation='h')
    fig.update_layout(title=title, title_x=0.5, width = 500, height=700, showlegend=False)
    fig.update_yaxes(categoryorder='total ascending')
    fig.update_xaxes(title='count')
    fig.update_layout(width=500, height=600, title_y=0.95, font=dict(family='Helvetica', color="Black", size=16), legend=dict(title_font_family = 'Helvetica', font=dict(size=16, color="Black")))
    if len(title)>20:
        fig.update_layout(title = title[0:23]+ '<br>' +title[23:], title_x = 0.45)
    else:
        fig.update_layout(title=title)
    fig.write_image(title+'.svg')
    return fig 

### To create a dataframe comparing the topics/methods based on the subgroups such work type, comfort, comp skill
def df_for_comparison(data, list_of_col=[], list_of_groups=[], slicing_column=[], sort_order=[], name=''):
    """
    Creates a dataframe for the different topics of interest/ preferable instructional methods by categorizing based on either work type, comfort level or computational skills. The dataframe has the percentage values in each of these categories. It also saves a csv file.
    data:pd.DataFrame.The whole dataframe from which the required columns will be sliced. 
    list_of_col: a list of the column names that needs to be sliced out of the dataframe. 
    list_of_groups: a list of the group names. For example- if it is work type the 'Imaging', 'Balanced' and 'Analyst' needs to be provided as a list.
    slicing_column: a list of the column name based on which the 'list_of_col' needs to be grouped. For example - if the topics of interest needs to be grouped based on work type then 'Work type' should be provided here.
    sort_order: a list of the order in which we want the csv to be created. 
    name: string. csv file is saved with this string name plus 'combined'. 
    """
    df = pd.DataFrame()
    cat_order = CategoricalDtype(sort_order,ordered=True)
    for i in list_of_col:
        for j in slicing_column:
            df_i= data.loc[:,[i, j]]
            df_group_i = df_i.groupby(j)
            for category in list_of_groups:
                df_i_cat = df_group_i.get_group(category)
                df_i_cat = df_i_cat.value_counts().to_frame().reset_index()
                df_i_cat[i +' '+ category] = (df_i_cat[0]/df_i_cat[0].sum())*100
                df_i_cat[i +' '+ category] = df_i_cat[i +' '+ category].round(decimals=1)
                final_df = df_i_cat.loc[:,[i,i +' '+ category]]
                final_df[i] = final_df[i].astype(cat_order)
                final_df = final_df.sort_values(i)
                final_df = final_df.set_index(i)                  
                df = pd.concat([df,final_df], axis=1)
        df.to_csv(name +'combined' +'.csv')
          
    return df 

### Creates subplots based on the dataframe provided
def fig_subgroups(combined_data, list_of_col=[], list_of_groups=[], colorkey=[], title=''):
    """
    Makes plotly subplots. 
    combined_data:pd.DataFrame. Possibly the output of the 'df_for_comparison' function.
    list_of_col: a list based on which the title of the subplots are provided. 
    list_of_groups: a list based on which the dataframe is filtered. For example- if it is work type the 'Imaging', 'Balanced' and 'Analyst' needs to be provided as a list.
    colorkey:a list of colors that the 'list_of_groups' will be colored accordingly. 
    title: a string. The graph will have the provided string as the title. 
    """
    fig = go.Figure()
    fig = make_subplots(rows=4, cols=2,shared_yaxes="all", subplot_titles=tuple(list_of_col))

    for i in range(0, len(list_of_groups)):
        df_1 = combined_data.filter(like=list_of_groups[i])
        fig.add_trace(go.Bar(name=list_of_groups[i], x=df_1.index, y=df_1.iloc[:,0], text=df_1.iloc[:,0], marker_color=colorkey[i], yaxis='y1'), 1,1)
        fig.add_trace(go.Bar(name=list_of_groups[i], x=df_1.index, y=df_1.iloc[:,1], text=df_1.iloc[:,1], marker_color=colorkey[i], showlegend =False), 1,2)
        fig.add_trace(go.Bar(name=list_of_groups[i], x=df_1.index, y=df_1.iloc[:,2], text=df_1.iloc[:,2], marker_color=colorkey[i], showlegend =False,yaxis='y3'), 2,1)
        fig.add_trace(go.Bar(name=list_of_groups[i], x=df_1.index, y=df_1.iloc[:,3], text=df_1.iloc[:,3], marker_color=colorkey[i], showlegend =False), 2,2)
        fig.add_trace(go.Bar(name=list_of_groups[i], x=df_1.index, y=df_1.iloc[:,4], text=df_1.iloc[:,4], marker_color=colorkey[i], showlegend =False,yaxis='y5'), 3,1)
        fig.add_trace(go.Bar(name=list_of_groups[i], x=df_1.index, y=df_1.iloc[:,5], text=df_1.iloc[:,5], marker_color=colorkey[i], showlegend =False), 3,2)
        fig.add_trace(go.Bar(name=list_of_groups[i], x=df_1.index, y=df_1.iloc[:,6], text=df_1.iloc[:,6], marker_color=colorkey[i], showlegend =False,yaxis='y7'), 4,1)

    fig.update_layout(width=650, height=1250, font=dict(family='Helvetica', color="Black", size=14), legend=dict(title_font_family = 'Helvetica', font=dict(size=14, color="Black")))
    fig.update_layout(yaxis=dict(title='Percent'), yaxis3=dict(title='Percent'),yaxis5=dict(title='Percent'), yaxis7=dict(title='Percent'))
    fig.update_layout(title=title, title_x =0.5)

    return fig 

### Another customwrap that is specifically used for the wordcount_barchart
def customwrap2(s,width=15):
    return "<br>".join(textwrap.wrap(str(s),width=width))

### Creates a barchart from the dataframe and also lists the total number of entries in the dataframe 
def wordcount_barchart(data, title='', total=()):
    """
    data: pd.DataFrame. The dataframe entries are filtered for the top 10 entries.
    title: a string. The title for the barchart
    total:pd.series. The total number of this particular series is tken as the total number of respondents and displayed in the figure
    """
    wordcount_df = data.filter(items=list(range(0,10)), axis=0)
    fig = px.bar(y=wordcount_df['ngram'].map(customwrap2), x=wordcount_df['frequency'],orientation='h', labels={'x':'frequency', 'y':''}, text_auto=True)
    fig.update_layout(width=500, height=600, title_y=0.95, font=dict(family='Helvetica', color="Black", size=16), legend=dict(title_font_family = 'Helvetica', font=dict(size=16, color="Black")))
    fig.update_yaxes(categoryorder='total ascending')
    fig.update_layout(title=title, title_x=0.5)
    fig.add_annotation(text="Total number of respondents = " + str(len(total)),xref="paper", yref="paper",x=0.5, y=1.05, showarrow=False)
    fig.write_image(title+'.svg')
    return fig
