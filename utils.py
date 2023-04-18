### Requirements
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from wordcloud import WordCloud, STOPWORDS
import matplotlib.pyplot as plt
import re 
import kaleido


#### Templates for the graphs
def tools_count(df, title=''):
    """
    Creates bar chart for commonly used and most used image analysis tools
    Parameters:
    df:pd.Series. Takes a dataframe column as input 
    title: title to be given for the graph
    """

    #creating the dcitionary of the unique values with the counts 
    list = df.dropna(how='all').tolist()
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
    tools_df = tools_df.drop('None')

    
    #plotting a graph 
    fig = px.bar(x=tools_df['counts'], y=tools_df.index, labels ={'x':'counts', 'y':''}, text_auto=True, orientation='h')                                              
    fig.update_layout(width=700, height=400, title=title, title_x=0.5, title_y=0.95, font=dict(family='Helvetica', color="Black", size=16), legend=dict(title_font_family = 'Helvetica', font=dict(size=16, color="Black")))
    fig.update_yaxes(categoryorder = 'total ascending')
    
    #saving the figure as svg file 
    fig.write_image(title+'.svg')

    return fig.show()


##### Stacked barchart
def stacked_barchart(columns=[], title='', order_of_axes=[], order_of_stacks=[]):
    """ 
    stacked bar chart for the types of images analyzed 
    Parameters:
    columns: list of all the columns in a dataframe to be analyzed
    title: title to be given to the plot.Images get saved in this name 
    order_of_axes: list of index names as strings. It defines the order of the x axes. 
    order_of_stacks: list of the column names as strings. It defines the order of the stacks 
    """
    df = pd.DataFrame()
    for i in columns:
        remove_na = i.dropna(how='all')       #removing the null values from the dataframe columns
        df_to_list = remove_na.tolist()        #converting the panda series to a list 
        list_split = [i.split(',') for i in df_to_list]         #the values inside the column are split by comma
        list_flat = [i for innerlist in list_split for i in innerlist]      #the nested list is made to a flatlist 
        list_wo_space = [i.replace(' ', '') for i in list_flat]     #since there were spaces in some of the entries, spaces were removed to make it uniform 
        dict_values = dict((x,list_wo_space.count(x)) for x in set(list_wo_space))      #the set has only the unique values and it is counted and made to a dictionary 
        dataframe = pd.DataFrame.from_dict(dict_values, orient='index')         #the dictionary with the unique values and the number of occurences are made into a dataframe 
        dataframe = dataframe.transpose() 
        col_name = i.name
        dataframe['kinds'] = col_name
        df = pd.concat([df, dataframe], axis=0)
        
    df.to_csv(title + '.csv') #saving the dataframe as csv file 
        
       
    #creating the stacked bar chart
    fig = go.Figure()
    df = df.set_index('kinds')
    df = df[order_of_stacks]
    

    for i in df.columns:
        fig.add_trace(go.Bar(name=i, y=df.index,x=df[i],orientation='h', text=df[i]))

    fig.update_layout(barmode='stack')
    fig.update_layout(title =title, 
    title_x=0.5, height =500, title_y=0.9, font=dict(family='Helvetica', color="Black", size=16), legend=dict(title_font_family = 'Helvetica', font=dict(size=16, color="Black")))
    if order_of_axes == []:
        fig.update_layout(yaxis={'categoryorder':'total ascending'})
    else:
        fig.update_yaxes(categoryorder='array', categoryarray = order_of_axes)
    fig.to_image(title+'.svg')
    return fig.show()

    
##### Percentage stacked bar charts 
def percentage_stackedcharts(columns=[], title='', order_of_axes=[], order_of_stacks=[]):
    """ 
    Percentage_stacked charts  
    Parameters:
    columns: list of all the columns in a dataframe to be analyzed
    title: title to be given to the plot.Images get saved in this name 
    order_of_axes: list of index names as strings. It defines the order of the x axes. 
    order_of_stacks: list of the column names as strings. It defines the order of the stacks 
    """
    
    df = pd.DataFrame()
    for i in columns:
        dataframe = i.value_counts().to_frame() # counting the number of occurences of each unique values
        df = pd.concat([df,dataframe], axis=1)  # Appending to the values to the dataframe 
    
    df = df.reset_index()           #resetting the index of the dataframe such that it is easier for making graphs
    df = df.rename(columns={'index':'interest'}) 
    df = df.set_index('interest').transpose()
    df.columns = df.columns.str.replace(' ', '')
    df.columns
    

    #creating a dataframe for the percentage values
    per_df = pd.DataFrame()         #creating a separate dataframe for percentage values
    for col in df.columns:
        per_df['percentage '+col] = [((i/df.iloc[0, 0:4].sum())*100) for i in df[col]]  #calculating the percentage values 
        per_df['percentage '+col] = per_df['percentage '+col].round(decimals=1)
    per_df.index =df.index  #defining the index of the dataframe to be similar as the previous dataframe 
    
    
    
    #saving the dataframe to a csv file 
    df.to_csv(title +'.csv')


    #creating the chart
    fig = go.Figure()
    df = df[order_of_stacks]

    for i in per_df.columns:
        name = i.split()
        fig.add_trace(go.Bar(name=name[1],x=per_df.index, y=per_df[i], text=[f'{val}%' for val in per_df[i]]))
    
    fig.update_layout(width=800, height=700, barmode='stack')
    fig.update_yaxes(title='Percent')
    fig.update_layout(title=title, title_x = 0.5, font=dict(family='Helvetica', color="Black", size=16), legend=dict(title_font_family = 'Helvetica', font=dict(size=16, color="Black")))
    if order_of_axes == []:
        fig.update_layout(yaxis={'categoryorder':'total ascending'})
    else:
        fig.update_xaxes(categoryorder='array', categoryarray = order_of_axes)
    fig.write_image(title+'.svg')
    return fig.show()


    
##### Wordcloud
# defining a function to make a wordcloud 
# input - Specific columns of a dataframe with the question on the survey data form as the column name
def wordcloud(df,extra_stopwords=[]):
    """
    Makes a wordcloud based on the words in a given column
    Parameters:
    df: pd.Series. A dataframe column 
    extra_stopwords: list of words that needs to be removed from the wordcloud 
    """
    input = ''.join(df.str.lower().str.split().dropna(how='all').astype(str).str.replace(r'[-./?!,":;()\']',' ')) #splitting the strings and replacing the spaces or special characters 
    words_to_remove = df.name.split() 
    stopwords_new = words_to_remove + list(STOPWORDS) + extra_stopwords
    wc_image = WordCloud(stopwords=stopwords_new, background_color='white', width=600, height=600, random_state=4).generate(input)
    plt.imshow(wc_image)
    plt.tight_layout()
    plt.title(df.name)
    plt.axis('off')
    plt.savefig('Image-' +df.name+ '.png')

    
#### Sunburst chart
#sunburst charts - a dataframe has to be given as the input
#path - order in which the sunburst charts needs to be plotted 
# color_column - the column in a dataframe based on which the colors need to be decided
def sunburst(df, path=[], color_column='', custom_colors={}, title=''):
    """
    Makes a sunburst chart for comparing the work type, computational skill and level of comfort in developing new computational skill 
    df: pd.DataFrame. A dataframe with all the columns that needs to be compared
    path: a list of the dataframe column names. This defines the order of the sunburst charts. 
    color_column: Column name in a string format. This defines the color of the sunburst 
    custom_colors: A dictionary having the categories as keys and the colors used as the values.

    """
    fig = px.sunburst(df, path=path,color = color_column, color_discrete_map = custom_colors, values='counts', width=500, height=500)
    fig.update_traces(textinfo="label+percent parent", insidetextorientation = 'radial')
    fig.update_layout(title=title, title_x=0.5, font=dict(family='Helvetica', color="Black", size=16))
    fig.write_image(title+'.svg')