### Requirements
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from wordcloud import WordCloud, STOPWORDS
import matplotlib.pyplot as plt
import re 
import kaleido
from sklearn.feature_extraction.text import CountVectorizer

#### Templates for the graphs
def barchart_vertical(series, title='', order_of_axes=[]):
    """
    Creates a vertical bar chart. 
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
    df = df.rename(columns={0:'counts'})
    
    
    #plotting a graph 
    fig = px.bar(x=df.index, y=df['counts'], labels ={'x':'', 'y':'counts'}, text_auto=True, orientation='v')                                              
    fig.update_layout(width=500, height=500, title=title, title_x=0.5, title_y=0.95, font=dict(family='Helvetica', color="Black", size=16), legend=dict(title_font_family = 'Helvetica', font=dict(size=16, color="Black")))
    if order_of_axes == []:
        fig.update_xaxes(categoryorder = 'total descending')
    else:
        fig.update_xaxes(categoryorder='array', categoryarray = order_of_axes)
    
    #saving the figure as svg file 
    fig.write_image(title+'.svg')

    return fig.show()
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
    df = df.rename(columns={0:'counts'})
    
    
    #plotting a graph 
    fig = px.bar(y=df.index, x=df['counts'], labels ={'x':'counts', 'y':''}, text_auto=True, orientation='h')                                              
    fig.update_layout(width=700, height=600, title=title, title_x=0.5, title_y=0.95, font=dict(family='Helvetica', color="Black", size=16), legend=dict(title_font_family = 'Helvetica', font=dict(size=16, color="Black")))
    if order_of_axes == []:
        fig.update_yaxes(categoryorder = 'total ascending')        
    else:
        fig.update_yaxes(categoryorder='array', categoryarray = order_of_axes)

    #saving the figure as svg file 
    fig.write_image(title+'.svg')

    return fig.show()
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
    fig = go.Figure()
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
    return fig.show()

    
##### Percentage stacked bar charts 
def percentage_stackedcharts(data, title='', order_of_axes=[], order_of_stacks=[]):
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
        fig.add_trace(go.Bar(name=i,x=per_df.index, y=per_df[i], text=[f'{val}%' for val in per_df[i]]))
    
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
##### Wordcloud
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
    plt.title(series.name)
    plt.axis('off')
    plt.savefig('Image-' +series.name+ '.svg')
#### Sunburst chart
#### Sunburst chart
def sunburst(df, order_list, color_column='', custom_colors={}, title=''):
    """
    Makes a sunburst chart for comparing the work type, computational skill and level of comfort in developing new computational skill 
    df: pd.DataFrame. A dataframe with all the columns that needs to be compared
    order_list: a list of the dataframe column names. This defines the order of the sunburst charts. The given dataframe will be grouped based on the last column name provided in this list. 
    color_column: Column name in a string format. This defines the color of the sunburst 
    custom_colors: A dictionary having the categories as keys and the colors used as the values.

    """
    #Creating the dataframe 
    df = df.groupby(order_list[-1]).value_counts().reset_index()  #creating the counts based on grouping the column that was last in the 'order_list'
    df = df.rename(columns={0:'counts'})
    
    #creating the sunburst chart
    fig = px.sunburst(df, path=order_list,color = color_column, color_discrete_map = custom_colors, values='counts', width=500, height=500)
    fig.update_traces(textinfo="label+percent parent", insidetextorientation = 'radial')
    fig.update_layout(title=title, title_x=0.5, font=dict(family='Helvetica', color="Black", size=16))
    fig.write_image(title+'.svg')
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
##### Filtering the CSV of ngrams for the conferences/topics of interest
def ngram_counts(df1,df2,second_df=False,title='', count_list_uni=[], count_list_bi=[]):
        """
        Creates a barchart checking the ngram csvs and the list of strings provided
        Parameters
        df1: unigram csv
        df2: bigram csv
        second_df: to be set to false if only unigram csv is provided
        title: csv filename 
        count_list_uni: list of unigram strings that has to be checked in the unigram csv
        count_list_bi: list of bigram strings that has to be checked in the bigram csv
        """
        #the following runs only if the second_df is set to True and df2 is provided along with the 'count_list_bi' 
        if second_df:
            df_counts=pd.DataFrame()
            for i in count_list_uni:
                df_uni = df1[df1['ngram'].str.contains(i)]
                df_counts = pd.concat([df_counts, df_uni], axis=0)
                        
            for i in count_list_bi:
                df_bi = df2[df2['ngram'].str.contains(i)]
                df_counts = pd.concat([df_counts, df_bi], axis=0)
            
        else:
            df_counts=pd.DataFrame()
            for i in count_list_uni:
                df_uni = df1[df1['ngram'].str.contains(i)]
                df_counts = pd.concat([df_counts, df_uni], axis=0)
        
        fig = go.Figure()

        fig = px.bar(x=df_counts['ngram'], y=df_counts['frequency'],labels={'x':'', 'y':'counts'}, text_auto=True)
        fig.update_layout(title=title, title_x=0.5, width = 500, height=500, showlegend=False, font=dict(family='Helvetica', color="Black", size=16))
        fig.update_xaxes(categoryorder='total descending', tickangle=45)
        
        fig.write_image(title+'.svg')

        return fig.show()