#!/usr/bin/env python
# coding: utf-8

# # Introduction

# As a project that aims to visually explore a dataset about board games, a relation will be established with regards to colours and typography in order to add the visual appeal of games without reducing the corporate seriousness that is required.
# 
# Overall, all plots have a light grey background that gives them a smoother visual, by reducing the high level of brightness of the default white colour while keeping a high contrast still. It also sets the plot in a frame other than a black line. The figure background is set to white when necessary in order to remove the transparency of the png files that can be generated from the Jupiter Notebook outputs; the png files can be easier observed and read without any transparency. The selection of colours for the plots themselves will be explained in detail according to each one.
# 
# In respect of typography, the Google font Play was the chosen one for Plotly plots. Play is a minimalistic sans-serif typeface that has a corporate, yet friendly appearance with high legibility and readability (Google Fonts, n.d.). Lucida Console was chosen for most of the Matplotlib, Seaborn, WordCloud and Altair titles and legends; and Lucida Sans Unicode, from the same family as Lucida Console, is being used for ticks with long texts as it has a tighter kerning. Lucida keeps some similarities with Play that will endorse a high level of unity across plots of different libraries. The available fonts for Matplotlib had to be installed beforehand.

# In[1]:


# Importing warnings and libraries

import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np
from collections import Counter

import matplotlib as mpl
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')

import seaborn as sns
sns.set(rc = {'axes.facecolor':'#f2f2f2', 'figure.facecolor':'#ffffff'})

import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from PIL import Image
from os import path
from wordcloud import WordCloud

import altair as alt
alt.data_transformers.disable_max_rows();


# In[2]:


# Installing fonts available for Matplotlib

#import matplotlib.font_manager
#from IPython.core.display import HTML

#def make_html(fontname):
#    return "<p>{font}: <span style='font-family:{font}; font-size: 24px;'>{font}</p>".format(font = fontname)

#code = '\n'.join([make_html(font) for font in sorted(set([f.name for f in matplotlib.font_manager.fontManager.ttflist]))])

#HTML("<div style='column-count: 2;'>{}</div>".format(code))


# # Data Preparation

# In[3]:


# Reading the dataset

missing_value_formats = ['n.a.', '?', 'NA', 'n/a', 'na', '--']
df = pd.read_csv('board_games.csv', na_values = missing_value_formats)


# In[4]:


# Discovering the shape of the dataset

df.shape


# In[5]:


# Obtaining basic information about the dataset

df.info()


# In[6]:


# Observing a sample of the dataframe

pd.set_option('display.max_columns', None)
df.head(10)


# **"Describe" insights**
# 
# Through the outputs for the function "describe", it is possible to extract a lot of useful insights:
# * The features "max_playtime" and "playing_time" have the same statistic summary and they are very close to the values for "min_playtime"; this will be analysed further on.
# * The count of unique values for the variables "description" and "name" differ from one another and both are less than the total of 10532 entries; this will be analysed further on.
# * While observing the lowest and highest values for "min_age", it is difficult to understand how reliable or accurate the observations for this variable are. For example, the highest value is 42 as the minimum age for the game "South African Railroads". On the website BoarGameGeek, this information is unavailable. By expanding the search on the web, an article about this game says:
# > Who can play? Age recommendation is 42+. That’s John Bohrer’s wry sense of humour, but then again, this is a somewhat demanding game. I’m sure my 11-year-old son would grasp the rules without issues, but I’m fairly sure he’d be pretty far from actually playing well. [2]
# * At least one game has the highest "min_playtime" of 60000 minutes. At first, this might look like a mistake that became an outlier (as the number is much higher than the third quartile one), but we can easily check that the information is actually correct. By searching the name of the game "The Campaign for North Africa" on the website BoardGameGeek[3], it is possible to confirm that the minimum time of 60000 minutes is correct.
# * Regarding the distribution of the "year_published", it can be seen that over the years, the number of games published increased exponentially, as the last 3 values (50%, 75% and max) are close to one another and much further from the min. This can be better observed through the boxplots that follow in the next section. A similar distribution happens for "average_rating" and the opposite happens with "users_rated", where the numbers lean towards the min and away from the max.
# * Finally, it is noticed that 558 entries for the variable "designer" are filled as "uncredited".
# 
# [2] https://mikkosgameblog.com/2017/11/south-african-railroads/
# 
# [3] https://boardgamegeek.com/boardgame/4815/campaign-north-africa-desert-war-1940-43

# In[7]:


# Obtaining a summary of the dataset basic statistics for the numerical columns

df.describe()


# In[8]:


# Obtaining a summary of the dataset basic statistics for the categorical columns

df.describe(include = 'object').T


# In[9]:


# Checking the game with highest "min_age"

df.sort_values(by = ['min_age'], ascending = False).head()


# In[10]:


# Checking the game with highest "min_playtime"

df.sort_values(by = ['min_playtime'], ascending = False).head()


# **Duplicates**
# 
# While checking for duplicated rows, there were no duplicates found. However, as seen in the previous section, the columns "max_playtime" and "playing_time" have the same statistic summary. By verifying the duplicates in the columns using the transpose property, "playing_time" displays the value True, which indicates that this feature is a duplicate of another one. This column is, then, compared with "max_playtime" with the function "equals", which confirms that they have the same values.
# 
# As seen previously, the feature "min_playtime" has values that are very similar to "max_playtime" and "playing_time". Comparing the two features, it is noticed that 8967 out of 10532 are exactly the same, which is 85% of similarity. This can mean that those values are not being correctly input into the source. Although the values for those features might not be reliable, there is no need to drop the columns of the dataset. It is important, however, to have this in mind if using one of them for visualization or even modelling.
# 
# Because "game_id" has a total of 10532 unique values, it would make sense that "description" and "name" had the same amount of unique values if each row was a distinct board game. However, "description" has 10528 unique values and "name" has 10357, which means that both features have some duplicated observations. By taking a look at some of these repeated values, it is possible to infer that:
# * Repeated values for "description" happen when there is more than one version of the same game.
# * Repeated values for "name" can happen for distinct games with the same name but different mechanics or updated versions of the same game, for example.
# 
# Therefore, it is possible to infer that all rows indeed represent unique games.

# In[11]:


# Ckecking for duplicates in rows

df.duplicated().sum()
print('There are a total of ' + (str(df.duplicated().sum()) + ' duplicates in the dataset.'))


# In[12]:


# Checking for duplicates in columns

df.T.duplicated()


# In[13]:


# Checking if columns "max_playtime" and "playing_time" are duplicates

df['max_playtime'].equals(df['playing_time'])


# In[14]:


# Checking when "max_playtime" is the same as "min_playtime"

df_time = df[df['max_playtime'] == df['min_playtime']]
df_time.shape


# In[15]:


# Checking the total of unique values in the columns "game_id", "description" and "name"

df['game_id'].nunique()
print('There are a total of ' + (str(df['game_id'].nunique()) + ' unique values in the "game_id" variable.'))

df['description'].nunique()
print('There are a total of ' + (str(df['description'].nunique()) + ' unique values in the "description" variable.'))

df['name'].nunique()
print('There are a total of ' + (str(df['name'].nunique()) + ' unique values in the "name" variable.'))


# In[16]:


# Observing the duplicated values in "description"

desc = df['description']
df[desc.isin(desc[desc.duplicated()])]


# In[17]:


# Observing the duplicated values in "name"

name = df['name']
df[name.isin(name[name.duplicated()])].sort_values('name', ascending = True)


# **Distribution and outliers**
# 
# Apart from observing the distribution of the variables through the function "describe", a group of boxplots is being used to display the distribution of the previously mentioned features "year_published", "average_rating" and "users_rated". Through this specific type of plot and those 3 examples, it is easier to understand that most of this dataset has a skewed distribution with outliers.
# 
# The boxplots were produced with the library Plotly to make some level of interaction possible. Through the hover tool, details of the distribution can be closely observed such as the values for min, max, mean etc. Zoom is another possible interaction, that can be more useful for the "user_rated" plot, which is the most skewed of the three. The legend was deactivated as they are only repeating the X label, thus they are not necessary.
# 
# Before the production, a custom template that will serve for all Plotly graphs was created. The custom template sets fonts and sizes for titles and labels as well as plot and paper (figure) background colour as discussed previously.
# 
# As the plot contains three boxes, the primary colours were selected. In gaming, colours are used to group or separate elements, like differentiating players, for example. For this reason, they need to be easily distinguishable among themselves and among the overall scenario or background (Interama Games, 2016). In this case, the primary colours work perfectly to make the three plots as distinct as possible while creating this link with the gaming atmosphere.
# 
# While some of the elements are placed to create unity, the variety of colours explicts that the features presented are well distinct from one another as are the primary colours. Variety in design is used to create visual interest and avoid monotony, improving the user's visual experience. One of the many ways of implementing variety is through colours (Chapman, 2019).
# 
# The outliers will not be treated. As previously mentioned, they do not seem to represent an imputation error. If they were removed or replaced, the graphs could mislead decision-making that would be based on a distorted visualization of the facts. They should be treated accordingly, though, when any machine learning model requires so.

# In[18]:


# Creating a custom template for plotly

custom_template = {'layout':
                   go.Layout(
                       font = {'family': 'Play, monospace',
                               'size': 12,
                               'color': '#707070'},
                       
                       title = {'font': {'family': 'Play, monospace',
                                         'size': 18,
                                          'color': '#1f1f1f'}},
                       
                       legend = {'font': {'family': 'Play, monospace',
                                          'size': 12,
                                          'color': '#1f1f1f'}},
                       
                       plot_bgcolor = '#f2f2f2',
                       paper_bgcolor = '#ffffff'
                   )}


# In[19]:


# Checking the distribution and outliers of the features 'year_published', 'average_rating', 'users_rated'

fig = make_subplots(rows = 1, cols = 3)

fig.add_trace(go.Box(y = df['year_published'], name = 'Year Published',
                    marker_color = 'red', showlegend = False), row = 1, col = 1)

fig.add_trace(go.Box(y = df['average_rating'], name = 'Average Rating',
                    marker_color ='green', showlegend = False), row = 1, col = 2)

fig.add_trace(go.Box(y = df['users_rated'], name = 'Users Rated',
                    marker_color = 'blue', showlegend = False), row = 1, col = 3)

fig.update_layout(height = 800, width = 800,
                  title_text = 'Distribution of Year Published, Average Rating and Users Rated', template = custom_template)

fig.update_layout(title = {
    'y':0.95,
    'x':0.5})

fig.show()


# **Missing values**
# 
# To better understand the proportions of missing values per column, a bar chart was plotted. As this is a visualization that does not require any type of interaction, Matplotlib was used to plot it.
# 
# The bars in this plot are red, as this colour is usually associated with warnings and danger in most of the occidental cultures. In games, red is also usually associated with enemies or the state of health and helps to focus on what is really important within the image (Gil, 2018). Thus, this colour is bringing the reader's attention to the importance of correcting any issues with the data source feeding instead of leaving the observations null so the dataset is more useful for gathering accurate insights and even modelling.
# 
# Apart from the general tweaks mentioned previously, in order to enhance the readability of the information, labels were added to the edge of the bars with the percentage value rounded.
# 
# No missing values will be dropped, in order to keep the most information from the dataset as possible. Thus, as all missing values are categorical, they were filled with "Unknown".

# In[20]:


# Checking if there are any missing values

df.isna().values.any()


# In[21]:


# Checking the total of missing values

print('There are a total of ' + (str(df.isna().sum().sum()) + ' missing values in the dataset.'))


# In[22]:


# Checking the total of missing values per column

df.isna().sum().sort_values(ascending = False)


# In[23]:


# Plotting the percentage of missing values per feature

df_na = df.isna().sum().div(df.shape[0]).mul(100).to_frame().sort_values(by = 0, ascending = False)
df_na = df_na[df_na[0] > 0]

fig, na = plt.subplots(figsize = (20, 10))

plot_na = na.bar(df_na.index, df_na.values.T[0], color = 'red')

na.bar_label(plot_na, label_type = 'edge', fmt = '%.2f', padding = 5)

plt.title('Percentage of missing values per feature', fontname = 'Lucida Console', fontsize = 18, pad = 20)
plt.xlabel('Features', fontname = 'Lucida Console', fontsize = 12)
plt.ylabel('Percentage of missing values',  fontname = 'Lucida Console', fontsize = 12)
plt.xticks(fontname = 'Lucida Sans Unicode', fontsize = 10)
plt.yticks(fontname = 'Lucida Sans Unicode', fontsize = 10);


# In[24]:


# Replacing the NaNs with "Unknown"

df2 = df.fillna('Unknown')
df2.head()


# In[25]:


# Checking if there are any missing values

df2.isna().values.any()


# ## Data Visualization

# ### 1. What are the top 5 "average rated" games?

# This visualization requires a fairly commom type of plot that was previously used (bars) and does not require interactivity. The library chosen in this case was Seaborn, that offers more customization options and can use the support of Matplotlib for adjusting details such as title, labels etc. The function "catplot" is being used instead of the usual "barplot". "This function provides access to several axes-level functions that show the relationship between a numerical and one or more categorical variables using one of several visual representations" (Waskom, 2021). The parameter "kind" accepts 8 types of plots, and "bar" is used in this case.
# 
# The colours of the bars used are assuming a colour-coded rank. There is not a single colour-coded rank for games, but many of them follow the a similar order, where orange (or gold) and purple are the most rare or precious items, sometimes called legendary and epic; blue and green are used for rare and uncommon; and grey and white are used for common itens. As white would not present a readability, the fifth colour for the graph is the grey. This specific order was based in the game World of Warcraft 2004 (Memmott, 2021).
# 
# Another tweak added to this graph is the limit of the y axis that was set to go from 8.5 to 9. As all games are rated within this range, this limit acts like a zoom in the bars, making the difference between the average ratings more noticeable.

# In[26]:


# Sorting the data per average rating and assigning the result to a new dataframe "top_rating"

top_rating = df2.sort_values(by = 'average_rating', ascending = False)
top_rating.head()


# In[27]:


# Plotting top 5 average rated games

sns.catplot(data = top_rating.head(), x = 'name', y = 'average_rating', kind = 'bar', height = 5, aspect = 3,
                            palette = ['orange', 'purple', 'blue', 'green', 'grey']).set(ylim = (8.5, 9))

plt.title('Top 5 average rated games', fontname = 'Lucida Console', fontsize = 18, pad = 20)
plt.xlabel('Games', fontname = 'Lucida Console', fontsize = 12)
plt.ylabel('Average rating', fontname = 'Lucida Console', fontsize = 12)
plt.xticks(fontname = 'Lucida Sans Unicode', fontsize = 10)
plt.yticks(fontname = 'Lucida Sans Unicode', fontsize = 10);


# ### 2. Is there a correlation between the “users_rated” and the “max_playtime”?

# This graph contains two different elements: dots and line. Regarding the colours, blue and orange were chosen for representing the elements of the graph as those are complementary at the colour wheel. Various competitive games make use of two factions that compete against each other and this dichotomy demands a clear visual communication that is usually represented with two clearly different colours (Interama Games, 2016).
# 
# The plot do not show much initially. As "max_playtime" scale is impacted by some outliers, the dots are overlapping a lot and the line looks completely horizontal, showing no upwards or downwards trend. A second graph is plotted, then, including a log scale along the axis y, which improves the visualization as the dots are overlapping less than before. However, the downward trend of the line is still minimal, which implies a very weak negative linear correlation.
# 
# As graphs are better supported with values, this can be confirmed by plotting a Seaborn "heatmap" with the correlation coefficient produced with the Numpy function "corrcoef". The result, as shown, is -0.004, which corroborates that the correlation between "users_rated" and "max_playtime" is so extremely weak that it can be said inexistent. 
# 
# The heatmap is displayed slightly differently than the other plots:
# * As there is not a lot of information to be shown, its size was reduced to (7, 7);
# * There is no need for x and y labels as only two variables are being compared with each other (where there is the value 1, it means the variable is being compared with itself);
# * Orange was chosen for the colour map, where the more faded the colour, the weaker the correlation.

# In[28]:


# Plotting a scatterplot to check the correlation between “users_rated” and “max_playtime”

fig = plt.gcf()
fig.set_size_inches(20, 10)

sns.regplot(data = df2, x = 'users_rated', y = 'max_playtime',
            scatter_kws = {'color': 'blue'}, line_kws = {'color': 'orange'})

plt.title('Correlation between "users_rated" and "max_playtime"', fontname = 'Lucida Console', fontsize = 18, pad = 20)
plt.xlabel('users_rated', fontname = 'Lucida Console', fontsize = 12)
plt.ylabel('max_playtime', fontname = 'Lucida Console', fontsize = 12)
plt.xticks(fontname = 'Lucida Sans Unicode', fontsize = 10)
plt.yticks(fontname = 'Lucida Sans Unicode', fontsize = 10);


# In[29]:


# Plotting a scatterplot to check the correlation between “users_rated” and “max_playtime” with log scale

fig = plt.gcf()
fig.set_size_inches(20, 10)

sns.regplot(data = df2, x = 'users_rated', y = 'max_playtime', ci = None,
            scatter_kws = {'color': 'blue'}, line_kws = {'color': 'orange'}).set_yscale('log')

plt.title('Correlation between "users_rated" and "max_playtime" with log scale',
          fontname = 'Lucida Console', fontsize = 18, pad = 20)
plt.xlabel('users_rated', fontname = 'Lucida Console', fontsize = 12)
plt.ylabel('max_playtime', fontname = 'Lucida Console', fontsize = 12)
plt.xticks(fontname = 'Lucida Sans Unicode', fontsize = 10)
plt.yticks(fontname = 'Lucida Sans Unicode', fontsize = 10);


# In[30]:


# Plotting a heatmap to check the correlation between “users_rated” and “max_playtime”

fig = plt.gcf()
fig.set_size_inches(7, 7)

sns.heatmap(np.corrcoef(df2['users_rated'], df2['max_playtime']), annot = True, cmap = 'Oranges')

plt.title('Correlation matrix for "users_rated" and "max_playtime"',
          fontname = 'Lucida Console', fontsize = 18, pad = 20)
plt.xticks(fontname = 'Lucida Sans Unicode', fontsize = 10)
plt.yticks(fontname = 'Lucida Sans Unicode', fontsize = 10);


# ### 3. What is the distribution of game categories?

# In this dataset, the column "category" has initially 3861 unique values. This can be explained by the fact that each game is associated with more than one category in the same observation, creating, then, multiple combinations of categories. In order to analyse individual categories, the functions "str.split" and "explode" are applied to the initial list and all the observations are split, where there is a comma, and exploded in new rows. After this step, 84 unique categories are identified with their respective count. Althout 84 is still a large number, it is much smaller than 3861.
# 
# As the list of unique categories has 84 items, this would be a lot of information for a pie chart. Hence, in this case, Seaborn "countplot" is being used to display, through bars, the counts of observations for each categorical bin. Once more, blue and orange were chosen as the colours of the bars. In this scenario, the dichotomy of the colours separate the most frequent categories from the less frequent ones. The "xticks" labels were rotated in 90 degrees to make it possible the display of all of them. From the bar chart plotted, it is possible to see that the most frequent category is "Card Game".
# 
# One way to plot a normalized distribution of the proportions of the categories is by applying the parameter "normalize = True" in the value counts list, which will transform the absolute count into a proportion. Then, Seaborn "kdeplot" is used to plot the distribution of the proportions, with the parameter "cut" set to zero, as no count can be negative. "Card Game" will be, for example, at the very end of the tail, as its frequency is 2981 out of the total of 27514 categories (over 10% or 0.10 as seen in the x-axis of the graph). In this case, it is possible to observe a highly skewed distribution.
# 
# Finally, another interesting way of displaying the frequency of the categories is through a word cloud. Although it is a less corporate format, the word cloud presented in the shape of the most frequent category makes it very quick to understand its importance for the industry. The "unique_cat_list" from the previous distribution graphs was exported to a .csv file in order to use it with the "WordCloud" library. The word cloud is generated with the background set to white colour, maximum of words to match the number of unique categories, a contour of blue colour and a colourful colormap called "prism", to facilitate the distinction between the categories. Also, an image of cards was imported and used as a mask, so the word cloud would be inside it.

# In[31]:


# Checking the total of unique values in the column "category"

df2['category'].nunique()
print('There are a total of ' + (str(df2['category'].nunique()) + ' unique values in the category variable.'))


# In[32]:


# Checking the count of each value in the column "category"

df2['category'].value_counts()


# In[33]:


# Splitting the values by the comma delimiter and exploding the individual values in a list of individual categories

unique_cat_list = df2['category'].str.split(',').explode('category')
unique_cat_list.value_counts()


# In[34]:


# Transforming the list of individual categories in a dataframe

unique_cat_list = unique_cat_list.to_frame()
unique_cat_list


# In[35]:


# Plotting the count distribution of the variable "category"

fig = plt.gcf()
fig.set_size_inches(20, 10)

sns.countplot(x = unique_cat_list['category'], order = unique_cat_list['category'].value_counts().index,
             palette = 'blend:#FF5733,#0000FF')

plt.title('Distribution of game categories', fontname = 'Lucida Console', fontsize = 18, pad = 20)
plt.xlabel('Category', fontname = 'Lucida Console', fontsize = 12)
plt.ylabel('Count', fontname = 'Lucida Console', fontsize = 12)
plt.xticks(rotation = 90, fontname = 'Lucida Sans Unicode', fontsize = 10)
plt.yticks(fontname = 'Lucida Sans Unicode', fontsize = 10);


# In[36]:


# Plotting the normalized distribution of the variable "category"

fig = plt.gcf()
fig.set_size_inches(20, 10)

sns.kdeplot(data = unique_cat_list.value_counts(normalize = True), cut = 0)

plt.title('Distribution of game categories', fontname = 'Lucida Console', fontsize = 18, pad = 20)
plt.xlabel('Category', fontname = 'Lucida Console', fontsize = 12)
plt.ylabel('Count', fontname = 'Lucida Console', fontsize = 12)
plt.xticks(rotation = 90, fontname = 'Lucida Sans Unicode', fontsize = 10)
plt.yticks(fontname = 'Lucida Sans Unicode', fontsize = 10);


# In[37]:


# Plotting the frequency of the variable "category"

d = path.dirname('unique_cat_list.csv')
text = open(path.join(d, 'unique_cat_list.csv')).read()

card_mask = np.array(Image.open(path.join(d, 'card-games.png')))

unique_cat_list_wc = WordCloud(background_color = 'white', max_words = 84, mask = card_mask,
                               contour_width = 3, contour_color = 'steelblue',
                               colormap = 'prism')

unique_cat_list_wc.generate(text)
unique_cat_list_wc.to_file(path.join(d, 'unique_cat_list_wc.png'))

plt.figure (figsize = (20, 10))
plt.title('Frequency of categories', fontname = 'Lucida Console', fontsize = 18, pad = 20)
plt.imshow(unique_cat_list_wc, interpolation = 'bilinear')
plt.axis('off');


# ### 4. Do older games (1992 and earlier) have a higher MEAN “average rating” than newer games (after 1992)?

# To analyse and compare the mean average rating throughout the years, a new dataframe "df_year_avgrate" was created with the variables needed ("year_published" and "average_rating") and sorted by year in ascending order. This new dataframe is then grouped by year while aggregating the average rate by the mean, generating a unique mean rating per year.
# 
# The "df_year_avgrate" generates, then, two new datasets: "df_year_avgrate_mean_older" which contains all the observations prior to 1993 and "df_year_avgrate_mean_newer" which keeps all the observations from 1993 onwards.
# 
# Again, Seaborn was the chosen library to plot a timeline of the mean average rating over the year. The final plot is a combination of two line plots for the "df_year_avgrate_mean_older" and "df_year_avgrate_mean_newer". And, once more, blue and orange were the chosen colours to show the contrast between older and newer games while keeping the visual unity with other plots.
# 
# With those two lines alone, it is already possible to visualize that older games have, instead, a lower mean average rating. However, to make it even quicker to obtain this information when visualizing the graph, two horizontal lines (axhline) were added with the overall mean for each period.
# 
# As per the tweaks, the line graph for the average rating of older games is thinner than the one for the newer games; the overall mean lines also differ in width and in style, while the line for older games is dashed, the line for newer games is composed by dashes and dots. Those changes would make it possible to read the graph even if it was printed in greyscale. Additionally, the "ci" was set to none in both graph lines as the confidence interval is not a parameter needed, and the legend for the horizontal lines was left at the top left corner, which is the most important place for information in the occidental culture.
# 
# The graph itself, the contrast of the colours and the position of the legend create an asymmetrical balance for the whole figure.

# In[38]:


# Creating a new dataframe with the columns "year_published" and "average_rating", sorted by year

df_year_avgrate = df2[['year_published', 'average_rating']].sort_values(by = 'year_published', ascending = True)
df_year_avgrate


# In[39]:


# Obtaining the average rating per year

df_year_avgrate_mean = df_year_avgrate.groupby('year_published', as_index = False).agg({'average_rating': 'mean'})
df_year_avgrate_mean


# In[40]:


# Obtaining the average rating for games published before 1993 and from this year on

df_year_avgrate_mean_older = df_year_avgrate[(df_year_avgrate['year_published'] <= 1992)]
df_year_avgrate_mean_newer = df_year_avgrate[1992 < (df_year_avgrate['year_published'])]


# In[41]:


# Plotting the average rating per year: oder games vs newer games

fig = plt.gcf()
fig.set_size_inches(20, 10)

ax = sns.lineplot(data = df_year_avgrate_mean_older, x = 'year_published', y = 'average_rating', color = 'blue', ci = None,
                 linewidth = 1)
ax.axhline(y = df_year_avgrate_mean_older['average_rating'].mean(), color = 'blue', ls = '--', lw = 1, xmax = 0.62,
           label = 'Average rating for older games')

ax = sns.lineplot(data = df_year_avgrate_mean_newer, x = 'year_published', y = 'average_rating', color = 'orange', ci = None,
                 linewidth = 3)
ax.axhline(df_year_avgrate_mean_newer['average_rating'].mean(), color = 'orange', ls = ':', lw = 2, xmin = 0.63,
           label = 'Average rating for newer games')

plt.title('Average rating per year', fontname = 'Lucida Console', fontsize = 18, pad = 20)
plt.xlabel('Year published', fontname = 'Lucida Console', fontsize = 12)
plt.ylabel('Average rating', fontname = 'Lucida Console', fontsize = 12)
plt.xticks(fontname = 'Lucida Sans Unicode', fontsize = 10)
plt.yticks(fontname = 'Lucida Sans Unicode', fontsize = 10)
plt.legend();


# In[42]:


#Printing the overall average rating for older and newer games

print('The average rating for older games is ' + (str(round(
    df_year_avgrate['average_rating'][(df_year_avgrate['year_published'] <= 1992)].mean(), 2)) + '.'))

print('The average rating for newer games is ' + (str(round(
    df_year_avgrate['average_rating'][1992 < (df_year_avgrate['year_published'])].mean(), 2)) + '.'))


# ### 5. What are the 3 most common “mechanics” in the dataset?

# This question follows a very similar approach to the first one (What are the top 5 "average rated" games?). Thus, to avoid repetition, a quicker overall of the steps will follow.
# 
# Regarding feature engineering, after splitting the observations by the commas and exploding the values, the list of unique mechanics is reduced from 3210 to 52 unique values. Then, those values were sorted in descending order, from the higher to the lower counts. From there, a new dataframe "unique_mec_list_top3" is created with the 3 larger counts, meaning the mechanics that are more frequent or more common among the board games.
# 
# Seaborn "catplot" is used to plot bars, but, in this case, only the first three colours are being used for rank: orange, purple and blue.

# In[43]:


# Checking the total of unique values in the column "category"

df2['mechanic'].nunique()
print('There are a total of ' + (str(df2['mechanic'].nunique()) + ' unique values in the category variable.'))


# In[44]:


# Checking the count of each value in the column "category"

df2['mechanic'].value_counts()


# In[45]:


# Splitting the values by the comma delimiter and exploding the individual values in a list of individual categories

unique_mec_list = df2['mechanic'].str.split(',').explode('mechanic')
unique_mec_list.value_counts()


# In[46]:


#Checking the total of unique mechanics

len(unique_mec_list.value_counts())


# In[47]:


# Creating a new dataframe with the count of the 3 most common mechanics

unique_mec_list_top3 = unique_mec_list.value_counts().nlargest(3)
unique_mec_list_top3 = unique_mec_list_top3.to_frame()
unique_mec_list_top3.rename(columns = {'mechanic': 'count'}, inplace = True)
unique_mec_list_top3['mechanic'] = unique_mec_list_top3.index
unique_mec_list_top3.reset_index(drop = True, inplace = True)
unique_mec_list_top3


# In[48]:


# Plotting the 3 most common mechanics

top3_mec = sns.catplot(data = unique_mec_list_top3, x = 'mechanic', y = 'count', kind = 'bar', height = 5, aspect = 3,
                      palette = ['orange', 'purple', 'blue'])

plt.title('3 most common mechanics', fontname = 'Lucida Console', fontsize = 18, pad = 20)
plt.xlabel('Mechanic', fontname = 'Lucida Console', fontsize = 12)
plt.ylabel('Count', fontname = 'Lucida Console', fontsize = 12)
plt.xticks(fontname = 'Lucida Sans Unicode', fontsize = 10)
plt.yticks(fontname = 'Lucida Sans Unicode', fontsize = 10);


# ### 6. What is the cumulative growth rate of the market in 2016, in terms of the number of games published?

# In 1950, the first year of the dataset, the total amount of games published was 4, that would be where y = 0 in the graph, representing the initial point from where the percentage change happens every year by increasing or decreasing the number of games published. The cumulative growth rate reaches 1037% (10.37) in 2016, after a 17% decrease in regards to the previous year.
# 
# To find these values, the original dataset "df" was used. In this original dataset, all values for "game_id" are unique, which means each line represents a unique game. So, the variable "count" was created and the number 1 is assigned to each row to represent the count for each unique game. After this, the two columns needed for this graph ("year_published" and "count") were assigned to a new dataframe called "df_games_year", which was sorted in ascending order by year and then grouped by year with the sum of games per year. Finally, the column "pct_change" was created in this new dataframe, by applying the homonym pandas function to the column "count", which calculates the percentage of change between the current and a prior element.
# 
# A line graph was, then, plotted with the cumulative percentage change throughout the years. This time, Plotly was used to allow some interaction. By moving the mouse cursor along the line, it is possible to see the specific percentage change for every year from 1951 to 2016. Since 1950 is the initial year, there is no prior value to which it can be compared.
# 
# Green is commonly related to health, healing or teammates in gaming (DVNC TECH LLC, 2018). In occidental cultures, green is usually associated with positive situations such as permission (green lights) and completion (as in done tasks). As this is a positive information of a growing market, the green colour was attributed to the line.

# In[49]:


# Creating a dataset with the count of games published and the percentage change of this count per year

df['count'] = 1
df_games_year = df[['year_published', 'count']].sort_values(
    by = 'year_published', ascending = True).groupby(by = ['year_published'], as_index = False).sum()

df_games_year['pct_change'] = df_games_year['count'].pct_change()
df_games_year


# In[50]:


# Plotting the cumulative growth rate of games published per year

fig = px.line(df_games_year, x = 'year_published', y = df_games_year['pct_change'].cumsum(),
              title = 'Cumulative growth rate of games published per year')

fig.update_layout(title = {
        'y':0.95,
        'x':0.5,
        'xanchor': 'center'})

fig.update_layout(xaxis_title = 'Year',
                  yaxis_title = 'Cumulative percentage of growth',
                  showlegend = False, template = custom_template,
                  width = 800)

fig.data[0].line.color = 'rgb(0, 204, 0)'

fig.show()


# ### 7. Display the average rating of the categories of games published in the last 10 year and gather some insights from the graph.

# The feature engineering applied in this question involves the previously mentioned function "str.split" and "explode", which, this time, are being applied in both "category" and "mechanic" variables. This exploded dataset is, then, sorted by "year_published" and a new dataframe is created "df_10y", where the year is larger than 2006.
# 
# Altair is the library used for this last graph. Although it was created especially for interactive plots, Altair has the powerful ability to work with categorical data and can easily bin continous data. Once more, blue and orange creates the dichotomy that makes it easy to identify the lowest and highest "average_ratings".
# 
# Some of the insights that can be extracted from this plot:
# * The top 5 categories of 2007 are: Industry / Manufacturing, Napoleonic, Negotiation, Pre-Napoleonic, Print & Play;
# * During 2008, 2009 and 2010, no category was rated above 8.
# * 4 categories were between the top rated in 2015 and 2016: American Revolutionary War, Fighting (after a bad performance in 2014), Vietnam War and Wargame (also after a bad performance in 2014).

# In[51]:


# Exploding the column "category" in the dataset

tag_cols = df2[['category']].columns
df2[tag_cols] = df2[tag_cols].apply(lambda col: col.str.split(','))

for col in tag_cols:
    df2 = df2.explode(col, ignore_index = True)
    
df2.head()


# In[52]:


# Displaying the shape of the exploded dataset

df2.shape


# In[53]:


# Creating a new dataset for the last 10 years

df_10y = df2.sort_values(by = 'year_published', ascending = True)
df_10y = df2[df2['year_published'] > 2006]
df_10y


# In[54]:


# Plotting a heatmap indicating the average rating across categories over the last 10 years

chart = alt.Chart(df_10y).mark_rect().encode(
    alt.X('year_published:Q', bin = True),
    alt.Y('category:N'),
    alt.Color('average_rating', bin = alt.Bin(maxbins = 10),
        scale = alt.Scale(scheme = 'blueorange'),
        legend = alt.Legend(title = 'Rating'))
    ).properties(width = 500, height = 800, title = 'Average rating of categories over the last 10 years')

chart.configure_title(
    font = 'Lucida Console',
    fontSize = 18,
    align = 'center')


# # Conclusion

# The market of Board Games has been growing in the past years and the internet plays an important role in this. In this report, Data Visualization techniques were applied in order to gather and analyze important insights that would help increase sales of a board game retail company.
# 
# Data Visualization is an important tool for communicating insights if used effectively. In order to make the plots more visually attractive, legible, readable and adequate for the business, fonts, colours and other aspects of each graph were changed and discussed.
# 
# The plots help the business to understand the top-rated games of all time, the most common categories and mechanics, the increase in the average rating of the newer games in comparison to the older ones, the exponential growth in the number of games published as well as the categories with the lowest and highest ratings over the last ten years. With those insights, decisions can be made in regard to sales strategy for the next year.
