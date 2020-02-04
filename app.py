from collections import OrderedDict
from dash.dependencies import Input, Output
from dash_table.Format import Format, Scheme, Sign, Symbol
from plotly.graph_objs import *
from scipy import stats
from sklearn.model_selection import train_test_split

import base64
import dash
import dash_core_components as dcc
import dash_html_components as html
import dash_table
import dash_table.FormatTemplate as FormatTemplate
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go


#--------- Pandas Dataframe
url = "https://raw.githubusercontent.com/SulmanK/PUBG-EDA-Dashboard-Univariate-App/master/data/PUBG_Player_Statistics.csv"
orig = pd.read_csv(url, nrows = 60000)

## Create a copy of the dataframe
df = orig.copy()
cols = np.arange(52, 152, 1)

## Drop columns after the 52nd index
df.drop(df.columns[cols], axis = 1, inplace = True)

## Drop player_name and tracker id
df.drop(df.columns[[0, 1]], axis = 1, inplace = True)

## Drop Knockout and Revives
df.drop(df.columns[[49]], axis = 1, inplace = True)
df.drop(columns = ['solo_Revives'], inplace = True)

## Drop the string solo from all strings
df.rename(columns = lambda x: x.lstrip('solo_').rstrip(''), inplace = True)

## Combine a few columns 
df['TotalDistance'] = df['WalkDistance'] + df['RideDistance']
df['AvgTotalDistance'] = df['AvgWalkDistance'] + df['AvgRideDistance']

# Create train and test set using Sci-Kit Learn
train, test = train_test_split(df, test_size = 0.1)
dev, test = train_test_split(test, test_size = 0.5)
df = train
data = df


#--------- Dashboard
## Importing Logo and encoding it
image_filename = 'assets/PUBG_logo.png' 
encoded_image = base64.b64encode(
    open(image_filename, 'rb').read())


## Bivariate Plots


def ScatterBoxPlot(data, discretized, cut, bins, labels, x, y):
    ##Creates a seaborn scatter and boxplot by discretizing a column to change into an interval (category)
    ## Use 
    # 1) Discretize a feature: convert it from numerical to categorical
    # 2) Plot a scatter plot
    # 3) Plot a box plot
    ## Function parameters:
    # data == dataframe
    # discretized == new feature (category)
    # cut == feature that is being
    # bins == the intervals
    # labels == string representations of the bins
    # x = x_axis of scatterplot
    # y = y_axis of scatterplot    
    
    # Discretize the data
    data[discretized] = pd.cut(data[cut], bins = bins, labels = labels)
    
    # Give a numerical label to the category
    c = data[discretized].astype('category')
    d = dict(enumerate(c.cat.categories))
    box_code = discretized + '_' + 'code'
    level_back = discretized + '_' + 'interval'
    data[box_code] = data[discretized].astype('category').cat.codes
    data[box_code] = data[box_code].replace(-1,0)
    data[level_back] = data[box_code].map(d)
    category_names = x + "_" + 'Interval'
    data[category_names] = data[level_back]      # we need this to color our boxplot
    
    # Scatter
    scatter = px.scatter(data, x = x, y = y,color = category_names,
                         color_discrete_sequence=px.colors.qualitative.Light24,
                         category_orders = {category_names : labels})
    scatter.update_xaxes(showline = True, linewidth = 1, linecolor = 'black', 
                          mirror = True, gridcolor = 'LightPink', automargin = True, 
                          zeroline = True, zerolinewidth = 2, zerolinecolor = 'LightPink', 
                          ticks = "outside", tickwidth = 2, tickcolor = 'black', ticklen = 10)
    scatter.update_yaxes(showline = True, linewidth = 2, linecolor = 'black', 
                          mirror = True, gridcolor = 'LightPink',
                          zeroline = True, zerolinewidth = 1, zerolinecolor = 'LightPink', 
                          ticks = "outside", tickwidth = 2, tickcolor = 'black', ticklen = 10)
    
    
    scatter.update_layout(
        legend = dict(
            x = 1,
            y = 1,
            traceorder = "normal",
            font = dict(
                family = "sans-serif",
                size = 14,
                color = "black"
            ),
            bgcolor = "#e5ecf6",
            bordercolor = "Black",
            borderwidth = 2
        )
    )

    # Boxplot
    box = px.box(data, x = discretized , y = y, color = category_names, 
                 color_discrete_sequence = px.colors.qualitative.Light24,
                         category_orders = { category_names : labels})
    box.update_xaxes(automargin = True, showline = True, linewidth = 1, linecolor = 'black', 
                          mirror = True, gridcolor = 'LightPink', 
                          zeroline = True, zerolinewidth = 2, zerolinecolor = 'LightPink', 
                          ticks = "outside", tickwidth = 2, tickcolor = 'black', ticklen = 10)
    box.update_yaxes(showline = True, linewidth = 2, linecolor = 'black', 
                          mirror = True, gridcolor = 'LightPink',
                          zeroline = True, zerolinewidth = 1, zerolinecolor = 'LightPink', 
                          ticks = "outside", tickwidth = 2, tickcolor = 'black', ticklen = 10)
    
    box.update_layout(
        legend = dict(
            x = 1,
            y = 1,
            traceorder = "normal",
            font = dict(
                family = "sans-serif",
                size = 14,
                color = "black"
            ),
            bgcolor = "#e5ecf6",
            bordercolor = "Black",
            borderwidth = 2
        )
    )
        
    return scatter, box




### KDR
bins =  [i for i in np.arange(0, 10, 1)] + [110]
labels = ["{0:.2f}".format(i) + '-' + "{0:.2f}".format(i + 0.99) for i in np.arange(0, 9, 1)] + ['9.0-100']

KDR_Scatter, KDR_Box = ScatterBoxPlot(data = data, discretized = 'KillDeathRatio_Box',
                                      cut = 'KillDeathRatio', bins = bins,
                                      labels = labels, x = 'KillDeathRatio', y = 'WinRatio')




KDR_Scatter.update_xaxes(title_text = 'Kill-Death Ratio', title_font = {'size': 24})
KDR_Scatter.update_yaxes(title_text = 'Win Ratio (%)', title_font = {'size': 24})
KDR_Scatter.update_layout(title_text = "Scatterplot of Kill-Death Ratios and Win Ratios", title_font = {'size': 30} )

KDR_Box.update_xaxes(title_text = 'Kills-Death Ratio', title_font = {'size': 24})
KDR_Box.update_yaxes(title_text = 'Win Ratio (%)', title_font = {'size': 24})
KDR_Box.update_layout(title_text = "Boxplot of Kill-Death Ratios and Win Ratios", title_font = {'size': 30} )



### Top 10s
bins =  [i for i in np.arange(0, 110, 10)]
labels = ["{0:.2f}".format(i) + '-' + "{0:.2f}".format(i + 9.99) for i in np.arange(0, 100, 10)] 
Top10_Scatter, Top10_Box = ScatterBoxPlot(data = data, discretized = 'Top10Ratio_Box' , cut = 'Top10Ratio',
                                          bins = bins , labels = labels, x = 'Top10Ratio', y = 'WinRatio')


Top10_Scatter.update_xaxes(title_text = 'Top 10 Ratio (%)', title_font = {'size': 24})
Top10_Scatter.update_yaxes(title_text = 'Win Ratio (%)', title_font = {'size': 24})
Top10_Scatter.update_layout(title_text = "Scatterplot of Top 10 Ratio and Win Ratios", title_font = {'size': 30} )

Top10_Box.update_xaxes(title_text = 'Top 10 Ratio (%)', title_font = {'size': 24})
Top10_Box.update_yaxes(title_text = 'Win Ratio (%)', title_font = {'size': 24})
Top10_Box.update_layout(title_text = "Boxplot of Top 10 Ratio and Win Ratios", title_font = {'size': 30} )


### Average Time Survived
bins = [i for i in np.arange(0, 2000, 200)] + [2200]
labels = [str(i) + '-' + str(i + 199) for i in np.arange(0, 1800 , 200)] + ['1800-2200']
ATS_Scatter, ATS_Box = ScatterBoxPlot(data = data, discretized = 'AvgSurvivalTime_Box' , cut = 'AvgSurvivalTime',
                                      bins = bins , labels = labels, x = 'AvgSurvivalTime', y = 'WinRatio')

ATS_Scatter.update_xaxes(title_text = 'Average Time Survived per round (s)', title_font = {'size': 24})
ATS_Scatter.update_yaxes(title_text = 'Win Ratio (%)', title_font = {'size': 24})
ATS_Scatter.update_layout(title_text = "Scatterplot of Average Time Survived per round and Win Ratios", title_font = {'size': 30} )

ATS_Box.update_xaxes(title_text = 'Average Time Survived per round (s)', title_font = {'size': 24})
ATS_Box.update_yaxes(title_text = 'Win Ratio (%)', title_font = {'size': 24})
ATS_Box.update_layout(title_text = "Boxplot of Average Time Survived per round and Win Ratios", title_font = {'size': 30} )


### Damage per round
bins = [i for i in range(0, 700 , 100)] + [2030]
labels = [str(i) + '-' + str(i + 9) for i in range(0,600,100)] + ['510 - 2030']
DPG_Scatter, DPG_Box = ScatterBoxPlot(data = data, discretized = 'DamagePg_Box' , cut = 'DamagePg',
                                      bins = bins , labels = labels, x = 'DamagePg', y = 'WinRatio')

DPG_Scatter.update_xaxes(title_text = 'Damage per round', title_font = {'size': 24})
DPG_Scatter.update_yaxes(title_text = 'Win Ratio (%)', title_font = {'size': 24})
DPG_Scatter.update_layout(title_text = "Scatterplot of Damage per round and Win Ratios", title_font = {'size': 30} )

DPG_Box.update_xaxes(title_text = 'Damage per round', title_font = {'size': 24})
DPG_Box.update_yaxes(title_text = 'Win Ratio (%)', title_font = {'size': 24})
DPG_Box.update_layout(title_text = "Boxplot of Damage per round and Win Ratios", title_font = {'size': 30} )






## Data Exploration
image_filename = 'assets/Distance_Win-Ratio.png' 
encoded_image19 = base64.b64encode(
    open(image_filename, 'rb').read())

image_filename = 'assets/Survival_Time-Win-Ratio.png' 
encoded_image20 = base64.b64encode(
    open(image_filename, 'rb').read())

image_filename = 'assets/Survival_Time-KDR.png' 
encoded_image21 = base64.b64encode(
    open(image_filename, 'rb').read())



## CSS stylesheet for formatting
external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

## Instantiating the dashboard application
app = dash.Dash(__name__,
                external_stylesheets=external_stylesheets)

app.config['suppress_callback_exceptions'] = True
server = app.server

## Setting up the dashboard layout
app.layout = html.Div(
    [


### Inserting Logo into Heading and centering it
        html.Div(
            [
                html.Img(src='data:image/png;base64,{}'
                         .format(encoded_image.decode())
                        )
            ],
            style = 
            {
                'display': 'flex', 'align-items': 'center',
                'justify-content': 'center'
            }
        ),







# Bivariate Relationship with Win Ratio
 
## Insert header for Bivariate Relationships
        html.Div(
            [
                html.H2("Bivariate Relationships with Win Ratio" )
            ]
        ),

# Insert Markdown for Bivariate Relationships       
        html.Div(
            [
                dcc.Markdown(
                    ''' 
                    * Bivariate relationships are analyzed using scatter and box plots. The features are discretized into larger intervals to minimize the number of box plots.
                    * All features below are positively correlated with win ratio, as you observe the increasing trend in the scatter plots.
                    * The box plots contain large variance due to the sporadic nature of this data.
                    '''
                )
            ]
        ),

# Insert KDR
        html.Div(
            [
                html.H3("Kill-Death Ratio")
            ]
        ),
        html.Div(
            [
                html.Div(
                    [
                        dcc.Graph(figure = KDR_Scatter),
                    ], className = "six columns"
                ), 
                html.Div(
                    [
                        dcc.Graph( figure = KDR_Box),
                    ], className = "six columns"
                ), 
            ], className = 'row'
        ),

# Insert Top 10s
        html.Div(
            [
                html.H3("Top 10s")
            ]
        ),
        html.Div(
            [
                html.Div(
                    [
                        dcc.Graph(figure = Top10_Scatter),
                    ], className = "six columns"
                ), 
                html.Div(
                    [
                        dcc.Graph( figure = Top10_Box),
                    ], className="six columns"
                ), 
            ], className = 'row'
        ), 

        
# Insert Average Time Survived
        html.Div(
            [
                html.H3("Average Time Survived per round")
            ]
        ),
        html.Div(
            [
                html.Div(
                    [
                        dcc.Graph(figure = ATS_Scatter),
                    ], className = "six columns"
                ), 
                html.Div(
                    [
                        dcc.Graph( figure = ATS_Box),
                    ], className = "six columns"
                ), 

    ], className = 'row'
        ), 
        
        
# Insert Damage per round
        html.Div(
            [
                html.H3("Damage per round")
            ]
        ),
        html.Div(
            [
                html.Div(
                    [
                        dcc.Graph(figure = DPG_Scatter),
                    ], className = "six columns"
                ), 
                html.Div(
                    [
                        dcc.Graph( figure = DPG_Box),
                    ], className = "six columns"
                ), 
            ], className = 'row'
        ),
        html.Div(
            [
                html.H2("Data Exploration")
            ]
        ),
# Insert Markdown for Data Exploration 1
        html.Div(
            [
                dcc.Markdown(
                    ''' 
                    ### Does traveling more distance per round correlate with a higher win rate? 
                    ##### Let's analyze this by comparing the lower 50% and the upper 50% of the population. 
                
                    '''
                )
            ]
        ),
        html.Div(
            [
                html.Img(src = 'data:image/png;base64,{}'
                         .format(encoded_image19.decode())
                        )
            ],
            style = 
            {
                'display': 'flex', 'align-items': 'center',
                'justify-content': 'center'
            }
        ),
# Insert Markdown for Data Exploration 1
        html.Div(
            [
                dcc.Markdown(
                    ''' 
                    * The lower 50% of the population has a 2.724% win ratio.
                    * The upper 50% of the population has a 7.326% win ratio.
                    * The mean of the population has a 5.025% win ratio.
                
                    '''
                )
            ]
        ),
# Insert Markdown for Data Exploration 2
        html.Div(
            [
                dcc.Markdown(
                    ''' 
                    ### Does average survival time per round correlate with a higher win rate? 
                    ##### Let's analyze this by comparing the lower 50% and the upper 50% of the population.   
                
                ''')
            ]
        ),
        html.Div(
            [
                html.Img(src = 'data:image/png;base64,{}'
                         .format(encoded_image20.decode())
                        )
            ],
            style = 
            {
                'display': 'flex', 'align-items': 'center',
                'justify-content': 'center'
            }
        ),
        html.Div(
            [
# Insert Markdown for Data Exploration 2
                dcc.Markdown(
                    ''' 
                    * The lower 50% of the population has a 2.323% win ratio.
                    * The upper 50% of the population has a 7.724% win ratio.
                    * The mean of the population has a 5.023% win ratio.          
                
                '''
                )
            ]
        ),
        
        html.Div(
            [
                
# Insert Markdown for Data Exploration 3
                dcc.Markdown(
                    ''' 
                    ### Does average survival time per round correlate with higher KDRs? 
                    ##### Let's analyze this by comparing the lower 50% and the upper 50% of the population. 
                
                    '''
                )
            ]
        ),

        html.Div(
            [
                html.Img(src = 'data:image/png;base64,{}'
                         .format(encoded_image21.decode())
                        )
            ],
            style = 
            {
                'display': 'flex', 'align-items': 'center',
                'justify-content': 'center'
            }
        ),

        html.Div(
            [
                
# Insert Markdown for Data Exploration 3
                
                dcc.Markdown(
                    ''' 
                    * The lower 50% of the population has a 1.424 KDR.
                    * The upper 50% of the population has a 2.313 KDR.
                    * The mean of the population has a 1.868 KDR.      
                
                    '''
                )
            ]
        ),
        html.Div(
            [
                html.H2("Remarks" )
            ]
        ),

# Insert Markdown for Remarks    
        html.Div(
            [
                dcc.Markdown(
                    ''' 
                    * From the analysis presented in the EDA section, the data is sporadic.
                    * Through the nature of the game despite having better combat skills than your opponents. There are external factors present in the game that can impact your success (wins).
                    * In the next section,  we'll cluster player behavior for identifying cheaters (hackers).
                    
                    
                    '''
                )
            ]
        ),
    ]
)




if __name__ == '__main__':
    app.run_server(debug = True)