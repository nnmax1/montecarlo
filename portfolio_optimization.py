
import pandas as pd
from pandas_datareader import data

import numpy as np, numpy.random
from numpy import mean

import random

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import cm

from datetime import datetime

from scipy.stats import norm 
from scipy.stats import shapiro
from scipy.stats import kstest
from scipy.stats import skew
from scipy.stats import kurtosis
from scipy import stats

#from statsmodels.tsa.stattools import adfuller



def extract_prices(start_date,end_date,symbols,portfolioWeights,portfolioValue,backtestduration=0):
    dim=len(symbols)
    for symbol in symbols:
        dfprices = data.DataReader(symbols, start=start_date, end=end_date, data_source='yahoo')
        dfprices = dfprices[['Adj Close']]
    dfprices.columns=[' '.join(col).strip() for col in dfprices.columns.values]

    priceAtEndDate=[]
    for symbol in symbols:
        priceAtEndDate.append(dfprices[[f'Adj Close {symbol}']][-(backtestduration+1):].values[0][0])
        
    noOfShares=[]
    portfolioValPerSymbol=[x * portfolioValue for x in portfolioWeights]
    for i in range(0,len(symbols)):
        noOfShares.append(portfolioValPerSymbol[i]/priceAtEndDate[i])
    noOfShares=[round(element, 5) for element in noOfShares]
    listOfColumns=dfprices.columns.tolist()   
    dfprices["Adj Close Portfolio"]=dfprices[listOfColumns].mul(noOfShares).sum(1)
    
    share_split_table=dfprices.tail(1).T
    share_split_table=share_split_table.iloc[:-1]
    share_split_table["Share"]=symbols
    share_split_table["No Of Shares"]=noOfShares
    share_split_table.columns=["Price At "+end_date,"Share Name","No Of Shares"]
    share_split_table["Value At "+end_date]=share_split_table["No Of Shares"]*share_split_table["Price At "+end_date]
    share_split_table.index=share_split_table["Share Name"]
    share_split_table=share_split_table[["Share Name","Price At "+end_date,"No Of Shares","Value At "+end_date]]
    share_split_table=share_split_table.round(3)
    share_split_table=share_split_table.append(share_split_table.sum(numeric_only=True), ignore_index=True)
    share_split_table.at[len(symbols),'No Of Shares']=np.nan
    share_split_table.at[len(symbols),'Price At '+end_date]=np.nan
    share_split_table.at[len(symbols),'Share Name']="Portfolio"
    share_split_table["Weights"]=portfolioWeights+["1"]
    share_split_table = share_split_table[['Share Name', 'Weights', 'Price At '+end_date, 'No Of Shares', "Value At "+end_date]] 
    
    print(f"Extracted {len(dfprices)} days worth of data for {len(symbols)} counters with {dfprices.isnull().sum().sum()} missing data")
    
    return dfprices, noOfShares, share_split_table


def calc_returns(dfprices,symbols):
    dfreturns=pd.DataFrame()
    columns = list(dfprices) 
    mean=[]
    stdev=[]
    for column in columns:
        dfreturns[f'Log Daily Returns {column}']=np.log(dfprices[column]).diff()
        mean.append(dfreturns[f'Log Daily Returns {column}'][1:].mean())
        stdev.append(dfreturns[f'Log Daily Returns {column}'][1:].std())
    dfreturns=dfreturns.dropna()
    
    if len(dfreturns.columns)==1:
        df_mean_stdev=pd.DataFrame(list(zip(symbols,mean,stdev)),columns =['Stock', 'Mean Log Daily Return','StdDev Log Daily Return']) 
    else:
        df_mean_stdev=pd.DataFrame(list(zip(symbols+["Portfolio"],mean,stdev)),columns =['Stock', 'Mean Log Daily Return','StdDev Log Daily Return'])
    
    return dfreturns ,df_mean_stdev

def plotpiechart(symbols,portfolioWeights,imagecounter,targetfolder):    
    labels = symbols
    sizes = portfolioWeights
    fig1, ax1 = plt.subplots()
    ax1.pie(portfolioWeights, labels=symbols, autopct='%1.1f%%',
        shadow=True, startangle=90)
    ax1.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
    plt.title("Portfolio Diversity")
    plt.savefig(f'static/{targetfolder}/{imagecounter}_01portfolioweights.png')

# covariance matrix
def create_covar(dfreturns):  
    try:
        returns=[]
        arrOfReturns=[]
        columns = list(dfreturns)
        for column in columns:
            returns=dfreturns[column].values.tolist()
            arrOfReturns.append(returns)
        Cov = np.cov(np.array(arrOfReturns))    
        return Cov
    except LinAlgError :
        Cov = nearPD(np.array(arrOfReturns), nit=10)
        print("WARNING -Original Covariance Matrix is NOT Positive Semi Definite And Has Been Adjusted To Allow For Cholesky Decomposition ")
        return Cov
        
def GBMsimulatorUniVar(So, mu, sigma, T, N):
    """
    Parameters

    seed:   seed of simulation
    So:     initial stocks' price
    mu:     expected return
    sigma:  volatility
    Cov:    covariance matrix
    T:      time period
    N:      number of increments
    """

    #np.random.seed(seed) turned off so Monte Carlo can be "randomised"
    dim = np.size(So)
    t = np.linspace(0., T, int(N))
    S = np.zeros([dim, int(N)])
    S[:, 0] = So
    for i in range(1, int(N)):    
        drift = (mu - 0.5 * sigma**2) * (t[i] - t[i-1])
        Z = np.random.normal(0., 1., dim)
        diffusion = sigma* Z * (np.sqrt(t[i] - t[i-1]))
        S[:, i] = S[:, i-1]*np.exp(drift + diffusion)
    return S, t

def GBMsimulatorMultiVar(So, mu, sigma, Cov, T, N):
    """
    Parameters

    seed:   seed of simulation
    So:     initial stocks' price
    mu:     expected return
    sigma:  volatility
    Cov:    covariance matrix
    T:      time period
    N:      number of increments
    """

    #np.random.seed(seed) turned off so Monte Carlo can be "randomised"
    dim = np.size(So)
    t = np.linspace(0., T, int(N))
    A = np.linalg.cholesky(Cov)
    S = np.zeros([dim, int(N)])
    S[:, 0] = So
    for i in range(1, int(N)):    
        drift = (mu - 0.5 * sigma**2) * (t[i] - t[i-1])
        Z = np.random.normal(0., 1., dim)
        diffusion = np.matmul(A, Z) * (np.sqrt(t[i] - t[i-1]))
        S[:, i] = S[:, i-1]*np.exp(drift + diffusion)
    return S, t

def MonteCarlo_GBM(start_date,end_date,backtest_duration,percentile_range,symbols,\
                       portfolioWeights,portfolioValue,T,N,NoOfIterationsMC,imagecounter,targetfolder):
    
    forecastresults=pd.DataFrame()
    percentiles=pd.DataFrame()
    
    extended_dates_future=[]
    lowerpercentile=int(percentile_range[1:3])
    upperpercentile=int(percentile_range[5:7])
 
    plotpiechart(symbols,portfolioWeights,imagecounter,targetfolder)

    if len(symbols)==1:
        dfpricesFULL, noOfSharesFULL, share_split_tableFULL = extract_prices(start_date,end_date,symbols,portfolioWeights,portfolioValue)
        backtest_end_date=dfpricesFULL.index[-(backtest_duration+1)].strftime("%Y-%m-%d")
        dfprices, noOfShares, share_split_table = extract_prices(start_date,backtest_end_date,symbols,portfolioWeights,portfolioValue)
        dfprices["Adj Close Portfolio"]=dfprices[list(dfprices.iloc[:,:-1].columns)].mul(noOfSharesFULL).sum(1)
        
    else:
        dfpricesFULL, noOfSharesFULL, share_split_tableFULL = extract_prices(start_date,end_date,symbols,portfolioWeights,portfolioValue)
        backtest_end_date=dfpricesFULL.index[-(backtest_duration+1)].strftime("%Y-%m-%d")
        dfprices, noOfShares, share_split_table = extract_prices(start_date,backtest_end_date,symbols,portfolioWeights,portfolioValue)
        dfprices["Adj Close Portfolio"]=dfprices[list(dfprices.iloc[:,:-1].columns)].mul(noOfSharesFULL).sum(1)
                  
    symbolsWPortfolio=symbols+["Portfolio"]

    dfreturns ,df_mean_stdev = calc_returns(dfprices,symbolsWPortfolio)
    
    S0=np.array(dfprices.tail(1).values.tolist()[0])
    mu=np.array(df_mean_stdev["Mean Log Daily Return"].values.tolist())
    sigma=np.array(df_mean_stdev["StdDev Log Daily Return"].values.tolist())   

    backtestdateslist=(list((dfpricesFULL.tail(backtest_duration+1).index)))
    backtestdates=[]
    for i in backtestdateslist:
        backtestdates.append(np.datetime64(datetime.strptime(str(i), '%Y-%m-%d %H:%M:%S').strftime("%Y-%m-%d")))
    
    for i in range(0,N-backtest_duration):
        extended_dates_future.append(np.busday_offset(end_date, i, roll='forward'))
        
    extended_dates=backtestdates[0:len(backtestdates)-1]+extended_dates_future
     
    if len(symbols)==1:
        for x in range(1,NoOfIterationsMC+1):
            stocks, time = GBMsimulatorUniVar(S0, mu, sigma, T, N)
            prediction=pd.DataFrame(stocks)
            prediction=prediction.T
            prediction.index=extended_dates
            prediction.columns=dfprices.columns
            prediction=prediction.add_prefix('Iter_'+str(x)+'_')
            forecastresults=pd.concat([forecastresults, prediction], axis=1)
        
        for x in range(1,NoOfIterationsMC+1):
            forecastresults["Iter_"+str(x)+"_Adj Close Portfolio"]=forecastresults["Iter_"+str(x)+"_Adj Close "+symbols[0]]*noOfSharesFULL

    else:
        Cov=create_covar(dfreturns)
        for x in range(1,NoOfIterationsMC+1):
            stocks, time = GBMsimulatorMultiVar(S0, mu, sigma, Cov, T, N)
            prediction=pd.DataFrame(stocks)
            prediction=prediction.T
            prediction.index=extended_dates
            prediction.columns=dfprices.columns
            prediction=prediction.add_prefix('Iter_'+str(x)+'_')
            forecastresults=pd.concat([forecastresults, prediction], axis=1)

    for y in range(0,len(symbolsWPortfolio)):
        percentiles["P"+str(lowerpercentile)+"_"+symbolsWPortfolio[y]]=forecastresults.filter(regex=symbolsWPortfolio[y]).quantile(float(lowerpercentile)/100,1)
        percentiles["P50_"+symbolsWPortfolio[y]]=forecastresults.filter(regex=symbolsWPortfolio[y]).quantile(0.5,1)
        percentiles["P"+str(upperpercentile)+"_"+symbolsWPortfolio[y]]=forecastresults.filter(regex=symbolsWPortfolio[y]).quantile(float(upperpercentile)/100,1)

        forecastresults=pd.concat([forecastresults,percentiles[["P"+str(lowerpercentile)+"_"+symbolsWPortfolio[y],"P50_"+symbolsWPortfolio[y],"P"+str(upperpercentile)+"_"+symbolsWPortfolio[y]]]], axis=1, sort=False)
    
    final=pd.concat([dfpricesFULL,forecastresults], axis=1, sort=False)
              
    for z in range(0,len(symbolsWPortfolio)):
        final.filter(regex="Adj Close "+symbolsWPortfolio[z]).tail(60).plot(legend=False,figsize = (20, 5),title=symbolsWPortfolio[z]+": Monte Carlo Simulations For "+str(NoOfIterationsMC)+" Iter-s")
        plt.axvline(x=end_date,linestyle='dashed')
        plt.savefig(f'static/{targetfolder}/{imagecounter}_totaliterations{z}.png')
        
        percentileplot=pd.DataFrame()
        percentileplot=pd.concat([final["Adj Close "+symbolsWPortfolio[z]],final.filter(regex="P??_"+symbolsWPortfolio[z])], axis=1, sort=False)
        percentileplot.tail(60).plot(legend=True,figsize = (20, 5),title=symbolsWPortfolio[z]+": Monte Carlo Simulations For "+percentile_range+" Range")
        plt.axvline(x=end_date,linestyle='dashed')
        if NoOfIterationsMC>0:
            plt.savefig(f'static/{targetfolder}/{imagecounter}_percentile{z}.png')
  
    ReturnsAtForecastEndDate=final.tail(1).iloc[:,-(len(symbolsWPortfolio))*3:].T
    HelperTable=pd.concat([dfpricesFULL.tail(1).round(3).T]*(3)) 
    HelperTable["Sym"]=HelperTable.index
    HelperTable['Sym'] =pd.Categorical(HelperTable["Sym"], list(dfprices.columns))
    HelperTable=HelperTable.sort_values(['Sym'])
    ReturnsAtForecastEndDate.insert(0, end_date, HelperTable.iloc[:,:-1].values)    
           
    ReturnsAtForecastEndDate["Returns Based On GBM"]=round((ReturnsAtForecastEndDate.iloc[:, 1]/ReturnsAtForecastEndDate.iloc[:, 0]-1)*100,2)
    
    return final, share_split_tableFULL , dfreturns , df_mean_stdev , ReturnsAtForecastEndDate, dfprices

def MonteCarlo_Bootstrap(start_date,end_date,backtest_duration,percentile_range,symbols,\
                       portfolioWeights,portfolioValue,T,N,NoOfIterationsMC,imagecounter,targetfolder):
    
    forecastresults=pd.DataFrame()
    percentiles=pd.DataFrame()
    
    extended_dates_future=[]
    lowerpercentile=int(percentile_range[1:3])
    upperpercentile=int(percentile_range[5:7])
    
    plotpiechart(symbols,portfolioWeights,imagecounter,targetfolder)

    if len(symbols)==1:
        dfpricesFULL, noOfSharesFULL, share_split_tableFULL = extract_prices(start_date,end_date,symbols,portfolioWeights,portfolioValue)
        backtest_end_date=dfpricesFULL.index[-(backtest_duration+1)].strftime("%Y-%m-%d")
        dfprices, noOfShares, share_split_table = extract_prices(start_date,backtest_end_date,symbols,portfolioWeights,portfolioValue)
        dfprices["Adj Close Portfolio"]=dfprices[list(dfprices.iloc[:,:-1].columns)].mul(noOfSharesFULL).sum(1)
        
    else:
        dfpricesFULL, noOfSharesFULL, share_split_tableFULL = extract_prices(start_date,end_date,symbols,portfolioWeights,portfolioValue)
        backtest_end_date=dfpricesFULL.index[-(backtest_duration+1)].strftime("%Y-%m-%d")
        dfprices, noOfShares, share_split_table = extract_prices(start_date,backtest_end_date,symbols,portfolioWeights,portfolioValue)
        dfprices["Adj Close Portfolio"]=dfprices[list(dfprices.iloc[:,:-1].columns)].mul(noOfSharesFULL).sum(1)
               
    symbolsWPortfolio=symbols+["Portfolio"]

    dfreturns ,df_mean_stdev = calc_returns(dfprices,symbolsWPortfolio)

    backtestdateslist=(list((dfpricesFULL.tail(backtest_duration+1).index)))
    backtestdates=[]
    for i in backtestdateslist:
        backtestdates.append(np.datetime64(datetime.strptime(str(i), '%Y-%m-%d %H:%M:%S').strftime("%Y-%m-%d")))
        
    for i in range(0,N-backtest_duration):
        extended_dates_future.append(np.busday_offset(end_date, i, roll='forward'))
        
    extended_dates=backtestdates[0:len(backtestdates)-1]+extended_dates_future

    for x in range(1,NoOfIterationsMC+1):      
        
        futurereturns=bootstrapforecast(dfreturns,T)
        futurereturns=np.exp(futurereturns)
        futurereturns=futurereturns.cumprod()
        stocks=pd.DataFrame()
        for i in range(0,len(symbolsWPortfolio)):
            futurereturns[str(i)+"Price"]=(futurereturns.iloc[:, i])*dfprices.tail(1).iloc[:, i][0]
        stocks=futurereturns[futurereturns.columns[-len(symbolsWPortfolio):]] 
        stocks.columns=list(dfreturns.columns)

        prediction=stocks
        prediction.index=extended_dates
        prediction.columns=dfprices.columns
        prediction=prediction.add_prefix('Iter_'+str(x)+'_')
        forecastresults=pd.concat([forecastresults,prediction], axis=1, sort=False)
    
    for y in range(0,len(symbolsWPortfolio)):
        percentiles["P"+str(lowerpercentile)+"_"+symbolsWPortfolio[y]]=forecastresults.filter(regex=symbolsWPortfolio[y]).quantile(float(lowerpercentile)/100,1)
        percentiles["P50_"+symbolsWPortfolio[y]]=forecastresults.filter(regex=symbolsWPortfolio[y]).quantile(0.5,1)
        percentiles["P"+str(upperpercentile)+"_"+symbolsWPortfolio[y]]=forecastresults.filter(regex=symbolsWPortfolio[y]).quantile(float(upperpercentile)/100,1)

        forecastresults=pd.concat([forecastresults,percentiles[["P"+str(lowerpercentile)+"_"+symbolsWPortfolio[y],"P50_"+symbolsWPortfolio[y],"P"+str(upperpercentile)+"_"+symbolsWPortfolio[y]]]], axis=1, sort=False)
    
    final=pd.concat([dfpricesFULL,forecastresults], axis=1, sort=False)
    
    for z in range(0,len(symbolsWPortfolio)):
        final.filter(regex="Adj Close "+symbolsWPortfolio[z]).tail(60).plot(legend=False,figsize = (20, 5),title=symbolsWPortfolio[z]+": Monte Carlo Simulations For "+str(NoOfIterationsMC)+" Iter-s")
        plt.axvline(x=end_date,linestyle='dashed')
        plt.savefig(f'static/{targetfolder}/{imagecounter}_totaliterations{z}.png')
        
        percentileplot=pd.DataFrame()
        percentileplot=pd.concat([final["Adj Close "+symbolsWPortfolio[z]],final.filter(regex="P??_"+symbolsWPortfolio[z])], axis=1, sort=False)
        percentileplot.tail(60).plot(legend=True,figsize = (20, 5),title=symbolsWPortfolio[z]+": Monte Carlo Simulations For "+percentile_range+" Range")
        plt.axvline(x=end_date,linestyle='dashed')
        if NoOfIterationsMC>1:
            plt.savefig(f'static/{targetfolder}/{imagecounter}_percentile{z}.png')
        
    if len(symbols)==1:
        ReturnsAtForecastEndDate=final.tail(1).iloc[:,-(len(symbolsWPortfolio))*3:].T
        HelperTable=pd.concat([dfpricesFULL.tail(1).round(3).T]*(3))
        HelperTable["Sym"]=HelperTable.index
        HelperTable['Sym'] =pd.Categorical(HelperTable["Sym"], list(dfprices.columns))
        HelperTable=HelperTable.sort_values(['Sym'])
        ReturnsAtForecastEndDate.insert(0, end_date, HelperTable.iloc[:,:-1].values) 
    else:
        ReturnsAtForecastEndDate=final.tail(1).iloc[:,-(len(symbolsWPortfolio))*3:].T
        HelperTable=pd.concat([dfpricesFULL.tail(1).round(3).T]*(3)) 
        HelperTable["Sym"]=HelperTable.index
        HelperTable['Sym'] =pd.Categorical(HelperTable["Sym"], list(dfprices.columns))
        HelperTable=HelperTable.sort_values(['Sym'])
        ReturnsAtForecastEndDate.insert(0, end_date, HelperTable.iloc[:,:-1].values)    
           
    ReturnsAtForecastEndDate["Returns Based On BStrp"]=round((ReturnsAtForecastEndDate.iloc[:, 1]/ReturnsAtForecastEndDate.iloc[:, 0]-1)*100,2)
    
    return final, share_split_tableFULL , dfreturns , df_mean_stdev, ReturnsAtForecastEndDate, dfprices


start_date = '2022-01-01'
end_date='2022-03-07'
forecast_end_date='2022-04-30'
backtest_duration=30 #day vs last day of available data
T=np.busday_count(end_date,forecast_end_date)+backtest_duration
N=T+1

symbols=['NVDA','AMD','CLF', 'UCO']
portfolioWeights=[0.40,0.20,0.2, 0.2]

portfolioValue=10000000

NoOfIterationsMC=100
NoOfIterationsInnerLoop=100
percentile_range="P10_P90"


dfprices, noOfShares, share_split_table=extract_prices(start_date,end_date,symbols,portfolioWeights,portfolioValue,backtestduration=0)
#print(share_split_table)
dfreturns ,df_mean_stdev=calc_returns(dfprices,symbols)

print(df_mean_stdev)
print(dfreturns)
#display(df_mean_stdev)

#display(df_mean_stdev)


# Monte Carlo Simulation for future prices
finalGBM, share_split_tableFULL , dfreturns , df_mean_stdev , ReturnsAtForecastEndDate, dfprices= MonteCarlo_GBM(start_date,end_date,backtest_duration,percentile_range,symbols,\
                       portfolioWeights,portfolioValue,T,N,NoOfIterationsMC,"GBM Simulation","efficientportfolio")
