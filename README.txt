This is a feed forward neural network for predicting stock prices using technical indicators. I read about it in an article you can find here:

https://www.sciencedirect.com/science/article/pii/S2405918815300179?via%3Dihub

There is no code in the article. I did my best to work backwards based off of the descriptions provided.

You need these installed to run the program

python
pandas
numpy
talib

HOW TO USE:

You can drag and drop an excel file with open, close, high, low, and dates of a specific stock into the same folder as Technical.py . The example I used is for Boeing and I got the data from yahoo/finance. Then, inside Technical.py change BA in df = pd.read_csv('BA.csv') into whatever your csv is with your stock data.

You will also need to change

train = df2.iloc[:1000, :]
and
test = df2.iloc[1000:, :]

and replace 1000 with the number of periods you want to train and the number of periods you want to test.

You should now be able to run SAM.py in your terminal. You can uncomment:

# print(f'Weights = {d_weights1}')

or

# print(f'Square = {square}')

to see if SAM is running properly in your terminal. If you're getting numbers and no errors then it's working.

The Jupyter notebook file imports SAM and will display with matplotlib how well the network did at predicting the stock prices compared to the actual prices over the test period.
