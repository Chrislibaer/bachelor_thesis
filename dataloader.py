import fxcmpy


TOKEN = '1527e7bd1a4dc334ef14b59ebc6c84df973a2c4f'
con = fxcmpy.fxcmpy(access_token=TOKEN, log_level='error', server='demo')
print("connected to the fxcm servers!")
eurusd = con.get_candles('EUR/USD', period='D1', number=10_000)
print("saving data")
con.close()
eurusd.to_csv('eurusd.csv')
