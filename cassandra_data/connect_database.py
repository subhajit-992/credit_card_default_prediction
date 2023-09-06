from cassandra.cluster import Cluster
from cassandra.auth import PlainTextAuthProvider
from creditcard.exception import CreditCardsException
from creditcard.logger import logging
import json
import csv
import os
import sys

def import_data(zip_file:str,json_file:str):
   try:
      # This secure connect bundle is autogenerated when you donwload your SCB, 
      # if yours is different update the file name below
      cloud_config= {'secure_connect_bundle': zip_file}

      # This token json file is autogenerated when you donwload your token, 
      # if yours is different update the file name below
      with open(json_file,"r") as f:
         secrets = json.load(f)

      CLIENT_ID = secrets["clientId"]
      CLIENT_SECRET = secrets["secret"]

      auth_provider = PlainTextAuthProvider(CLIENT_ID, CLIENT_SECRET)
      cluster = Cluster(cloud=cloud_config, auth_provider=auth_provider)
      session = cluster.connect()

      row = session.execute("select release_version from system.local").one()
      if row:
         print(row[0])
      else:
         print("An error occurred.")

      keyspace = "credit_card"
      table = "uci_credit_card"

      # Execute a SELECT query
      query = f"SELECT * FROM {keyspace}.{table}"
      result = session.execute(query)
      # Define the CSV filename
      csv_filename = "data.csv"

      # Write the data to the CSV file
      with open(csv_filename, "w", newline="") as csvfile:
         csv_writer = csv.writer(csvfile)
    
         # Write headers
         csv_writer.writerow(result.column_names)
    
         # Write rows
         for row in result:
            csv_writer.writerow(row)

      session.shutdown()
      cluster.shutdown()
      return csv_filename
   except Exception as e:
      raise CreditCardsException(e,sys)