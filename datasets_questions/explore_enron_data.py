#!/usr/bin/python

""" 
    starter code for exploring the Enron dataset (emails + finances) 
    loads up the dataset (pickled dict of dicts)

    the dataset has the form
    enron_data["LASTNAME FIRSTNAME MIDDLEINITIAL"] = { features_dict }

    {features_dict} is a dictionary of features associated with that person
    you should explore features_dict as part of the mini-project,
    but here's an example to get you started:

    enron_data["SKILLING JEFFREY K"]["bonus"] = 5600000
    
"""

import pickle

enron_data = pickle.load(open(file="../final_project/final_project_dataset.pkl", mode="rb"))
dataset_length = len(enron_data)
features_length = len(enron_data["SKILLING JEFFREY K"])

poi_count = 0
quantified_salary = 0
known_email_address = 0
undefined_payment = 0
poi_undefined_payment = 0
max_exercised_stock_option = 0;
min_exercised_stock_option = 9223372036854775807;
for j in enron_data:
    if enron_data[j]["poi"]:
        poi_count = poi_count + 1
    if enron_data[j]["email_address"] != "NaN":
       known_email_address = known_email_address + 1
    if enron_data[j]["salary"] != "NaN":
        quantified_salary = quantified_salary + 1
    if enron_data[j]["total_payments"] == "NaN":
        undefined_payment = undefined_payment + 1
    if enron_data[j]["poi"] and enron_data[j]["total_payments"] == "NaN":
          poi_undefined_payment = poi_undefined_payment + 1;
    if enron_data[j]["exercised_stock_options"]  != "NaN" and  enron_data[j]["exercised_stock_options"] > max_exercised_stock_option:
        max_exercised_stock_option = enron_data[j]["exercised_stock_options"];
    if enron_data[j]["exercised_stock_options"]  != "NaN" and  enron_data[j]["exercised_stock_options"] < min_exercised_stock_option:
        min_exercised_stock_option = enron_data[j]["exercised_stock_options"];

print("amount of people in dataset:", dataset_length)
print("amount of features:", features_length)
print("amount of persons of interest:", poi_count)
print("James Prentice total stock value:", enron_data["PRENTICE JAMES"]["total_stock_value"])
print("emails from Wesley Colwell to poi:", enron_data["COLWELL WESLEY"]["from_this_person_to_poi"])
print("value stock options of Jeffrey Skilling", enron_data["SKILLING JEFFREY K"]["exercised_stock_options"])
print("total payment to Jeffrey Skilling", enron_data["SKILLING JEFFREY K"]["total_payments"])
print("total payment to Kenneth Lay", enron_data["LAY KENNETH L"]["total_payments"])
print("total payment to Andrew fastow", enron_data["FASTOW ANDREW S"]["total_payments"])
print("amount of people with quantified salaries:", quantified_salary)
print ("amount of people with emails:", known_email_address)
print ("amount of people with undefined payment:", undefined_payment)
print ("amount of poi with undefined payment:", poi_undefined_payment)
print ("max exercised stock options",max_exercised_stock_option)
print ("min exercised stock options", min_exercised_stock_option)
