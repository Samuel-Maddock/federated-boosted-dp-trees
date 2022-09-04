import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_classification
from collections import Counter
from pmlb import fetch_data
import random

import os

dirname = os.path.split(os.path.split(os.path.dirname(__file__))[0])[0]
data_path = dirname + "/data/"

class DataLoader():
    def __init__(self, seeds=None):
        self.seeds = [random.randint(0, 2**30)] if seeds is None else seeds

    def _load_synthetic(self, data_dict, seed, test_size, verbose=True, num_samples=100000, num_features=500, informative=5, redundant=2, repeated=2, weights=None):
        name = "synthetic_" + str(num_samples) + "_" + str(num_features) + "_" + str(informative) + "_" + str(redundant) + "_" + str(repeated) + "_" + str(weights)
        X,y = make_classification(round(num_samples/test_size), num_features, n_informative=informative, n_redundant=redundant, n_repeated=repeated, weights=weights, shuffle=True)

        if verbose:
            self.print_data_info(name, X, y)

        data = train_test_split(X, y, test_size=test_size, random_state=seed, stratify=y)
        data_dict[name+"_"+str(seed)] = data

    def print_data_info(self, name, X, y):
        print(name, "shape:", X.shape)
        print(pd.DataFrame(X).corrwith(pd.Series(y)))
        print((pd.DataFrame(X).corrwith(pd.Series(y)).abs() >= 0.5).sum()/X.shape[1])
        print(np.array(list(Counter(y).values()))/len(y), "\n")

    def _load_pmlb_dataset(self, name, data_dict, seed, test_size, verbose):
         X, y = fetch_data(name, return_X_y=True, local_cache_dir=data_path)
         if verbose:
             self.print_data_info(name, X, y)
         data = train_test_split(X, y, test_size=test_size, random_state=seed, shuffle=True, stratify=y)
         data_dict[name+"_"+str(seed)] = data

    def _load_credit1(self, data_dict, seed, test_size, remove_missing, verbose):
        credit1_df = pd.read_csv(data_path + "Kaggle Credit 1/credit1-training.csv")
        credit1_df = credit1_df.drop('Unnamed: 0', axis=1)

        if remove_missing:
            num_rows = credit1_df.shape[0]
            credit1_df = credit1_df.dropna()
            print("[Data Loader] Removing nans from Credit 1", num_rows, "vs", credit1_df.shape[0])

        credit1_y = credit1_df["SeriousDlqin2yrs"]
        credit1_X = credit1_df.drop("SeriousDlqin2yrs", axis=1)
        credit1_counter = Counter(credit1_y)
        print("[Data Loader] Credit 1 Class Balance:", credit1_counter[1] / sum(credit1_counter.values()), "\n")

        if verbose:
            self.print_data_info("Credit 1", credit1_X, credit1_y)

        print(seed)
        credit1_data = train_test_split(credit1_X, credit1_y, test_size=test_size, random_state=seed, stratify=credit1_y)
        data_dict["Credit 1"+"_"+str(seed)] = credit1_data

    def _load_credit2(self, data_dict, seed, test_size, remove_missing, verbose):
        credit2_df = pd.read_csv(data_path + "UCI Credit 2/UCI_Credit_Card.csv").drop("ID", axis=1)

        if remove_missing:
            num_rows = credit2_df.shape[0]
            credit2_df = credit2_df.dropna()
            print("[Data Loader] Removing nans from Credit 2", num_rows, "vs", credit2_df.shape[0])

        credit2_y = credit2_df["default.payment.next.month"]
        credit2_X = credit2_df.drop("default.payment.next.month", axis=1)
        credit2_data = train_test_split(credit2_X, credit2_y, test_size=test_size, random_state=seed, stratify=credit2_y)
        credit2_counter = Counter(credit2_y)
        print("[Data Loader] Credit 2 Class Balance:", credit2_counter[1] / sum(credit2_counter.values()), "\n")

        if verbose:
            self.print_data_info("Credit 2", credit2_X, credit2_y)

        data_dict["Credit 2"+"_"+str(seed)] = credit2_data

    def _load_credit_card(self, data_dict, seed, test_size, remove_missing, verbose):
        credit_df = pd.read_csv(data_path + "Kaggle Credit Fraud/creditcard.csv")

        if remove_missing:
            num_rows = credit_df.shape[0]
            credit_df = credit_df.dropna()
            print("[Data Loader] Removing nans from Credit", num_rows, "vs", credit_df.shape[0])

        credit_y = credit_df["Class"]
        credit_X = credit_df.drop("Class", axis=1)
        credit_data = train_test_split(credit_X, credit_y, test_size=test_size, random_state=seed, stratify=credit_y)
        credit_counter = Counter(credit_y)
        print("[Data Loader] Credit Card Class Balance:", credit_counter[1] / sum(credit_counter.values()), "\n")

        if verbose:
            self.print_data_info("Credit Card", credit_X, credit_y)

        data_dict["Credit Fraud"+"_"+str(seed)] = credit_data

    def _load_sensor(self, data_dict, seed, test_size, remove_missing, verbose):
        sensor_data = pd.read_csv(data_path + "UCI Sensorless Drive Diagnosis/Sensorless_drive_diagnosis.txt", delimiter=" ", lineterminator="\n",
                                  header=None)
        if remove_missing:
            num_rows = sensor_data.shape[0]
            sensor_data = sensor_data.dropna()
            print("[Data Loader] Removing nans from Sensor", num_rows, "vs", sensor_data.shape[0])

        sensor_y = np.array(sensor_data.iloc[:, 48].values)
        class_val = 1
        sensor_y[sensor_y <= class_val] = 1
        sensor_y[sensor_y > class_val] = 0
        sensor_X = sensor_data.drop(48, axis=1)
        sensor_data = train_test_split(sensor_X, sensor_y, test_size=test_size, random_state=seed, stratify=sensor_y)

        sensor_counter = Counter(sensor_y)
        print("[Data Loader] Sensor Class Balance:", sensor_counter[1] / sum(sensor_counter.values()), "\n")

        if verbose:
         print("Sensor", "shape:", sensor_X.shape)
         print(pd.DataFrame(sensor_X).corrwith(pd.Series(sensor_y)))
         print((pd.DataFrame(sensor_X).corrwith(pd.Series(sensor_y)).abs() >= 0.5).sum()/sensor_X.shape[1])
         print(np.array(list(Counter(sensor_y).values()))/len(sensor_y), "\n")

        data_dict["sensor"+"_"+str(seed)] = sensor_data

    def _load_nomao(self, data_dict, seed, test_size, remove_missing, verbose):
        nomao_data = pd.read_csv(data_path + "UCI Nomao/Nomao.data", delimiter=",", lineterminator="\n", header=None)
        no_missing_cols = []
        for col in nomao_data.columns:
            if (nomao_data[col] != "?").sum() == 34465 and nomao_data[col].dtype != object:
                no_missing_cols.append(col)

        nomao_data = nomao_data[no_missing_cols]

        if remove_missing:
            num_rows = nomao_data.shape[0]
            nomao_data = nomao_data.dropna()
            print("[Data Loader] Removing nans from Nomao", num_rows, "vs", nomao_data.shape[0])

        nomao_y = np.array(nomao_data[119].values)
        nomao_y[nomao_y == 1] = 0
        nomao_y[nomao_y == -1] = 1
        nomao_X = nomao_data.drop(119, axis=1)
        nomao_data = train_test_split(nomao_X, nomao_y, test_size=test_size, random_state=seed, stratify=nomao_y)

        nomao_counter = Counter(nomao_y)
        print("[Data Loader] Nomao Class Balance:", nomao_counter[1] / sum(nomao_counter.values()), "\n")

        if verbose:
            self.print_data_info("Nomao", nomao_X, nomao_y)

        data_dict["nomao"+"_"+str(seed)] = nomao_data

    def _load_adult(self, data_dict, seed, test_size, remove_missing, verbose):
        adult = pd.read_csv(data_path + "UCI Adult/adult.data", header=None, na_values=" ?")
        print(adult.columns)

        if remove_missing:
            num_rows = adult.shape[0]
            adult = adult.dropna()
            print("[Data Loader] Removing nans from Adult", num_rows, "vs", adult.shape[0])

        adult_y = pd.factorize(adult[14])[0]
        adult_X = adult.drop(14, axis=1)

        for col in adult_X.columns:
            adult_X[col] = pd.Categorical(adult_X[col]).codes

        # adult_X = pd.get_dummies(adult_X) # One-hot encoding is expensive for hist-based dp
        adult_data = train_test_split(adult_X, adult_y, test_size=test_size, random_state=seed, stratify=adult_y)
        adult_counter = Counter(adult_y)
        print(adult_counter)
        print("[Data Loader] Adult Class Balance:", adult_counter[1] / sum(adult_counter.values()), "\n")

        if verbose:
            self.print_data_info("Adult", adult_X, adult_y)

        data_dict["adult"+"_"+str(seed)] = adult_data

    def _load_higgs(self, data_dict, seed, test_size, remove_missing, verbose, load_sampled=False):
        if load_sampled:
            higgs_data = pd.read_csv(data_path + "UCI Higgs/higgs-200k.csv", delimiter=",", lineterminator="\n", header=None)
        else:
            higgs_data = pd.read_csv(data_path + "UCI Higgs/HIGGS_sampled.csv", delimiter=",", lineterminator="\n", header=None)

        # higgs_data = higgs_data.sample(n=200000)
        # higgs_data.to_csv(data_path + "UCI Higgs/higgs-200k.csv", index=False,)

        if remove_missing:
            num_rows = higgs_data.shape[0]
            higgs_data = higgs_data.dropna()
            print("Removing nans from higgs", num_rows, "vs", higgs_data.shape[0], "\n")

        higgs_y = higgs_data[1].astype("int32")
        higgs_X = higgs_data.drop([0, 1], axis=1)
        # print("Higgs:", higgs_y.sum() / len(higgs_y))
        higgs_data = train_test_split(higgs_X, higgs_y, test_size=test_size, random_state=seed, shuffle=True, stratify=higgs_y)
        higgs_counter = Counter(higgs_y)
        print("[Data Loader] Higgs Class Balance:", higgs_counter[1] / sum(higgs_counter.values()), "\n")

        if verbose:
            self.print_data_info("Higgs", higgs_X, higgs_y)

        if load_sampled:
            data_dict["higgs-sample"+"_"+str(seed)] = higgs_data
        else:
            data_dict["higgs"+"_"+str(seed)] = higgs_data

    def _load_diabetes(self, data_dict, seed, test_size, remove_missing, verbose):
        diabetes_df = pd.read_csv(data_path + "UCI Diabetes130US/diabetic_data.csv")

        if remove_missing:
            num_rows = diabetes_df.shape[0]
            bank_df = diabetes_df.dropna()
            print("[Data Loader] Removing nans from Diabetes Failure", num_rows, "vs", bank_df.shape[0])

        diabetes_y = (diabetes_df["readmitted"]=="<30") | (diabetes_df["readmitted"]==">30")
        diabetes_X = diabetes_df.drop(["readmitted", "encounter_id", "patient_nbr"], axis=1)

        for col in diabetes_X.columns:
            diabetes_X[col] = pd.Categorical(diabetes_X[col]).codes

        aps_counter = Counter(diabetes_y)
        print("[Data Loader] Diabetes Class Balance:", aps_counter[1] / sum(aps_counter.values()), "\n")

        if verbose:
            self.print_data_info("Diabetes", diabetes_X, diabetes_y)

        diabetes_data = train_test_split(diabetes_X, diabetes_y, test_size=test_size, random_state=seed, stratify=diabetes_y)
        data_dict["Diabetes"+"_"+str(seed)] = diabetes_data

    def _load_aps_failure(self, data_dict, seed, test_size, remove_missing, verbose):
        aps_df = pd.read_csv(data_path + "UCI APS Failure/aps.csv")
        aps_df = aps_df.replace("na", -1)

        if remove_missing:
            num_rows = aps_df.shape[0]
            aps_df = aps_df.dropna()
            print("[Data Loader] Removing nans from APS Failure", num_rows, "vs", aps_df.shape[0])

        aps_y = aps_df["class"]=="pos"
        aps_X = aps_df.drop("class", axis=1)
        for col in aps_X.columns:
            aps_X[col] = aps_X[col].astype(np.float32)
        aps_counter = Counter(aps_y)
        print("[Data Loader] APS Class Balance:", aps_counter[1] / sum(aps_counter.values()), "\n")

        if verbose:
            self.print_data_info("APS", aps_X, aps_y)

        aps_data = train_test_split(aps_X, aps_y, test_size=test_size, random_state=seed, stratify=aps_y)
        data_dict["APS"+"_"+str(seed)] = aps_data

    def _load_bank_marketing(self, data_dict, seed, test_size, remove_missing, verbose):
        bank_df = pd.read_csv(data_path + "UCI Bank Marketing/bank-full.csv", sep=";")

        if remove_missing:
            num_rows = bank_df.shape[0]
            bank_df = bank_df.dropna()
            print("[Data Loader] Removing nans from Bank", num_rows, "vs", bank_df.shape[0])

        bank_y = bank_df["y"]=="yes"
        bank_X = bank_df.drop("y", axis=1)
        for col in bank_X.columns:
            bank_X[col] = pd.Categorical(bank_X[col]).codes

        aps_counter = Counter(bank_y)
        print("[Data Loader] Bank Class Balance:", aps_counter[1] / sum(aps_counter.values()), "\n")

        if verbose:
            self.print_data_info("Bank", bank_X, bank_y)

        aps_data = train_test_split(bank_X, bank_y, test_size=test_size, random_state=seed, stratify=bank_y)
        data_dict["Bank"+"_"+str(seed)] = aps_data

    def load_datasets(self, data_list, remove_missing=False, return_dict=False, verbose=False):
        test_size = 0.3
        data_dict = {}

        for dataset_name in data_list:
            for seed in self.seeds:
                # Credit 1
                if dataset_name == "Credit 1":
                    self._load_credit1(data_dict, seed, test_size, remove_missing, verbose)

                # Credit 1 Noised
                if dataset_name == "Credit 1 Noised":
                    credit1_df = pd.read_csv(data_path + "Kaggle Credit 1/credit1-training.csv")
                    credit1_df = credit1_df.drop('Unnamed: 0', axis=1)

                    if remove_missing:
                        num_rows = credit1_df.shape[0]
                        credit1_df = credit1_df.dropna()
                        print("[Data Loader] Removing nans from Credit 1", num_rows, "vs", credit1_df.shape[0])

                    credit1_y = credit1_df["SeriousDlqin2yrs"]
                    credit1_X = credit1_df.drop("SeriousDlqin2yrs", axis=1)
                    credit1_noised_X = credit1_X.copy()
                    for i in range(0, 40):
                        credit1_noised_X[i] = pd.Series(np.random.normal(0, 1000, credit1_noised_X.shape[0])).values

                    credit1_noised_data = train_test_split(credit1_noised_X, credit1_y, test_size=test_size, random_state=seed, stratify=credit1_y)
                    data_dict["Credit 1 Noised"+"_"+str(seed)] = credit1_noised_data

                if "synthetic" in dataset_name:
                    n = int(dataset_name.split("n=")[1].split("_")[0])
                    m = int(dataset_name.split("m=")[1].split("_")[0])
                    informative = int(dataset_name.split("informative=")[1].split("_")[0])

                    self._load_synthetic(data_dict, seed, test_size, num_samples=n, num_features=m, informative=informative)

                # Credit 2
                if dataset_name == "Credit 2":
                    self._load_credit2(data_dict, seed, test_size, remove_missing, verbose)

                # Credit Fraud
                if dataset_name == "Credit Card":
                    self._load_credit_card(data_dict, seed, test_size, remove_missing, verbose)

                # Adult
                if dataset_name == "adult":
                    self._load_adult(data_dict, seed, test_size, remove_missing, verbose)

                # Sensor
                if dataset_name == "sensor":
                    self._load_sensor(data_dict, seed, test_size, remove_missing, verbose)

                # Nomao
                if dataset_name == "nomao":
                   self._load_nomao(data_dict, seed, test_size, remove_missing, verbose)

                # HIGGS Dataset
                if dataset_name == "higgs":
                   self._load_higgs(data_dict, seed, test_size, remove_missing, verbose, load_sampled=False)

                if dataset_name == "higgs_sampled":
                   self._load_higgs(data_dict, seed, test_size, remove_missing, verbose, load_sampled=True)

                # APS Failure Dataset
                if dataset_name == "APS":
                   self._load_aps_failure(data_dict, seed, test_size, remove_missing, verbose)

                # Bank Marketing Dataset
                if dataset_name == "Bank":
                    self._load_bank_marketing(data_dict, seed, test_size, remove_missing, verbose)

                # Diabetes Datasets
                if dataset_name == "Diabetes":
                   self._load_diabetes(data_dict, seed, test_size, remove_missing, verbose)

                if dataset_name == "synthetic":
                    self._load_synthetic(data_dict, seed, test_size, verbose)

                # Load a pmlb dataset
                if dataset_name in ["breast_cancer_wisconsin", "hypothyroid", "tokyo1", "coil2000", "sonar", "spambase", "magic", "mushroom", "churn", # Binary classification
                             "537_houses", "1201_BNG_breastTumor", "564_fried", # regression
                             "nursery", "sleep", "connect_4", "mnist"]: # multi-class


                    self._load_pmlb_dataset(dataset_name, data_dict, seed, test_size, verbose)

        if return_dict:
            return data_dict
        else:
            return [data_dict[dataset] for dataset in data_list]