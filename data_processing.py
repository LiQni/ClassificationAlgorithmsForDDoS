import pandas as pd
import numpy as np
from datetime import datetime


# IP转数字
def ip_to_int(ip):
    parts = ip.split('.')
    if len(parts) != 4:
        raise ValueError("Invalid IP address format")
    try:
        parts = [int(part) for part in parts]
        for part in parts:
            if part < 0 or part > 255:
                raise ValueError("Each part of the IP address must be between 0 and 255")
    except ValueError:
        raise ValueError("Invalid IP address format")
    ip_int = (parts[0] << 24) + (parts[1] << 16) + (parts[2] << 8) + parts[3]
    return ip_int


# 时间转数字
def time_to_float(time):
    datetime_obj = datetime.strptime(time, "%Y-%m-%d %H:%M:%S.%f")
    timedelta_obj = datetime_obj - datetime(1970, 1, 1)
    timestamp_s = timedelta_obj.total_seconds()
    return timestamp_s


def data_preprocessing(filepath):
    df = pd.read_csv(filepath).drop(['Flow ID', 'SimillarHTTP'], axis=1)
    df.columns = df.columns.str.strip()
    df = df[~df.isin([np.nan, np.inf, -np.inf]).any(axis=1)]  # 去除空值、无穷大
    df['Source IP'] = df['Source IP'].astype(str)
    df['Destination IP'] = df['Destination IP'].astype(str)
    df['Source IP'] = df['Source IP'].apply(ip_to_int)
    df['Destination IP'] = df['Destination IP'].apply(ip_to_int)
    df['Timestamp'] = df['Timestamp'].apply(time_to_float)
    df.Label.loc[df.Label == "BENIGN"] = 0
    df.Label.loc[df.Label != 0] = 1
    return df


def import_data():
    print("LDAP")
    LDAP = data_preprocessing("D:\\dev\\DDoS\\data\\03-11\\LDAP.csv")

    print("MSSQL")
    MSSQL = data_preprocessing("D:\\dev\\DDoS\\data\\03-11\\MSSQL.csv")

    print("NetBIOS")
    NetBIOS = data_preprocessing("D:\\dev\\DDoS\\data\\03-11\\NetBIOS.csv")

    print("Syn")
    Syn = data_preprocessing("D:\\dev\\DDoS\\data\\03-11\\Syn.csv")

    print("UDP")
    UDP = data_preprocessing("D:\\dev\\DDoS\\data\\03-11\\UDP.csv")

    print("UDPLag")
    UDPLag = data_preprocessing("D:\\dev\\DDoS\\data\\03-11\\UDPLag.csv")

    print("Portmap")
    Portmap = data_preprocessing("D:\\dev\\DDoS\\data\\03-11\\Portmap.csv")

    df = LDAP.append([MSSQL, NetBIOS, Syn, UDP, UDPLag, Portmap])

    df.to_csv("ddos_dataset2.csv", index=False)
    print("finish")


import_data()