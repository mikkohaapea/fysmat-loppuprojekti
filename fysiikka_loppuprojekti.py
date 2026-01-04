#required libraries
import streamlit as st
import pandas as pd
import folium
from streamlit_folium import st_folium
from scipy.signal import butter, filtfilt
import matplotlib.pyplot as plt
import numpy as np
from math import radians, cos, sin, asin, sqrt

#file paths for experiment data
ACCEL_DATA_PATH = "./Loppuprojekti/data/experimentAccelData.csv"
LOCATION_DATA_PATH = "./Loppuprojekti/data/experimentLocationData.csv"

df_a = pd.read_csv(ACCEL_DATA_PATH)
df_loc = pd.read_csv(LOCATION_DATA_PATH)

#rajataan datasta pois se osa, jonka aikana ei liikuttu (alusta ja lopusta)
df_a = df_a[(df_a['Time (s)'] > 24) & (df_a['Time (s)'] < 174)].reset_index(drop=True)
df_loc = df_loc[(df_loc['Time (s)'] > 24) & (df_loc['Time (s)'] < 174)].reset_index(drop=True)
#df_loc = df_loc.reset_index(drop=True)

#Jaksollinen liike havaitaan parhaiten kiihtyvyyden y-komponentista, joten käytetään sitä jatkokäsittelyssä

def butter_lowpass_filter(data, cutoff, nyq, order):
    # alipäästösuodatin, jolla suodatetaan y-komponentista askeltaajuutta korkeammat taajuudet
    normal_cutoff = cutoff / nyq
    # Get the filter coefficients
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    y = filtfilt(b,a,data)
    return y

y_accel = df_a['Linear Acceleration y (m/s^2)']
t_accel = df_a['Time (s)']

T_tot = t_accel.max() - t_accel.min()
n = len(t_accel) #datapisteiden lkm
fs = n / T_tot #näytteenottotaajuus (oletetaan vakioksi)
nyq = fs/2 #Nyqvistin taajuus (suurin taajuus, joka datasta voidaan havaita)
order = 3
cutoff = 3 #tätä suuremmat taajuudet alipäästösuodatin poistaa datasta

filtered_y = butter_lowpass_filter(y_accel, cutoff, nyq, order)

# Maksimien perusteella
steps = 0
for i in range(1, len(filtered_y)-1):
    if (filtered_y[i] > filtered_y[i-1] and filtered_y[i] > filtered_y[i+1] and filtered_y[i] > 1):
        #jos y:n arvo pisteessä i on suurempi kuin y:n arvo pisteissä (i-1) ja (i+1) --> i on maksimi
        #   (y:n arvo pisteessä i täytyy myös olla suurempi kuin 1, jotta kävelyvauhtien väliset tauot eivät nosta askelsummaa)
        steps = steps + 1

t_0 = t_accel - t_accel[t_accel.first_valid_index()] #kalibroidaan kiihtyvyysdata alkamaan ajanhetkestä t=0
N = len(y_accel)    #havaintojen lkm
dt = np.max(t_0)/N  #näytteenottoväli (oletuksena vakio)

fourier = np.fft.fft(y_accel, N) #fourier-muunnos
psd = fourier * np.conj(fourier) / N #tehospektri
freq = np.fft.fftfreq(N, dt)    #taajuudet
L = np.arange(1, int(N/2))  #negatiivisten taajuuksien ja nollataajuuden rajaus

# tehospektristä voidaan havaita kolme dominoivaa taajuutta, jotka todennäköisesti
# vastaavat kolmea eri kävelynopeutta mittauksen aikana (hidas kävely, nopea kävely, juoksu)

f_slow = freq[L][psd[L] == np.max(psd[L][freq[L] < 2])][0] #hitaan kävelyn taajuuspiikki on kuvaajan perusteella alle 2 Hz
f_medium = freq[L][psd[L] == np.max(psd[L][(freq[L] < 2.5) & (freq[L] > 2)])][0] #nopean kävelyn taajuuspiikki on kuvaajan perusteella yli 2Hz, mutta alle 2.5Hz
f_fast = freq[L][psd[L] == np.max(psd[L][freq[L] > 2.5])][0] #juoksun taajuuspiikki on kuvaajan perusteella yli 2.5Hz, mutta alle 3Hz

#hidas kävely:  ~24s ... 75s
#nopea kävely:  ~76s ... 128s
#      juoksu:  ~128s ... 174s

#Askeleeseen kuluva aika, eli jaksonaika (oletetaan, että dominoiva
# taajuus on askeltaajuus)
T_slow = 1/f_slow
T_medium = 1/f_medium
T_fast = 1/f_fast

#Askelmäärät eri kävelynopeuksilla
steps1 = int(f_slow * (75 - 24)) #hidas kävely tapahtuu aikavälillä 24s ... 75s
steps2 = int(f_medium * (128 - 76)) #nopea kävely tapahtuu aikavälillä 76s ... 128s
steps3 = int(f_fast * (174 - 128)) #juoksu tapahtuu aikavälillä 128s ... 174s

#Keskinopeus gps-datasta
avg_v = round(df_loc['Velocity (m/s)'].mean(), 3)

# kuljettu matka gps-datasta (käyttäen Haversinen kaavaa)
def haversine(lon1, lat1, lon2, lat2):
    """
    Calculate the great circle distance in kilometers between two points
    on the earth (specified in decimal degrees)
    """
    # convert decimal degrees to radians
    lon1, lat1, lon2, lat2 = map(radians, [lon1, lat1, lon2, lat2])
    # haversine formula
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = sin(dlat / 2) ** 2 + cos(lat1) * cos(lat2) * sin(dlon / 2) ** 2
    c = 2 * asin(sqrt(a))
    r = 6371 #radius of the Earth in kilometers
    return c * r

#alustetaan dataframeen uusi sarake (johon laskettu etäisyys tallennetaan)
df_loc['calculated_distance'] = np.zeros(len(df_loc))

#lasketaan välimatka havaintopisteiden välillä käyttäen for-looppia
for i in range(len(df_loc) - 1):
    lon1 = df_loc['Longitude (°)'][i]
    lon2 = df_loc['Longitude (°)'][i+1]
    lat1 = df_loc['Latitude (°)'][i]
    lat2 = df_loc['Latitude (°)'][i+1]
    df_loc.loc[i+1, 'calculated_distance'] = haversine(lon1, lat1, lon2, lat2)

#lasketaan kokonaismatka mittapisteiden välisestä matkasta
df_loc['total_distance'] = df_loc['calculated_distance'].cumsum()

#calculated_distance on ilmoitettu kilometreinä
total_dist = df_loc['total_distance'][df_loc.last_valid_index()]

#askelpituus
total_steps = steps1+steps2+steps3
stepLength = (total_dist / total_steps) * 1000 #muutetaan kilometrit metreiksi

st.title("Loppuprojekti")

st.write("Askelmäärä laskettuna suodatuksen avulla: {} askelta".format(steps))
st.write("Askelmäärä laskettuna Fourier-analyysin avulla: {} askelta".format(steps1 + steps2 + steps3))
st.write("Keskinopeus: {:.2f} m/s".format(avg_v))
st.write("Kokonaismatka: {} km".format(round(total_dist, 2)))
st.write("Askelpituus on {} cm".format(round(stepLength * 100, 1))) #muutetaan metrit senttimetreiksi

st.header("Kiihtyvyyden y-komponentti")
fig1 = plt.figure(figsize=(12,5))
plt.plot(t_accel,filtered_y)
plt.grid()
plt.xlabel("Aika [s]")
plt.ylabel("Kiihtyvyys [m/s^2]")
plt.title("Suodatettu kiihtyvyyden y-komponentti (koko väli)")
st.pyplot(fig=fig1)

fig2 = fig1
plt.xlim((23,76))
plt.ylim((-4, 4))
plt.title("Suodatettu kiihtyvyyden y-komponentti (hidas kävely)")
st.pyplot(fig=fig2)

fig3 = fig1
plt.xlim((76,128))
plt.ylim((-7.5, 7.5))
plt.title("Suodatettu kiihtyvyyden y-komponentti (nopea kävely)")
st.pyplot(fig=fig3)

fig4 = fig1
plt.xlim((128,175))
plt.ylim((-10, 10))
plt.title("Suodatettu kiihtyvyyden y-komponentti (juoksu)")
st.pyplot(fig=fig4)

st.header("Tehospektri")

fig5 = plt.figure(figsize=(15,6))
plt.plot(freq[L], psd[L].real)
plt.xlabel('Taajuus [Hz]')
plt.ylabel('Teho')
plt.xlim([0,5])
plt.ylim([0,20000])
plt.title("Kiihtyvyyden y-komponentin tehospektri")
st.pyplot(fig=fig5)

st.header("Kuljettu reitti kartalla")

start_lat = df_loc['Latitude (°)'].mean()
start_long = df_loc['Longitude (°)'].mean()
map = folium.Map(location = [start_lat, start_long], zoom_start = 16)

folium.PolyLine(df_loc[['Latitude (°)','Longitude (°)']], color='blue', weight=3.5, opacity=1).add_to(map)

st_map = st_folium(map, width=900, height=650)