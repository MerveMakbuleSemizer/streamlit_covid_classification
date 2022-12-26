import streamlit as st
import pandas as pd
import numpy as np

st.set_page_config(
    page_title="Covid Survive Classifier",
    page_icon="https://upload.wikimedia.org/wikipedia/commons/8/82/SARS-CoV-2_without_background.png"
    )

st.title("Covid Survive Classifier Project")

st.markdown("This website was created to predict whether a patient will succumb to Covid-19 disease based on some current symptoms and medical history. The model was trained by the dataset provided by the Mexican government.")
st.image("http://basiskele.meb.gov.tr/meb_iys_dosyalar/2020_08/06012704_842020140246coronavirus.jpg")

st.header("Metadata")
# df = pd.read_csv("Covid Data.csv")
# st.table(df.sample(5, random_state=42))
st.markdown("[The dataset](https://datos.gob.mx/busca/dataset/informacion-referente-a-casos-covid-19-en-mexico) was provided by the Mexican government. This dataset contains an enormous number of anonymized patient-related information including pre-conditions. The raw dataset consists of 21 unique features and 1,048,576 unique patients.") 
st.markdown("**- sex:** of the patient")
st.markdown("**- age:** of the patient.")
st.markdown("**- classification:** covid test findings. ")
st.markdown("**- patient type:** type of care the patient received in the unit. returned home or hospitalization.")
st.markdown("**- pneumonia:** whether the patient already have air sacs inflammation or not.")
st.markdown("**- pregnancy:** whether the patient is pregnant or not.")
st.markdown("**- diabetes:** whether the patient has diabetes or not.")
st.markdown("**- copd:** Indicates whether the patient has Chronic obstructive pulmonary disease or not.")
st.markdown("**- asthma:** whether the patient has asthma or not.")
st.markdown("**- inmsupr:** whether the patient is immunosuppressed or not.")
st.markdown("**- hypertension:** whether the patient has hypertension or not.")
st.markdown("**- cardiovascular:** whether the patient has heart or blood vessels related disease.")
st.markdown("**- renal chronic:** whether the patient has chronic renal disease or not.")
st.markdown("**- other disease:** whether the patient has other disease or not.")
st.markdown("**- obesity:** whether the patient is obese or not.")
st.markdown("**- tobacco:** whether the patient is a tobacco user.")
st.markdown("**- usmr:** Indicates whether the patient treated medical units of the first, second or third level.")
st.markdown("**- medical unit:** type of institution of the National Health System that provided the care.")
st.markdown("**- intubed:** whether the patient was connected to the ventilator.")
st.markdown("**- icu:** Indicates whether the patient had been admitted to an Intensive Care Unit.")
st.markdown("**- date died:** If the patient died indicate the date of death, and 9999-99-99 otherwise.")

st.sidebar.markdown("**Choose** the features below to see the result!")

#sidebar
sex = st.sidebar.selectbox("Sex",('Female', 'Male'))
age = st.sidebar.number_input("Age",min_value=0,format="%d")
usmer = st.sidebar.selectbox("USMER",(1,2),help = "Indicates whether the patient treated medical units of the first or second level.")
patient_type = st.sidebar.selectbox("Patient Type",("Returned Home","Hospitalization"))
pneumonia = st.sidebar.checkbox("Pneumonia",help="Whether the patient already have air sacs inflammation or not.")
if sex == "Male":
    pregnancy = False
else:
    pregnancy = st.sidebar.checkbox("Pregnancy")
diabetes = st.sidebar.checkbox("Diabetes")
copd = st.sidebar.checkbox("Copd",help="Indicates whether the patient has Chronic obstructive pulmonary disease or not.")
asthma= st.sidebar.checkbox("Asthma")
inmsupr = st.sidebar.checkbox("INMSUPR",help="whether the patient is immunosuppressed or not.")
hipertension = st.sidebar.checkbox("Hipertension")
cardiovascular = st.sidebar.checkbox("Cardiovascular",help="whether the patient has heart or blood vessels related disease.")
other_disease = st.sidebar.checkbox("Other Disease")
obesity = st.sidebar.checkbox("Obesity")
renal_chronic = st.sidebar.checkbox("Renal Chronic",help="whether the patient has chronic renal disease or not.")
tobacco = st.sidebar.checkbox("Tobacco")
classification = st.sidebar.checkbox("PCR Test Positive")


#model import
from joblib import load

model = load('decision_tree_oversampling.pkl')


input_df = pd.DataFrame({
    'age': [age],
    "usmer":[usmer],
    'sex': [sex],
    "patient_type":[patient_type],
    "pneumonia":[pneumonia],
    "pregnant":[pregnancy],
    "diabetes":[diabetes],
    "copd":[copd],
    "asthma":[asthma],
    "inmsupr":[inmsupr],
    "hipertension":[hipertension],
    "other_disease":[other_disease],
    "cardiovascular":[cardiovascular],
    "obesity":[obesity],
    "renal_chronic":[renal_chronic],
    "tobacco":[tobacco],
    "classification":[classification]
})
# input_df.columns = [column.upper() for column in input_df.columns]
input_df.sex = input_df.sex.replace(["Female","Male"],[0,1])
input_df["age"] = [1 if 0 <= age & age <= 40 else (2 if 41 <= age & age <= 80 else 3) for age in input_df.age]
input_df.patient_type = input_df.patient_type.replace(["Returned Home","Hospitalization"],[1,0])
input_df = input_df.replace([True,False],[1,0])

# input_df = pd.get_dummies(input_df,columns = ["usmer","sex","patient_type","pneumonia","pregnant","diabetes","copd","asthma","inmsupr","hipertension","other_disease","cardiovascular","obesity","renal_chronic","tobacco","classification"],drop_first = True)
# input_df = pd.get_dummies(input_df,columns = ["USMER","SEX","PATIENT_TYPE","PNEUMONIA","PREGNANT","DIABETES","COPD","ASTHMA","INMSUPR","HIPERTENSION","OTHER_DISEASE","CARDIOVASCULAR","OBESITY","RENAL_CHRONIC","TOBACCO","CLASSIFICATION"],drop_first = True)
# st.markdown(input_df.values)
pred = model.predict(input_df.values)
pred_probability = np.round(model.predict_proba(input_df.values), 2)

#results
st.header("Results")

# Sonuç Ekranı
if st.sidebar.button("Submit"):

    # Info mesajı oluşturma
    st.info("You can find the result below.")

    # Sonuçları Görüntülemek için DataFrame
    results_df = pd.DataFrame({
    'Prediction': [pred],
    'Probability of Survival': [pred_probability[:,:1]],
    'Probability of Death': [pred_probability[:,1:]]
    })

    results_df["Prediction"] = results_df["Prediction"].apply(lambda x: str(x).replace("0","Alive"))
    results_df["Prediction"] = results_df["Prediction"].apply(lambda x: str(x).replace("1","Dead"))

    st.table(results_df)

    if pred == 0:
        st.image("https://www.pngplay.com/wp-content/uploads/9/Life-PNG-Free-File-Download.png")
    else:
        st.image("https://www.nicepng.com/png/full/11-117450_rip-gravestone-dessin-tombe.png")
else:
    st.markdown("Please click the *Submit Button*!")