import streamlit as st
from perceptron import train, predict

class Utils:
    @staticmethod
    def display_training_data(X_train, y_train):
        st.header("Données d'entraînement")
        st.write("""
        Nous utilisons un ensemble de données d'entraînement simple pour démontrer le fonctionnement du perceptron.
        """)
        st.dataframe("X_train:", X_train)
        st.dataframe("y_train:", y_train)   

    @staticmethod
    def display_prediction_results(X_test, y_pred):
        st.header("Résultats des prédictions")
        st.write("""
        Voici les résultats des prédictions effectuées par le perceptron sur les données de test.
        """)
        st.dataframe("X_test:", X_test)
        st.dataframe("y_pred:", y_pred)
