import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from perceptron import train, predict


st.sidebar.title("Perceptron Binaire")
st.sidebar.header("Paramètres d'entraînement")

learning_rate = st.sidebar.number_input(
    "Taux d'apprentissage (η)",
    min_value=0.0001,
    max_value=1.0,
    value=0.1,
    step=0.0001,
    format="%.4f"
)

epochs = st.sidebar.slider(
    "Nombre d'époques",
    min_value=10,
    max_value=10000,
    value=1000,
    step=50
)


st.title("Perceptron : Classification bineaire")
st.write("""
Ce projet implémente un perceptron simple pour la classification binaire en utilisant la régression logistique.
""")

st.header("Données d'entraînement")
st.write("""
Nous utilisons un ensemble de données d'entraînement simple répresentant deux classes d'Iris setosa et Versicolor pour démontrer le fonctionnement du perceptron.
""")
df =pd.read_csv("iris/iris_100_2d.csv")
st.dataframe(df.head(5))

st.subheader("nettoyage des données et préparation")
st.write("""
Nous extrayons les caractéristiques pertinentes et convertissons les étiquettes de classe en valeurs binaires. car le perceptron est conçu pour la classification binaire.
""")

df["label"]=df["label"].apply(lambda x: 1 if x=="Setosa" else 0)

st.dataframe(df.head(5))

X= df[["petal_length","petal_width"]].to_numpy()
y= df["label"].to_numpy() # Converti les étiquettes en valeurs binaires


st.header("Entraînement du Perceptron")
st.write("""
Nous entraînons le perceptron en utilisant les données préparées. Le modèle ajuste ses poids en fonction du taux d'apprentissage et du nombre d'iterations spécifiés. 
""")

col1, col2 = st.columns(2)
B = st.button("🎞️ Animation de l'entraînement")

with col1:
    st.subheader("Entraînement du Perceptron")
    weights, bias, cost = train(X, y, learning_rate, epochs)

    final_W = weights[-1]     
    final_b = bias[-1]   

    fig2, ax2 = plt.subplots()
    ax2.set_xlabel("Époques")
    ax2.set_ylabel("Perte")
    ax2.plot(range(epochs), cost, color='blue', label='Perte')
    ax2.set_title("Courbe de perte")
    st.pyplot(fig2)
    plt.close(fig2)

with col2:
    st.subheader("Données + Frontière de décision")

    # Placeholder pour l’animation
    plot_placeholder = st.empty()

    # Si l’utilisateur active l’animation
    if B:

        for i in range(0, epochs, max(1, epochs // 100)):

            w1 = weights[i][0][0]
            w2 = weights[i][1][0]
            b  = bias[i]

            x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
            x_line = np.linspace(x_min, x_max, 200)
            y_line = -(w1 * x_line + b) / w2

            fig, ax = plt.subplots()
            scatter = ax.scatter(X[:, 0], X[:, 1], c=y, cmap='bwr', alpha=0.7)
            ax.plot(x_line, y_line, "--", linewidth=2, label="Frontière")
    

            ax.set_xlabel("Longueur des pétales")
            ax.set_ylabel("Largeur des pétales")
            ax.set_title(f"Époque {i+1}")
            ax.legend()

            plot_placeholder.pyplot(fig)

        st.success("Animation terminée !")

    else:
        # Frontière finale
        w1 = final_W[0][0]
        w2 = final_W[1][0]
        b  = final_b

        x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
        x_line = np.linspace(x_min, x_max, 200)
        y_line = -(w1 * x_line + b) / w2

        fig, ax = plt.subplots()
        scatter = ax.scatter(X[:, 0], X[:, 1], c=y, cmap='bwr', alpha=0.7)
        ax.plot(x_line, y_line, "--", linewidth=2)
        ax.set_xlabel("Longueur des pétales")
        ax.set_ylabel("Largeur des pétales")
        st.pyplot(fig)
        plt.close(fig)
