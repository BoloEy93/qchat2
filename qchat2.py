import streamlit as st
import joblib
import pandas as pd
import random

st.write("Dépistage TSA - Q-CHAT-10")

# Gender input
gender = st.selectbox("Sexe de l'enfant ?", ["Garçon", "Fille"])

col1, col2, col3 = st.columns(3)

# Q-CHAT-10 Questions

q1 = col1.selectbox("1. Est-ce que votre enfant vous regarde lorsque vous l’appelez par son prénom?",
                    ["Toujours", "Habituellement", "Parfois", "Rarement", "Jamais"])

q2 = col2.selectbox("2. Est-ce que c'est facile d’établir un contact visuel avec votre enfant?",
                    ["Très facile", "Assez facile", "Parfois difficile", "Très difficile", "Impossible"])

q3 = col3.selectbox("3. Est-ce que votre enfant pointe du doigt pour demander quelque chose ou parce qu’il en a besoin?",
                    ["Oui, souvent", "Oui, parfois", "Rarement", "Jamais"])

q4 = col1.selectbox("4. Est-ce que votre enfant joue à faire semblant, comme par exemple faire semblant de parler au téléphone ou faire semblant de nourrir une poupée?",
                    ["Oui, souvent", "Oui, parfois", "Rarement", "Jamais"])

q5 = col2.selectbox("5. Est-ce que votre enfant regarde dans la même direction que vous lorsque vous regardez quelque chose?",
                    ["Oui, souvent", "Oui, parfois", "Rarement", "Jamais"])

q6 = col3.selectbox("6. Si vous êtes bouleversé ou triste, est-ce que votre enfant montre des signes de vouloir vous réconforter?",
                    ["Oui, souvent", "Oui, parfois", "Rarement", "Jamais"])

q7 = col1.selectbox("7. Est-ce que votre enfant utilise spontanément des gestes simples de communication, comme saluer de la main ou montrer du doigt?",
                    ["Oui, souvent", "Oui, parfois", "Rarement", "Jamais"])

q8 = col2.selectbox("8. Est-ce que les premiers mots de votre enfant étaient clairs et utilisés de façon appropriée?",
                    ["Oui, très clair", "Plutôt clair", "Légèrement inhabituel", "Très inhabituel", "Mon enfant ne parle pas"])

q9 = col3.selectbox("9. Est-ce que votre enfant vous montre des objets juste pour partager son intérêt, pas parce qu'il en a besoin?",
                    ["Oui, souvent", "Oui, parfois", "Rarement", "Jamais"])

q10 = col1.selectbox("10. Est-ce que votre enfant montre une réaction inhabituelle à des bruits forts ou des objets en mouvement?",
                     ["Oui, souvent", "Oui, parfois", "Rarement", "Jamais"])

# Create a DataFrame to hold the answers
df_pred = pd.DataFrame([[gender, q1, q2, q3, q4, q5, q6, q7, q8, q9, q10]],
                       columns=['gender', 'q1', 'q2', 'q3', 'q4', 'q5', 'q6', 'q7', 'q8', 'q9', 'q10'])

# Transforming categorical inputs into numerical values
def transform_response(response):
    mapping = {
        "Toujours": 0, "Habituellement": 1, "Parfois": 2, "Rarement": 3, "Jamais": 4,
        "Très facile": 0, "Assez facile": 1, "Parfois difficile": 2, "Très difficile": 3, "Impossible": 4,
        "Oui, souvent": 0, "Oui, parfois": 1, "Rarement": 2, "Jamais": 3,
        "Oui, très clair": 0, "Plutôt clair": 1, "Légèrement inhabituel": 2, "Très inhabituel": 3, "Mon enfant ne parle pas": 4
    }
    return mapping.get(response, 3)

# Apply transformations
df_pred['q1'] = df_pred['q1'].apply(transform_response)
df_pred['q2'] = df_pred['q2'].apply(transform_response)
df_pred['q3'] = df_pred['q3'].apply(transform_response)
df_pred['q4'] = df_pred['q4'].apply(transform_response)
df_pred['q5'] = df_pred['q5'].apply(transform_response)
df_pred['q6'] = df_pred['q6'].apply(transform_response)
df_pred['q7'] = df_pred['q7'].apply(transform_response)
df_pred['q8'] = df_pred['q8'].apply(transform_response)
df_pred['q9'] = df_pred['q9'].apply(transform_response)
df_pred['q10'] = df_pred['q10'].apply(transform_response)

# Gender transformation
df_pred['gender'] = df_pred['gender'].apply(lambda x: 1 if x == 'Garçon' else 0)

#st.write("Réponses converties :")
#st.write(df_pred)

model = joblib.load('autism_rf_model.pkl')
prediction = model.predict(df_pred)

#prediction = random.choice([0, 1])

if st.button('Predire'):
    
# Display prediction result
    st.write("Prédiction : ")
    if prediction == 1:
          st.write("Le résultat indique un risque potentiel d'autisme. Veuillez consulter un spécialiste.")
    else:
          st.write("Le résultat ne montre pas de signes clairs de risque d'autisme.")



#    if(prediction[0]==0):
#        st.write('<p class="big-font">Vous n etes probablement pas atteint d Autisme.</p>',unsafe_allow_html=True)
#    else:
#        st.write('<p class="big-font">Vous etes probablement atteint d Autisme.</p>',unsafe_allow_html=True)

