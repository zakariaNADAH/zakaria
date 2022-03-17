# -*- coding: utf-8 -*-
"""
Created on Tue Mar 15 00:13:22 2022

@author: asus
"""

import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import streamlit.components.v1 as stc


#st.set_page_config(page_title="Distillation Column Design")
hide_menu_style = """
    <style>
    #MainMenu {visibility: hidden; }
    footer {visibility: hidden;}
    </style>
    """
st.markdown(hide_menu_style, unsafe_allow_html=True)

col = st.columns(4)
#with col[0]:
#    st.image("Logo_UT3.JPG")
    
with col[3]:
    st.image("ensiacet.JPG")

st.markdown('---')


#st.sidebar.image('UT3.PNG')
st.sidebar.header("Définition des paramètres")

    

HTML_BANNER = """
    <h1 style="color:#DD985C;text-align:center;">Colonne de distillation</h1>
    <p style="color:#DD985C;text-align:center;">Méthode de MacCabe-Thiele</p>
    </div>
    """
stc.html(HTML_BANNER)
st.markdown("Le dimensionnement d'une colonne de distillation consiste à déterminer son diamètre et sa hauteur. Dans cet application, on va voir les étapes clés pour déterminer **la hauteur d'une colonne de distillation** par la méthode de **MacCabe-Thiele**.")


# =============================================================================
# Définir la fonction d'équilibre
# =============================================================================

# st.subheader("1. Définir la courbe d'équilibre")
    
# liste_équi = ["Volatilité relative", "Valeurs exp"]
# courbe_équi = st.radio('', liste_équi, index=0)

# if courbe_équi == liste_équi[0]:
alpha = st.sidebar.text_input("Volatilité relative", 2.5)
alpha = float(alpha)

# st.write("La courbe d'équilibre est déterminée par :")
# st.markdown(r'### <p style="text-align: center;">$$y_{eq}=\frac{\alpha x_{eq}}{1 + x_{eq}(\alpha - 1}$$</p>', unsafe_allow_html=True)
def equi(alpha):
    x_eq = np.linspace(0, 1, 101)
    y_eq = alpha*x_eq/(1+(alpha-1)*x_eq)
    return x_eq, y_eq

x_eq, y_eq = equi(alpha)

# else:
    
#     col_x = st.columns(6)
#     text_x, x_1, x_2, x_3, x_4, x_5 = col_x[0].text_input("", 'x(-)'), col_x[1].text_input("",0.1), col_x[2].text_input("",0.3), col_x[3].text_input("",0.5), col_x[4].text_input("",0.7), col_x[5].text_input("",0.9)
#     x_1, x_2, x_3, x_4, x_5 = float(x_1), float(x_2), float(x_3), float(x_4), float(x_5)
#     xx = [0, x_1, x_2, x_3, x_4, x_5, 1]
    
#     col_y = st.columns(6)
#     text_y, y_1, y_2, y_3, y_4, y_5 = col_y[0].text_input("", 'y(-)'), col_y[1].text_input("",0.21), col_y[2].text_input("",0.51), col_y[3].text_input("",0.72), col_y[4].text_input("",0.85), col_y[5].text_input("",0.96)
#     y_1, y_2, y_3, y_4, y_5 = float(y_1), float(y_2), float(y_3), float(y_4), float(y_5)
#     yy = [0, y_1, y_2, y_3, y_4, y_5, 1]

#     def interpolation(xx, yy):
#         cs=CubicSpline(xx,yy)
#         x_eq=np.arange(0.00001,1,0.00001)
#         y_eq=cs(x_eq)
#         return x_eq, y_eq
    
#     x_eq, y_eq = interpolation(xx, yy)
#     alpha = y_eq*(x_eq-1)/(y_eq*x_eq - x_eq)
#     alpha = alpha.mean()

st.write("**1. Le nombre minimum d'étages théoriques ($NET_{min}$ à Reflux total)**")
# =============================================================================
# Définir quelques paramétres
# =============================================================================
q = st.sidebar.number_input("La fraction du vapeur dans l'alimentation (q)", value= 1.0)
q = float(q)
if q==1:
    q=0.9999999999

X_F = st.sidebar.slider("Titre molaire de l'alimentation (en plus volatil XF)",min_value=0.0, max_value=100.0, step=0.1, value=40.0)
X_F = X_F/100
X_D = st.sidebar.slider("Titre molaire du distillat (en plus volatil XD)",min_value=0.0, max_value=100.0, step=0.1, value=95.5)
X_D = X_D/100
X_W = st.sidebar.slider("Titre molaire du résidu (en plus volatil XW)",min_value=0.0, max_value=100.0, step=0.1, value=06.0)
X_W = X_W/100


# =============================================================================
# Déterminer Rmin : il faut définir une fonction qui nous retourne le point 
# d'intersection entre la courbe d'alimentation et la courbe d'équilibre
# =============================================================================

def inter(q, X_F, alpha):
    c1 = (q*(alpha-1))
    c2 = q + X_F*(1-alpha) - alpha*(q-1)
    c3 = -X_F
    coeff = [c1, c2, c3]
    r = np.sort(np.roots(coeff))
    
    if r[0]>0:
        x_ae = r[0]
    else:
        x_ae = r[1]
   
    y_ae = alpha*x_ae/(1+ x_ae*(alpha-1))
    if q == 1:
        x_fed = [X_F, X_F]
        y_fed = [X_F, y_ae]
    else:
        x_fed = np.linspace(X_F, x_ae, 51)
        y_fed = q/(q-1)*x_fed - X_F/(q-1)
    
    return x_ae, y_ae, y_fed, x_fed
x_ae, y_ae, y_fed, x_fed = inter(q, X_F, alpha)

# =============================================================================
# NET min
# =============================================================================
R = 1000

x_inter = (X_F/(q-1)+X_D/(R+1))/(q/(q-1)-R/(R+1))
y_inter = R/(R+1)*x_inter + X_D/(R+1)

# =============================================================================
# Section de rectification : établissement de la courbe d'enrichissement
# =============================================================================

def rect(R, X_D, x_inter):
    x_rect = np.linspace(X_D, x_inter, 51)
    y_rect = R/(R+1)*x_rect +X_D/(R+1)
    return x_rect, y_rect

x_rect, y_rect = rect(R, X_D, x_inter)

# =============================================================================
# Section d'allimentation : établissement de la courbe d'allimentation
# =============================================================================

def alim(X_F, q, x_inter):
    x_alim = np.linspace(X_F, x_inter)
    y_alim = q/(q-1)*x_alim - X_F/(q-1)
    return x_alim, y_alim

x_alim, y_alim = alim(X_F, q, x_inter)

# =============================================================================
# Section d'appauvrissement : établissement de la courbe d'appauvrissement
# =============================================================================

def appau(X_W, x_inter, y_inter):
    x_appau = np.linspace(X_W, x_inter, 51)
    y_appau = (y_inter - X_W)/(x_inter - X_W) * (x_appau - X_W) +X_W
    return x_appau, y_appau
x_appau, y_appau = appau(X_W, x_inter, y_inter) 

# =============================================================================
# Construction des étages
# =============================================================================
s = np.zeros((1000,5)) # Empty array (s) to calculate coordinates of stages

for i in range(1,1000):
    # (s[i,0],s[i,1]) = (x1,y1) --> First point
    # (s[i,2],s[i,3]) = (x2,y2) --> Second point
    # Joining (x1,y1) and (x2,y2) will result into stages
    
    s[0,0] = X_D
    s[0,1] = X_D
    s[0,2] = s[0,1]/(alpha-s[0,1]*(alpha-1))
    s[0,3] = s[0,1]
    s[0,4] = 0
# x1
    s[i,0] = s[i-1,2]
    
    # Breaking step once (x1,y1) < (xW,xW)
    if s[i,0] < X_W:
        s[i,1] = s[i,0] 
        s[i,2] = s[i,0]
        s[i,3] = s[i,0]
        s[i,4] = i
        break
        # y1
    if s[i,0] > x_inter:
        s[i,1] = R/(R+1)*s[i,0] + X_D/(R+1)
    else :
        s[i,1] = ((y_inter - X_W)/(x_inter - X_W))*(s[i,0]-X_W) + X_W
        
    # x2
    if s[i,0] > X_W:
        s[i,2] = s[i,1]/(alpha-s[i,1]*(alpha-1))
    else:
        s[i,2] = s[i,0]
    
    # y2
    s[i,3] = s[i,1]
    
    # Nbr des étages
    if s[i,0] < x_inter:
        s[i,4] = i
    else:
        s[i,4] = 0

s = s[~np.all(s == 0, axis=1)] # Clearing up zero containing rows 
s_rows = s.shape[0] 

S = np.zeros((s_rows*2,2)) # Empty array to rearragne 's' array for plotting

for i in range(0,s_rows):
    S[i*2,0] = s[i,0]
    S[i*2,1] = s[i,1]
    S[i*2+1,0] = s[i,2]
    S[i*2+1,1] = s[i,3]

# =============================================================================
# Déterminier le nombre des étages théoriques
# =============================================================================
x_s = s[:,2:3]
y_s = s[:,3:4]

stages = np.char.mod('%d', np.linspace(1,s_rows-1,s_rows-1))

st.set_option('deprecation.showPyplotGlobalUse', False)

for label, x, y in zip(stages, x_s, y_s):
    plt.annotate(label,
                  xy=(x, y),
                  xytext=(0,5),
                  textcoords='offset points', 
                  ha='right')

plt.grid(linestyle='dotted')
#plt.title('Distillation Column Design (MacCabe-Thiele Method)')
# if courbe_équi == liste_équi[1]:
#     plt.scatter(xx, yy, marker='o', s=10)
plt.plot(x_eq,y_eq,'-', label="Courbe d'équilibre")
plt.plot([0, 1],[0, 1],'black')

plt.scatter(X_D,X_D, color='r', s=20)
plt.scatter(X_F,X_F, color='r', s=20)
plt.scatter(X_W,X_W, color='r', s=20)

plt.plot(S[:,0],S[:,1],'r-.', label="Etages")

plt.legend(loc="upper left")
plt.xlabel("x (-)")
plt.ylabel("y (-)")

st.pyplot()

stages_min = s_rows -1
#st.write("Le nombre des étages theoriques minimal pour réaliser cette séparation est:", étages_min,"étages")
st.write(r''' $$\hspace*{5.2cm} NET_{min} =$$''', stages_min)
if stages_min > 15:
    st.error("Attention!! le nombre des étages theoriques minimal est trop élevé. La distillation à ces conditions n'est pas raisonnable ")
else :
    st.success("Parfait! l'opération qu'on souhaite mise en oeuvre est raisonnable")




# =============================================================================
# Rmin
# =============================================================================
st.write(r'**2. Le taux de reflux minimum ($R_{min}$ à nombre de plateaux infini)**')

def Rmin(X_D, x_ae, y_ae):
    x_Rmin = np.linspace(X_D, 0, 51)
    y_Rmin = (y_ae - X_D)/(x_ae - X_D) * (x_Rmin - X_D) +X_D
    return x_Rmin, y_Rmin
x_Rmin, y_Rmin = Rmin(X_D, x_ae, y_ae) 

######## R_min & R (new) ########
R_min = (X_D-y_ae)/(y_ae - x_ae)
ordo = X_D/(R_min +1)

# =============================================================================
# plot
# =============================================================================

plt.grid(visible=True, which='major',linestyle=':',alpha=0.6)
plt.grid(visible=True, which='minor',linestyle=':',alpha=0.3)
plt.minorticks_on()
#plt.title('Distillation Column Design (MacCabe-Thiele Method)')

plt.plot(x_eq,y_eq,'-', label="Courbe d'équilibre")
plt.plot([0, 1],[0, 1],'black')

plt.scatter(X_D, X_D, color='r', s=20)
plt.scatter(X_F, X_F, color='r', s=20)
plt.scatter(x_ae, y_ae, color='r', s=20)

plt.plot(x_Rmin, y_Rmin, label="Courbe d'enrichissement")
plt.plot(x_fed,y_fed, label="Courbe d'alimentation")

plt.legend(loc="best")
plt.xlabel("x (-)")
plt.ylabel("y (-)")

plt.scatter(0,ordo, color='r', s=20)
plt.text(0.01,ordo-0.08,'($\\frac{X_{D}}{R_{min}+1}$)',horizontalalignment='center')
st.pyplot()

st.markdown(r'### <p style="text-align: center;">$$R_{min}=\frac{X_{D}}{Y_{min}}-1$$</p>', unsafe_allow_html=True)
st.write("$\hspace*{5.2cm} R_{min} =$",round(R_min,3))


st.write("**3. Le nombre d'étages théoriques NET requis pour un taux de reflux R :**")
col1 = st.columns(2)

with col1[1]:
    Coeff = st.slider("Coeff",min_value=1.0, max_value=2.0, step=0.01, value=1.21)
    R = Coeff*R_min

with col1[0]:
    st.write("Le taux de reflux réel $R$ est définit par :")
    st.write("$R = Coeff \\times R_{min}$")
    st.write("$R =$",round(R,3))


# =============================================================================
# le point d'intersection entre la courbe d'alimentation et la courbe d'enrichissement
# =============================================================================

x_inter = (X_F/(q-1)+X_D/(R+1))/(q/(q-1)-R/(R+1))
y_inter = R/(R+1)*x_inter + X_D/(R+1)

# =============================================================================
# Section de rectification : établissement de la courbe d'enrichissement
# =============================================================================

def rect(R, X_D, x_inter):
    x_rect = np.linspace(X_D, x_inter, 51)
    y_rect = R/(R+1)*x_rect +X_D/(R+1)
    return x_rect, y_rect

x_rect, y_rect = rect(R, X_D, x_inter)

# =============================================================================
# Section d'allimentation : établissement de la courbe d'allimentation
# =============================================================================

def alim(X_F, q, x_inter):
    x_alim = np.linspace(X_F, x_inter)
    y_alim = q/(q-1)*x_alim - X_F/(q-1)
    return x_alim, y_alim

x_alim, y_alim = alim(X_F, q, x_inter)

# =============================================================================
# Section d'appauvrissement : établissement de la courbe d'appauvrissement
# =============================================================================

def appau(X_W, x_inter, y_inter):
    x_appau = np.linspace(X_W, x_inter, 51)
    y_appau = (y_inter - X_W)/(x_inter - X_W) * (x_appau - X_W) +X_W
    return x_appau, y_appau
x_appau, y_appau = appau(X_W, x_inter, y_inter) 

# =============================================================================
# Construction des étages
# =============================================================================
s = np.zeros((1000,5)) # Empty array (s) to calculate coordinates of stages

for i in range(1,1000):
    # (s[i,0],s[i,1]) = (x1,y1) --> First point
    # (s[i,2],s[i,3]) = (x2,y2) --> Second point
    # Joining (x1,y1) and (x2,y2) will result into stages
    
    s[0,0] = X_D
    s[0,1] = X_D
    s[0,2] = s[0,1]/(alpha-s[0,1]*(alpha-1))
    s[0,3] = s[0,1]
    s[0,4] = 0
# x1
    s[i,0] = s[i-1,2]
    
    # Breaking step once (x1,y1) < (xW,xW)
    if s[i,0] < X_W:
        s[i,1] = s[i,0] 
        s[i,2] = s[i,0]
        s[i,3] = s[i,0]
        s[i,4] = i
        break
        # y1
    if s[i,0] > x_inter:
        s[i,1] = R/(R+1)*s[i,0] + X_D/(R+1)
    else :
        s[i,1] = ((y_inter - X_W)/(x_inter - X_W))*(s[i,0]-X_W) + X_W
        
    # x2
    if s[i,0] > X_W:
        s[i,2] = s[i,1]/(alpha-s[i,1]*(alpha-1))
    else:
        s[i,2] = s[i,0]
    
    # y2
    s[i,3] = s[i,1]
    
    # Nbr des étages
    if s[i,0] < x_inter:
        s[i,4] = i
    else:
        s[i,4] = 0

s = s[~np.all(s == 0, axis=1)] # Clearing up zero containing rows 
s_rows = s.shape[0] 

S = np.zeros((s_rows*2,2)) # Empty array to rearragne 's' array for plotting

for i in range(0,s_rows):
    S[i*2,0] = s[i,0]
    S[i*2,1] = s[i,1]
    S[i*2+1,0] = s[i,2]
    S[i*2+1,1] = s[i,3]

# =============================================================================
# Déterminier le nombre des étages théoriques
# =============================================================================
# (x2,y2) from 's' array as (x_s,y_s) used for stage numbering
x_s = s[:,2:3]
y_s = s[:,3:4]

stages = np.char.mod('%d', np.linspace(1,s_rows-1,s_rows-1))

NET = s_rows-1

# '''
# localiser l'étage d'alimentation
# '''
s_f = s_rows-np.count_nonzero(s[:,4:5], axis=0)

# =============================================================================
# FINALE
# =============================================================================
#st.set_option('deprecation.showPyplotGlobalUse', False)

fig = plt.figure(num=None, figsize=(10, 8))

for label, x, y in zip(stages, x_s, y_s):
    plt.annotate(label,
                  xy=(x, y),
                  xytext=(0,5),
                  textcoords='offset points', 
                  ha='right')

plt.grid(linestyle='dotted')
plt.title('Distillation Column Design (MacCabe-Thiele Method)')

plt.plot(x_eq,y_eq,'-', label="Courbe d'équilibre")
plt.plot([0, 1],[0, 1],'black')


plt.scatter(X_D,X_D, color='r' )
plt.scatter(X_F,X_F, color='r' )
plt.scatter(X_W,X_W, color='r' )

plt.scatter(x_inter,y_inter )
plt.plot(x_alim, y_alim, label="Courbe d'alimentation")
plt.plot(x_appau, y_appau, label="Courbe d'appauvrissement")
plt.plot(x_rect, y_rect, label="Courbe d'enrichissement")
# plt.plot(x_fed,y_fed, color='black' )

plt.plot(S[:,0],S[:,1],'-.', label="Etages")

plt.legend(loc="upper left")
plt.xlabel("x (-)")
plt.ylabel("y (-)")

st.pyplot()

st.write(r''' $$\hspace*{5.2cm} NET =$$''', s_rows -1)

st.write("**4. Hauteur de la colonne**")

menu = ["Colonne à plateaux","Colonne à garnissage"]
techno = st.selectbox("Technologies",menu)

st.markdown("La hauteur de la colonne résulte:")
st.write("$~~~~$- du nombre d'étages théoriques nécessaires")

if techno == menu[0]:
    st.write("$~~~~$- de l'efficacité de chaque plateau réel (eff)")
    st.write("$~~~~$- de l'espacement entre plateaux (TS pour Tray Spacing)")
    st.markdown(r'### <p style="text-align: center;">$$H=\frac{NET}{eff} \times TS$$</p>', unsafe_allow_html=True)
    
    col2 = st.columns(2)
    with col2[0]:
        eff = st.text_input("efficacité des plateaux (%) ", 90)
        eff = float(eff)
    
    with col2[1]:
        TS = st.text_input("espacement entre plateaux (m) ", 0.4)
        TS = float(TS)
    st.write("- Hauteur de la colonne =", round(NET/(eff/100)*TS,2),"m")
else:
    st.write("$~~~~$- de la hauteur equivalente à un plateau théorique (HEPT)")
    st.markdown(r'### <p style="text-align: center;">$$H=NET \times HEPT$$</p>', unsafe_allow_html=True)
    col2 = st.columns(2)
    with col2[0]:
        HEPT = st.text_input("Hauteur Equivalente à un Plateau Théorique (m)", 0.8)
        HEPT = float(HEPT)
    st.write("- Hauteur de garnissage à installer =",round(NET*HEPT,2),"m" )


Button = st.expander("Get In Touch With Me!")
with Button:
    col31, col32, col33 = st.columns(3)
    col31.write("[Zakaria NADAH](https://www.linkedin.com/in/zakaria-nadah-00ab81160/)")
    col31.write("Ingénieur Procédés Junior")
    col31.write("+336.28.80.13.40")
    col31.write("zakariaenadah@gmail.com")
    
    col33.image("profil_.jpg")
