"""Bonjour Monsieur,Voici notre programme permettant de comparer toutes les méthodes entre-elles"""
"""Ici nous avons choisi avec certains de nos camarades les mêmes fonctions à étudier afin de pouvoir comparer nos résultats entre nous"""


from math import *
import numpy as np
import matplotlib.pyplot as plt


"""Voici notre fonction établie lors des TP précédents nous permettant de calculer les solutions avec la méthode du Point fixe""" 

def PointFixe(g,x0,epsilon,NiterMax):
    n=1 
    xold=x0
    erreur= g(xold) - xold
    xnew=g(xold) 
    while abs(erreur) >= epsilon and n < NiterMax:
        n = n+1
        xnew=g(xold)
        erreur =(xnew-xold)
        xold=xnew
    return(xnew,n,abs(xnew-xold))

"""Voici notre fonction établie lors des TP précédents nous permettant de calculer les solutions avec la méthode de Newton""" 

def Newton(f,fprim,x0,epsilon,NiterMax):
    n=1 
    xold=x0
    xnew=xold-(f(xold))/(fprim(xold))
    erreur= xnew - xold
    while abs(erreur) >= epsilon and n < NiterMax:
        n = n+1
        xnew=xold-(f(xold))/(fprim(xold)) 
        erreur =(xnew-xold)
        xold=xnew
    return(xnew,n,abs(xnew-xold))

"""Voici notre fonction établie lors de ce TP permettant de calculer les solutions avec la méthode de Dichotomie""" 

def Dichotomie(f,a0,b0,epsilon,Nitermax): 
    n=1
    x0 , x1 =min(a0,b0) , max(a0,b0)
    while abs(x1-x0) >=  epsilon and n < Nitermax:
        n=n+1
        c0= (x0+x1)/2
        if f(c0) == 0:
            return c0
        elif f(x0)*f(c0) < 0:
            x1=c0
        else:
            x0=c0
    return ((x0+x1)/2,n)

"""Voici notre fonction établie lors de ce TP permettant de calculer les solutions avec la méthode de la sécante""" 

def Secante(f,x0,x1,epsilon,Nitermax):
    n=1
    while abs(x1-x0) > epsilon and n < Nitermax:
        n=n+1
        x2=x0-f(x0)*(x1-x0)/(f(x1)-f(x0))
        x0=x1
        x1=x2
    return (x0,n)

"""Rappelons les fonctions que nous avons décider d'étudier pour comparer les méthodes drésolutions entre-elles"""
def g3(x):
    return log(7/x)

def g6(x):
    return log((x**2.0)+3)

def g7(x):
    return (7-4*log(x))/3

def g9(x):
    return log(2*sin(x)+7)

def g10(x):
    return log(10)-log(log(x**2+4))
def f3(x):
    return x*exp(x) - 7

def f6(x):
    return exp(x)- x**2 - 3

def f7(x):
    return 3*x + 4*log(x) - 7

def f9(x):
    return exp(x) - 2*sin(x) - 7

def f10(x):
    return (log(x**2 + 4))*exp(x) - 10

def f3prim(x):
    return exp(x)*(x+1)

def f6prim(x):
    return exp(x)-2*x

def f7prim(x):
    return 3 + (4/x)

def f9prim(x):
    return exp(x)-2*cos(x)

def f10prim(x):
    return ((2*x)/(x**2+4))*exp(x) + exp(x)*(log(x**2+4))


"""Voici les fonctions necessaires à la comparaison des méthodes """

print ("="*5,"Exercice 1","="*5)

def Liste_Dichotomie_n(f,a0,b0,epsilon,Nitermax): 
    n=0
    liste_n=[]
    liste_x0=[]
    liste_erreur=[]
    gamma=Dichotomie(f,1,2,1E-14,1E4)
    print(gamma[0])
    x0 , x1 =min(a0,b0) , max(a0,b0)
    while abs(x1-x0) >=  epsilon and n < Nitermax:
        n=n+1
        c0= (x0+x1)/2
        if f(c0) == 0:
            return c0
        elif f(x0)*f(c0) < 0:
            x1=c0
        else:
            x0=c0
        liste_n.append(n)
        liste_x0.append(c0)
        liste_erreur.append(c0-gamma[0])
    return liste_n

def Liste_Dichotomie_Erreur(f,a0,b0,epsilon,Nitermax): 
    n=0
    liste_n=[]
    liste_x0=[]
    liste_erreur=[]
    gamma=Dichotomie(f,1,2,1E-14,1E4)
    print(gamma[0])
    x0 , x1 =min(a0,b0) , max(a0,b0)
    while abs(x1-x0) >=  epsilon and n < Nitermax:
        n=n+1
        c0= (x0+x1)/2
        if f(c0) == 0:
            return c0
        elif f(x0)*f(c0) < 0:
            x1=c0
        else:
            x0=c0
        liste_n.append(n)
        liste_x0.append(c0)
        liste_erreur.append(c0-gamma[0])
    return liste_erreur

def Liste_Secante_n(f,x0,x1,epsilon,Nitermax):
    n=0
    liste_n=[]
    liste_x0=[]
    liste_erreur=[]
    delta=Secante(f,1,2,1E-14,1E4)
    print(delta[0])
    while abs(x1-x0) > epsilon and n < Nitermax:
        n=n+1
        x2=x0-f(x0)*(x1-x0)/(f(x1)-f(x0))
        x0=x1
        x1=x2
        liste_n.append(n)
        liste_x0.append(x2)
        liste_erreur.append(x2-delta[0])
    return liste_n

def Liste_Secante_Erreur(f,x0,x1,epsilon,Nitermax):
    n=0
    liste_n=[]
    liste_x0=[]
    liste_erreur=[]
    delta=Secante(f,1,2,1E-14,1E4)
    print(delta[0])
    while abs(x1-x0) > epsilon and n < Nitermax:
        n=n+1
        x2=x0-f(x0)*(x1-x0)/(f(x1)-f(x0))
        x0=x1
        x1=x2
        liste_n.append(n)
        liste_x0.append(x2)
        liste_erreur.append(x2-delta[0])
    return liste_erreur

def Liste_Newton_n(f,fprim,x0,epsilon,NiterMax):
    n=0 
    xold=x0
    xnew=xold-(f(xold))/(fprim(xold))
    erreur= xnew - xold
    liste_n=[]
    liste_x0=[]
    liste_erreur=[]
    beta=Newton(f,fprim,1,1E-14,1E4)
    print(beta[0])
    while abs(erreur) >= epsilon and n < NiterMax:
        n = n+1
        xnew=xold-(f(xold))/(fprim(xold)) 
        erreur =(xnew-xold)
        xold=xnew
        liste_n.append(n)
        liste_x0.append(xnew)
        liste_erreur.append(xnew-beta[0])
    return liste_n

def Liste_Newton_Erreur(f,fprim,x0,epsilon,NiterMax):
    n=0 
    xold=x0
    xnew=xold-(f(xold))/(fprim(xold))
    erreur= xnew - xold
    liste_n=[]
    liste_x0=[]
    liste_erreur=[]
    beta=Newton(f,fprim,1,1E-14,1E4)
    print(beta[0])
    while abs(erreur) >= epsilon and n < NiterMax:
        n = n+1
        xnew=xold-(f(xold))/(fprim(xold)) 
        erreur =(xnew-xold)
        xold=xnew
        liste_n.append(n)
        liste_x0.append(xnew)
        liste_erreur.append(xnew-beta[0])
    return liste_erreur

def Liste_PointFixe_n(g,x0,epsilon,NiterMax):
    n=0
    xold=x0
    erreur= g(xold) - xold
    xnew=g(xold) 
    liste_n=[]
    liste_x0=[]
    liste_erreur=[]
    alpha=PointFixe(g,1,1E-14,1E4)
    print(alpha[0])
    while abs(erreur) >= epsilon and n < NiterMax:
        n = n+1
        xnew=g(xold)
        erreur =(xnew-xold)
        xold=xnew
        liste_n.append(n)
        liste_x0.append(xnew)
        liste_erreur.append(xnew-alpha[0])
    return liste_n

def Liste_PointFixe_Erreur(g,x0,epsilon,NiterMax):
    n=0
    xold=x0
    erreur= g(xold) - xold
    xnew=g(xold) 
    liste_n=[]
    liste_x0=[]
    liste_erreur=[]
    alpha=PointFixe(g,1,1E-14,1E4)
    print(alpha[0])
    while abs(erreur) >= epsilon and n < NiterMax:
        n = n+1
        xnew=g(xold)
        erreur =(xnew-xold)
        xold=xnew
        liste_n.append(n)
        liste_x0.append(xnew)
        liste_erreur.append(xnew-alpha[0])
    return liste_erreur

"""Dans la partie suivante, le programme calcule les solutions ainsi que le nombre d'itérations pour chaque méthode"""

print ("="*5,"Exercice 2","="*5)

def f(x):
    return (2*x) - (1 + sin (x))

def fprim(x):
    return 2 + cos(x)

def g(x):
    return (1 + sin(x)) / 2


print("_"*20)

print("Avec Dichotomie:")
c = Dichotomie(f, 0, 1, 1E-10, 1E4)
print(c)

print("_"*20)

print(" Avec Sécante:")
d = Secante(f, 0, 1, 1E-10, 1E4)
print(d)

print("_"*20)

print("Avec Newton:")
b = Newton(f, fprim, 0, 1E-10, 1E4)
print(b)

print("_"*20)

print("Avec point fixe:")
a = PointFixe(g, 0, 1E-10, 1E4)
print(a)

print("_"*20)

print ("="*5,"Exercice 3","="*5)

print(" Avec Dichotomie:")

a1 = Dichotomie(f3, 1, 2, 1E-10, 1E4)
print("Question 3:",a1)
b1 = Dichotomie(f6, 1, 2, 1E-10, 1E4)
print("Question 6:",b1)
c1 = Dichotomie(f7, 1, 2, 1E-10, 1E4)
print("Question 7:",c1)
d1 = Dichotomie(f9, 1, 2, 1E-10, 1E4)
print("Question 9:",d1)
e1 = Dichotomie(f10, 1, 2, 1E-10, 1E4)
print("Question 10:",e1)

print("_"*20)

print("Avec Sécante:")

a2 = Secante(f3, 1, 2, 1E-10, 1E4)
print("Question 3:",a2)
b2 = Secante(f6, 1, 2, 1E-10, 1E4)
print("Question 6:",b2)
c2 = Secante(f7, 1, 2, 1E-10, 1E4)
print("Question 7:",c2)
d2 = Secante(f9, 1, 2, 1E-10, 1E4)
print("Question 9:",d2)
e2 = Secante(f10,1,2, 1E-10, 1E4)
print("Question 10:",e2)

print("_"*20)

print("Avec Newton:")

a3 = Newton(f3, f3prim,1, 1E-10, 1E4)
print("Question 3:",a3)
b3 = Newton(f6, f6prim,1, 1E-10, 1E4)
print("Question 6:",b3)
c3 = Newton(f7, f7prim,1, 1E-10, 1E4)
print("Question 7:",c3)
d3 = Newton(f9, f9prim,1, 1E-10, 1E4)
print("Question 9:",d3)
e3 = Newton(f10, f10prim,1, 1E-10, 1E4)
print("Question 10:",e3)


print("_"*20)

print("Avec Point Fixe:")

a4=PointFixe(g3,1,1E-10,1E4)
print("Question 3:",a4)
b4=PointFixe(g6,1,1E-10,1E4)
print("Question 6:",b4)
c4=PointFixe(g7,1,1E-10,1E4)
print("Question 7:",c4)
d4=PointFixe(g9,1,1E-10,1E4)
print("Question 9:",d4)
e4=PointFixe(g10,1,1E-10,1E4)
print("Question 10:",e4)


"""La partie finale est la partie réalisée avec Matplotlib permettant de tracer les graphes"""
"""Elle complète l'exercice 2 et 3 et repond aux dernieres questions du TP"""

    
x1 = np.array(Liste_Dichotomie_n(f, 1, 2, 1E-10, 1E4))
y1 = np.array(Liste_Dichotomie_Erreur(f, 0, 1, 1E-10, 1E4))

x2 = np.array(Liste_Secante_n(f, 1, 2, 1E-10, 1E4))
y2 = np.array(Liste_Secante_Erreur(f, 1, 2, 1E-10, 1E4))

x3 = np.array(Liste_Newton_n(f, fprim, 1, 1E-10, 1E4))
y3 = np.array(Liste_Newton_Erreur(f, fprim, 1, 1E-10, 1E4))

x4= np.array(Liste_PointFixe_n(g, 1, 1E-10, 1E4))
y4= np.array(Liste_PointFixe_Erreur(g, 1, 1E-10, 1E4))

plt.figure(figsize=(12,8))
plt.subplot(2,1,1)
plt.grid()
plt.semilogy(x1,y1,label="Dichotomie",c="g")
plt.semilogy(x2,y2,label="Sécante",c="r")
plt.semilogy(x3,y3,label="Newton",c="b")
plt.semilogy(x4,y4,label="Point Fixe",c="y")
plt.title("Comparaison de l'équation de l'exercice 2 pour les 4 méthodes")
plt.ylabel("Evolution de l'erreur \n (échelle log10)")
plt.legend()

plt.subplot(2,1,2)
plt.grid()
plt.plot(x1,y1,label="Dichotomie",c="g")
plt.plot(x2,y2,label="Sécante",c="r")
plt.plot(x3,y3,label="Newton",c="b")
plt.plot(x4,y4,label="Point Fixe",c="y")
plt.title("Comparaison de l'équation de l'exercice 2 pour les 4 méthodes")
plt.ylabel("Evolution de l'erreur \n (échelle linéaire)")
plt.legend()
  

xa1 = np.array(Liste_Dichotomie_n(f3,1,2,1E-10,1E4))
ya1 = np.array(Liste_Dichotomie_Erreur(f3,1,2,1E-10,1E4))

xa2 = np.array(Liste_Secante_n(f3, 1, 2, 1E-10, 1E4))
ya2 = np.array(Liste_Secante_Erreur(f3, 1, 2, 1E-10, 1E4))

xa3 = np.array(Liste_Newton_n(f3, f3prim,1, 1E-10, 1E4))
ya3 = np.array(Liste_Newton_Erreur(f3, f3prim,1, 1E-10, 1E4))

xa4= np.array(Liste_PointFixe_n(g3,1,1E-10,1E4))
ya4= np.array(Liste_PointFixe_Erreur(g3,1,1E-10,1E4))

plt.figure(figsize=(12,8))
plt.subplot(2,1,1)
plt.grid()
plt.semilogy(xa1,ya1,label="Dichotomie",c="g")
plt.semilogy(xa2,ya2,label="Sécante",c="r")
plt.semilogy(xa3,ya3,label="Newton",c="b")
plt.semilogy(xa4,ya4,label="Point Fixe",c="y")
plt.title("Comparaison de l'équation E3 pour les 4 méthodes")
plt.ylabel("Evolution de l'erreur \n (échelle log10)")
plt.legend()

plt.subplot(2,1,2)
plt.grid()
plt.plot(xa1,ya1,label="Dichotomie",c="g")
plt.plot(xa2,ya2,label="Sécante",c="r")
plt.plot(xa3,ya3,label="Newton",c="b")
plt.plot(xa4,ya4,label="Point Fixe",c="y")
plt.title("Comparaison de l'équation E3 pour les 4 méthodes")
plt.ylabel("Evolution de l'erreur \n (échelle linéaire)")
plt.legend()

"""Question 6 """

xb1 = np.array(Liste_Dichotomie_n(f6, 1, 2, 1E-10, 1E4))
yb1 = np.array(Liste_Dichotomie_Erreur(f6, 1, 2, 1E-10, 1E4))

xb2 = np.array(Liste_Secante_n(f6, 1, 2, 1E-10, 1E4))
yb2 = np.array(Liste_Secante_Erreur(f6, 1, 2, 1E-10, 1E4))

xb3 = np.array(Liste_Newton_n(f6, f6prim, 1, 1E-10, 1E4))
yb3 = np.array(Liste_Newton_Erreur(f6, f6prim, 1, 1E-10, 1E4))

xb4=np.array(Liste_PointFixe_n(g6,1,1E-10,1E4))
yb4=np.array(Liste_PointFixe_Erreur(g6,1,1E-10,1E4))


plt.figure(figsize=(12,8))
plt.subplot(2,1,1)
plt.grid()
plt.semilogy(xb1,yb1,label="Dichotomie",c="g")
plt.semilogy(xb2,yb2,label="Sécante",c="r")
plt.semilogy(xb3,yb3,label="Newton",c="b")
plt.semilogy(xb4,yb4,label="Point Fixe",c="y")
plt.title("Comparaison de l'équation E6 pour les 4 méthodes")
plt.ylabel("Evolution de l'erreur \n (échelle log10)")
plt.legend()

plt.subplot(2,1,2)
plt.grid()
plt.plot(xb1,yb1,label="Dichotomie",c="g")
plt.plot(xb2,yb2,label="Sécante",c="r")
plt.plot(xb3,yb3,label="Newton",c="b")
plt.plot(xb4,yb4,label="Point Fixe",c="y")
plt.xlabel("Nombre d'itérations n")
plt.ylabel("Evolution de l'erreur \n (échelle linéaire)")
plt.legend()

"""Question 7"""

xc1 = np.array(Liste_Dichotomie_n(f7, 1, 2, 1E-10, 1E4))
yc1 = np.array(Liste_Dichotomie_Erreur(f7, 1, 2, 1E-10, 1E4))

xc2 = np.array(Liste_Secante_n(f7, 1, 2, 1E-10, 1E4))
yc2 = np.array(Liste_Secante_Erreur(f7, 1, 2, 1E-10, 1E4))

xc3 = np.array(Liste_Newton_n(f7, f7prim, 1, 1E-10, 1E4))
yc3 =np.array(Liste_Newton_Erreur(f7, f7prim, 1, 1E-10, 1E4))

xc4=np.array(Liste_PointFixe_n(g7,1,1E-10,1E4))
yc4=np.array(Liste_PointFixe_Erreur(g7,1,1E-10,1E4))

plt.figure(figsize=(12,8))
plt.subplot(2,1,1)
plt.grid()
plt.semilogy(xc1,yc1,label="Dichotomie",c="g")
plt.semilogy(xc2,yc2,label="Sécante",c="r")
plt.semilogy(xc3,yc3,label="Newton",c="b")
plt.semilogy(xc4,yc4,label="Point Fixe",c="y")
plt.title("Comparaison de l'équation E7 pour les 4 méthodes")
plt.ylabel("Evolution de l'erreur \n (échelle log10)")
plt.legend()

plt.subplot(2,1,2)
plt.grid()
plt.plot(xc1,yc1,label="Dichotomie",c="g")
plt.plot(xc2,yc2,label="Sécante",c="r")
plt.plot(xc3,yc3,label="Newton",c="b")
plt.plot(xc4,yc4,label="Point Fixe",c="y")
plt.xlabel("Nombre d'itérations n")
plt.ylabel("Evolution de l'erreur \n (échelle linéaire)")
plt.legend()

"""Question 9"""

xd1 = np.array(Liste_Dichotomie_n(f9, 1, 2, 1E-10, 1E4))
yd1 = np.array(Liste_Dichotomie_Erreur(f9, 1, 2, 1E-10, 1E4))

xd2 = np.array(Liste_Secante_n(f9, 1, 2, 1E-10, 1E4))
yd2 = np.array(Liste_Secante_Erreur(f9, 1, 2, 1E-10, 1E4))

xd3 = np.array(Liste_Newton_n(f9, f9prim, 1, 1E-10, 1E4))
yd3 = np.array(Liste_Newton_Erreur(f9, f9prim, 1, 1E-10, 1E4))

xd4=np.array(Liste_PointFixe_n(g9,1,1E-10,1E4))
yd4=np.array(Liste_PointFixe_Erreur(g9,1,1E-10,1E4))

plt.figure(figsize=(12,8))
plt.subplot(2,1,1)
plt.grid()
plt.semilogy(xd1,yd1,label="Dichotomie",c="g")
plt.semilogy(xd2,yd2,label="Sécante",c="r")
plt.semilogy(xd3,yd3,label="Newton",c="b")
plt.semilogy(xd4,yd4,label="Point Fixe",c="y")
plt.title("Comparaison de l'équation E9 pour les 4 méthodes")
plt.ylabel("Evolution de l'erreur \n (échelle log10)")
plt.legend()

plt.subplot(2,1,2)
plt.grid()
plt.plot(xd1,yd1,label="Dichotomie",c="g")
plt.plot(xd2,yd2,label="Sécante",c="r")
plt.plot(xd3,yd3,label="Newton",c="b")
plt.plot(xd4,yd4,label="Point Fixe",c="y")
plt.xlabel("Nombre d'itérations n")
plt.ylabel("Evolution de l'erreur \n (échelle linéaire)")
plt.legend()

"""Question 10"""

xe1 = np.array(Liste_Dichotomie_n(f10, 1, 2, 1E-10, 1E4))
ye1 = np.array(Liste_Dichotomie_Erreur(f10, 1, 2, 1E-10, 1E4))

xe2 = np.array(Liste_Secante_n(f10, 1, 2, 1E-10, 1E4))
ye2 = np.array(Liste_Secante_Erreur(f10, 1, 2, 1E-10, 1E4))

xe3 = np.array(Liste_Newton_n(f10, f10prim, 1, 1E-10, 1E4))
ye3 = np.array(Liste_Newton_Erreur(f10, f10prim, 1, 1E-10, 1E4))

xe4=np.array(Liste_PointFixe_n(g10,1,1E-10,1E4))
ye4=np.array(Liste_PointFixe_Erreur(g10,1,1E-10,1E4))

plt.figure(figsize=(12,8))
plt.subplot(2,1,1)
plt.grid()
plt.semilogy(xe1,ye1,label="Dichotomie",c="g")
plt.semilogy(xe2,ye2,label="Sécante",c="r")
plt.semilogy(xe3,ye3,label="Newton",c="b")
plt.semilogy(xe4,ye4,label="Point Fixe",c="y")
plt.title("Comparaison de l'équation E10 pour les 4 méthodes")
plt.ylabel("Evolution de l'erreur \n (échelle log10)")
plt.legend()

plt.subplot(2,1,2)
plt.grid()
plt.plot(xe1,ye1,label="Dichotomie",c="g")
plt.plot(xe2,ye2,label="Sécante",c="r")
plt.plot(xe3,ye3,label="Newton",c="b")
plt.plot(xe4,ye4,label="Point Fixe",c="y")
plt.xlabel("Nombre d'itérations n")
plt.ylabel("Evolution de l'erreur \n (échelle linéaire)")
plt.legend()

