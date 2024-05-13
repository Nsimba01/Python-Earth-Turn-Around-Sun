#!/usr/bin/env python3

# Remarque: la fonction read_kmc_system() fait usage d'expressions regulieres (module re), non vues ce semestre
# Les appels a re.match() et re.search() permettent de recuperer le bon texte de la ligne du fichier en cours de lecture

import re
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

####################################################################################################
def read_kmc_system(filename):
    # Lecture des parametres du systeme dans le fichier 'filename'
    # Le format est fourni dans le fichier exemple "reactions.dat"
    # Rappel pour le calcul des h_i:
    #   X   -> ...    rtype=0
    #   X+X -> ...    rtype=1
    #   X+Y -> ...    rtype=2
    # Les donnees renvoyees ont les types suivants:
    #  nspec   : int => nombre d'especes chimiques
    #  sname[] : liste de str => nom de chaque espece
    #  sc0[]   : liste de float => concentrations initiales de chaque espece
    #  nreac   : int => nombre de reactions chimiques
    #  rstr[]  : liste de str => equation bilan de chaque reaction
    #  rtype[] : np.array[nreac] de int => type de chaque reaction: 0,1,2
    #  rid[][] : np.array[nreac][2] de int => no. de chaque espece reactive pour chaque reaction [-1 pour le 2eme si inutile]
    #  rpop[][]: np.array[nreac][nspec] de int => variation des populations de chaque espece lors de chaque reaction
    #  rk[]    : liste de float => constante de vitesse de chaque reaction
    # lecture des donnees brutes:
    nspec = 0
    sname = []
    sc0 = []
    nreac = 0
    rstr = []
    rk = []
    fic = open(filename, "r")
    for line in fic:
        if re.match(r'^specie *: *', line, re.I):
            s = re.search(r'^ *specie *: *(\S+) *c0 *= *(\S+)', line, re.I)
            sname.append(s.group(1))
            sc0.append(float(s.group(2)))
            nspec = nspec + 1
        elif re.match(r'^reaction *: *', line, re.I):
            s = re.search(r'^ *reaction *: *(\S.+\S) *k *= *(\S+)', line, re.I)
            rs = re.sub(r' +', "", s.group(1))
            rstr.append(rs)
            rk.append(float(s.group(2)))
            nreac = nreac + 1
    fic.close
    # creation du tableau de populations:
    err = ""
    rtype = np.zeros((nreac), int)
    rid = np.ones((nreac, 2), int)
    rid = -rid
    rpop = np.zeros((nreac, nspec), int)
    for i in range(nreac):
        s = re.search(r'^(.+)->(.+)$', rstr[i], re.I)
        # gestion des reactifs => rtype, rid et rpop:
        rs = s.group(1).split("+")
        nrtot = 0
        nrmax = 0
        for j in range(nspec):
            nj = sname[j]
            for k in rs:
                if re.match(r'^\d+{:s}$'.format(nj), k, re.I):
                    n = int(re.search(r'^(\d+){:s}$'.format(nj), k, re.I).group(1))
                    rpop[i][j] = rpop[i][j] - n
                    if rid[i][0] == -1:
                        rid[i][0] = j
                    else:
                        rid[i][1] = j
                    nrtot = nrtot + n
                    if n > nrmax:
                        nrmax = n
                elif re.match(r'^{:s}$'.format(nj), k, re.I):
                    rpop[i][j] = rpop[i][j] - 1
                    if rid[i][0] == -1:
                        rid[i][0] = j
                    elif rid[i][0] != j:
                        rid[i][1] = j
                    nrtot = nrtot + 1
                    if 1 > nrmax:
                        nrmax = 1
        if nrtot == 2:
            if nrmax == 1:
                rtype[i] = 2
            elif nrmax == 2:
                rtype[i] = 1
            else:
                # => erreur detectee si trop de reactifs pour faire un type 0,1 ou 2
                err = err + "> unsupported reactant number in reaction: \033[92m" + rstr[i] + "\033[0m\n"
        elif nrtot == 1:
            rtype[i] = 0
        else:
            # => erreur detectee si trop de reactifs pour faire un type 0,1 ou 2
            err = err + "> unsupported reactant number in reaction: \033[92m" + rstr[i] + "\033[0m\n"
        # gestion des produits => seulement rpop:
        ps = s.group(2).split("+")
        for j in range(nspec):
            nj = sname[j]
            for k in ps:
                if re.match(r'^\d+{:s}$'.format(nj), k, re.I):
                    n = int(re.search(r'^(\d+){:s}$'.format(nj), k, re.I).group(1))
                    rpop[i][j] = rpop[i][j] + n
                elif re.match(r'^{:s}$'.format(nj), k, re.I):
                    rpop[i][j] = rpop[i][j] + 1
    if err == "":
        return nspec, sname, sc0, nreac, rstr, rtype, rid, rpop, rk
    else:
        print("\n\033[91m*** read_kmc_system() error:\033[0m\n{:s}".format(err))
        exit()
#################################################################################### read_kmc_system

####################################################################################################
def display_kmc_system(nspec, sname, sc0, nreac, rstr, rtype, rpop, rk):
    # > Renvoie une chaine de caracteres a afficher sur le terminal afin de visualiser les
    #   parametres du systeme decrit par la liste des variables d'entree
    # > Cette fonction ne contient que de l'affichage formate vu en cours
    # Notes: - la sequence \033[95m active le texte en violet
    #        - la sequence \033[96m active le texte en cyan
    #        - la sequence \033[0m remet le texte en normal
    s = "\n\n\n                                  =====================\n"
    s = s + "                           ---==# SUMMARY OF KMC SYSTEM #==---\n"
    s = s + "                                  =====================\n"
    s = s + "\n\n\033[95m  *** list of chemical species ***\033[0m\n\n"
    s = s + "   id    name    c0 (mol/L)\n ===========================\n"
    for i in range(nspec):
        s = s + "{:5d}      {:4s}{:12.4e}\n".format(i, sname[i], sc0[i])
    s = s + "\n\n\033[96m  *** list of chemical reactions ***\033[0m\n\n"
    s = s + "   id  equation        type      k      unit       matrix\n"
    s = s + " ==================================================" + "=" * (nspec * 3 + 2) + "\n"
    for i in range(nreac):
        if rtype[i] == 0:
            unit = "s^-1"
        else:
            unit = "L/(mol.s)"
        s = s + "{:5d}  {:18s}{:1d}{:12.3e}  {:9s}  {:s}\n".format(i, rstr[i], rtype[i], rk[i], unit, str(rpop[i]))
    return s

# Constante gravitationnelle (G)
G = 6.67408e-11  # en m^3 kg^-1 s^-2

# Définition de la classe pour les corps célestes
class CelestialBody:
    def __init__(self, mass, position, velocity):
        self.mass = mass
        self.position = np.array(position)
        self.velocity = np.array(velocity)

# Fonction pour calculer la force gravitationnelle
def gravitational_force(body1, body2):
    distance_vector = body2.position - body1.position
    distance = np.linalg.norm(distance_vector)
    force_magnitude = G * body1.mass * body2.mass / distance**2
    force_direction = distance_vector / distance
    return force_magnitude * force_direction

# Fonction pour mettre à jour la position et la vitesse
def update_position_velocity(body, force, timestep):
    # Mise à jour de la vitesse
    body.velocity += force / body.mass * timestep
    # Mise à jour de la position
    body.position += body.velocity * timestep

# Création des corps célestes
# Exemple: Soleil et Terre
sun = CelestialBody(mass=1.989e30, position=[0, 0], velocity=[0, 0])
earth = CelestialBody(mass=5.972e24, position=[0, 1.496e11], velocity=[2.978e4, 0])

# Choix du pas de temps (τ)
timestep = 60 * 60  # 1 heure en secondes

# Simulation sur une période donnée
simulation_time = 365 * 24 * 60 * 60  # 1 an en secondes
current_time = 0

# Liste pour stocker les positions pour la visualisation
positions = []

while current_time < simulation_time:
    force_on_earth = gravitational_force(sun, earth)
    update_position_velocity(earth, force_on_earth, timestep)
    positions.append(earth.position.copy())
    current_time += timestep

# Sauvegarde des positions dans un fichier
np.savetxt('earth_trajectory.txt', positions)

# Visualisation et animation
positions = np.loadtxt('earth_trajectory.txt')

# Créer une figure pour l'animation
fig, ax = plt.subplots()

# Trouver la distance maximale de la Terre par rapport au Soleil
max_distance = np.max(np.linalg.norm(positions, axis=1))

# Ajuster les limites de l'axe en fonction de la distance maximale
ax.set_xlim(-max_distance * 1.1, max_distance * 1.1)
ax.set_ylim(-max_distance * 1.1, max_distance * 1.1)

# Initialiser le point représentant la Terre et le Soleil
earth, = ax.plot([], [], 'bo', ms=5)
sun, = ax.plot([0], [0], 'yo', ms=10)

# Fonction d'initialisation pour l'animation
def init():
    earth.set_data([], [])
    sun.set_data([0], [0])
    return earth, sun

# Fonction d'animation pour mettre à jour la position de la Terre
def animate(i):
    earth.set_data([positions[i, 0]], [positions[i, 1]])
    return earth, sun

# Créer l'animation
ani = animation.FuncAnimation(fig, animate, frames=len(positions), init_func=init, blit=True, interval=50)

# Afficher l'animation
plt.show()

# Pour sauvegarder l'animation, décommentez la ligne suivante :
# ani.save('earth_orbit.mp4', writer='ffmpeg')
