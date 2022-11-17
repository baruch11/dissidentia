# Annotation guidelines

## Description générale du label

Contexte : les phrases du dataset sont issues des réponses à la question du grand débat : "Que pensez-vous de l'organisation de l'Etat et des administrations en France ? De quelle manière cette organisation devrait-elle évoluer ?" 

Dans chaque phrase, on cherche à détecter les critiques outrancières du pouvoir en place.
On qualifie chaque phrase de critique outrancière sur 2 critères : véhémence de la critique, et cible de la critique.


### Véhémence 

Rentre dans ce critère tout ce qui est outrancier, haineux, ou calomnieux (c-a-d sans preuve évidente, juste sur la base de on-dit). Les phrases critique mais relativement neutres ne rentre pas dans ce critère.

**exemples** : 
"Tous pourris !" (outrance), "Les fonctionnaires ne recrutent que par copinage" (grosse généralité sans preuve)
"supprimer commissions qui servent à rien" (agacement, supposition inutilité)
"incompétences des fonctionnaires" (grosse généralité infondée)
"suppression du senat"

**contre-exemples** : 
"Il faut réduire le mille feuille administratif"
"Il simplifier le système", 
"A mon avis les fonctionnaires sont trop nombreux" (critique pas forcement justifiée mais pas outrancière)
"Trop de députés, ministres"
"La cour des comptes a souvent épinglé cette gabegie financière mais rien n’est fait !" (s'appuie sur un avis d'une organisation légitime)


### Cible

Rentre dans ce critère tout ce qui est dirigé contre le pouvoir en place, c'est-à-dire l'Etat, les élus et toute l'administration (fonctionnaires, députés, ministres, maires, entreprise/service public etc.)

**exemples** : 
   - "Les fonctionnaires sont des feignasses"
   - "Les hauts fonctionnaires sont corrompus"
   - "Nulle." (car ça se rapporte à l'organisation, cf la question)

**contre-exemples** : "Les chômeurs sont des feignasses", "Il faut dégager les immigrés", ou critique des impots


### Labels en pratique

3 labels en pratique: _dissident_, _non dissident_, _inclassable_ 

**dissident**: 
Phrase véhémente et dirigé contre le pouvoir en place et son administration. 
S'il y a plusieurs phrases (le split en phrases a mal fonctionné) et au moins l'une d'elle est violente contre le pouvoir, on labellise "dissident.
S'il y a un seul mot ou une phrase très courte, on considère que ça se rapporte à la question "Que pensez vous de l'organisation de l'Etat"

**non dissident**
Propos qui ne vérifie pas l'une des 2 conditions (cible, véhémence). Par exemple une critique modérée, argumentée de l'Etat. 
Ou alors une phrase hors sujet par rapport à la question (par ex. situation personnelle)

**inclassable**:
Quand on ne sait pas labelliser: on a un doute et on manque de contexte pour savoir plus. 
Souvent des phrases qu'on ne veut pas retenir dans le dataset d'entrainement.

