import cv2
import pytesseract
from PIL import Image, ImageEnhance, ImageOps, ImageFilter
import numpy as np
import json 
import re 
# Chemin vers l'exécutable de Tesseract
pytesseract.pytesseract.tesseract_cmd = '/opt/homebrew/bin/tesseract'

video_path = '/Users/ridha/ocr/godar.mp4'
cap = cv2.VideoCapture(video_path)
blocc=10
def nettoyer_sous_titre(sous_titre):
    # Si l'entrée n'est pas une chaîne de caractères, la convertir en chaîne
    sous_titre = str(sous_titre) if not isinstance(sous_titre, str) else sous_titre

    # Appliquer la suppression des caractères spéciaux en début de chaîne
    sous_titre = re.sub(r'^[^a-zA-Z0-9\u0600-\u06FF]+', '', sous_titre)

    # Vérifier s'il y a des points à la fin et les déplacer au début
    while sous_titre.endswith('.'):
        # Retirer un point de la fin
        sous_titre = sous_titre[:-1]
        # Ajouter un point au début
        sous_titre = '.' + sous_titre

    return sous_titre

# Vérification de l'ouverture de la vidéo
if not cap.isOpened():
    print("Erreur lors de l'ouverture du fichier vidéo.")
    exit()

# Taux de rafraîchissement de la vidéo
fps = int(cap.get(cv2.CAP_PROP_FPS))
subtitles = []  # Liste pour stocker les sous-titres
frame_number = 0
start_time = 0
subtitle_index = 1  # Indice pour le numéro de sous-titre

try:
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if frame_number :  # Traitement toutes les frames 
            

            # Extraction de la région d'intérêt (ROI) avec des paramètres par défaut
            roi = frame[360:480, 250:650]

            # Vérification si la ROI est vide
            if roi.size == 0:
                print(f"Erreur : ROI vide pour la frame {frame_number}.")
                continue  # Passer à la prochaine frame

            # Conversion en PIL Image pour traitement OCR
            try:
                img = Image.fromarray(cv2.cvtColor(roi, cv2.COLOR_BGR2RGB))
            except cv2.error as e:
                print(f"Erreur de conversion de l'image : {e}")
                continue  # Passer à la prochaine frame

            # Appliquer le processus d'amélioration sur la ROI
            enhancer = ImageEnhance.Contrast(img)
            img = enhancer.enhance(2.0)
            img = img.convert("L")
            img = ImageOps.autocontrast(img, cutoff=2)

            threshold = 249
            img = img.point(lambda p: 255 if p > threshold else 0)
            img = ImageOps.invert(img)
            img = img.filter(ImageFilter.GaussianBlur(radius=1))

            # Affichage de l'image traitée avant OCR
            #img_cv = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
            #cv2.imshow('Image pour OCR', img_cv)  # Affiche l'image
            #cv2.waitKey(100)  # Attendre 100 ms pour voir l'image

            # Extraction du texte avec pytesseract
            extracted_text = nettoyer_sous_titre(pytesseract.image_to_string(img, lang='ara').strip())
            

            # Gestion des sous-titres
            end_time = (frame_number / fps)

            subtitles.append(f"{subtitle_index}\n")
            subtitles.append(f"{int(start_time // 3600):02}:{int((start_time % 3600) // 60):02}:{int(start_time % 60):02},{int((start_time % 1) * 1000):03} --> {int(end_time // 3600):02}:{int((end_time % 3600) // 60):02}:{int(end_time % 60):02},{int((end_time % 1) * 1000):03}\n")

            if len(extracted_text)> 2 :  # Si du texte a été extrait
                subtitles.append(f"{extracted_text}\n\n")  # Ajouter le texte extrait
            else:  # Aucun texte extrait
                subtitles.append("Aucun sous-titre n'a été extrait\n\n\n")  # Ajouter le message de défaut

            subtitle_index += 1  # Incrémenter le numéro de sous-titre

            # Mettre à jour le timecode de début pour le prochain sous-titre
            start_time = end_time

        frame_number += 1  # Incrémenter le numéro de frame

finally:
    cap.release()
    cv2.destroyAllWindows()  # Fermer toutes les fenêtres d'affichage

    # Écriture des sous-titres dans un fichier
    with open('/Users/ridha/ocr/ocr_dar/OUTPUT.srt', 'w', encoding='utf-8') as f:
        f.writelines(subtitles)

def lire_fichier(fichier):
    with open(fichier, 'r', encoding='utf-8') as f:
        contenu = f.read()

    # Séparer le contenu en sections, une section pour chaque bloc (par numéro)
    blocs = contenu.split('\n\n')
    time_codes = []
    sous_titres = []

    for bloc in blocs:
        lignes = bloc.strip().split('\n')
        if len(lignes) > 1:  # S'assurer qu'il y a bien plusieurs lignes
            time_code = lignes[1]
            time_codes.append(time_code)

            # Nettoyer chaque ligne du sous-titre individuellement
            sous_titres_lignes_nettoyes = [nettoyer_sous_titre(ligne) for ligne in lignes[2:]]  # Appliquer nettoyage
            
            # Joindre les lignes nettoyées en ajoutant ' /' entre elles
            sous_titre = ' /'.join(sous_titres_lignes_nettoyes).strip()# Joindre les lignes de sous-titre en insérant "/n" à la fin de chaque ligne sauf la dernière

            sous_titres.append(sous_titre)

    # Retourner les sous-titres et les time_codes extraits
    return sous_titres, time_codes

def ecrire_fichier(nouveau_fichier, sous_titres, time_codes):
    with open(nouveau_fichier, 'w', encoding='utf-8') as f:
        for i, (sous_titre, time_code) in enumerate(zip(sous_titres, time_codes), start=1):
            # Écrire le numéro, le time code et le sous-titre avec les "/n" pour représenter les sauts de ligne
            f.write(f"{i}\n{time_code}\n{sous_titre}\n\n")

s,t=lire_fichier("/Users/ridha/ocr/ocr_dar/OUTPUT.srt")
a="/Users/ridha/ocr/ocr_dar/OUTPUT1.srt"
ecrire_fichier(a,s,t)

def analyser_repetitions(nouveau_fichier, fichier_sortie):
    with open(nouveau_fichier, 'r', encoding='utf-8') as f:
        contenu = f.read()

    blocs = contenu.strip().split('\n\n')
    resultats = {}  # Dictionnaire pour stocker les sous-titres les plus répétés
    
    # Traiter les blocs par groupe de fps taille
    
    for i in range(0, len(blocs), blocc):
        bloc_courant = blocs[i:i + blocc]
        sous_titres_bloc = []
        for bloc in bloc_courant:
            lignes = bloc.strip().split('\n')
            if len(lignes) > 2:  # S'assurer qu'il y a bien trois lignes
                sous_titres_bloc.append(lignes[2])  # Ajouter le sous-titre
            else:
                sous_titres_bloc.append("")  # Ajouter une ligne vide si pas assez de lignes

        max_repetitions = 0
        sous_titre_max = None

        # Comparer les sous-titres pour détecter les répétitions successives
        j = 0
        while j < len(sous_titres_bloc) - 1:
            compteur = 1
            # Avancer tant que les sous-titres sont identiques
            while (j + 1 < len(sous_titres_bloc) and 
                   sous_titres_bloc[j] == sous_titres_bloc[j + 1]):
                compteur += 1
                j += 1

            # Mettre à jour le maximum si nécessaire
            if compteur > max_repetitions:
                max_repetitions = compteur
                sous_titre_max = sous_titres_bloc[j]

            j += 1  # Passer au sous-titre suivant

        # Stocker le sous-titre le plus répété ou indiquer aucune répétition
        if sous_titre_max is not None:
            resultats[f"Bloc {i // blocc + 1}"] = sous_titre_max
        else:
            resultats[f"Bloc {i // blocc + 1}"] = "Aucune répétition"

    # Écrire le résultat dans un fichier JSON
    with open(fichier_sortie, 'w', encoding='utf-8') as sortie:
        json.dump(resultats, sortie, ensure_ascii=False, indent=4)  # Indenté pour la lisibilité

analyser_repetitions("/Users/ridha/ocr/ocr_dar/OUTPUT1.srt","/Users/ridha/ocr/ocr_dar/OUTPUT2.srt")


def chaine_similaire(sous_titre_repere, sous_titre_comparaison):
    sous_titre_repere = nettoyer_sous_titre(sous_titre_repere)
    sous_titre_comparaison = nettoyer_sous_titre(sous_titre_comparaison)
    
    if sous_titre_repere[-2:] != sous_titre_comparaison[-2:]:
        return "", 0
    
    chaine_similaire = ""
    compteur_similaire = 0
    
    for i in range(1, min(len(sous_titre_repere), len(sous_titre_comparaison)) + 1):
        if sous_titre_repere[-i] == sous_titre_comparaison[-i]:
            chaine_similaire = sous_titre_repere[-i] + chaine_similaire
            compteur_similaire += 1
        else:
            break
    
    return chaine_similaire, compteur_similaire
        
def comparer_sous_titres(sous_titres_repetes_fichier, ocr_fichier):
    # Lire les sous-titres répétés à partir du fichier JSON
    with open(sous_titres_repetes_fichier, 'r', encoding='utf-8') as f:
        sous_titres_repetes = json.load(f)

    # Lire tous les sous-titres de ocr_n1.txt
    sous_titres_ocr,t = lire_fichier(ocr_fichier)

    # Créer un dictionnaire pour stocker les résultats de comparaison
    resultats_comparaison = {}
    # Traiter chaque sous-titre répété
    for i, (bloc_num, sous_titre_repere) in enumerate(sous_titres_repetes.items()):
        # Prendre les blocs sous-titres correspondant dans ocr_n1.txt
        debut_bloc = i * blocc
        fin_bloc = debut_bloc + blocc
        sous_titres_bloc = sous_titres_ocr[debut_bloc:fin_bloc]
        resultats_comparaison[bloc_num] = {
            "Sous-titre répété": sous_titre_repere,
            "Comparaisons": []
        }
        
        # Comparer le sous-titre répété avec chaque sous-titre du bloc
        chaine_triple = None
        for j, sous_titre_comparaison in enumerate(sous_titres_bloc):
            # Si la chaîne de référence est "Aucun sous-titre n'a été extrait", comparer normalement
            if sous_titre_repere == "Aucun sous-titre n'a été extrait":
                
                for sous_titre in sous_titres_bloc:
                    mots = sous_titre.split()
                for mot in mots:
                    if len(mot) >= 3:  # Vérifier si le mot a au moins 3 caractères
                        for j in range(len(mot) - 2):  # Parcourir les caractères du mot
                            if mot[j].isalpha() and mot[j + 1].isalpha() and mot[j + 2].isalpha():# Vérifier s'il y a trois lettres successives dans le mot
                                chaine_triple = mot
                                break
                if chaine_triple:  # Sortir de la boucle si trouvé
                    break
            
            # Modifier la clé avec la chaîne trouvée
            if chaine_triple:
                sous_titre_repere_modifie = chaine_triple
                chaine, compteur = chaine_similaire(sous_titre_repere, sous_titre_comparaison)
                resultats_comparaison[bloc_num]["Comparaisons"].append({
                    "Sous-titre": sous_titre_repere_modifie,
                    "Chaîne similaire": chaine,
                    "Nombre de caractères similaires": compteur
                })
            # Sinon, comparer uniquement si ce n'est pas le même sous-titre que celui de référence
            elif sous_titre_comparaison != sous_titre_repere:
                chaine, compteur = chaine_similaire(sous_titre_repere, sous_titre_comparaison)
                resultats_comparaison[bloc_num]["Comparaisons"].append({
                    "Sous-titre": sous_titre_comparaison,
                    "Chaîne similaire": chaine,
                    "Nombre de caractères similaires": compteur
                })
    # Écrire les résultats de la comparaison dans un fichier JSON
    fichier_resultat = '/Users/ridha/OCR/ocr_dar/comparaison_resultats.json'
    with open(fichier_resultat, 'w', encoding='utf-8') as f_resultat:
        json.dump(resultats_comparaison, f_resultat, ensure_ascii=False, indent=4)


comparer_sous_titres("/Users/ridha/ocr/ocr_dar/OUTPUT2.srt","/Users/ridha/ocr/ocr_dar/OUTPUT1.srt")

import json

def extraire_chaine(fichier_json):
    # Lire le fichier JSON contenant les résultats de comparaison
    with open(fichier_json, 'r', encoding='utf-8') as f:
        resultats_comparaison = json.load(f)

    meilleures_similitudes = []

    # Parcourir chaque bloc de résultats
    for bloc_num, resultats in resultats_comparaison.items():
        comparaisons = resultats.get("Comparaisons", [])
        sous_titre_repete = resultats.get("Sous-titre répété", "")

        # Trouver la comparaison avec le plus grand nombre de caractères similaires
        meilleure_comparaison = max(comparaisons, key=lambda x: x["Nombre de caractères similaires"], default=None)

        if meilleure_comparaison and meilleure_comparaison["Nombre de caractères similaires"] > 0:
            meilleure_chaine = meilleure_comparaison["Chaîne similaire"]
            meilleures_similitudes.append(meilleure_chaine)
        else:
            # Si aucune comparaison ou aucune chaîne similaire n'a été trouvée, utiliser le "Sous-titre répété"
            meilleures_similitudes.append(sous_titre_repete)


    
    return meilleures_similitudes
# Exécution de la fonction
chaines_similaires = extraire_chaine("/Users/ridha/ocr/ocr_dar/comparaison_resultats.json")
list=chaines_similaires

def ajouter_bonne_chaine(fichier_json, liste_chaînes, fichier_sortie):
    # Charger l'ancien fichier JSON
    with open(fichier_json, 'r', encoding='utf-8') as f:
        resultats_comparaison = json.load(f)

    # Ajouter la nouvelle clé "bonne_chaine" dans chaque bloc en itérant sur les blocs et la liste des chaînes
    for i, (bloc_num, resultats) in enumerate(resultats_comparaison.items()):
        if i < len(liste_chaînes):  # Vérifier si on a une chaîne correspondante dans la liste
            resultats["bonne_chaine"] = liste_chaînes[i]
        else:
            resultats["bonne_chaine"] = ""  # Si on n'a plus de chaîne à associer, mettre une chaîne vide

    # Sauvegarder le fichier avec les nouveaux résultats (incluant la clé "bonne_chaine")
    with open(fichier_sortie, 'w', encoding='utf-8') as f_sortie:
        json.dump(resultats_comparaison, f_sortie, ensure_ascii=False, indent=4)

ajouter_bonne_chaine("/Users/ridha/ocr/ocr_dar/comparaison_resultats.json",list,"/Users/ridha/ocr/ocr_dar/comparaison_resultats1.json")

def generer_segments_sous_titres(fichier_ocr, fichier_comparaison):
    """Lire le fichier de sous-titres et générer une liste contenant les segments et sous-titres associés."""
    segments_sous_titres = []  # Liste pour stocker les segments et sous-titres associés

    with open(fichier_ocr, 'r', encoding='utf-8') as f:
        lignes = f.readlines()
        
        for i in range(0, len(lignes), 4):  # Chaque bloc fait 4 lignes (numéro, time code, sous-titre, ligne vide)
            time_code = lignes[i + 1].strip()  # Récupérer le time code
            sous_titre = lignes[i + 2].strip()  # Récupérer le sous-titre
            # Formater la chaîne
            chaine_avec_segment = (time_code, sous_titre)  
            segments_sous_titres.append(chaine_avec_segment)  

    # Lire les sous-titres répétés à partir du fichier JSON
    with open(fichier_comparaison, 'r', encoding='utf-8') as f:
        sous_titres_repetes = json.load(f)

    sous_titres_reperes = []
    for bloc_num, sous_titre_repere in sous_titres_repetes.items():
        
        if isinstance(sous_titre_repere, dict):
            sous_titre_repere = sous_titre_repere.get("bonne_chaine", "")
            sous_titres_reperes.append(sous_titre_repere)

    return segments_sous_titres, sous_titres_reperes

t,s=generer_segments_sous_titres("/Users/ridha/ocr/ocr_dar/OUTPUT1.srt","/Users/ridha/ocr/ocr_dar/comparaison_resultats1.json")

def extraire_time_codes(segments_sous_titres, sous_titres_reperes):
    """Comparer les sous-titres repérés avec les blocs de sous-titres et extraire les time codes."""
    resultats = []  # Liste pour stocker les résultats
    sous_titres_uniques = set()  # Ensemble pour vérifier les doublons

    # Boucler à travers chaque sous-titre repéré
    for sous_titre_repere in sous_titres_reperes:
        debut_time_code = None
        fin_time_code = None

        # Boucler à travers les segments de sous-titres par tranche de blocc = 10
        for i in range(0, len(segments_sous_titres), blocc):
            # Récupérer le bloc de 10 sous-titres
            bloc = segments_sous_titres[i:i + blocc]

            # Variable pour suivre si le sous-titre repéré a été trouvé dans le bloc
            trouve = False

            # Comparer le sous-titre repéré avec chaque sous-titre du bloc
            for time_code, sous_titre in bloc:
                if sous_titre_repere in sous_titre:
                    # Si on trouve le sous-titre repéré
                    if debut_time_code is None:
                        # Enregistrer le time code de début si c'est la première occurrence
                        debut_time_code = time_code
                    # Toujours mettre à jour le time code de fin
                    fin_time_code = time_code
                    trouve = True

            # Si le sous-titre repéré a été trouvé au moins une fois
            if trouve and debut_time_code is not None and fin_time_code is not None:
                # Créer une clé unique pour vérifier les doublons
                cle = (sous_titre_repere, debut_time_code, fin_time_code)
                if cle not in sous_titres_uniques:
                    sous_titres_uniques.add(cle)  # Ajouter à l'ensemble
                    # Ajouter les résultats (sous-titre repéré, time code de début et time code de fin)
                    resultats.append((sous_titre_repere, debut_time_code, fin_time_code))
            debut_time_code = fin_time_code

    return resultats

resultats = extraire_time_codes(t, s)
print(resultats)

resultats_final = [
    (sous_titre_repere, f"{debut_time_code[0:17]}{fin_time_code[-12:]}")
    if "00:00:00,000" in debut_time_code
    else (sous_titre_repere, f"{debut_time_code[17:29]} --> {fin_time_code[-12:]}")
    for sous_titre_repere, debut_time_code, fin_time_code in resultats
]

def generer_fichier_srt(resultats, nom_fichier):
    """Générer un fichier SRT à partir des résultats."""
    with open(nom_fichier, 'w', encoding='utf-8') as f:
        for index, (sous_titre_repere, time_codes) in enumerate(resultats):
            # Décomposer les time codes

            # Écrire le contenu dans le fichier SRT
            f.write(f"{index + 1}\n")  # Numérotation des sous-titres
            f.write(f"{time_codes}\n")  # Time codes
            f.write(f"{sous_titre_repere}\n\n")  # Sous-titre avec deux sauts de ligne

generer_fichier_srt(resultats_final,"/Users/ridha/ocr/ocr_dar/OUTPUT3.srt")

def filtrer_sous_titres(input_file, output_file):
    with open(input_file, 'r', encoding='utf-8') as f:
        lignes = f.readlines()

    # Liste pour stocker les blocs valides (sans "Aucun sous-titre n'a été extrait")
    sous_titres_filtres = []
    index = 1  # Nouvelle numérotation

    # Parcourir les lignes par blocs de 4 lignes (numéro, time code, sous-titre, ligne vide)
    for i in range(0, len(lignes), 4):
        sous_titre = lignes[i + 2].strip()  # Récupérer la ligne contenant le sous-titre

        # Si le sous-titre ne contient pas "Aucun sous-titre n'a été extrait", on le garde
        if "Aucun sous-titre n'a été extrait" not in sous_titre:
            sous_titres_filtres.append(f"{index}\n")
            sous_titres_filtres.append(lignes[i + 1])  # Time code
            sous_titres_filtres.append(lignes[i + 2])  # Sous-titre
            sous_titres_filtres.append("\n")  # Ligne vide
            index += 1  # Incrémenter le numéro de bloc

    # Écrire les blocs filtrés dans le nouveau fichier
    with open(output_file, 'w', encoding='utf-8') as f:
        f.writelines(sous_titres_filtres)

# Appel de la fonction
filtrer_sous_titres("/Users/ridha/ocr/ocr_dar/OUTPUT3.srt", "/Users/ridha/ocr/ocr_dar/OUTPUT4.srt")



def traiter_final_txt(fichier_entree, fichier_sortie="/Users/ridha/ocr/ocr_dar/OUTPUT4.srt"):
    """Traite le fichier SRT en parcourant de l'arrière vers l'avant et remplace '/' par des sauts de ligne."""
    
    # Lire le fichier SRT d'entrée
    with open(fichier_entree, 'r', encoding='utf-8') as f:
        contenu = f.read()

    # Parcourir le contenu de la fin vers le début et remplacer '/' par '\n'
    contenu_inverse = contenu[::-1]  # Inverser tout le contenu du fichier
    contenu_remplace = contenu_inverse.replace('/', '\n')  # Remplacer '/' par '\n' dans le texte inversé

    # Ré-inverser le texte pour revenir à l'ordre original mais avec les remplacements effectués
    contenu_final = contenu_remplace[::-1]

    # Écrire le contenu traité dans le fichier de sortie
    with open(fichier_sortie, 'w', encoding='utf-8') as f:
        f.write(contenu_final)

    print(f"Fichier {fichier_sortie} traité pour remplacer '/' par des sauts de ligne.")

# Appel de la fonction
traiter_final_txt("/Users/ridha/ocr/ocr_dar/OUTPUT4.srt")


def nettoyer_sous_titre1(sous_titre):
    """Nettoie le texte des sous-titres."""
    sous_titre = str(sous_titre) if not isinstance(sous_titre, str) else sous_titre

    # Vérifier s'il y a des points à la fin et les déplacer au début
    while sous_titre.endswith('.'):
        sous_titre = sous_titre[:-1]
        sous_titre = '.' + sous_titre

    return sous_titre

def lire_fichier_srt1(nom_fichier):
    """Lit un fichier SRT et retourne une liste de tuples avec l'index, le time code, et le texte."""
    with open(nom_fichier, 'r', encoding='utf-8') as f:
        contenu = f.read()

    # Séparation des blocs de sous-titres
    blocs = contenu.strip().split('\n\n')
    sous_titres = []

    for bloc in blocs:
        lignes = bloc.strip().split('\n')
        if len(lignes) >= 3:  # S'assurer qu'il y a au moins 3 lignes
            index = lignes[0]
            time_code = lignes[1]

            # Appliquer nettoyage sur les lignes de sous-titre
            sous_titres_lignes_nettoyes = [nettoyer_sous_titre1(ligne) for ligne in lignes[2:]]
            # Joindre les lignes de sous-titre en insérant un slash pour garder une trace
            sous_titre = ' / '.join(sous_titres_lignes_nettoyes).strip()
            sous_titres.append((index, time_code, sous_titre))

    return sous_titres

def fusionner_sous_titres(sous_titres):
    """Fusionne les sous-titres identiques consécutifs en un seul et met à jour les index."""
    resultats = []

    if not sous_titres:
        return resultats

    # Initialiser avec le premier sous-titre
    index1, temps1, texte1 = sous_titres[0]
    temps_debut, temps_fin = temps1.split(' --> ')
    index_compteur = 1  # Compteur pour les nouveaux index

    for i in range(1, len(sous_titres)):
        index2, temps2, texte2 = sous_titres[i]

        # Vérifier si les sous-titres sont identiques
        if texte1.strip() == texte2.strip():
            # Si identiques, mettre à jour le temps de fin
            temps_fin = temps2.split(' --> ')[1]
        else:
            # Si pas identiques, ajouter le sous-titre précédent avec le nouvel index
            resultats.append((index_compteur, f"{temps_debut} --> {temps_fin}", texte1))
            index_compteur += 1  # Incrémenter le compteur d'index
            # Réinitialiser avec le nouveau sous-titre
            index1, temps1, texte1 = index2, temps2, texte2
            temps_debut, temps_fin = temps1.split(' --> ')

    # Ajouter le dernier sous-titre avec l'index mis à jour
    resultats.append((index_compteur, f"{temps_debut} --> {temps_fin}", texte1))

    return resultats

def ecrire_fichier_srt(sous_titres, nom_fichier):
    """Écrit les sous-titres fusionnés dans un fichier SRT."""
    with open(nom_fichier, 'w', encoding='utf-8') as f:
        for index, (index_srt, temps, texte) in enumerate(sous_titres, start=1):
            f.write(f"{index}\n")
            f.write(f"{temps}\n")
            # Remplacer ' / ' par '\n' avant d'écrire
            texte_final = texte.replace(' / ', '\n')
            f.write(f"{texte_final}\n\n")  # Écrire le texte final

# Test de la fonction avec un fichier SRT et impression du résultat
sous_titres = lire_fichier_srt1("/Users/ridha/ocr/ocr_dar/OUTPUT4.srt")
sous_titres_fusionnes = fusionner_sous_titres(sous_titres)
ecrire_fichier_srt(sous_titres_fusionnes, "/Users/ridha/ocr/ocr_dar/OUTPUT5.srt")

def traiter_final_txt(fichier_entree, fichier_sortie="/Users/ridha/ocr/ocr_dar/OUTPUT6.srt"):
    """Traite le fichier SRT en parcourant de l'arrière vers l'avant et remplace '/' par des sauts de ligne."""
    
    # Lire le fichier SRT d'entrée
    with open(fichier_entree, 'r', encoding='utf-8') as f:
        contenu = f.read()

    # Remplacer '/' par '\n' dans le contenu
    contenu_final = contenu.replace('/', '\n')  

    # Écrire le contenu traité dans le fichier de sortie
    with open(fichier_sortie, 'w', encoding='utf-8') as f:
        f.write(contenu_final)

traiter_final_txt("/Users/ridha/ocr/ocr_dar/OUTPUT5.srt")