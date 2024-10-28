import cv2
import pytesseract
from PIL import Image, ImageEnhance, ImageOps, ImageFilter
import numpy as np
import json 
import re 
# Chemin vers l'exécutable de Tesseract
pytesseract.pytesseract.tesseract_cmd = '/opt/homebrew/bin/tesseract'

blocc=30
def nettoyer_sous_titre(sous_titre):
    # Si l'entrée n'est pas une chaîne de caractères, la convertir en chaîne
    sous_titre = str(sous_titre) if not isinstance(sous_titre, str) else sous_titre
    while sous_titre.endswith('.'):
        # Retirer le point de la fin
        sous_titre = sous_titre[:-1]
        # Ajouter le point au début
        sous_titre = '.' + sous_titre

    # Appliquer la suppression des caractères spéciaux en fin de chaîne comme c'est l'arabe la fin c'est le début
    sous_titre = re.sub(r'[^a-zA-Z0-9\u0600-\u06FF]+$', '', sous_titre)

    return sous_titre

def extract_subtitles(video_path, output_path, threshold1=250, contrast_factor=2.0, text_lang='ara'):
    #création d'objet cap pour acceder au frame de la vidéo 
    cap = cv2.VideoCapture(video_path)
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    subtitles = []  # Liste pour stocker les sous-titres
    frame_number = 0
    start_time = 0
    subtitle_index = 1  # Indice pour le numéro de sous-titre

    # Coordonnées et autres valeurs
    c, d = 640, 632
    e, f = 640, 624
    x, y, w, h = 400, 633, 480, 58

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # Comparaison de la couleur pour déterminer la région d'intérêt
            b, g, r = frame[f, e]
            ba, ga, ra = frame[d, c]
            e1, e2, e3 = abs(ba - b), abs(ga - g), abs(ra - r)
            
            if e1 < threshold1 and e2 < threshold1 and e3 < threshold1:
                y, h = 575, 116  # Ajustement de la région d'intérêt
            
            # Extraction de la sous-région
            roi = frame[y:y+h, x:x+w]
            img = Image.fromarray(cv2.cvtColor(roi, cv2.COLOR_BGR2RGB))  # Convertir en format PIL RGB

            # Augmentation du contraste
            enhancer = ImageEnhance.Contrast(img)
            img = enhancer.enhance(contrast_factor)

            
            img = img.convert("L") #Conversion en niveaux de gris et traitement pour OCR
            img = ImageOps.autocontrast(img, cutoff=2) #Ajustement automatique du contraste
            img = img.point(lambda p: 255 if p > 249 else 0) #Seuil de binarisation (noir et blanc)
            img = ImageOps.invert(img) #Inversion des couleurs
            img = img.filter(ImageFilter.GaussianBlur(radius=1)) #Application d'un flou gaussien

            # Extraction du texte avec pytesseract et le nettoyer 
            extracted_text = nettoyer_sous_titre(pytesseract.image_to_string(img, lang=text_lang))

            # Calcul du timecode
            end_time = frame_number / fps
            subtitles.append(f"{subtitle_index}\n")
            subtitles.append(f"{int(start_time // 3600):02}:{int((start_time % 3600) // 60):02}:{int(start_time % 60):02},{int((start_time % 1) * 1000):03} --> {int(end_time // 3600):02}:{int((end_time % 3600) // 60):02}:{int(end_time % 60):02},{int((end_time % 1) * 1000):03}\n")

            if len(extracted_text) > 2:
                subtitles.append(f"{extracted_text}\n\n\n")
            else:
                subtitles.append("Aucun sous-titre n'a été extrait\n\n\n")

            # Préparation pour le prochain sous-titre
            subtitle_index += 1
            start_time = end_time
            frame_number += 1  # Incrément du numéro de frame
            print(frame_number)

    finally:
        #libérer les ressources
        cap.release()
        cv2.destroyAllWindows()

        # Écriture des sous-titres dans un fichier SRT
        with open(output_path, 'w', encoding='utf-8') as f:
            f.writelines(subtitles)

def lire_fichier(fichier):
    #cette fonction est pour re-traiter le premier fichier sous titre et concaténer les sous titre dans une seule ligne et renvoie
    # les sous titres concaténés et leurs time code.
    with open(fichier, 'r', encoding='utf-8') as f:
        contenu = f.read()

    # Séparer le contenu en  blocs, une section pour chaque bloc (chaque bloc est separé par 2 saute de lignes)
    blocs = contenu.split('\n\n\n')
    time_codes = []
    sous_titres = []
    for bloc in blocs:
        lignes = bloc.strip().split('\n')
        if len(lignes) > 1:  # S'assurer qu'il y a bien plusieurs lignes
            time_code = lignes[1]
            time_codes.append(time_code)
            sous_titres_lignes = [ligne for ligne in lignes[2:4]]  
            # Joindre les lignes nettoyées en ajoutant ' /' entre elles
            sous_titre = '/'.join(sous_titres_lignes).strip()# Joindre les lignes de sous-titre en insérant "/" à la fin de chaque ligne sauf la dernière
            sous_titres.append(sous_titre)
            
    # Retourner les sous-titres et les time_codes extraits
    return sous_titres, time_codes

def lire_fichier1(fichier):
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
            sous_titre = lignes[2] 
            sous_titres.append(sous_titre)
            
    # Retourner les sous-titres et les time_codes extraits
    return sous_titres, time_codes

def ecrire_fichier(nouveau_fichier, sous_titres, time_codes):
    with open(nouveau_fichier, 'w', encoding='utf-8') as f:
        for i, (sous_titre, time_code) in enumerate(zip(sous_titres, time_codes), start=1):
            # Écrire le numéro, le time code et le sous-titre avec les "/n" pour représenter les sauts de ligne
            f.write(f"{i}\n{time_code}\n{sous_titre}\n\n")


def analyser_repetitions(nouveau_fichier, fichier_sortie):
    #la fonction est pour chercher la chaine de caractere la plus repetée sucssusivement.
    # et renvoie un json avec chaque bloc et sa chaine la plus repetée sucssusivement.
    with open(nouveau_fichier, 'r', encoding='utf-8') as f:
        contenu = f.read()

    blocs = contenu.strip().split('\n\n')
    resultats = {}  # Dictionnaire pour stocker les sous-titres les plus répétés

    for i in range(0, len(blocs), blocc):
        bloc_courant = blocs[i:i + blocc]
        sous_titres_bloc = []
        for bloc in bloc_courant:
            lignes = bloc.strip().split('\n')
            if len(lignes) > 2:  # S'assurer qu'il y a bien trois lignes
                sous_titres_bloc.append(lignes[2])  # Ajouter le sous-titre
            else:
                sous_titres_bloc.append("Aucun sous-titre n'a été extrait")  # Ajouter une ligne vide si pas assez de lignes

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

        # Si aucun sous-titre n'a été extrait, chercher un mot avec 3 lettres successives
        if sous_titre_max == "Aucun sous-titre n'a été extrait":
            chaine_triple = None
            for idx, sous_titre in enumerate(sous_titres_bloc):
                if sous_titre != "Aucun sous-titre n'a été extrait":
                    # Parcourir chaque mot du sous-titre
                    
                    if len(sous_titre) >= 3:  # Vérifier si le mot a au moins 3 caractères
                            # Parcourir les lettres du mot en sens inverse pour trouver 3 lettres successives
                        if sous_titre[0]!=" " and sous_titre[1]!=" " and sous_titre[2]!=" ":
                            if idx + 2 < len(sous_titres_bloc) and sous_titre == sous_titres_bloc[idx + 1]== sous_titres_bloc[idx + 2]:
                                chaine_triple = sous_titre
                                break
                        if chaine_triple:  # Sortir de la boucle si trouvé
                            break
                if chaine_triple:  # Sortir si une chaîne valide est trouvée
                    break

            # Stocker la chaîne trouvée comme sous-titre max
            resultats[f"Bloc {i // blocc + 1}"] = chaine_triple if chaine_triple else "Aucun sous-titre n'a été extrait"
        else:
            resultats[f"Bloc {i // blocc + 1}"] = sous_titre_max

    # Écrire le résultat dans un fichier JSON
    with open(fichier_sortie, 'w', encoding='utf-8') as sortie:
        json.dump(resultats, sortie, ensure_ascii=False, indent=4)  # Indenté pour la lisibilité

def chaine_similaire(sous_titre_repere, sous_titre_comparaison):
    #la fonction est pour comparer deux chaine et extraire les lettres qui se ressemblent. 

    # au début on vérifier si les deux dernières lettres correspondent, sinon on renvoie vide
    if sous_titre_repere[0:2] != sous_titre_comparaison[0:2]:
        return "", 0
    
    chaine_similaire = ""
    compteur_similaire = 0
    
    # Comparer les caractères de la fin des deux chaînes vers le début
    for i in range(0, min(len(sous_titre_repere), len(sous_titre_comparaison))):
        if sous_titre_repere[i] == sous_titre_comparaison[i]:
            chaine_similaire += sous_titre_repere[i]  # On ajoute les caractères 
            compteur_similaire += 1
        else:
            break
    
    return chaine_similaire, compteur_similaire

def comparer_sous_titres(sous_titres_repetes_fichier, ocr_fichier):
    #la fonction est pour comparer les sous titres qu'on a reperer avant avec tout le bloc pour trouver la bonne chaine de caractere.

    # Lire les sous-titres répétés à partir du fichier JSON
    with open(sous_titres_repetes_fichier, 'r', encoding='utf-8') as f:
        sous_titres_repetes = json.load(f)
    
    # Lire tous les sous-titres du fichier ou on a extrait tout les sous titres
    sous_titres_ocr,t = lire_fichier1(ocr_fichier)
    
    # Créer un dictionnaire pour stocker les résultats de comparaison
    resultats_comparaison = {}

    # Traiter chaque sous-titre répété
    for i, (bloc_num, sous_titre_repere) in enumerate(sous_titres_repetes.items()):
        # Prendre les ous titres par bloc 
        debut_bloc = i * blocc
        fin_bloc = debut_bloc + blocc
        sous_titres_bloc = sous_titres_ocr[debut_bloc:fin_bloc]


        resultats_comparaison[bloc_num] = {
            "Sous-titre répété": sous_titre_repere,
            "Comparaisons": []
        }
        
        # Comparer le sous-titre répété avec chaque sous-titre du bloc
        for j, sous_titre_comparaison in enumerate(sous_titres_bloc):
            # Si la chaîne de référence est "Aucun sous-titre n'a été extrait", comparer normalement
            if sous_titre_repere == "Aucun sous-titre n'a été extrait":
                chaine, compteur = chaine_similaire(sous_titre_repere, sous_titre_comparaison)
                resultats_comparaison[bloc_num]["Comparaisons"].append({
                    "Sous-titre": sous_titre_comparaison,
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
    fichier_resultat = '/Users/ridha/ocr/ocr_dar/comparaison_resultats.json'
    with open(fichier_resultat, 'w', encoding='utf-8') as f_resultat:
        json.dump(resultats_comparaison, f_resultat, ensure_ascii=False, indent=4)

def extraire_chaine(fichier_json):
    #la fonction est pour recuperer les bonnes chaines de chaque bloc qu'on a trouvé avant  
    #si on a trouvé la chaine on la prend sinon on prend la chaine la plus répétée 
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

#aprés avoir trouver la liste des bonnes chaines on va chercher leurs segments.  
def generer_segments_sous_titres(fichier_ocr):
    #on va faire une liste de liste qui contient chaque sous titre avec son segment 

    #Lire le fichier de sous-titres et générer une liste contenant les segments et sous-titres associés
    segments_sous_titres = []  # Liste pour stocker les segments et sous-titres associés

    with open(fichier_ocr, 'r', encoding='utf-8') as f:
        lignes = f.readlines()
        
        for i in range(0, len(lignes), 4):  # Chaque bloc fait 4 lignes (numéro, time code, sous-titre, ligne vide)
            time_code = lignes[i + 1].strip()  # Récupérer le time code
            sous_titre = lignes[i + 2].strip()  # Récupérer le sous-titre
            # Formater la chaîne
            chaine_avec_segment = (time_code, sous_titre)  
            segments_sous_titres.append(chaine_avec_segment)  



    return segments_sous_titres

#après on va chercher chaque chaine de caractere repère, dans la liste des sous titres et les segments d'avant 
#et pour chaque bloc on stocke (sous_titre_repere, segment_apparition, segment_disparition) 
def extraire_segments(sous_titres_reperes, segments_sous_titres):
    resultat = []
    index_segment = 0
    for sous_titre_repere in sous_titres_reperes:
        # Initialisation des segments d'apparition et de disparition
        segment_apparition = None
        segment_disparition = None

        # Prendre le bloc de 3 segments consécutifs
        bloc = segments_sous_titres[index_segment:index_segment + blocc]
        
        # Parcourir les segments du bloc de 
        for segment in bloc:
            temps_segment, texte_segment = segment
            
            # Comparaison avec le sous-titre repère
            if sous_titre_repere in texte_segment:
                if segment_apparition is None:
                    segment_apparition = temps_segment  # Premier segment où le sous-titre apparaît
                segment_disparition = temps_segment  # Dernier segment où le sous-titre apparaît

        # Si on a trouvé une apparition du sous-titre repère
        if segment_apparition and segment_disparition:
            resultat.append((sous_titre_repere, segment_apparition, segment_disparition))

        # Passer au bloc suivant
        index_segment += blocc

    return resultat


def generer_fichier_srt(resultats, nom_fichier):
    #la fonction est pour generer le fichier .srt qui contient les bonnes chaines avec leurs bons segments. 
    with open(nom_fichier, 'w', encoding='utf-8') as f:
        for index, (sous_titre_repere, time_codes) in enumerate(resultats):
            # Écrire le contenu dans le fichier SRT
            f.write(f"{index + 1}\n")  # Numérotation des sous-titres
            f.write(f"{time_codes}\n")  # Time codes
            f.write(f"{sous_titre_repere}\n\n")  # Sous-titre avec deux sauts de ligne


def filtrer_sous_titres(input_file, output_file):
    #la fonction est pour filtrer les blocs de "Aucun sous-titre n'a été extrait" du fichier .srt 
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


def traiter_final_txt(fichier_entree, fichier_sortie="/Users/ridha/ocr/ocr_dar/OUTPUT4.srt"):
    #la fonction est pour traiter le fichier SRT en parcourant de l'arrière vers l'avant
    #et remplacer '/' par des sauts de ligne
    
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


def lire_fichier_srt1(nom_fichier):
    #Lit un fichier SRT et retourne une liste de tuples avec l'index, le time code, et le texte 
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
            sous_titres_lignes_nettoyes = [(ligne) for ligne in lignes[2:]]
            # Joindre les lignes de sous-titre en insérant un slash pour garder une trace
            sous_titre = ' / '.join(sous_titres_lignes_nettoyes).strip()
            sous_titres.append((index, time_code, sous_titre))

    return sous_titres

import Levenshtein
def fusionner_sous_titres(sous_titres):
    #Fusionne les sous-titres identiques consécutifs en un seul et met à jour les segments.
    resultats = []

    if not sous_titres:
        return resultats

    # Initialiser avec le premier sous-titre
    index1, temps1, texte1 = sous_titres[0]
    temps_debut, temps_fin = temps1.split(' --> ')
    index_compteur = 1  # Compteur pour les nouveaux index

    for i in range(1, len(sous_titres)):
        index2, temps2, texte2 = sous_titres[i]
        # Calculer la distance de Levenshtein
        distance = Levenshtein.distance(texte1.strip(), texte2.strip())
        similarite = 1 - (distance / max(len(texte1.strip()), len(texte2.strip())))

        # Vérifier si les sous-titres sont identiques
        if similarite >= 0.40:
            texte_reference = max(texte1, texte2, key=len)
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
    resultats.append((index_compteur, f"{temps_debut} --> {temps_fin}", texte_reference if similarite >= 0.40 else texte1))
    return resultats

def ecrire_fichier_srt(sous_titres, nom_fichier):
    #fonction pour écrire les sous-titres fusionnés dans un fichier SRT
    with open(nom_fichier, 'w', encoding='utf-8') as f:
        for index, (index_srt, temps, texte) in enumerate(sous_titres, start=1):
            f.write(f"{index}\n")
            f.write(f"{temps}\n")
            # Remplacer ' / ' par '\n' avant d'écrire
            texte_final = texte.replace(' / ', '\n')
            f.write(f"{texte_final}\n\n")  # Écrire le texte final


def lire_et_filtrer_fichier_srt(nom_fichier, fichier_sortie):
    # la fonction Lit un fichier SRT, filtre les sous-titres ayant moins de deux caractères et les sous titres contient *,-<>#«َّ_:“"
    # et écrit le résultat dans un nouveau fichier avec de nouveaux index
    with open(nom_fichier, 'r', encoding='utf-8') as f:
        contenu = f.read()
    # Séparer les blocs de sous-titres
    blocs = contenu.strip().split('\n\n')
    sous_titres_valides = []
    for bloc in blocs:
        lignes = bloc.strip().split('\n')
        if len(lignes) >= 3:  # S'assurer qu'il y a au moins 3 lignes
            time_code = lignes[1]
            sous_titre = ' /'.join(lignes[2:]).strip()  # Joindre les lignes de sous-titre
            # Vérifier si la longueur du sous-titre est inférieure à 2 caractères
            if len(sous_titre) <= 3 or any(char in sous_titre for char in '*,-<>#«َّ_:“"')or any(char.isdigit() for char in sous_titre) or any(len(mot) < 2 for mot in sous_titre.split()):
                continue  # Ignorer ce bloc
            # Ajouter le bloc valide à la liste sans l'index
            sous_titres_valides.append((time_code, sous_titre))

    # Écrire les sous-titres filtrés dans le fichier de sortie avec de nouveaux index
    with open(fichier_sortie, 'w', encoding='utf-8') as f:
        for index, (temps, texte) in enumerate(sous_titres_valides, start=1):
            f.write(f"{index}\n")
            f.write(f"{temps}\n")
            f.write(f"{texte}\n\n")  # Écrire le texte final


def traiter_final_sous_titres(fichier_entree, fichier_sortie):
    # Traite le fichier SRT en remplaçant '/' par des sauts de ligne pour la derniere fois 

    # Lire le fichier SRT d'entrée
    with open(fichier_entree, 'r', encoding='utf-8') as f:
        contenu = f.read()

    # Remplacer '/' par '\n' dans le contenu
    contenu_final = contenu.replace('/', '\n')  

    # Écrire le contenu traité dans le fichier de sortie
    with open(fichier_sortie, 'w', encoding='utf-8') as f:
        f.write(contenu_final)



# Appel des fonctions
extract_subtitles('/Users/ridha/ocr/ocr_dar/GodarSubS01E1_6.mp4', '/Users/ridha/ocr/ocr_dar/OUTPUT.srt')
s,t=lire_fichier("/Users/ridha/ocr/ocr_dar/OUTPUT.srt")
a="/Users/ridha/ocr/ocr_dar/OUTPUT1.srt"
ecrire_fichier(a,s,t)
analyser_repetitions("/Users/ridha/ocr/ocr_dar/OUTPUT1.srt", "/Users/ridha/ocr/ocr_dar/OUTPUT2.srt")
comparer_sous_titres("/Users/ridha/ocr/ocr_dar/OUTPUT2.srt","/Users/ridha/ocr/ocr_dar/OUTPUT1.srt")
chaines_similaires = extraire_chaine("/Users/ridha/ocr/ocr_dar/comparaison_resultats.json")
segment_soustitres=generer_segments_sous_titres("/Users/ridha/ocr/ocr_dar/OUTPUT1.srt")
resultats = extraire_segments(chaines_similaires,segment_soustitres)

resultats_final = [
    (sous_titre_repere, f"{debut_time_code[0:17]}{fin_time_code[-12:]}")
    if "00:00:00,000" in debut_time_code
    else (sous_titre_repere, f"{debut_time_code[0:12]} --> {fin_time_code[-12:]}")
    for sous_titre_repere, debut_time_code, fin_time_code in resultats
]
generer_fichier_srt(resultats_final,"/Users/ridha/ocr/ocr_dar/OUTPUT3.srt")
filtrer_sous_titres("/Users/ridha/ocr/ocr_dar/OUTPUT3.srt", "/Users/ridha/ocr/ocr_dar/OUTPUT4.srt")
traiter_final_txt("/Users/ridha/ocr/ocr_dar/OUTPUT4.srt")
sous_titres = lire_fichier_srt1("/Users/ridha/ocr/ocr_dar/OUTPUT4.srt")
sous_titres_fusionnes = fusionner_sous_titres(sous_titres)
ecrire_fichier_srt(sous_titres_fusionnes, "/Users/ridha/ocr/ocr_dar/OUTPUT5.srt")
lire_et_filtrer_fichier_srt("/Users/ridha/ocr/ocr_dar/OUTPUT5.srt", "/Users/ridha/ocr/ocr_dar/OUTPUT6.srt")
traiter_final_sous_titres("/Users/ridha/ocr/ocr_dar/OUTPUT6.srt","/Users/ridha/ocr/ocr_dar/OUTPUT6.srt")
b=lire_fichier_srt1("/Users/ridha/ocr/ocr_dar/OUTPUT6.srt")
s_b=fusionner_sous_titres(b)
ecrire_fichier_srt(s_b, "/Users/ridha/ocr/ocr_dar/OUTPUT7.srt")