from random import randint
from time import perf_counter
import numpy as np
from scipy.spatial.distance import cdist
from entrainement import Entrainement

class Clustering:
    def __init__(self, cerveau: Entrainement, k: int, n: int, knn: int) -> None:
        self.cerveau = cerveau
        self.k = k
        self.n = n
        self.knn = knn
        self.length = len(self.cerveau.vocabulaire)
        self.centroides = []
        self.iter_courante = []
        self.iter_precedente = []
        
    def __initialiser_arrays(self):
        self.centroides = np.zeros((self.k, self.length))
        self.iter_courante = np.zeros((self.length,), dtype=int)
        self.iter_precedente = np.zeros((self.length,), dtype=int)
        
    def __initialiser_centroides(self):
        rand_cen = []
        for index_cen in range(len(self.centroides)):
            rand_idx = randint(0, self.length - 1)
            if rand_idx not in rand_cen:
                self.centroides[index_cen] = self.cerveau.matrice[rand_idx]
                rand_cen.append(rand_idx)
                               
    def __calculer_centroide(self):
        self.centroides = np.zeros((self.k, self.length))
        self.count_par_cent = np.zeros(self.k)
        
        for i in range(self.length):
            self.centroides[self.iter_courante[i]] += self.cerveau.matrice[i]
            self.count_par_cent[self.iter_courante[i]] += 1
        
        for k in range(self.k):
            if self.count_par_cent[k] > 0:
                self.centroides[k] /= self.count_par_cent[k]
    
    def __imprimer_iterations(self, it_idx, time, changements):
        print(f'Itération {it_idx} effectuées en {time} secondes ({changements} changements) \n')
        
        for idx, count in enumerate(self.count_par_cent):
            print(f'Il y a {count} mots appartenant au centroide {idx}')
        
        print('\n************************************************************************')
        
    def __compute_votes_scores(self, reversed_dict, minimum, noms_cgrams, iter):
        scores = []
        for idx_mot in range(len(self.iter_courante)):
            if self.iter_courante[idx_mot] == iter:
                scores.append([reversed_dict[idx_mot], min(minimum[idx_mot])])
        scores = sorted(scores, key=lambda t: t[1], reverse=False)
        
        votes = {cgram : 0 for cgram in noms_cgrams.values()}
        for mot, distance in scores[:self.knn]:
            if mot in noms_cgrams:
                votes[noms_cgrams[mot]] += 1/(distance**2 + 1)
                
        return scores, votes
        
    def __imprimer_resultats_finaux(self, temps_total, it_index, minimum, noms_cgrams:dict):
        reversed_dict = {v: k for k, v in self.cerveau.vocabulaire.items()}
        
        print(f'Clustering effectuée en {it_index} itérations, en un temps de {temps_total} secondes.')
        for iter in range(self.k):
            scores, votes = self.__compute_votes_scores(reversed_dict, minimum, noms_cgrams, iter)
            max_vote_key = max(votes, key=votes.get)
            print(f'\nCentroïde {iter} -> cgram: {max_vote_key} ({votes[max_vote_key]} votes)')
            for _, result in zip(range(self.n), scores):
                if result[0] in noms_cgrams:
                    w1 = f'({noms_cgrams[result[0]]})'
                else:
                    w1 = '(NONE)'
                
                print("{:<13}{:<17} --> {:<30}".format(w1, result[0], result[1]))
            
    def __func_knn(self):
        SEP = '\t'
        CHEMIN_TSV = '../tsv/Lexique382.tsv'
        
        def extract(ch, enc, fonc):
            with open(ch, encoding = enc) as f:
                lines = f.read().splitlines()
            noms = lines[0].split(SEP)
            data = np.array([fonc(line) for line in lines[1:]])

            return noms, data
    
        def extract_feature(line):
            return [x for x in line.split(SEP)]
        
        noms_features, features = extract(CHEMIN_TSV, 'utf-8', extract_feature)
        
        features = [list(feats) for feats in zip(*features)]
            
        unique_dict = {}
        
        for feat_0, feat_3, feat_9 in zip(features[0], features[3], features[9]):
            feat_9 = float(feat_9)
            if feat_0 not in unique_dict or feat_9 > unique_dict[feat_0][1]:
                unique_dict[feat_0] = (feat_3, feat_9)
                
        ortho_category_dict = {key: value[0] for key, value in unique_dict.items()}
        
        return ortho_category_dict


    def regrouper(self):
        self.__initialiser_arrays()
        self.__initialiser_centroides()
        
        changement = True
        it_index = 0
        temps_total = 0
        minimum = []
        while changement:
            t = perf_counter()

            minimum = cdist(self.cerveau.matrice, self.centroides)
            self.iter_courante = np.argmin(minimum, axis=1)
            nbr_changements = np.count_nonzero(self.iter_courante != self.iter_precedente)
            
            if nbr_changements == 0:
                changement = False
                
            self.__calculer_centroide()
        
            t = perf_counter() - t
            temps_total += t
            self.__imprimer_iterations(it_index, t, nbr_changements)
            it_index += 1
            
            np.copyto(self.iter_precedente, self.iter_courante)
        
        noms_cgrams = self.__func_knn()
        self.__imprimer_resultats_finaux(temps_total, it_index, minimum, noms_cgrams)
        