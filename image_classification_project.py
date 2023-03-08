import cv2
import numpy as np
import pickle
import matplotlib.pyplot as plt
import sys
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KNeighborsClassifier

def print_same_line(string):
	sys.stdout.write('\r' + string)
	sys.stdout.flush()


# Le Dataset se nomme DCIFAR-10 : "Learning Multiple Layers of Features from Tiny Images"
# Il provien d'Alex Krizhevsky (2009)

# Le code de la ligne 20 à 89 a été récupéré. Il permet d'extraire les images du dataset.
# J'ai ajouté quelques commentaires dans la première partie récupérée afin d'expliquer l'intégralité de la démarche

class CIFAR10:
	def __init__(self, data_path):
		# Extrait les données CIFAR10 depuis data_path
		file_names = ['data_batch_%d' % i for i in range(1,6)]
		file_names.append('test_batch')

		X = []
		y = []
		# Boucle permettant de récupérer les données dans les variables X et y
		for file_name in file_names:
			with open(data_path + file_name, 'rb') as fin:
				data_dict = pickle.load(fin, encoding='bytes')
			X.append(data_dict[b'data'].ravel())
			y = y + data_dict[b'labels']
 
        # On fixe les dimensions de la matrice X
		# 60000 lignes --> 60000 images
		# 32 * 32 --> représente les dimesnsion de chaque image extraite (32x32 pixels)
		# 3 --> représente les trois couches RGB relatives aux couleurs primaires composants chaque image
		# 32*32*3 = 3072 --> 3072 colonnes
		self.X = np.asarray(X).reshape(60000, 32*32*3)
		# Pour la matrice y, la dimension est de 1
		# Les valeurs seront comprises entre 0 et 9
		# Chaque entier correpsondra à une image
		# Exemple : 0 -> 'Avion' / 1 -> 'Bateau' / etc.
		self.y = np.asarray(y)
 
        # On fait correspondre tous les chiffres de la matrice y compris 0 et 9 que l'on va lié avec son image de la matrice X
		fin = open(data_path + 'batches.meta', 'rb')
		self.LABEL_NAMES = pickle.load(fin, encoding='bytes')[b'label_names']
		fin.close()

# On crée nos matrices d'entraînement (pour entraîner l'algorithme)
# Ainsi que nos matrices test (pour tester la précision de l'algorithme)
# Sur 60000 images, 50000 seront dédiées à l'entraînement et 10000 au test
# Il est important de séparer les données d'entraînement et de test afin d'éviter que l'algorithme apprenne par coeur les images
# De ce fait lors du test on pourra réellement évaluer la précision de l'algorithme
	def train_test_split(self):
		X_train = self.X[:50000]
		y_train = self.y[:50000]
		X_test = self.X[50000:]
		y_test = self.y[50000:]

		return X_train, y_train, X_test, y_test

	def all_data(self):
		return self.X, self.y

	def __prep_img(self, idx):
		img = self.X[idx].reshape(3,32,32).transpose(1,2,0).astype(np.uint8)
		img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
		return img

# Retourne l'image liée à l'index renseigné
	def show_img(self, idx):
		cv2.imshow(self.LABEL_NAMES[self.y[idx]], self.__prep_img(idx))
		cv2.waitKey(0)
		cv2.destroyAllWindows()

# Retourne 25 images aléatoires
	def show_examples(self):
		fig, axes = plt.subplots(5, 5)
		fig.tight_layout()
		for i in range(5):
			for j in range(5):
				rand = np.random.choice(range(self.X.shape[0]))
				axes[i][j].set_axis_off()
				axes[i][j].imshow(self.__prep_img(rand))
				axes[i][j].set_title(self.LABEL_NAMES[self.y[rand]].decode('utf-8'))
		plt.show()	
        


class NearestNeighbor:
    def __init__(self, distance_func='l1'):
        self.distance_func = distance_func
    
    def train(self, X, y):
        # X_tr est la  matrice d'entraînement dans laquelle chaque ligne représente une image
        # y_tr est la matrice de dimension 1 de valeurs correctes (integer compris entre 0 et 9)
		# On met les valeurs au format float32 afin d'obtenir des valeurs négatives le cas échéant
        self.X_tr = X.astype(np.float32)
        self.y_tr = y
    
    def predict(self, X):
        # X_te est la matrice composée d'images utilisées dans le cadre du test la précision de l'algorithme
		# On met les valeurs au format float32 afin d'obtenir des valeurs négatives le cas échéant
        X_te = X.astype(np.float32)
	# On recupère le nombre ligne de la matrice X (50000 lignes)
	# Utile si l'on ne souhaite pas 50000 lignes 
        num_test_examples = X.shape[0]
        y_pred = np.zeros(num_test_examples, self.y_tr.dtype)
        for i in range(num_test_examples):
            if self.distance_func == 'l2':
		# On calcule la distance entre une image de la matrice de test d'indice i et l'ensemble des images de la matrice d'entraînement (avec la méthode valeur absolue ou racine carré)
		# axis = 1 car on soustrait des lignes entre elles
                distances = np.sum(np.square(self.X_tr - X_te[i]), axis = 1)
            else:
                distances = np.sum(np.abs(self.X_tr - X_te[i]), axis = 1)
		
		# On sélectionne l'indice pour lequel la distance est la plus petite
            smallest_dist_idx = np.argmin(distances)
	    # On sélectionne la valeur de la matrice d'entraînement correspondant à l'indice de la plus petite valeur (plus petite distance)
		# Puis on intègre cette valeur dans la matrice y de prédiction pour l'image i
            y_pred[i] = self.y_tr[smallest_dist_idx]
            
        return y_pred

# On renseigne le chemin du fichier
dataset = CIFAR10("C:\Users\PC\Projects IA\Projet Classificateur d'images\image-classifier.zip\cifar-10-batches-py")
X_train, y_train, X_test, y_test = dataset.train_test_split()
X, y = dataset.all_data()

dataset.show_examples()

# Retournes les dimensions de nos matrices d'entraînement et de test
print(X_train.shape)
print(y_train.shape)
print(X_test.shape)
print(y_test.shape)


nn = NearestNeighbor()
# On applique la fonction train() aux données d'entraînement
nn.train(X_train, y_train)
# Si on veut un nombre de test plus restreint, on peut prendre les 200 premiers par exemple
y_pred = nn.predict(X_test[:200])

# Permet de calculer la précision de l'algorithme pour les 100 premières images 
# On compte le nombre de fois où la prédiction est juste puis on divise par le nombre total d'image (Ici, dénominateur = 100)
accuracy = np.mean(y_test[:200] == y_pred)
print(accuracy)

# K Nearest Neighbors
# Le classificateur KNeighborsClassifier() nous permet de renseigner des informations concernant les hyper paramètres
# n_neighbors --> combien de plus proches voisins souhaitons nous
# p = quelle méthode de calcul de la distance souhaitons nous (valeur obsolue OU racine carré)
# n_jobs=-1 est un paramètre annexe jouant sur la vitesse d'entraînement et d'execution et n'impactant pas les données
knn = KNeighborsClassifier(n_neighbors = 5, p = 1, n_jobs = -1)
knn.fit(X_train, y_train)
y_pred = knn.predict(X_test)

# Permet de calculer la précision de l'algorithme
# On compte le nombre de fois où la prédiction est juste puis on divise par le nombre total d'image (dénominateur = 60000)
accuracy = np.mean(y_test == y_pred)
print(accuracy)

# Paramétrage des hyper paramatères avec une grille de recherche (k voisins ET choix du calcul de la distance)
# Ici, on renseigne une liste de k voisins et les deux méthodes de calcul possibles
# L'algorithme éxecutera toutes les combinaisons possibles et nous pourrons ainsi déterminer la plus précise 
param_grid = {'n_neighbors': [1,3,5,10,20,50,100],'p': [1,2]}

# Cependant, réaliser toute les combinaisons risque d'être long, pour cela on utilise GridSearchCV()
# Cela nous permettra de renseigner des critères précis et ainsi réduire le temps d'éxecution
# cv=5 --> représente le nombre de phase de validation
# La phase de validation représente différentes parties des données d'entraînement où l'algorithme va choisir les hyper paramètres 
# Ces données vont constamment changer afin d'éviter que l'algorithme mémorise les données
grid_search = GridSearchCV(knn, param_grid, cv=5, n_jobs=-1)

# Réalise la validation croisée (Cross Validation) sur les données d'entraînement afin de trouver les meilleures paramètres (k ET p) à appliquer sur les données de test
grid_search.fit(X_train,y_train)
print(grid_search.best_params_)
