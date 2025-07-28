import numpy as np
from PIL import Image

# Crée une image noire (3 canaux, RGB)
img = np.ones((224, 224, 3), dtype=np.uint8) * 255

# Ajoute un carré vert (G = 255) dans le coin supérieur droit
# Par exemple : 40x40 pixels dans le coin haut-droit
img[10:50, 170:210] = [0, 255, 0]

# Sauvegarde ou affiche l'image
image = Image.fromarray(img)
image.save("test_vert_haut_droite.png")
image.show()
