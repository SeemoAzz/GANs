# Conditional GANs (cGANs)

Ce projet implémente des GANs conditionnels (cGANs) permettant de générer des images conditionnées sur des labels spécifiques. Il utilise le dataset MNIST pour générer des chiffres manuscrits (0-9) en fonction de labels donnés.

---

## **Description du Projet**

Un Conditional GAN (cGAN) est une variante des GANs classiques qui ajoute une information auxiliaire (un label) comme condition pour la génération d'images.
Dans ce projet :
- Le **Générateur** apprend à produire des images correspondant à un label spécifique.
- Le **Discriminateur** évalue à la fois l'authenticité des images (réelles ou générées) et leur compatibilité avec le label fourni.

### **Objectifs :**
- Apprendre à générer des chiffres manuscrits basés sur des labels précis.
- Comprendre l’architecture des cGANs et leur application à un problème concret.

---

## **Fonctionnement du Modèle**

### **1. Générateur (Generator)**
Le Générateur prend comme entrées :
- Un vecteur de bruit `z` (tiré d'une distribution normale).
- Un label `y` spécifiant la classe de l'image souhaitée.

Ces entrées sont fusionnées et passées à travers plusieurs couches pour produire une image réaliste correspondant au label.

### **2. Discriminateur (Discriminator)**
Le Discriminateur prend comme entrées :
- Une image (réelle ou générée).
- Un label associé à cette image.

Il prédit si l'image est réelle et si elle correspond au label fourni.

---

## **Architecture**

### **Générateur**
- Input : Bruit (vecteur `z`) + Label (représenté en One-Hot Encoding).
- Plusieurs couches linéaires avec activations `LeakyReLU` et normalisation batch (`BatchNorm1d`).
- Output : Une image réaliste de dimension 28x28.

### **Discriminateur**
- Input : Image (aplatie en vecteur) + Label (One-Hot Encoding).
- Plusieurs couches linéaires avec `LeakyReLU`.
- Output : Probabilité (valeur entre 0 et 1) indiquant si l'image est réelle et compatible avec le label.

---

## **Données**

- **Dataset** : MNIST (10 classes, chiffres manuscrits de 0 à 9).
- **Prétraitement** :
  - Les images sont redimensionnées en `28x28` pixels.
  - Normalisation dans l’intervalle [-1, 1].

---

## **Entraînement**

Le modèle est entraîné pendant plusieurs époques avec les étapes suivantes :
1. Le Discriminateur est entraîné à distinguer les images réelles des images générées tout en tenant compte des labels.
2. Le Générateur est entraîné à produire des images capables de tromper le Discriminateur.

---

## **Génération d’Images**

Après l'entraînement, le modèle peut générer des images basées sur des labels donnés.

Exemple de code pour générer des images :

```python
# Exemple de génération d'images avec des labels donnés
labels = torch.tensor([0, 1, 2, 3, 4, 5, 6, 7, 8, 9]).to(device)
z = torch.randn(10, latent_dim).to(device)
generated_images = generator(z, labels).detach().cpu()

# Affichage des images
import matplotlib.pyplot as plt
for i in range(10):
    plt.subplot(2, 5, i + 1)
    plt.imshow(generated_images[i][0], cmap="gray")
    plt.title(f"Label: {labels[i].item()}")
    plt.axis("off")
plt.show()
```

---

## **Résultats**

- Le modèle est capable de générer des images manuscrites correspondant à des labels donnés.
- Les images générées ont une qualité dépendante du nombre d'époques d'entraînement.


