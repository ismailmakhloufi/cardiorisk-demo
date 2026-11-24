# CardioRisk AI ❤️‍🩹

**Prédiction IA de réhospitalisation chez l'insuffisant cardiaque (IC à FE réduite) — 19 variables + espace patient/médecin connecté.**

> 🌐 **Démo en ligne (gratuite, sans carte bancaire) :** [https://cardiorisk-demo.onrender.com](https://cardiorisk-demo.onrender.com)
>
> ⏳ L'application est hébergée sur Render (plan gratuit) : elle "dort" après ~15 min d'inactivité. Au premier chargement, un écran *Waking up...* s'affiche pendant 30 à 50 secondes, c'est normal.

---

## 📋 Sommaire

1. [Présentation](#-présentation)
2. [Fonctionnalités](#-fonctionnalités)
3. [Le modèle IA](#-le-modèle-ia)
4. [Les 19 variables](#-les-19-variables)
5. [Architecture technique](#-architecture-technique)
6. [Installation locale](#-installation-locale)
7. [Utilisation](#-utilisation)
8. [API](#-api)
9. [Déploiement](#-déploiement)
10. [Démonstration (script vidéo)](#-démonstration-script-vidéo)
11. [Structure du dépôt](#-structure-du-dépôt)

---

## 🎯 Présentation

L'insuffisance cardiaque à fraction d'éjection réduite (IC-FEr) reste la première cause de réhospitalisation cardiovasculaire. **CardioRisk AI** est une application web qui :

- **estime le risque de réhospitalisation** d'un patient à partir de **19 variables cliniques** facilement recueillies à la sortie d'hospitalisation ;
- **génère des alertes et des recommandations** personnalisées pour le cardiologue ;
- **connecte le patient à son cardiologue** : messagerie sécurisée, partage de bilans (PDF/JPG), suivi longitudinal ;
- **automatise la création du compte patient** : le médecin saisit les données, l'application crée automatiquement l'espace patient et affiche l'adresse e-mail générée.

---

## ✨ Fonctionnalités

### Espace Médecin
- **Inscription / connexion** médecin (compte isolé par praticien).
- **Calcul du score de risque** en 1 clic à partir du formulaire clinique (19 variables + ordonnance).
- **Génération d'un rapport IA** complet (alertes, recommandations, interprétation).
- **Tableau de bord** : patients à risque, historique des consultations, statistiques.
- **Création automatique du compte patient** : e-mail `prenom.nom@cardiorisk.ai`, mot de passe `0000`, affiché à l'écran après le calcul du score.
- **Messagerie** avec chaque patient (isolation stricte : un médecin ne voit que *ses* patients).
- **Bilans** : upload et téléchargement de documents (PDF/JPG), réponse du médecin.
- **Import / Export Excel** de la liste des patients.
- **Profil & mot de passe** modifiables.

### Espace Patient
- **Connexion** avec l'e-mail généré et le mot de passe temporaire `0000`.
- **Changement de mot de passe forcé à la première connexion** (une notification contenant le nouveau mot de passe est envoyée dans la messagerie).
- **Accès à son score**, à ses alertes, à ses recommandations.
- **Messagerie sécurisée** avec son cardiologue assigné uniquement.
- **Bilans** : dépôt de documents et réponses du médecin.
- **Isolation totale** : un patient ne voit jamais les données d'un autre patient ni d'un autre médecin.

### Général
- Interface **100 % en français**, responsive (PC, tablette, mobile).
- Page de connexion affichée à chaque ouverture (aucune session restaurée).
- Les libellés affichés côté interface : **BNP** (au lieu de ProBNP) et **dose lasilix** (au lieu de `dose_lasilix_sup120`).

---

## 🧠 Le modèle IA

| Critère | Valeur |
|---|---|
| **Type de modèle** | Régression logistique régularisée **Ridge** |
| **Variables** | 19 |
| **Seuil de décision (τ)** | **0.19** |
| **Sensibilité** | 72,7 % |
| **Spécificité** | 72,5 % |
| **Base d'apprentissage** | 245 patients (196 en entraînement / 49 en test) |

**Matrices de confusion :**

| Jeu | Vrais positifs | Faux négatifs | Faux positifs | Vrais négatifs |
|---|---|---|---|---|
| Entraînement (196) | 111 | 44 | 4 | 37 |
| Test (49) | 28 | 10 | 3 | 8 |

- Le seuil τ = 0.19 a été choisi pour un **bon équilibre sensibilité/spécificité** (priorité à la détection des patients à risque).
- Deux variables supplémentaires sont utilisées comme **indicateurs de vigilance** dans l'interface :
  - `QT_corrige ≥ 0,01`
  - `dose_lasilix_sup120 ≥ 0,10`
- Modèle déployé : `models/modele_19vars_dose120contrainte/modele_final_19vars.joblib`.

---

## 📊 Les 19 variables

| # | Variable (internement) | Libellé affiché | Type / bornes |
|---|---|---|---|
| 1 | `espace_PR` | Espace PR | 50 – 600 ms |
| 2 | `cause_valvulaire` | Cause valvulaire | 0 / 1 |
| 3 | `PAD` | PAD | 20 – 200 mmHg |
| 4 | `PAS` | PAS | 50 – 300 mmHg |
| 5 | `OG` | OG | 15 – 100 mm |
| 6 | `Uree` | Urée | 0,5 – 600 mg/dL |
| 7 | `statine` | Statine | 0 / 1 |
| 8 | `ATCD_d_hospitalisation` | ATCD d'hospitalisation | ≥ 0 |
| 9 | `IMC` | IMC | 10 – 80 |
| 10 | `IEC_dose` | Dose d'IEC | 0 – 2 |
| 11 | `FQ_ECG_sortie` | Fréquence ECG sortie | 25 – 220 bpm |
| 12 | `HTAP` | HTAP | 0 / 1 / 2 (saisie en mmHg, seuils 40 et 60) |
| 13 | `TP` | TP | 5 – 600 % |
| 14 | `lymphocyte` | Lymphocyte | 0 – 100 % |
| 15 | `ARM` | ARM | 0 / 1 |
| 16 | `QT_corrige` | QT corrigé | 0,2 – 1,0 s |
| 17 | `Glycemie_a_jeun` | Glycémie à jeun | 0,2 – 10 g/L |
| 18 | `ProBNP` → affiché **BNP** | **BNP** | 0 – 50 000 pg/mL |
| 19 | `dose_lasilix_sup120` → affiché **dose lasilix** | **dose lasilix** | 0 / 1 |

> Les **bornes de saisie** sont validées côté serveur (Pydantic) et côté interface : une saisie hors bornes renvoie un message d'erreur explicite (HTTP 422 journalisé).

---

## 🏗️ Architecture technique

```
┌────────────────────────┐      ┌──────────────────────────────┐
│   Frontend (React)     │      │   Backend (FastAPI)          │
│   Vite + lucide-react  │ ───► │   /api/predict, /api/report  │
│   frontend/dist        │      │   /api/patients, /api/...    │
└────────────────────────┘      └──────────────┬───────────────┘
                                               │
                     ┌─────────────────────────┼────────────────────────┐
                     ▼                         ▼                        ▼
              models/*.joblib           data/*.json              src/generer_rapport_patient.py
              (modèle Ridge 19 vars)    (patients, doctors,       (prédiction, alertes,
                                        messages, bilans)         recommandations, RAG/Ollama)
```

- **Backend :** FastAPI + Uvicorn, Python 3.10+.
- **Frontend :** React + Vite, bundlé dans `frontend/dist` et **servi par le backend** → une seule URL en production.
- **Persistance :** fichiers JSON dans `data/` (pas de base de données externe → démo sans infra).
- **RAG (optionnel) :** `chromadb` + `sentence-transformers` + Ollama pour enrichir les recommandations ; l'application fonctionne **sans** Ollama (repli sur des règles cliniques codées).
- **Configuration :** `VITE_API_URL` (front) ; en production la valeur est vide → appels relatifs vers le même domaine.

---

## 💻 Installation locale

### Prérequis
- Python **3.10+**
- Node.js **18+**
- (Optionnel) Ollama pour les recommandations générées par LLM

### 1. Backend

```powershell
# depuis la racine du projet
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt

# lancer le serveur
.venv\Scripts\uvicorn.exe backend.main:app --reload --port 8000
```

### 2. Frontend

```powershell
cd frontend
npm install
npm run dev        # http://localhost:5173  (API auto-proxy vers localhost:8000)
```

### 3. Version de production (une seule URL)

```powershell
cd frontend
npm run build      # génère frontend/dist

# le backend sert automatiquement frontend/dist sur /
cd ..
.venv\Scripts\uvicorn.exe backend.main:app --port 8000
# → http://localhost:8000
```

---

## 🕹️ Utilisation

### Premier usage (démo)

1. **Accueil** → *Espace Médecin* → **Inscription** (nom, e-mail, mot de passe) → **Connexion**.
2. **Ajouter un patient** → saisir les 19 variables (et éventuellement l'ordonnance) → **Calculer le score et enregistrer**.
3. L'application **crée automatiquement le compte patient** et affiche la carte :
   > 📧 `prenom.nom@cardiorisk.ai` — mot de passe `0000`
4. **Espace Patient** → connexion avec cet e-mail et `0000` → **changement de mot de passe obligatoire** à la première connexion.
5. Après le changement, un **message de confirmation contenant le nouveau mot de passe** arrive dans la messagerie du patient.
6. **Messagerie / Bilans** : échange patient ↔ médecin (isolation stricte par `doctor_id`).

### Comptes de test

| Rôle | Identifiant | Mot de passe |
|---|---|---|
| Médecin | créé à l'inscription | au choix |
| Patient | `prenom.nom@cardiorisk.ai` | `0000` (puis changé) |

> Les données sont stockées dans `data/*.json`. Pour repartir de zéro (avant une vidéo/démo), remplacez leur contenu par `[]` et videz `data/uploads/`.

---

## 🔌 API

Base : `/api` — toutes les réponses en JSON.

### Santé
| Méthode | Endpoint | Description |
|---|---|---|
| GET | `/api/health` | Vérification de l'état du service |

### Médecins
| Méthode | Endpoint | Description |
|---|---|---|
| POST | `/api/doctors/signup` | Inscription médecin |
| POST | `/api/doctors/login` | Connexion médecin |
| PUT | `/api/doctors/profile` | Mise à jour du profil / mot de passe |
| GET | `/api/doctors/list` | Liste des médecins (pour l'espace patient) |

### Prédiction & rapports
| Méthode | Endpoint | Description |
|---|---|---|
| POST | `/api/predict` | Score de risque (19 variables) |
| POST | `/api/report` | Score + rapport IA (alertes, recommandations) |

### Patients (côté médecin)
| Méthode | Endpoint | Description |
|---|---|---|
| POST | `/api/patients` | Enregistrer un patient (+ création auto du compte) |
| GET | `/api/patients?doctor_id=` | Liste des patients du médecin |
| GET | `/api/patients/critical?doctor_id=` | Patients à risque élevé |
| GET | `/api/patients/{id}` | Détail d'un patient |
| GET | `/api/patients/{id}/auth` | Identifiants du compte patient généré |
| POST | `/api/patients/import?doctor_id=` | Import Excel |
| GET | `/api/patients/export?doctor_id=` | Export Excel |

### Espace patient
| Méthode | Endpoint | Description |
|---|---|---|
| POST | `/api/patient/signup` | Inscription patient (liée à un médecin) |
| POST | `/api/patient/login` | Connexion patient |
| GET | `/api/patient/me` | Profil patient connecté |
| POST | `/api/patient/change-password` | Changement mot de passe (+ notification en messagerie) |

### Messagerie & bilans
| Méthode | Endpoint | Description |
|---|---|---|
| POST | `/api/messages` | Envoyer un message (contrôle d'appartenance) |
| GET | `/api/messages?patient_auth_id=&doctor_id=` | Messages du patient |
| GET | `/api/messages/doctor?doctor_id=` | Toutes les conversations du médecin |
| POST | `/api/bilans/upload` | Déposer un bilan (PDF/JPG) |
| GET | `/api/bilans?patient_auth_id=&doctor_id=` | Liste des bilans |
| GET | `/api/bilans/download/{id}` | Télécharger un bilan |
| POST | `/api/bilans/{id}/reply` | Réponse du médecin sur un bilan |

> **Sécurité d'isolation :** chaque endpoint patient/messages vérifie que `patient.doctor_id == doctor_id` appelant ; sinon `403 Forbidden`.

---

## 🚀 Déploiement

### Render (actif, gratuit, sans carte bancaire)

- **URL :** [https://cardiorisk-demo.onrender.com](https://cardiorisk-demo.onrender.com)
- **Repo GitHub :** [ismailmakhloufi/cardiorisk-demo](https://github.com/ismailmakhloufi/cardiorisk-demo)
- Déploiement **automatique à chaque push** sur `main`.
- Le backend sert le frontend construit (`frontend/dist`) → **une seule URL**.
- Plan gratuit : mise en veille après ~15 min d'inactivité (réveil ~30-50 s).

**Commande de build :**

```powershell
npm run build --prefix frontend
```

**Commande de démarrage :**

```powershell
uvicorn backend.main:app --host 0.0.0.0 --port $PORT
```

### Docker

```powershell
docker build -t cardiorisk .
docker run -p 8000:8000 cardiorisk
```

> Les fichiers `Dockerfile` et `fly.toml` sont conservés pour un déploiement alternatif sur Fly.io.

---

## 🎬 Démonstration (script vidéo)

1. **Accueil** — présentation du titre et des deux espaces.
2. **Inscription médecin** → tableau de bord vide (données neuves).
3. **Ajout patient** — saisie des 19 variables → *Calculer le score* → score + alertes + recommandations + **carte e-mail patient**.
4. **Connexion patient** (`prenom.nom@cardiorisk.ai` / `0000`) → **changement de mot de passe forcé** → message de confirmation reçu.
5. **Messagerie** patient ↔ médecin, **dépôt d'un bilan**, réponse du médecin.
6. **Dashboard médecin** — patients à risque, historique, export Excel.
7. **Générer rapport IA** — rapport complet téléchargeable.

---

## 📁 Structure du dépôt

```
stage/
├── backend/
│   └── main.py                 # API FastAPI (25 endpoints), auth, isolation, service du frontend
├── frontend/
│   ├── src/
│   │   ├── main.jsx            # Application React (espace médecin + espace patient)
│   │   └── styles.css
│   └── dist/                   # Build de production (servi par le backend)
├── models/
│   └── modele_19vars_dose120contrainte/   # Modèle Ridge 19 variables (joblib)
├── src/
│   ├── generer_rapport_patient.py         # Prédiction, alertes, recommandations, RAG
│   ├── modele_ridge_final.py              # Entraînement / évaluation du modèle
│   └── constrained_logistic.py            # Contraintes métier sur les coefficients
├── data/                       # Persistance JSON (patients, médecins, messages, bilans)
│   ├── patients.json
│   ├── doctors.json
│   ├── patients_auth.json
│   ├── messages.json
│   ├── bilans.json
│   └── uploads/                # Bilans déposés (PDF/JPG)
├── rapport/                    # Rapport de stage (LaTeX)
├── requirements.txt
├── Dockerfile
├── fly.toml
└── README.md                   # ← ce fichier
```

---

## 📝 Notes

- Tout le code et l'interface sont **en français**.
- Libellés affichés : **BNP** (variable `ProBNP`) et **dose lasilix** (variable `dose_lasilix_sup120`).
- Les données patients sont **anonymisées** : aucun identifiant nominal n'est stocké côté modèle.
- Projet réalisé dans le cadre d'un **stage de recherche** — l'application est destinée à la **démonstration / compétition**, pas à un usage clinique réel.

---

**CardioRisk AI** — Anticipez la décompensation, personnalisez la prise en charge, transformez le suivi cardiaque. 💙
