"""
Pipeline patient -> Ridge -> RAG -> LLM Ollama -> rapport clinique.
"""
import json
from pathlib import Path
from urllib.request import Request, urlopen

import chromadb
import joblib
import numpy as np
from sentence_transformers import SentenceTransformer

PROJECT_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = PROJECT_DIR / "models" / "modele_19vars_dose120contrainte" / "modele_final_19vars.joblib"
CHROMA_DIR = PROJECT_DIR / "data" / "rag_index"
REPORT_DIR = PROJECT_DIR / "reports" / "rapports_patients"
REPORT_DIR.mkdir(parents=True, exist_ok=True)

COLLECTION_NAME = "guidelines_ic"
EMBEDDING_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
OLLAMA_MODEL = "llama3.1"
OLLAMA_URL = "http://localhost:11434/api/generate"

VARIABLE_LABELS = {
    "espace_PR": "Espace PR (ms)",
    "cause_valvulaire": "Cause valvulaire",
    "PAD": "Pression arterielle diastolique (mmHg)",
    "PAS": "Pression arterielle systolique (mmHg)",
    "OG": "Oreillette gauche (mm)",
    "Uree": "Uree (mmol/L)",
    "statine": "Statine",
    "ATCD_d_hospitalisation": "Antecedent d'hospitalisation",
    "IMC": "Indice de masse corporelle",
    "IEC_dose": "Dose d'IEC",
    "FQ_ECG_sortie": "Frequence cardiaque de sortie (bpm)",
    "HTAP": "HTAP",
    "TP": "Taux de prothrombine (%)",
    "lymphocyte": "Lymphocytes (G/L)",
    "ARM": "ARM",
    "QT_corrige": "QT corrige (ms)",
    "Glycemie_a_jeun": "Glycemie a jeun (g/L)",
    "ProBNP": "BNP (pg/mL)",
    "dose_lasilix_sup120": "Dose Lasilix >=120 mg/j",
    "dose_de_lasilix": "Dose de Lasilix",
    "Hb": "Hemoglobine (g/dL)",
    "FEVG": "Fraction d'ejection ventriculaire gauche (%)",
    "creat": "Creatinine (umol/L)",
    "cause_ischémique": "Cause ischemique",
    "QT_corrige_sup_450": "QT corrige > 450 ms",
    "lasilix": "Lasilix",
    "betabloquants": "Betabloquant",
    "dose_BB_pourcentage": "Dose betabloquant (%)",
    "plavix_cardiocine": "Antiagregant plaquettaire",
    "sintrome_aod": "Anticoagulation",
    "plavix": "Plavix",
    "cardiocine100": "Cardiocine 100",
    "sintrom": "Sintrom",
    "AOD": "AOD",
    "INH_SGLT2": "ISGLT2",
    "Ivabradine": "Ivabradine",
}


PATIENT_EXEMPLE = {
    "espace_PR": 160,
    "cause_valvulaire": 0,
    "PAD": 60,
    "PAS": 91,
    "OG": 50,
    "Uree": 8.5,
    "statine": 1,
    "ATCD_d_hospitalisation": 1,
    "IMC": 22.87,
    "IEC_dose": 12.5,
    "FQ_ECG_sortie": 72,
    "HTAP": 0,
    "TP": 75,
    "lymphocyte": 0.8,
    "ARM": 0,
    "QT_corrige": 519,
    "Glycemie_a_jeun": 1.1,
    "ProBNP": 2500,
    "dose_lasilix_sup120": 0,
}


def predict_patient(patient):
    bundle = joblib.load(MODEL_PATH)
    features = bundle["features"]
    patient_values = dict(patient)
    if "QT_corrige_sup_450" in features and "QT_corrige_sup_450" not in patient_values:
        patient_values["QT_corrige_sup_450"] = 1 if patient_values.get("QT_corrige", 0) > 450 else 0
    if "dose_lasilix_sup120" in features and "dose_lasilix_sup120" not in patient_values:
        dose = patient_values.get("dose_de_lasilix", patient_values.get("dose_lasilix", 0)) or 0
        try:
            dose = float(dose)
        except:
            dose = 0
        patient_values["dose_lasilix_sup120"] = 1 if dose >= 120 else 0
    values = np.array([[patient_values[f] for f in features]], dtype=float)
    scaled = bundle["scaler"].transform(bundle["imputer"].transform(values))
    model = bundle["model"]
    proba = float(model.predict_proba(scaled)[0, 1])
    score = float(model.decision_function(scaled)[0])
    threshold = float(bundle["threshold"])

    coefs = model.coef_[0]
    contributions = []
    for i, feature in enumerate(features):
        contributions.append({
            "variable": feature,
            "valeur": float(patient_values[feature]),
            "coefficient": float(coefs[i]),
            "contribution": float(coefs[i] * scaled[0, i]),
        })
    contributions.sort(key=lambda x: abs(x["contribution"]), reverse=True)
    return {
        "features": features,
        "proba": proba,
        "score": score,
        "threshold": threshold,
        "classe": "a risque" if proba >= threshold else "faible risque",
        "niveau_risque": risk_level(proba),
        "contributions": contributions,
    }


def build_rag_query(patient, prediction):
    top = prediction["contributions"][:6]
    vars_txt = ", ".join([f"{x['variable']}={x['valeur']}" for x in top])
    dose_val = patient.get('dose_de_lasilix', 0) or 0
    try:
        dose_f = float(dose_val)
    except:
        dose_f = 0
    sup120 = patient.get('dose_lasilix_sup120', 1 if dose_f >= 120 else 0)
    ordonnance_txt = (
        f"ordonnance: Lasilix={patient.get('lasilix', 'non renseigne')} dose={dose_val} (sup120={sup120}), "
        f"betabloquant={patient.get('betabloquants', 'non renseigne')} dose={patient.get('dose_BB_pourcentage', 'non renseignee')}, "
        f"IEC_dose={patient.get('IEC_dose', 'non renseignee')}, ARM={patient.get('ARM', 'non renseigne')}, "
        f"ISGLT2={patient.get('INH_SGLT2', 'non renseigne')}, Ivabradine={patient.get('Ivabradine', 'non renseigne')}."
    )
    bio_txt = f"Biologie: ProBNP={patient.get('ProBNP','non renseigne')} pg/mL, Uree={patient.get('Uree','non renseignee')}, TP={patient.get('TP','non renseigne')}%, Glycemie={patient.get('Glycemie_a_jeun','non renseignee')} g/L, HTAP={patient.get('HTAP','non renseigne')}, espace_PR={patient.get('espace_PR','non renseigne')} ms."
    return (
        "Patient insuffisant cardiaque avec risque de rehospitalisation a 3 mois. "
        f"Score de risque {prediction['proba']:.1%}, niveau {prediction['niveau_risque']}. "
        f"Variables principales: {vars_txt}. "
        f"{ordonnance_txt} {bio_txt} "
        "Besoin de recommandations sur sortie d'hospitalisation, suivi ambulatoire, "
        "optimisation therapeutique selon ordonnance, titration, ajout ou reevaluation de medicaments, fonction renale, FEVG, signes d'alerte, ProBNP et dose Lasilix >=120 mg."
    )


def clinical_flags(patient, prediction):
    flags = []
    add = flags.append

    if prediction["proba"] >= 0.40:
        add("Risque estime eleve selon le modele")
    elif prediction["proba"] >= prediction["threshold"]:
        add("Risque estime au-dessus du seuil de vigilance")
    else:
        add("Risque estime sous le seuil, a interpreter avec le contexte clinique")

    if patient.get("PAS", 999) < 100:
        add("PAS basse (<100 mmHg) : vigilance hemodynamique et tolerance therapeutique")
    if patient.get("OG", 0) >= 45:
        add("Oreillette gauche dilatee : marqueur de pressions de remplissage chroniquement elevees")
    if patient.get("FEVG", 100) < 40:
        add("FEVG reduite (<40%) : profil IC-FEr")
    if patient.get("creat", 0) >= 120:
        add("Creatinine elevee : surveillance fonction renale et equilibre decongestion/tolerance")
    if patient.get("QT_corrige", 0) > 450:
        add("QT corrige >450 ms : vigilance ECG, medicaments allongeant le QT et troubles ioniques")
    if patient.get("Hb", 99) < 12:
        add("Hemoglobine basse : rechercher anemie/carence martiale selon contexte")
    if patient.get("lymphocyte", 99) < 1:
        add("Lymphocytes bas : possible marqueur de fragilite ou inflammation")
    if patient.get("ProBNP", 0) >= 2000:
        add("ProBNP eleve (>=2000 pg/mL) : marqueur de surcharge et de pronostic, surveillance etroite")
    elif patient.get("ProBNP", 0) >= 1000:
        add("ProBNP modere (1000-2000 pg/mL) : a interpreter avec clinique et fonction renale")
    if patient.get("dose_lasilix_sup120", 0) == 1 or patient.get("dose_de_lasilix", 0) >= 120:
        add("Dose Lasilix >=120 mg/j : facteur de risque majeur de rehospitalisation (contrainte modele)")
    elif patient.get("dose_de_lasilix", 0) >= 80:
        add("Dose elevee de diuretique de l'anse : possible congestion ou dependance diuretique")
    if patient.get("HTAP", 0) == 1:
        add("HTAP presente : evaluer retentissement et optimisation therapeutique")
    if patient.get("espace_PR", 0) > 200:
        add("Espace PR allonge (>200 ms) : vigilance conduction auriculo-ventriculaire")
    if patient.get("Glycemie_a_jeun", 0) >= 1.26:
        add("Glycemie a jeun elevee (>=1.26 g/L) : depistage/equilibre diabete")
    if patient.get("Uree", 0) >= 10:
        add("Uree elevee (>=10 mmol/L) : surveillance fonction renale et hydratation")
    if patient.get("TP", 100) < 70:
        add("TP bas (<70%) : verifier anticoagulation et fonction hepatique")
    if patient.get("ARM", 0) == 0 and patient.get("FEVG", 100) < 40:
        add("ARM absent avec FEVG reduite : discuter introduction selon kaliemie et fonction renale")
    if patient.get("ATCD_d_hospitalisation", 0) >= 1:
        add("Antecedent d'hospitalisation : facteur de risque de nouvelle decompensation")
    if patient.get("lasilix") == 0 and (patient.get("OG", 0) >= 45 or prediction["proba"] >= prediction["threshold"]):
        add("Lasilix absent malgre profil congestif/risque : discuter indication selon signes cliniques de congestion")
    if patient.get("betabloquants") == 0 and patient.get("FEVG", 100) < 40 and patient.get("PAS", 999) >= 100:
        add("Betabloquant absent avec FEVG reduite : discuter introduction si patient stable et non congestif")
    if patient.get("betabloquants") == 1 and patient.get("dose_BB_pourcentage") not in (None, "") and patient.get("dose_BB_pourcentage", 0) < 50:
        add("Dose de betabloquant basse : discuter titration progressive selon frequence cardiaque, PAS et tolerance")
    if patient.get("IEC_dose", 0) <= 0 and patient.get("FEVG", 100) < 40 and patient.get("PAS", 999) >= 100:
        add("IEC non renseigne ou dose nulle avec FEVG reduite : discuter IEC/ARAII/ARNI selon tolerance et fonction renale")
    if patient.get("ARM") == 0 and patient.get("FEVG", 100) < 40:
        add("ARM absent avec FEVG reduite : discuter introduction selon kaliemie et fonction renale")
    if patient.get("INH_SGLT2") == 0 and patient.get("FEVG", 100) < 40:
        add("ISGLT2 absent avec FEVG reduite : discuter introduction selon contre-indications")
    if patient.get("Ivabradine") == 0 and patient.get("FQ_ECG_sortie", 0) >= 70 and patient.get("FEVG", 100) <= 35:
        add("Ivabradine absente avec FC elevee et FEVG basse : discuter si rythme sinusal et traitement de fond optimise")
    if patient.get("cause_ischémique", 0) == 1 and patient.get("plavix_cardiocine", 0) == 0:
        add("Etiologie ischemique sans antiagregant renseigne : verifier indication de prevention secondaire")
    if patient.get("sintrome_aod", 0) == 0 and patient.get("ACFA", 0) == 1:
        add("ACFA sans anticoagulation renseignee : verifier indication et contre-indications")
    if patient.get("cause_ischémique", 0) == 1:
        add("Etiologie ischemique : verifier prevention secondaire et optimisation coronarienne")
    if patient.get("cause_valvulaire", 0) == 1:
        add("Etiologie valvulaire : verifier severite et strategie de prise en charge")
    return flags


def risk_level(proba):
    if proba < 0.20:
        return "faible"
    if proba < 0.40:
        return "modere"
    if proba < 0.70:
        return "eleve"
    return "tres eleve"


def build_dashboard_recommendations(patient, prediction):
    alerts = []
    cards = []

    def add_alert(title, detail, priority):
        alerts.append({"title": title, "detail": detail, "priority": priority})

    def add_card(category, action, justification, priority):
        cards.append({"category": category, "action": action, "justification": justification, "priority": priority})

    if prediction["proba"] >= prediction["threshold"]:
        add_alert(
            "Risque de rehospitalisation au-dessus du seuil",
            f"Probabilite estimee {prediction['proba']:.1%}, niveau {risk_level(prediction['proba'])}.",
            "haute" if prediction["proba"] >= 0.40 else "moderee",
        )

    if patient.get("PAS", 999) < 100:
        add_alert("PAS basse", f"PAS={patient['PAS']} mmHg : verifier tolerance hemodynamique.", "haute")
        add_card("Optimisation therapeutique", "Verifier la tolerance des traitements de fond avant toute intensification.", "La PAS basse peut limiter la titration et impose une adaptation selon la clinique.", "haute")

    if patient.get("FEVG", 100) < 40:
        add_alert("FEVG reduite", f"FEVG={patient['FEVG']}% : profil IC-FEr.", "haute")
        add_card("Optimisation therapeutique", "Verifier que les traitements de fond de l'IC-FEr sont optimises selon tolerance.", "La FEVG reduite justifie une strategie structuree d'optimisation et de suivi.", "haute")

    if patient.get("OG", 0) >= 45:
        add_alert("OG dilatee", f"OG={patient['OG']} mm : pressions de remplissage probablement elevees.", "moderee")
        add_card("Gestion de la congestion", "Surveiller poids, dyspnee, oedemes et signes de congestion apres la sortie.", "Une OG dilatee peut traduire une charge chronique et un risque de decompensation.", "haute")

    if patient.get("creat", 0) >= 120:
        add_alert("Creatinine elevee", f"Creatinine={patient['creat']} umol/L : surveillance renale necessaire.", "haute")
        add_card("Biologie", "Programmer ionogramme, creatinine et fonction renale dans les 3 a 7 jours.", "La fonction renale influence la decongestion et la tolerance des traitements.", "haute")

    if patient.get("QT_corrige", 0) > 450:
        add_alert("QT corrige prolonge", f"QTc={patient['QT_corrige']} ms : vigilance ECG et troubles ioniques.", "moderee")
        add_card("Biologie", "Verifier kaliemie, magnesemie si disponible, et medicaments allongeant le QT.", "Un QTc >450 ms impose une vigilance rythmique et medicamenteuse.", "moderee")

    if patient.get("Hb", 99) < 12:
        add_alert("Hemoglobine basse", f"Hb={patient['Hb']} g/dL : rechercher anemie/carence martiale selon contexte.", "moderee")
        add_card("Biologie", "Discuter bilan d'anemie et statut martial selon disponibilite clinique.", "L'anemie peut aggraver fatigue, dyspnee et fragilite en insuffisance cardiaque.", "moderee")

    if patient.get("ProBNP", 0) >= 2000:
        add_alert("ProBNP eleve", f"ProBNP={patient['ProBNP']} pg/mL : surcharge et pronostic pejoratif.", "haute")
        add_card("Biologie", "Surveillance ProBNP serielle et optimisation decongestion selon clinique et fonction renale.", "ProBNP >=2000 est facteur predictif majeur (OR 1.39) et justifie un suivi rapproche.", "haute")
    elif patient.get("ProBNP", 0) >= 1000:
        add_alert("ProBNP modere", f"ProBNP={patient['ProBNP']} pg/mL : a interpreter avec clinique.", "moderee")
        add_card("Biologie", "Controler ProBNP et fonction renale a 7-14 jours selon evolution.", "ProBNP modere necessite une interpretation integree.", "moderee")

    if patient.get("dose_lasilix_sup120", 0) == 1 or patient.get("dose_de_lasilix", 0) >= 120:
        add_alert("Dose Lasilix >=120 mg", f"Dose={patient.get('dose_de_lasilix', patient.get('dose_lasilix_sup120'))} mg/j : risque majeur de rehospitalisation.", "haute")
        add_card("Gestion de la congestion", "Reevaluer indication et dose de Lasilix >=120 mg : verifier congestion residuelle, fonction renale, ionogramme et envisager strategie de decongestion alternative.", "Dose >=120 mg est facteur de risque contraint (OR 1.26) et marque une congestion/dependance diuretique.", "haute")
    elif patient.get("dose_de_lasilix", 0) >= 80:
        add_alert("Dose Lasilix elevee", f"Dose={patient['dose_de_lasilix']} mg/j : congestion ou dependance.", "moderee")
        add_card("Gestion de la congestion", "Reevaluer la dose de Lasilix selon poids, diurese, congestion residuelle.", "Dose elevee refleter congestion persistante.", "moderee")

    if patient.get("ATCD_d_hospitalisation", 0) >= 1:
        add_alert("Antecedent d'hospitalisation", "Antecedent present : risque de nouvelle decompensation.", "haute")
        add_card("Suivi rapproche", "Planifier un contact precoce apres sortie et une consultation rapprochee.", "Un antecedent d'hospitalisation augmente le risque de rehospitalisation precoce.", "haute")

    if patient.get("HTAP", 0) == 1:
        add_alert("HTAP", "HTAP presente : evaluer retentissement droit et optimisation.", "moderee")
        add_card("Suivi rapproche", "Discuter echographie et prise en charge HTAP selon etiologie.", "HTAP contribue au risque et necessite un suivi specialise.", "moderee")

    if patient.get("lasilix") == 0 and (patient.get("OG", 0) >= 45 or prediction["proba"] >= prediction["threshold"]):
        add_card("Gestion de la congestion", "Discuter l'indication d'un diuretique de l'anse si signes cliniques de congestion.", "L'ordonnance ne renseigne pas Lasilix alors que le profil impose une surveillance de congestion.", "haute")

    if patient.get("betabloquants") == 0 and patient.get("FEVG", 100) < 40 and patient.get("PAS", 999) >= 100:
        add_card("Optimisation therapeutique", "Discuter l'introduction d'un betabloquant si le patient est stable, euvolemique et tolerant.", "La FEVG reduite justifie les traitements de fond, avec titration prudente selon PAS et frequence cardiaque.", "haute")
    elif patient.get("betabloquants") == 1 and patient.get("dose_BB_pourcentage") not in (None, "") and patient.get("dose_BB_pourcentage", 0) < 50 and patient.get("PAS", 999) >= 100:
        add_card("Optimisation therapeutique", "Discuter une augmentation progressive du betabloquant selon FC, PAS et tolerance clinique.", "La dose renseignee est basse et peut etre optimisee si l'etat hemodynamique le permet.", "moderee")

    if patient.get("IEC_dose", 0) <= 0 and patient.get("FEVG", 100) < 40 and patient.get("PAS", 999) >= 100:
        add_card("Optimisation therapeutique", "Verifier l'indication d'un IEC, ARAII ou ARNI selon tolerance tensionnelle et fonction renale.", "Aucun IEC dose n'est renseigne alors que la FEVG est reduite.", "haute")
    elif patient.get("IEC_dose", 0) > 0 and patient.get("FEVG", 100) < 40 and patient.get("PAS", 999) >= 100:
        add_card("Optimisation therapeutique", "Discuter la titration prudente de l'IEC selon PAS, creatinine et kaliemie.", "La dose IEC est renseignee et peut etre ajustee uniquement si la tolerance biologique et tensionnelle est correcte.", "moderee")

    if patient.get("ARM") == 0 and patient.get("FEVG", 100) < 40:
        add_card("Optimisation therapeutique", "Discuter l'ajout d'un ARM selon kaliemie, creatinine et contre-indications.", "L'ARM fait partie de l'optimisation de l'IC-FEr mais depend de la tolerance renale et ionique.", "haute")

    if patient.get("INH_SGLT2") == 0 and patient.get("FEVG", 100) < 40:
        add_card("Optimisation therapeutique", "Discuter l'ajout d'un ISGLT2 selon contre-indications et contexte clinique.", "L'ordonnance ne renseigne pas d'ISGLT2 chez un patient avec FEVG reduite.", "haute")

    if patient.get("Ivabradine") == 0 and patient.get("FQ_ECG_sortie", 0) >= 70 and patient.get("FEVG", 100) <= 35:
        add_card("Optimisation therapeutique", "Discuter l'ivabradine si rythme sinusal, FC persistante elevee et traitement de fond optimise.", "La frequence de sortie reste elevee avec FEVG basse ; l'indication depend du rythme et de la tolerance.", "moderee")

    if patient.get("cause_ischémique", 0) == 1 and patient.get("plavix_cardiocine", 0) == 0:
        add_card("Prevention secondaire", "Verifier l'indication d'un antiagregant plaquettaire et d'une statine selon le contexte coronarien.", "L'etiologie ischemique impose de revoir la prevention secondaire, sans presumer d'une indication automatique.", "moderee")

    if patient.get("sintrome_aod", 0) == 0 and patient.get("ACFA", 0) == 1:
        add_card("Prevention thromboembolique", "Verifier l'indication d'une anticoagulation et les contre-indications.", "Une ACFA renseignee sans anticoagulation dans l'ordonnance necessite une verification clinique.", "haute")

    add_card("Education patient", "Renforcer poids quotidien, observance, regime hyposode adapte et signes d'alerte.", "L'education aide a detecter precocement la congestion et a reduire les recours non programmes.", "haute" if prediction["proba"] >= prediction["threshold"] else "moderee")

    next_steps = [
        "Contact infirmier ou appel de suivi dans les 7 a 14 jours." if prediction["proba"] >= prediction["threshold"] else "Suivi ambulatoire programme selon parcours habituel.",
        "Bilan biologique dans les 3 a 7 jours si creatinine elevee, traitement diuretique ou risque ionique.",
        "Surveillance quotidienne du poids, dyspnee, oedemes et tolerance tensionnelle.",
        "Consultation cardiologique rapprochee selon disponibilite et evolution clinique.",
    ]

    seen = set()
    unique_cards = []
    for card in cards:
        key = (card["category"], card["action"])
        if key not in seen:
            seen.add(key)
            unique_cards.append(card)

    return {
        "risk_level": risk_level(prediction["proba"]),
        "alerts": alerts,
        "recommendation_cards": unique_cards[:8],
        "next_steps": next_steps,
    }


def treatment_summary(patient):
    plavix_cardiocine = {0: "aucun", 1: "Plavix", 2: "Cardiocine 100"}.get(patient.get("plavix_cardiocine"), "non renseigne")
    sintrome_aod = {0: "aucun", 1: "Sintrom", 2: "AOD"}.get(patient.get("sintrome_aod"), "non renseigne")
    yes_no = {0: "non", 1: "oui"}
    rows = [
        f"- Lasilix: {yes_no.get(patient.get('lasilix'), 'non renseigne')} | dose: {patient.get('dose_de_lasilix', 'non renseignee')} mg/j",
        f"- Betabloquant: {yes_no.get(patient.get('betabloquants'), 'non renseigne')} | dose: {patient.get('dose_BB_pourcentage', 'non renseignee')}%",
        f"- IEC: dose {patient.get('IEC_dose', 'non renseignee')} mg/j",
        f"- Plavix/Cardiocine: {plavix_cardiocine}",
        f"- Sintrom/AOD: {sintrome_aod}",
        f"- ARM: {yes_no.get(patient.get('ARM'), 'non renseigne')}",
        f"- ISGLT2: {yes_no.get(patient.get('INH_SGLT2'), 'non renseigne')}",
        f"- Ivabradine: {yes_no.get(patient.get('Ivabradine'), 'non renseigne')}",
    ]
    return "\n".join(rows)


def retrieve_context(query, n_results=5):
    embedder = SentenceTransformer(EMBEDDING_MODEL)
    client = chromadb.PersistentClient(path=str(CHROMA_DIR))
    collection = client.get_collection(COLLECTION_NAME)
    query_embedding = embedder.encode([query], normalize_embeddings=True).tolist()[0]
    results = collection.query(query_embeddings=[query_embedding], n_results=n_results)

    chunks = []
    for doc, meta in zip(results["documents"][0], results["metadatas"][0]):
        chunks.append({"text": doc, "source": meta["source"], "topic": meta["topic"]})
    return chunks


def build_prompt(patient, prediction, context, recommendations):
    top_risk = [x for x in prediction["contributions"] if x["contribution"] > 0][:5]
    top_protective = [x for x in prediction["contributions"] if x["contribution"] < 0][:5]

    flags = clinical_flags(patient, prediction)
    patient_txt = "\n".join([f"- {VARIABLE_LABELS.get(k, k)} ({k}): {v}" for k, v in patient.items()])
    risk_txt = "\n".join([
        f"- {VARIABLE_LABELS.get(x['variable'], x['variable'])} ({x['variable']}={x['valeur']}): contribution modele +{x['contribution']:.3f}"
        for x in top_risk
    ])
    protective_txt = "\n".join([
        f"- {VARIABLE_LABELS.get(x['variable'], x['variable'])} ({x['variable']}={x['valeur']}): contribution modele {x['contribution']:.3f}"
        for x in top_protective
    ])
    flags_txt = "\n".join([f"- {flag}" for flag in flags])
    treatment_txt = treatment_summary(patient)
    alerts_txt = "\n".join([f"- [{a['priority']}] {a['title']}: {a['detail']}" for a in recommendations["alerts"]])
    cards_txt = "\n".join([
        f"- {c['category']} | Priorite: {c['priority']} | Action: {c['action']} | Justification: {c['justification']}"
        for c in recommendations["recommendation_cards"]
    ])
    next_steps_txt = "\n".join([f"- {step}" for step in recommendations["next_steps"]])
    context_txt = "\n\n".join([
        f"[{i+1}] {c['text']}\nSource: {c['source']} | Topic: {c['topic']}"
        for i, c in enumerate(context)
    ])

    return f"""
Tu es un assistant medical francophone integre dans un tableau de bord d'insuffisance cardiaque.
Ta mission est de produire un rapport court, coherent, directement exploitable par un cardiologue.
Le rapport doit surtout proposer des recommandations personnalisees de sortie et de suivi ambulatoire.

Contraintes strictes:
- Ne presente jamais le modele comme un diagnostic absolu ; c'est un outil d'aide a la decision.
- Ne transforme pas les contributions statistiques en causalite.
- Ne dis jamais "PAS elevee" si PAS <100 mmHg ; dire "PAS basse".
- Dis "creatinine" et jamais "creatines".
- Si QT_corrige >450 ms, le decrire comme facteur de vigilance ECG/rythmique, meme si sa contribution modele est faible.
- Ne recommande pas d'augmenter ou diminuer un traitement sans formule prudente : "a verifier", "a discuter", "selon tolerance", "selon bilan".
- Ne donne pas de posologie nouvelle.
- Base l'optimisation therapeutique sur l'ordonnance fournie : traitement present/absent et dose renseignee.
- Si un traitement est absent, proposer "a discuter" ou "verifier l'indication", jamais une prescription automatique.
- Si une dose est faible ou elevee, proposer une titration, reduction ou reevaluation uniquement "selon PAS, frequence cardiaque, congestion, creatinine, kaliemie et tolerance".
- Ne cite pas specifiquement les betabloquants si cette information n'est pas fournie ; dire plutot "traitements de fond de l'insuffisance cardiaque".
- OG est mesuree en mm, jamais en mmHg.
- Ecrire "PAS basse" si PAS <100 mmHg.
- Utilise le contexte RAG comme support des recommandations.
- Les recommandations calculees par regles cliniques sont prioritaires : les reformuler, ne pas les contredire.
- Commence directement par "## Resume executif". Ne dis pas "Voici le rapport".
- Les delais doivent rester prudents : suivi precoce 7 a 14 jours si risque eleve, bilan biologique 3 a 7 jours si fonction renale/ionogramme a surveiller. Ne propose 24-48h que si signe d'alerte clinique majeur.

DONNEES PATIENT:
{patient_txt}

ORDONNANCE ACTUELLE:
{treatment_txt}

RESULTAT DU MODELE:
- Probabilite estimee de rehospitalisation a 3 mois: {prediction['proba']:.1%}
- Formulation dashboard recommandee: score de risque {prediction['proba']:.1%}, niveau {prediction['niveau_risque']}
- Seuil utilise: {prediction['threshold']:.2f}
- Classification technique interne: {prediction['classe']}
- Score logistique: {prediction['score']:.3f}

FACTEURS QUI AUGMENTENT LE RISQUE SELON LE MODELE:
{risk_txt}

FACTEURS QUI DIMINUENT LE RISQUE SELON LE MODELE:
{protective_txt}

SIGNAUX CLINIQUES A CONSIDERER:
{flags_txt}

ALERTES CALCULEES PAR REGLES CLINIQUES:
{alerts_txt}

RECOMMANDATIONS CALCULEES PAR REGLES CLINIQUES:
{cards_txt}

PROCHAINES ETAPES CALCULEES:
{next_steps_txt}

CONTEXTE RAG:
{context_txt}

Structure obligatoire du rapport:

## Resume executif
- 2 a 4 phrases maximum.
- Dire exactement que le patient possede un score de risque de {prediction['proba']:.1%}, niveau {prediction['niveau_risque']}.
- Ne pas utiliser l'expression "classe a risque" dans le resume executif.

## Facteurs majeurs expliquant le risque
- 4 a 6 puces maximum.
- Pour chaque facteur : valeur, interpretation clinique, impact pratique.

## Recommandations personnalisees
Presenter 5 cartes courtes avec exactement ces titres si pertinentes:
- Optimisation therapeutique
- Gestion de la congestion
- Biologie
- Education patient
- Suivi rapproche
- Prevention secondaire
- Prevention thromboembolique
Pour chaque carte, respecter exactement ce format:
**Titre de la carte**
- Action: ...
- Justification: ...
- Priorite: haute/moderee/faible
Les cartes therapeutiques doivent dire clairement s'il faut discuter l'ajout d'un medicament, la titration, la diminution ou la reevaluation de dose, selon l'ordonnance actuelle.

## Alertes actives
- Lister uniquement les alertes presentes chez ce patient.
- Exemples : PAS basse, creatinine elevee, QT >450 ms, FEVG reduite, antecedent d'hospitalisation.

## Prochaines etapes
- Proposer un calendrier simple compatible tableau de bord : consultation/appel, bilan biologique, surveillance poids/symptomes.
- Ne pas inventer de date exacte.

## Limites
- 2 phrases maximum.

Style: professionnel, concis, sans remplissage, adapte a une interface clinique.
""".strip()


def call_ollama(prompt):
    payload = {
        "model": OLLAMA_MODEL,
        "prompt": prompt,
        "stream": False,
        "options": {"temperature": 0.1, "top_p": 0.85},
    }
    req = Request(
        OLLAMA_URL,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urlopen(req, timeout=300) as response:
        data = json.loads(response.read().decode("utf-8"))
    report = data["response"].strip()
    start = report.find("## Resume executif")
    if start != -1:
        report = report[start:].strip()
    return report


def main():
    patient = PATIENT_EXEMPLE
    prediction = predict_patient(patient)
    recommendations = build_dashboard_recommendations(patient, prediction)
    query = build_rag_query(patient, prediction)
    context = retrieve_context(query)
    prompt = build_prompt(patient, prediction, context, recommendations)
    report = call_ollama(prompt)

    output = {
        "patient": patient,
        "prediction": {
            "proba": prediction["proba"],
            "threshold": prediction["threshold"],
            "classe": prediction["classe"],
            "score": prediction["score"],
        },
        "rag_context": context,
        "dashboard": recommendations,
        "rapport": report,
    }

    (REPORT_DIR / "rapport_patient_exemple.md").write_text(report, encoding="utf-8")
    (REPORT_DIR / "rapport_patient_exemple.json").write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")

    print("=" * 80)
    print("RAPPORT PATIENT GENERE")
    print("=" * 80)
    print(f"Probabilite: {prediction['proba']:.1%}")
    print(f"Classe: {prediction['classe']}")
    print(f"Rapport: {REPORT_DIR / 'rapport_patient_exemple.md'}")
    print("\n" + report.encode("cp1252", errors="replace").decode("cp1252"))


if __name__ == "__main__":
    main()
